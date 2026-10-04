//! KV storage shared by chunked prefill and single-token decoding stacks.
use super::*;

pub(crate) struct KvCache {
    pub capacity: usize,
    pub layers: Vec<(hrx::Buffer, hrx::Buffer)>,
}
impl KvCache {
    pub fn new(
        stream: &mut hrx::Stream,
        layers: usize,
        context: usize,
        width: usize,
    ) -> Result<Arc<Self>> {
        let capacity = context.div_ceil(16) * 16 + 16;
        let mut buffers = Vec::new();
        for _ in 0..layers {
            buffers.push((
                stream.allocate_zeroed(capacity * width * 2)?,
                stream.allocate_zeroed(capacity * width * 2)?,
            ));
        }
        Ok(Arc::new(Self {
            capacity,
            layers: buffers,
        }))
    }
}
pub(super) struct CachedAttention {
    cache: Arc<KvCache>,
    attention: crate::compile::Kernel,
    copy: crate::compile::Kernel,
    offset: usize,
}
impl Stack {
    pub(crate) fn use_kv_cache(
        &mut self,
        c: &Compiler,
        stream: &mut hrx::Stream,
        cache: Arc<KvCache>,
    ) -> Result<()> {
        if !self.d.causal
            || self.d.head_dim != 128
            || self.d.heads != self.d.kv_heads * 8
            || cache.layers.len() != self.layers
        {
            return err("cached attention requires Qwen GQA dimensions");
        }
        let ns = "h3.attention_qwen_cached.";
        let cfg = vec![
            (format!("{ns}q_stride"), self.d.inner().to_string()),
            (format!("{ns}kv_stride"), self.d.kv_inner().to_string()),
            (format!("{ns}out_stride"), self.attn_width.to_string()),
            (format!("{ns}tokens"), self.tokens.to_string()),
            (format!("{ns}token_capacity"), self.capacity.to_string()),
            (format!("{ns}kv_capacity"), cache.capacity.to_string()),
            (
                format!("{ns}scale"),
                num(1.0 / (self.d.head_dim as f64).sqrt()),
            ),
        ];
        self.causal_cache = Some(CachedAttention {
            attention: c.get(
                stream,
                "attention_qwen_cached",
                "h3_attention_qwen_cached",
                &cfg,
            )?,
            copy: c.get(stream, "copy_u16", "h3_copy_u16", &vec![])?,
            cache,
            offset: 0,
        });
        Ok(())
    }
    pub(crate) fn cache_offset(&mut self, offset: usize) -> Result<()> {
        let c = self
            .causal_cache
            .as_mut()
            .ok_or_else(|| crate::compile::Error::Io("missing KV cache".into()))?;
        if offset
            .checked_add(self.tokens)
            .is_none_or(|n| n > c.cache.capacity - 16)
        {
            return err("KV cache context exceeded");
        }
        c.offset = offset;
        Ok(())
    }
    pub(super) fn attend_cached<'g>(
        &'g self,
        sink: &mut Sink<'_, 'g>,
        prof: &mut Profile,
        layer: usize,
        q: View<'g>,
        k: View<'g>,
        v: View<'g>,
    ) -> Result<()> {
        let c = self.causal_cache.as_ref().expect("checked by caller");
        let width = self.d.kv_inner();
        let count = self.tokens * width;
        let (keys, values) = &c.cache.layers[layer];
        for (src, dst) in [(k, keys), (v, values)] {
            emit(
                sink,
                &c.copy,
                Some(prof),
                "KV append",
                [(count as u32).div_ceil(256), 1, 1],
                [256, 1, 1],
                &[count as u32],
                &[src, dst.slice(c.offset * width * 2, count * 2)],
                &[count * 2, count * 2],
            )?;
        }
        emit(
            sink,
            &c.attention,
            Some(prof),
            "cached attention",
            [(self.tokens as u32).div_ceil(16), self.d.kv_heads as u32, 1],
            [256, 1, 1],
            &[c.offset as u32],
            &[q, keys.binding(), values.binding(), self.attn.binding()],
            &[
                self.capacity * self.d.inner() * 2,
                c.cache.capacity * width * 2,
                c.cache.capacity * width * 2,
                self.tokens * self.attn_width * 2,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires gfx1151 and provisioned HRX; synthetic weights only"]
    fn cached_blocks_match_full_prefix_and_share_weights() {
        use crate::weights::rows_of;
        use safetensors::tensor::{serialize, Dtype, TensorView};
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blocks.safetensors");
        let mut tensors = Vec::new();
        for layer in 0..2 {
            for (name, rows, cols) in [
                ("qkv.q", 1280, 256),
                ("out.q", 256, 1024),
                ("gu.q", 1024, 256),
                ("down.q", 256, 512),
            ] {
                let data: Vec<_> = (0..rows * cols)
                    .flat_map(|i| {
                        half::bf16::from_f32(((i * 17 + layer * 7) % 101) as f32 / 5000. - 0.01)
                            .to_le_bytes()
                    })
                    .collect();
                tensors.push((
                    format!("blocks.{layer}.{name}"),
                    Dtype::BF16,
                    vec![rows, cols],
                    data,
                ));
            }
            for (name, size) in [
                ("norm1", 256),
                ("norm2", 256),
                ("qnorm", 128),
                ("knorm", 128),
            ] {
                tensors.push((
                    format!("blocks.{layer}.{name}"),
                    Dtype::F32,
                    vec![size],
                    bytemuck::cast_slice(&vec![1f32; size]).to_vec(),
                ));
            }
        }
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        std::fs::write(&path, serialize(views, None).unwrap()).unwrap();
        // Safety: this temporary checkpoint is immutable until all owners drop.
        let weights = unsafe {
            Weights::open(&path, |ck, out| {
                for (name, e) in ck.entries() {
                    let pitch = if e.shape.len() == 2 {
                        gemm_pitch(e.shape[1], 16) * 2
                    } else {
                        0
                    };
                    out.insert(name.clone(), rows_of(ck, &[name], pitch)?);
                }
                Ok(())
            })
        }
        .unwrap();
        let mut stream = hrx::Stream::open().unwrap();
        let c = Compiler::new(None, std::path::PathBuf::new());
        let constants = Constants::new(&mut stream).unwrap();
        let d = StackDims {
            hidden: 256,
            heads: 8,
            kv_heads: 1,
            head_dim: 128,
            ffn: 512,
            rope_dim: 128,
            classes: 1,
            wbits: 16,
            eps: 1e-6,
            bias: false,
            gate_first: true,
            causal: true,
            attn_i4: false,
            attn_qk_bits: 16,
            bf16: true,
        };
        let make = |stream: &mut hrx::Stream, n| {
            Stack::new(
                &c,
                stream,
                d.clone(),
                n,
                2,
                &weights,
                |i| format!("blocks.{i}."),
                true,
                constants.ones.clone(),
                "local-test",
            )
            .unwrap()
        };
        let n = 19;
        let input: Vec<f32> = (0..n * 256)
            .map(|i| (i as f32 * 0.03).sin() * 0.2)
            .collect();
        let mut full = make(&mut stream, n);
        let mut run = |stack: &mut Stack, offset: usize, count: usize| {
            let x = stream.allocate_zeroed(stack.capacity() * 256 * 4).unwrap();
            stream
                .upload_at(
                    &x,
                    0,
                    bytemuck::cast_slice(&input[offset * 256..(offset + count) * 256]),
                )
                .unwrap();
            let cls = crate::dispatch::Classes::zeroed(&mut stream, stack.capacity()).unwrap();
            let pos: Vec<_> = (offset..offset + count)
                .flat_map(|i| [i as f64; 3])
                .collect();
            let mut co = vec![0.; count * 64];
            let mut si = co.clone();
            crate::rope::te(&pos, &mut co, &mut si);
            let cos = stream.allocate(co.len() * 4).unwrap();
            let sin = stream.allocate(si.len() * 4).unwrap();
            stream
                .upload(cos.binding(), bytemuck::cast_slice(&co))
                .unwrap();
            stream
                .upload(sin.binding(), bytemuck::cast_slice(&si))
                .unwrap();
            let cond = constants.identity();
            let cond_fn = |_| LayerCond { ..cond };
            stack
                .forward(
                    &mut stream,
                    &mut Profile::default(),
                    x.binding(),
                    cls.all(),
                    cos.binding(),
                    sin.binding(),
                    &cond_fn,
                    0,
                    None,
                )
                .unwrap();
            let bytes = stream
                .read(x.slice(0, count * 256 * 4))
                .unwrap()
                .wait(&mut stream)
                .unwrap();
            bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
        };
        let expected = run(&mut full, 0, n);

        let cache = KvCache::new(&mut stream, 2, n, 128).unwrap();
        let mut actual = Vec::new();
        let mut offset = 0;
        for count in [15, 1, 3] {
            let mut stack = make(&mut stream, count);
            stack.use_kv_cache(&c, &mut stream, cache.clone()).unwrap();
            stack.cache_offset(offset).unwrap();
            assert!(Arc::ptr_eq(&full.block(0).qkv_q, &stack.block(0).qkv_q));
            let x = stream.allocate_zeroed(stack.capacity() * 256 * 4).unwrap();
            stream
                .upload_at(
                    &x,
                    0,
                    bytemuck::cast_slice(&input[offset * 256..(offset + count) * 256]),
                )
                .unwrap();
            let cls = crate::dispatch::Classes::zeroed(&mut stream, stack.capacity()).unwrap();
            let pos: Vec<_> = (offset..offset + count)
                .flat_map(|i| [i as f64; 3])
                .collect();
            let mut co = vec![0.; count * 64];
            let mut si = co.clone();
            crate::rope::te(&pos, &mut co, &mut si);
            let cos = stream.allocate(co.len() * 4).unwrap();
            let sin = stream.allocate(si.len() * 4).unwrap();
            stream
                .upload(cos.binding(), bytemuck::cast_slice(&co))
                .unwrap();
            stream
                .upload(sin.binding(), bytemuck::cast_slice(&si))
                .unwrap();
            let cond = constants.identity();
            let cond_fn = |_| LayerCond { ..cond };
            stack
                .forward(
                    &mut stream,
                    &mut Profile::default(),
                    x.binding(),
                    cls.all(),
                    cos.binding(),
                    sin.binding(),
                    &cond_fn,
                    0,
                    None,
                )
                .unwrap();
            let bytes = stream
                .read(x.slice(0, count * 256 * 4))
                .unwrap()
                .wait(&mut stream)
                .unwrap();
            actual.extend_from_slice(bytemuck::cast_slice::<u8, f32>(&bytes));
            offset += count;
        }
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 0.003, "{a} vs {b}");
        }
    }
}
