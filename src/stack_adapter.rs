//! Experimental BF16 low-rank branches alongside the original quantized base.
//! Base projections stop before activation/gating, then the two paths are combined.
use super::*;

struct Projection {
    prepare: Prepare,
    rank_prepare: Prepare,
    down: Gemm,
    up: Gemm,
    base: Gemm,
    finish: crate::compile::Kernel,
    output: usize,
    result_width: usize,
    mode: usize,
    weights: Vec<(hrx::Buffer, hrx::Buffer)>,
}

pub(super) struct AdapterRuntime {
    projections: Vec<Projection>,
    input: Option<hrx::Buffer>,
    rank: hrx::Buffer,
    rank_operand: hrx::Buffer,
    base: hrx::Buffer,
    delta: hrx::Buffer,
    classes: usize,
}

#[allow(clippy::too_many_arguments)]
fn matrix(
    stream: &mut hrx::Stream,
    adapter: &crate::adapter::Adapter,
    name: &str,
    rows: usize,
    cols: usize,
    padded_rows: usize,
    padded_cols: usize,
    scale: f32,
    interleave: bool,
) -> Result<hrx::Buffer> {
    let entry = adapter
        .checkpoint()
        .at(name)
        .map_err(|e| crate::compile::Error::Io(e.to_string()))?;
    let source = adapter.checkpoint().bytes(entry);
    let mut bytes = vec![0u8; padded_rows * padded_cols * 2];
    for row in 0..rows {
        let src_row = if interleave {
            (row / 32) * 16 + row % 16 + if row % 32 >= 16 { rows / 2 } else { 0 }
        } else {
            row
        };
        for col in 0..cols {
            let at = (src_row * cols + col) * 2;
            let value = half::bf16::from_le_bytes([source[at], source[at + 1]]).to_f32() * scale;
            if !value.is_finite() {
                return err(format!("non-finite adapter weight in {name}"));
            }
            let target = (row * padded_cols + col) * 2;
            bytes[target..target + 2].copy_from_slice(&half::bf16::from_f32(value).to_le_bytes());
        }
    }
    let out = stream.allocate(bytes.len())?;
    stream.upload(out.binding(), &bytes)?;
    adapter.checkpoint().done_with(source);
    Ok(out)
}

impl AdapterRuntime {
    pub(super) fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
        stack: &Stack,
        adapter: &crate::adapter::Adapter,
        refiner: bool,
    ) -> Result<Self> {
        let d = &stack.d;
        if d.hidden != HID || d.heads != HEADS || d.ffn != FFN || d.bias || !d.gate_first {
            return err("Turbo adapters require the unbiased H3 DiT/refiner stack");
        }
        let elem = if d.wbits == 8 {
            "i8"
        } else if d.bf16 {
            "bf16"
        } else {
            return err("unsupported Turbo base weight precision");
        };
        let mut projections = Vec::new();
        for (op, input, output, mode, rank) in [
            ("attn.qkv_proj", HID, QKV, 0, 384usize),
            ("attn.out_proj", INNER, HID, 2, 128),
            ("mlp.fc1", HID, 2 * FFN, 1, 128),
            ("mlp.fc2", FFN, HID, 2, 128),
        ] {
            // Prepare works on multiples of 256; padding the rank adds exact zero terms.
            let rank_width = rank.div_ceil(256) * 256;
            let ip = gemm_pitch(input, 16);
            let rp = gemm_pitch(rank_width, 16);
            let result_width = if mode == 1 { output / 2 } else { output };
            let mut weights = Vec::new();
            for layer in 0..stack.layers {
                let group = if refiner {
                    "token_refiner.blocks"
                } else {
                    "blocks"
                };
                let prefix = format!("diffusion_model.{group}.{layer}.{op}");
                let spec = adapter
                    .projections
                    .iter()
                    .find(|s| s.prefix == prefix)
                    .ok_or_else(|| {
                        crate::compile::Error::Io(format!("adapter missing {prefix}"))
                    })?;
                weights.push((
                    matrix(
                        stream,
                        adapter,
                        &format!("{prefix}.lora_A.weight"),
                        rank,
                        input,
                        rank_width,
                        ip,
                        1.0,
                        false,
                    )?,
                    matrix(
                        stream,
                        adapter,
                        &format!("{prefix}.lora_B.weight"),
                        output,
                        rank,
                        output,
                        rp,
                        spec.scale,
                        mode == 1,
                    )?,
                ));
            }
            let finish = c.get(
                stream,
                "adapter_finish",
                "h3_adapter_finish",
                &vec![
                    ("h3.adapter_finish.width".into(), result_width.to_string()),
                    ("h3.adapter_finish.mode".into(), mode.to_string()),
                    ("h3.adapter_finish.classes".into(), d.classes.to_string()),
                ],
            )?;
            projections.push(Projection {
                prepare: Prepare::build(
                    c,
                    stream,
                    if op == "attn.qkv_proj" || op == "mlp.fc1" {
                        "norm"
                    } else {
                        "plain"
                    },
                    "bf16",
                    input,
                    d.eps,
                    d.classes,
                    ip,
                )?,
                rank_prepare: Prepare::build(c, stream, "plain", "bf16", rank_width, d.eps, 1, rp)?,
                down: Gemm::build(
                    c,
                    stream,
                    "plain",
                    "bf16",
                    false,
                    true,
                    input,
                    rank_width,
                    stack.tokens,
                    d.classes,
                    ip,
                    Tile::Plain,
                    0,
                )?,
                up: Gemm::build(
                    c,
                    stream,
                    "plain",
                    "bf16",
                    false,
                    true,
                    rank_width,
                    output,
                    stack.tokens,
                    d.classes,
                    rp,
                    Tile::Plain,
                    0,
                )?,
                base: Gemm::build(
                    c,
                    stream,
                    "f32",
                    elem,
                    false,
                    true,
                    input,
                    output,
                    stack.tokens,
                    d.classes,
                    gemm_pitch(input, d.wbits),
                    Tile::Plain,
                    0,
                )?,
                finish,
                output,
                result_width,
                mode,
                weights,
            });
        }
        let cap = stack.capacity;
        let result = Self {
            projections,
            // The original BF16 input is dead after A*x. The base GEMM writes
            // later in the same dependency chain, so its FP32 scratch can own it.
            input: if env_once("H3_REUSE_SCRATCH") == Some("1") {
                None
            } else {
                Some(stream.allocate(cap * gemm_pitch(FFN, 16) * 2)?)
            },
            rank: stream.allocate(cap * 512 * 2)?,
            rank_operand: stream.allocate(cap * gemm_pitch(512, 16) * 2)?,
            base: stream.allocate(cap * 2 * FFN * 4)?,
            delta: stream.allocate(cap * 2 * FFN * 2)?,
            classes: d.classes,
        };
        c.flush(stream)?;
        Ok(result)
    }

    fn input_view(&self) -> View<'_> {
        self.input.as_ref().unwrap_or(&self.base).binding()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit<'g>(
        &'g self,
        sink: &mut Sink<'_, 'g>,
        prof: &mut Profile,
        projection: usize,
        layer: usize,
        tokens: u32,
        original: View<'g>,
        norm: Option<(View<'g>, View<'g>, ClassRows<'g>)>,
        base_input: View<'g>,
        base_weight: View<'g>,
        scales: Option<(View<'g>, View<'g>)>,
        output: View<'g>,
        residual: Option<(View<'g>, ClassRows<'g>)>,
    ) -> Result<()> {
        let p = &self.projections[projection];
        let (a, b) = &p.weights[layer];
        let (base_stage, rank_stage, up_stage) = match projection {
            0 => ("gemm qkv f32", "adapter rank qkv", "adapter up qkv"),
            1 => ("gemm out f32", "adapter rank out", "adapter up out"),
            2 => (
                "gemm gate/up f32",
                "adapter rank gate/up",
                "adapter up gate/up",
            ),
            _ => ("gemm down f32", "adapter rank down", "adapter up down"),
        };
        p.prepare.emit(
            sink,
            Some(prof),
            "adapter prepare",
            tokens,
            original,
            norm,
            self.input_view(),
            None,
        )?;
        p.down.emit(
            sink,
            Some(prof),
            rank_stage,
            tokens,
            self.input_view(),
            a.binding(),
            None,
            self.rank.binding(),
            None,
            None,
        )?;
        p.rank_prepare.emit(
            sink,
            Some(prof),
            "adapter rank prepare",
            tokens,
            self.rank.binding(),
            None,
            self.rank_operand.binding(),
            None,
        )?;
        p.up.emit(
            sink,
            Some(prof),
            up_stage,
            tokens,
            self.rank_operand.binding(),
            b.binding(),
            None,
            self.delta.binding(),
            None,
            None,
        )?;
        p.base.emit(
            sink,
            Some(prof),
            base_stage,
            tokens,
            base_input,
            base_weight,
            scales,
            self.base.binding(),
            None,
            None,
        )?;
        let rows = tokens as usize;
        let (gate, cls) = match residual {
            Some((gate, cls)) => (gate, cls.against("adapter finish", rows, self.classes)?),
            None => (self.base.binding(), self.base.binding()),
        };
        emit(
            sink,
            &p.finish,
            Some(prof),
            "adapter finish",
            [(rows * p.result_width).div_ceil(256) as u32, 1, 1],
            [THREADS, 1, 1],
            &[tokens],
            &[self.base.binding(), self.delta.binding(), output, gate, cls],
            &[
                rows * p.output * 4,
                rows * p.output * 2,
                rows * p.result_width * if p.mode == 2 { 4 } else { 2 },
                if p.mode == 2 {
                    self.classes * p.result_width * 4
                } else {
                    0
                },
                if p.mode == 2 { rows * 4 } else { 0 },
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::{bf16, f16};

    fn read(stream: &mut hrx::Stream, buffer: &hrx::Buffer, bytes: usize) -> Vec<u8> {
        read_view(stream, buffer.binding(), bytes)
    }
    fn read_view(stream: &mut hrx::Stream, view: View<'_>, bytes: usize) -> Vec<u8> {
        let mut out = vec![0; bytes];
        stream
            .read_blocking(view.slice(0, bytes).unwrap(), &mut out)
            .unwrap();
        out
    }
    fn bfloat(bytes: &[u8], i: usize) -> f32 {
        bf16::from_le_bytes(bytes[2 * i..2 * i + 2].try_into().unwrap()).to_f32()
    }
    fn close(actual: &[f32], expected: &[f32], context: &str) {
        let error: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(a - b).powi(2))
            .sum();
        let norm: f64 = expected.iter().map(|b| f64::from(*b).powi(2)).sum();
        let relative = (error / norm.max(1e-30)).sqrt();
        eprintln!("{context}: relative L2 {relative:.6}");
        assert!(relative < 0.005, "{context}: relative L2 {relative}");
        assert!(actual.iter().all(|x| x.is_finite()));
    }

    #[test]
    #[ignore = "requires idle gfx1151, base checkpoint and both cached Turbo adapters"]
    fn real_adapted_stacks_match_eager_and_recorded_execution() {
        assert!(
            std::env::var_os("H3_ADAPTER_BLOCK_FIXTURE").is_none()
                || env_once("H3_REUSE_SCRATCH") != Some("1"),
            "export reference intermediates with separate scratch; test reused scratch without H3_ADAPTER_BLOCK_FIXTURE"
        );
        let c = Compiler::new(None, std::path::PathBuf::new());
        let mut stream = hrx::Stream::open().unwrap();
        let path = crate::models::Resolver::new()
            .offline(true)
            .find(crate::models::DIT_FL2VA)
            .unwrap();
        // Safety: these cached fixtures remain immutable throughout the test.
        let weights = unsafe { Weights::open(path, crate::plan::dit::plan) }.unwrap();
        let constants = Constants::new(&mut stream).unwrap();
        let rows = 17;
        for preset in [
            crate::adapter::TurboPreset::Four,
            crate::adapter::TurboPreset::Eight,
        ] {
            let path = preset.resolve(true).unwrap();
            let checkpoint = unsafe { crate::adapter::Adapter::open(&path) }.unwrap();
            for refiner in [false, true] {
                let classes = if refiner { 1 } else { 4 };
                let dims = StackDims {
                    hidden: HID,
                    heads: HEADS,
                    kv_heads: HEADS,
                    head_dim: HEAD_DIM,
                    ffn: FFN,
                    rope_dim: ROPE_DIM,
                    classes,
                    wbits: if refiner { 16 } else { 8 },
                    eps: 1e-5,
                    bias: false,
                    gate_first: true,
                    causal: false,
                    attn_i4: false,
                    attn_qk_bits: if refiner { 16 } else { 8 },
                    bf16: refiner,
                };
                let mut stack = Stack::new(
                    &c,
                    &mut stream,
                    dims,
                    rows,
                    1,
                    &weights,
                    |_| {
                        if refiner {
                            "h3.refiner.0.".into()
                        } else {
                            "blocks.0.".into()
                        }
                    },
                    true,
                    constants.ones.clone(),
                    "adapter-graph-test",
                )
                .unwrap();
                stack
                    .attach_adapter(&c, &mut stream, &checkpoint, refiner)
                    .unwrap();
                let x = stream.allocate(stack.capacity * HID * 4).unwrap();
                stream.fill(x.binding(), 0).unwrap();
                let mut cls = crate::dispatch::Classes::zeroed(&mut stream, rows).unwrap();
                cls.write(
                    &mut stream,
                    &(0..rows).map(|r| (r % classes) as i32).collect::<Vec<_>>(),
                    classes,
                )
                .unwrap();
                let cos = stream.allocate(rows * ROPE_DIM / 2 * 4).unwrap();
                let sin = stream.allocate(rows * ROPE_DIM / 2 * 4).unwrap();
                stream
                    .upload(
                        cos.binding(),
                        bytemuck::cast_slice(&vec![1.0f32; rows * ROPE_DIM / 2]),
                    )
                    .unwrap();
                stream.fill(sin.binding(), 0).unwrap();
                let table = stream.allocate(classes * 2 * HID * 4).unwrap();
                let gate = stream.allocate(classes * HID * 4).unwrap();
                stream
                    .upload(
                        table.binding(),
                        bytemuck::cast_slice(
                            &(0..classes * 2 * HID)
                                .map(|i| (i % 13) as f32 / 100.0)
                                .collect::<Vec<_>>(),
                        ),
                    )
                    .unwrap();
                stream
                    .upload(
                        gate.binding(),
                        bytemuck::cast_slice(
                            &(0..classes * HID)
                                .map(|i| 0.1 + (i % 7) as f32 / 20.0)
                                .collect::<Vec<_>>(),
                        ),
                    )
                    .unwrap();
                let cond = |_| LayerCond {
                    table_msa: table.binding(),
                    gate_msa: gate.binding(),
                    table_mlp: table.binding(),
                    gate_mlp: gate.binding(),
                };
                let mut graph = None;
                for iteration in 0..3 {
                    let input: Vec<f32> = (0..rows * HID)
                        .map(|i| ((i * 17 + iteration * 31) % 127) as f32 / 64.0 - 1.0)
                        .collect();
                    stream
                        .upload(x.binding(), bytemuck::cast_slice(&input))
                        .unwrap();
                    stack
                        .forward(
                            &mut stream,
                            &mut Profile::default(),
                            x.binding(),
                            cls.slice(0, rows),
                            cos.binding(),
                            sin.binding(),
                            &cond,
                            0,
                            None,
                        )
                        .unwrap();
                    let expected = read(&mut stream, &x, rows * HID * 4);
                    stream
                        .upload(x.binding(), bytemuck::cast_slice(&input))
                        .unwrap();
                    // Record explicitly so this test exercises replay even when
                    // the caller has not enabled the optional H3_GRAPH path.
                    if graph.is_none() {
                        let mut recording = stream.graph().unwrap();
                        stack
                            .emit(
                                &mut Sink::Graph {
                                    graph: &mut recording,
                                    after: Default::default(),
                                },
                                &mut Profile::default(),
                                x.binding(),
                                cls.slice(0, rows),
                                cos.binding(),
                                sin.binding(),
                                &cond,
                                0,
                                None,
                                false,
                            )
                            .unwrap();
                        graph = Some(recording.finish().unwrap());
                    }
                    stream.launch(graph.as_mut().unwrap()).unwrap();
                    let actual = read(&mut stream, &x, rows * HID * 4);
                    assert!(
                        actual == expected,
                        "{preset:?} refiner={refiner} replay={iteration}"
                    );
                    assert!(actual
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite()));
                    if iteration == 0 {
                        if let Some(dir) = std::env::var_os("H3_ADAPTER_BLOCK_FIXTURE") {
                            let dir = std::path::PathBuf::from(dir).join(format!(
                                "{preset:?}-{}",
                                if refiner { "refiner" } else { "dit" }
                            ));
                            std::fs::create_dir_all(&dir).unwrap();
                            std::fs::write(dir.join("input.f32"), bytemuck::cast_slice(&input))
                                .unwrap();
                            std::fs::write(dir.join("output.f32"), &actual).unwrap();
                            for (name, view, bytes) in [
                                ("attention.f16", stack.attn.binding(), rows * INNER * 2),
                                ("hidden.f16", stack.gu_view(), rows * FFN * 2),
                            ] {
                                std::fs::write(dir.join(name), read_view(&mut stream, view, bytes))
                                    .unwrap();
                            }
                            // Fused QKV is overwritten by gate/up when scratch reuse is enabled.
                            if stack.gu.is_some() {
                                std::fs::write(
                                    dir.join("qkv.f16"),
                                    read(&mut stream, &stack.fused, rows * QKV * 2),
                                )
                                .unwrap();
                            }
                            std::fs::write(
                                dir.join("table.f32"),
                                read(&mut stream, &table, classes * 2 * HID * 4),
                            )
                            .unwrap();
                            std::fs::write(
                                dir.join("gate.f32"),
                                read(&mut stream, &gate, classes * HID * 4),
                            )
                            .unwrap();
                            std::fs::write(dir.join("fixture.json"), format!("{{\"rows\":{rows},\"classes\":{classes},\"refiner\":{refiner},\"preset\":\"{preset:?}\",\"rope\":\"identity\",\"layer\":0}}\n")).unwrap();
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "requires idle gfx1151, base checkpoint and both cached Turbo adapters"]
    fn real_low_rank_branches_match_independent_cpu_products() {
        let c = Compiler::new(None, std::path::PathBuf::new());
        let mut stream = hrx::Stream::open().unwrap();
        let path = crate::models::Resolver::new()
            .offline(true)
            .find(crate::models::DIT_FL2VA)
            .unwrap();
        // Safety: these cached fixtures remain immutable throughout the test.
        let weights = unsafe { Weights::open(path, crate::plan::dit::plan) }.unwrap();
        let constants = Constants::new(&mut stream).unwrap();
        let rows = 3;
        let mut cls = crate::dispatch::Classes::zeroed(&mut stream, rows).unwrap();
        cls.write(&mut stream, &vec![0; rows], 1).unwrap();
        for preset in [
            crate::adapter::TurboPreset::Four,
            crate::adapter::TurboPreset::Eight,
        ] {
            let path = preset.resolve(true).unwrap();
            let checkpoint = unsafe { crate::adapter::Adapter::open(&path) }.unwrap();
            for refiner in [false, true] {
                let dims = StackDims {
                    hidden: HID,
                    heads: HEADS,
                    kv_heads: HEADS,
                    head_dim: HEAD_DIM,
                    ffn: FFN,
                    rope_dim: ROPE_DIM,
                    classes: 1,
                    wbits: if refiner { 16 } else { 8 },
                    eps: 1e-5,
                    bias: false,
                    gate_first: true,
                    causal: false,
                    attn_i4: false,
                    attn_qk_bits: 16,
                    bf16: refiner,
                };
                let mut stack = Stack::new(
                    &c,
                    &mut stream,
                    dims,
                    rows,
                    1,
                    &weights,
                    |_| {
                        if refiner {
                            "h3.refiner.0.".into()
                        } else {
                            "blocks.0.".into()
                        }
                    },
                    true,
                    constants.ones.clone(),
                    "adapter-test",
                )
                .unwrap();
                stack
                    .attach_adapter(&c, &mut stream, &checkpoint, refiner)
                    .unwrap();
                let adapter = stack.adapter.as_ref().unwrap();
                let block = &stack.blocks[0];
                for (op, input_width, op_name) in [
                    (0, HID, "attn.qkv_proj"),
                    (1, INNER, "attn.out_proj"),
                    (2, HID, "mlp.fc1"),
                    (3, FFN, "mlp.fc2"),
                ] {
                    let p = &adapter.projections[op];
                    let original_values: Vec<f32> = (0..rows * input_width)
                        .map(|i| ((i * 17 % 127) as f32 - 63.0) / 64.0)
                        .collect();
                    let original_bytes = if op == 0 || op == 2 {
                        bytemuck::cast_slice(&original_values).to_vec()
                    } else {
                        original_values
                            .iter()
                            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                            .collect()
                    };
                    let original = stream.allocate(original_bytes.len()).unwrap();
                    stream.upload(original.binding(), &original_bytes).unwrap();
                    let norm = (op == 0 || op == 2).then_some((
                        constants.ones.binding(),
                        constants.zeros.binding(),
                        cls.slice(0, rows),
                    ));
                    let prepare = match op {
                        0 | 2 => &stack.prep_norm,
                        1 => stack.prep_attn.as_ref().unwrap(),
                        _ => stack.prep_down.as_ref().unwrap(),
                    };
                    prepare
                        .run(
                            &mut stream,
                            None,
                            "base prepare",
                            rows as u32,
                            original.binding(),
                            norm,
                            stack.a_q.binding(),
                            Some(stack.a_s.binding()),
                        )
                        .unwrap();
                    let (w, scale) = match op {
                        0 => (&block.qkv_q, &block.qkv_s),
                        1 => (&block.out_q, &block.out_s),
                        2 => (&block.gu_q, &block.gu_s),
                        _ => (&block.down_q, &block.down_s),
                    };
                    let output = stream.allocate(rows * p.result_width * 4).unwrap();
                    stream.fill(output.binding(), 0).unwrap();
                    let residual =
                        (p.mode == 2).then_some((constants.ones.binding(), cls.slice(0, rows)));
                    // Snapshot before the full projection reuses this input storage.
                    p.prepare
                        .run(
                            &mut stream,
                            None,
                            "adapter prepare fixture",
                            rows as u32,
                            original.binding(),
                            norm,
                            adapter.input_view(),
                            None,
                        )
                        .unwrap();
                    let ip = gemm_pitch(input_width, 16);
                    let input = read_view(&mut stream, adapter.input_view(), rows * ip * 2);
                    adapter
                        .emit(
                            &mut Sink::Stream(&mut stream),
                            &mut Profile::default(),
                            op,
                            0,
                            rows as u32,
                            original.binding(),
                            norm,
                            stack.a_q.binding(),
                            w.binding(),
                            scale.as_ref().map(|s| (s.binding(), stack.a_s.binding())),
                            output.binding(),
                            residual,
                        )
                        .unwrap();
                    let group = if refiner {
                        "token_refiner.blocks"
                    } else {
                        "blocks"
                    };
                    let prefix = format!("diffusion_model.{group}.0.{op_name}");
                    let spec = checkpoint
                        .projections
                        .iter()
                        .find(|s| s.prefix == prefix)
                        .unwrap();
                    let a_entry = checkpoint
                        .checkpoint()
                        .at(&format!("{prefix}.lora_A.weight"))
                        .unwrap();
                    let b_entry = checkpoint
                        .checkpoint()
                        .at(&format!("{prefix}.lora_B.weight"))
                        .unwrap();
                    let a = checkpoint.checkpoint().bytes(a_entry);
                    let b = checkpoint.checkpoint().bytes(b_entry);
                    let rank_width = spec.rank.div_ceil(256) * 256;
                    let rp = gemm_pitch(rank_width, 16);
                    let ranks = read(&mut stream, &adapter.rank_operand, rows * rp * 2);
                    let mut expected_rank = vec![0.0; rows * spec.rank];
                    let mut actual_rank = expected_rank.clone();
                    for row in 0..rows {
                        for r in 0..spec.rank {
                            let mut sum = 0.0f32;
                            for k in 0..input_width {
                                sum +=
                                    bfloat(&input, row * ip + k) * bfloat(a, r * input_width + k);
                            }
                            expected_rank[row * spec.rank + r] = bf16::from_f32(
                                f16::from_f32(sum.clamp(-65472.0, 65472.0)).to_f32(),
                            )
                            .to_f32();
                            actual_rank[row * spec.rank + r] = bfloat(&ranks, row * rp + r);
                        }
                    }
                    close(
                        &actual_rank,
                        &expected_rank,
                        &format!("{preset:?} {prefix} A"),
                    );
                    let delta = read(&mut stream, &adapter.delta, rows * p.output * 2);
                    let mut expected = vec![0.0; rows * p.output];
                    let mut actual = expected.clone();
                    for row in 0..rows {
                        for column in 0..p.output {
                            // Original B keeps whole gate/up halves, unlike the GPU's packed rows.
                            let original_column = if spec.gate_up {
                                (column / 32) * 16
                                    + column % 16
                                    + if column % 32 >= 16 { p.output / 2 } else { 0 }
                            } else {
                                column
                            };
                            let mut sum = 0.0f32;
                            for r in 0..spec.rank {
                                sum += bfloat(&ranks, row * rp + r)
                                    * bf16::from_f32(
                                        bfloat(b, original_column * spec.rank + r) * spec.scale,
                                    )
                                    .to_f32();
                            }
                            expected[row * p.output + column] =
                                f16::from_f32(sum.clamp(-65472.0, 65472.0)).to_f32();
                            actual[row * p.output + column] = f16::from_le_bytes(
                                delta[(row * p.output + column) * 2
                                    ..(row * p.output + column) * 2 + 2]
                                    .try_into()
                                    .unwrap(),
                            )
                            .to_f32();
                        }
                    }
                    close(&actual, &expected, &format!("{preset:?} {prefix} B"));
                }
                stream.synchronize().unwrap();
            }
        }
    }
}
