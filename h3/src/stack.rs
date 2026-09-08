//! The generic transformer stack: one block loop that the DiT, the token refiner, the text encoder and
//! the video decoder all run through, differing only in their dimensions and their operand types.
//!
//! What a stack decides at construction is which kernels it will use, and those choices depend on each
//! other: the operand element type follows the checkpoint's rows, the attention kernel follows the head
//! count and the QK^T width, and whether a projection needs a `prepare` at all follows from whether the
//! kernel before it already wrote its operand at the right pitch.
use crate::compile::{num, Cfg, Compiler};
use crate::dispatch::{launch, Gemm, Prepare, Profile, Tile};
use crate::model::*;
use crate::weights::Weights;
use hrx::sys::BufferRef;
use std::sync::{Arc, OnceLock};

pub type Result<T> = std::result::Result<T, crate::compile::Error>;

fn err<T>(message: impl Into<String>) -> Result<T> {
    Err(crate::compile::Error::Io(message.into()))
}

/// One stack's shape and operand types.
#[derive(Clone)]
pub struct StackDims {
    pub hidden: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub rope_dim: usize,
    pub classes: usize,
    /// 8: the checkpoint's int8 rows; 16: its f16 rows, or bf16 rows with `bf16` set.
    pub wbits: usize,
    pub eps: f32,
    pub bias: bool,
    pub gate_first: bool,
    pub causal: bool,
    /// QK^T in int4 (`prepare_qk_i4` operands).
    pub attn_i4: bool,
    /// 8: int8 operands (`prepare_qk_i8`, `attention_i8qk_*`); 4 is `attn_i4`.
    pub attn_qk_bits: usize,
    pub bf16: bool,
}

impl StackDims {
    pub fn elem(&self) -> &'static str {
        if self.wbits == 8 {
            "i8"
        } else if self.bf16 {
            "bf16"
        } else {
            "f16"
        }
    }
    pub fn inner(&self) -> usize {
        self.heads * self.head_dim
    }
    pub fn kv_inner(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    pub fn qkv(&self) -> usize {
        self.inner() + 2 * self.kv_inner()
    }
}

/// The AdaLN rows a layer modulates with: a table and a gate for each of the two residual writes.
///
/// The sizes are not optional and are not checked by the runtime: a table is `(scale, shift)` per
/// class, so each `table_*` must hold **two** rows of the hidden width per class, and each `gate_*` one
/// row per class. A buffer half that size reads past its allocation and produces a plausible-looking
/// wrong answer rather than a fault — a stack whose blocks modulate with a scale alone still needs a
/// `2 * hidden` zero table, not a `hidden` one.
#[derive(Clone, Copy)]
pub struct LayerCond {
    pub table_msa: BufferRef,
    pub gate_msa: BufferRef,
    pub table_mlp: BufferRef,
    pub gate_mlp: BufferRef,
}

/// The constant rows a stack modulates with when it does not really modulate.
///
/// A stack with no AdaLN still hands the kernels a table and a gate; these are the identity ones. Both
/// are sized for the widest consumer in the pipeline rather than the caller at hand, because the cost
/// is a fraction of a megabyte and the failure mode of guessing too small is an out-of-bounds read that
/// produces a plausible wrong answer instead of a fault.
pub struct Constants {
    /// `HID` ones: a residual gate of one, and the per-head norm weights of a stack with none
    pub ones: Arc<hrx::Buffer>,
    /// `2 * TE_FFN` zeros: a (scale, shift) table of zero at any width the pipeline uses
    pub zeros: Arc<hrx::Buffer>,
}

impl Constants {
    pub fn new(gpu: &hrx::Gpu) -> Result<Self> {
        let ones = gpu.alloc(HID * 4)?;
        let row: Vec<u8> = (0..HID).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        gpu.h2d(&ones, &row)?;
        let zeros = gpu.alloc(2 * TE_FFN * 4)?;
        gpu.memset(&zeros, 0, 2 * TE_FFN * 4)?;
        Ok(Self {
            ones: Arc::new(ones),
            zeros: Arc::new(zeros),
        })
    }

    /// The layer conditioning of a stack that neither shifts nor gates: a zero table and a gate of one.
    pub fn identity(&self) -> LayerCond {
        LayerCond {
            table_msa: self.zeros.binding(),
            gate_msa: self.ones.binding(),
            table_mlp: self.zeros.binding(),
            gate_mlp: self.ones.binding(),
        }
    }
}

/// One layer's weights, already on the device.
pub struct Block {
    pub qkv_q: Arc<hrx::Buffer>,
    pub qkv_s: Option<Arc<hrx::Buffer>>,
    pub out_q: Arc<hrx::Buffer>,
    pub out_s: Option<Arc<hrx::Buffer>>,
    pub gu_q: Arc<hrx::Buffer>,
    pub gu_s: Option<Arc<hrx::Buffer>>,
    pub down_q: Arc<hrx::Buffer>,
    pub down_s: Option<Arc<hrx::Buffer>>,
    pub qkv_b: Option<Arc<hrx::Buffer>>,
    pub out_b: Option<Arc<hrx::Buffer>>,
    pub gu_b: Option<Arc<hrx::Buffer>>,
    pub down_b: Option<Arc<hrx::Buffer>>,
    pub norm1: Arc<hrx::Buffer>,
    pub norm2: Arc<hrx::Buffer>,
    pub qnorm: Arc<hrx::Buffer>,
    pub knorm: Arc<hrx::Buffer>,
    pub scale1: Option<Arc<hrx::Buffer>>,
    pub scale2: Option<Arc<hrx::Buffer>>,
}

/// Environment knobs, read once: changing one mid-run would change kernels already built.
pub fn env_once(name: &'static str) -> Option<&'static str> {
    static CACHE: OnceLock<
        std::sync::Mutex<std::collections::HashMap<&'static str, Option<&'static str>>>,
    > = OnceLock::new();
    let cache = CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()));
    let mut map = cache.lock().expect("not poisoned");
    *map.entry(name).or_insert_with(|| {
        std::env::var(name)
            .ok()
            .map(|v| &*Box::leak(v.into_boxed_str()))
    })
}

pub struct Stack {
    d: StackDims,
    tokens: usize,
    capacity: usize,
    layers: usize,
    tag: String,
    waves: usize,
    calls: usize,
    direct_attn: bool,
    direct_down: bool,
    blocks: Vec<Block>,

    prep_norm: Prepare,
    prep_attn: Option<Prepare>,
    prep_down: Option<Prepare>,
    gemm_qkv: Option<Gemm>,
    gemm_gu: Gemm,
    gemm_out: Gemm,
    gemm_down: Gemm,

    qkv_rope: Option<Arc<hrx::Kernel>>,
    qkv_group: u32,
    rope: Option<Arc<hrx::Kernel>>,
    attention: Arc<hrx::Kernel>,
    colmean: Option<Arc<hrx::Kernel>>,
    prep_q: Option<Arc<hrx::Kernel>>,
    prep_k: Option<Arc<hrx::Kernel>>,
    transpose: Option<Arc<hrx::Kernel>>,

    a_q: hrx::Buffer,
    a_s: hrx::Buffer,
    fused: hrx::Buffer,
    /// When the QKV projection is fused with rope, these are views into `fused`; otherwise they own
    /// their own allocations.
    qkv_split: Option<(hrx::Buffer, hrx::Buffer, hrx::Buffer)>,
    attn: hrx::Buffer,
    gu: hrx::Buffer,
    /// The integer QK^T operands, when the attention runs on them.
    int_qk: Option<IntQk>,
}

struct IntQk {
    qi: hrx::Buffer,
    ki: hrx::Buffer,
    qs: hrx::Buffer,
    ks: hrx::Buffer,
    kmean: hrx::Buffer,
    zmean: hrx::Buffer,
    vt: hrx::Buffer,
}

impl Stack {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        c: &Compiler,
        gpu: &hrx::Gpu,
        d: StackDims,
        tokens: usize,
        layers: usize,
        w: &Weights,
        prefix: impl Fn(usize) -> String,
        qk_weights: bool,
        ones_head: Arc<hrx::Buffer>,
        tag: &str,
    ) -> Result<Self> {
        let elem = d.elem();
        let bits = d.wbits;
        let quant = quantised(elem);

        // The decoder's measured tiles, both restricted to its own shape.
        let decoder_shape = elem == "f16" && d.head_dim == 64 && d.bias && !d.gate_first;
        let wide = decoder_shape && env_once("H3_VAE_WIDE") == Some("1");
        let fast = !wide && decoder_shape && env_once("H3_VAE_FAST") != Some("0");
        let tile = if wide {
            Tile::Wide
        } else if fast {
            Tile::Fast
        } else {
            Tile::Plain
        };

        let wbytes = |n: usize, k: usize| if quant { n * k } else { n * k * 2 };
        let pitch = |k: usize| gemm_pitch(k, bits);

        let waves = if d.causal || tokens >= 4096 { 8 } else { 4 };
        let mut capacity = (tokens + 16).div_ceil(32) * 32;
        capacity = capacity.max(tokens.div_ceil(16 * waves) * (16 * waves));
        capacity = capacity.max(tokens.div_ceil(256) * 256);

        let mut blocks = Vec::with_capacity(layers);
        for i in 0..layers {
            let p = prefix(i);
            let wpad = |name: &str, n: usize, k: usize| -> Result<Arc<hrx::Buffer>> {
                w.rows(gpu, name, n, wbytes(1, k), wbytes(1, pitch(k)))
                    .map_err(|e| crate::compile::Error::Io(e.to_string()))
            };
            let scale = |name: &str, n: usize| -> Result<Option<Arc<hrx::Buffer>>> {
                if !quant {
                    return Ok(None);
                }
                w.at(gpu, name, n * 4)
                    .map(Some)
                    .map_err(|e| crate::compile::Error::Io(e.to_string()))
            };
            let vec_at = |name: &str, n: usize| -> Result<Arc<hrx::Buffer>> {
                w.at(gpu, name, n * 4)
                    .map_err(|e| crate::compile::Error::Io(e.to_string()))
            };

            let (qnorm, knorm) = if qk_weights {
                (
                    vec_at(&format!("{p}qnorm"), d.head_dim)?,
                    vec_at(&format!("{p}knorm"), d.head_dim)?,
                )
            } else {
                (ones_head.clone(), ones_head.clone())
            };
            let (scale1, scale2) = if w.has(&format!("{p}scale1")) {
                (
                    Some(vec_at(&format!("{p}scale1"), d.hidden)?),
                    Some(vec_at(&format!("{p}scale2"), d.hidden)?),
                )
            } else {
                (None, None)
            };
            blocks.push(Block {
                qkv_q: wpad(&format!("{p}qkv.q"), d.qkv(), d.hidden)?,
                qkv_s: scale(&format!("{p}qkv.s"), d.qkv())?,
                out_q: wpad(&format!("{p}out.q"), d.hidden, d.inner())?,
                out_s: scale(&format!("{p}out.s"), d.hidden)?,
                gu_q: wpad(&format!("{p}gu.q"), 2 * d.ffn, d.hidden)?,
                gu_s: scale(&format!("{p}gu.s"), 2 * d.ffn)?,
                down_q: wpad(&format!("{p}down.q"), d.hidden, d.ffn)?,
                down_s: scale(&format!("{p}down.s"), d.hidden)?,
                qkv_b: if d.bias {
                    Some(vec_at(&format!("{p}qkv.b"), d.qkv())?)
                } else {
                    None
                },
                out_b: if d.bias {
                    Some(vec_at(&format!("{p}out.b"), d.hidden)?)
                } else {
                    None
                },
                gu_b: if d.bias {
                    Some(vec_at(&format!("{p}gu.b"), 2 * d.ffn)?)
                } else {
                    None
                },
                down_b: if d.bias {
                    Some(vec_at(&format!("{p}down.b"), d.hidden)?)
                } else {
                    None
                },
                norm1: vec_at(&format!("{p}norm1"), d.hidden)?,
                norm2: vec_at(&format!("{p}norm2"), d.hidden)?,
                qnorm,
                knorm,
                scale1,
                scale2,
            });
        }

        let prep_norm = Prepare::build(
            c,
            gpu,
            "norm",
            elem,
            d.hidden,
            d.eps,
            d.classes,
            pitch(d.hidden),
        )?;
        // f16 rows: the attention output is the out projection's operand as it stands, because the
        // kernel writes it at the padded pitch. The gate|up product is written at `ffn`, so the down
        // projection needs a prepare back whenever that pitch is padded — unless a decoder tile, which
        // writes the padded pitch itself.
        let direct_attn = elem == "f16";
        let direct_down = elem == "f16" && (wide || fast || pitch(d.ffn) == d.ffn);
        let prep_attn = if direct_attn {
            None
        } else {
            Some(Prepare::build(
                c,
                gpu,
                "plain",
                elem,
                d.inner(),
                1e-5,
                1,
                pitch(d.inner()),
            )?)
        };
        let prep_down = if direct_down {
            None
        } else {
            Some(Prepare::build(
                c,
                gpu,
                "plain",
                elem,
                d.ffn,
                1e-5,
                1,
                pitch(d.ffn),
            )?)
        };

        let fused_qkv = fast
            && !d.causal
            && waves == 4
            && d.hidden == 2048
            && d.heads == 32
            && d.kv_heads == 32
            && d.head_dim == 64
            && d.rope_dim == 48;
        let (mut qkv_rope, mut qkv_group, mut gemm_qkv) = (None, 1u32, None);
        if fused_qkv {
            qkv_group = vae_fast_m_group_for(tokens, d.hidden, 3 * d.hidden);
            let stem = "gemm_f16_qkvropehm_256b";
            let ns = format!("h3.{stem}.");
            qkv_rope = Some(c.get(
                gpu,
                stem,
                &format!("h3_{stem}"),
                &vec![
                    (format!("{ns}k_size"), "2048".into()),
                    (format!("{ns}n_size"), "6144".into()),
                    (format!("{ns}k_stride"), pitch(d.hidden).to_string()),
                    (format!("{ns}m_group"), qkv_group.to_string()),
                    (format!("{ns}token_capacity"), capacity.to_string()),
                    (format!("{ns}eps"), num(f64::from(d.eps))),
                ],
            )?);
        } else {
            gemm_qkv = Some(Gemm::build(
                c,
                gpu,
                "plain",
                elem,
                d.bias,
                true,
                d.hidden,
                d.qkv(),
                tokens,
                1,
                pitch(d.hidden),
                tile,
                0,
            )?);
        }
        let gemm_gu = Gemm::build(
            c,
            gpu,
            "swiglu",
            elem,
            d.bias,
            d.gate_first,
            d.hidden,
            2 * d.ffn,
            tokens,
            1,
            pitch(d.hidden),
            tile,
            pitch(d.ffn),
        )?;
        let gemm_out = Gemm::build(
            c,
            gpu,
            "resid",
            elem,
            d.bias,
            true,
            d.inner(),
            d.hidden,
            tokens,
            d.classes,
            pitch(d.inner()),
            tile,
            0,
        )?;
        let gemm_down = Gemm::build(
            c,
            gpu,
            "resid",
            elem,
            d.bias,
            true,
            d.ffn,
            d.hidden,
            tokens,
            d.classes,
            pitch(d.ffn),
            tile,
            0,
        )?;

        let rope = if fused_qkv {
            None
        } else {
            let stem = if d.head_dim == 64 {
                "rope64_qknorm_f16"
            } else if d.rope_dim == 128 {
                "rope128_qknorm_f16"
            } else {
                "rope_qknorm_f16"
            };
            let ns = format!("h3.{stem}.");
            Some(c.get(
                gpu,
                stem,
                &format!("h3_{stem}"),
                &vec![
                    (format!("{ns}row_stride"), d.qkv().to_string()),
                    (format!("{ns}heads"), d.heads.to_string()),
                    (format!("{ns}kv_heads"), d.kv_heads.to_string()),
                    (format!("{ns}k_offset"), d.inner().to_string()),
                    (format!("{ns}eps"), num(f64::from(d.eps))),
                ],
            )?)
        };

        let qk_int = d.attn_i4 || d.attn_qk_bits == 8;
        let qk_head_major = !d.attn_i4 && d.attn_qk_bits == 8 && waves == 8;
        let pqk = if d.attn_i4 {
            "prepare_qk_i4"
        } else if qk_head_major {
            "prepare_qk_i8hm"
        } else {
            "prepare_qk_i8"
        };
        let (mut colmean, mut prep_q, mut prep_k, mut transpose) = (None, None, None, None);
        if qk_int {
            if d.causal || d.head_dim != 128 {
                return err("integer QK^T attention: MHA with head 128 only");
            }
            let pq = format!("h3.{pqk}.");
            colmean = Some(c.get(
                gpu,
                "colmean_f32",
                "h3_colmean_f32",
                &vec![("h3.colmean_f32.width".into(), d.inner().to_string())],
            )?);
            let mut cfg: Cfg = vec![
                (format!("{pq}row_stride"), d.inner().to_string()),
                (format!("{pq}head_offset"), "0".into()),
                (format!("{pq}heads"), d.heads.to_string()),
            ];
            if qk_head_major {
                cfg.push((format!("{pq}token_capacity"), capacity.to_string()));
            }
            cfg.push((
                format!("{pq}extra_scale"),
                num(1.0 / (d.head_dim as f64).sqrt() / 128.0),
            ));
            prep_q = Some(c.get(gpu, pqk, &format!("h3_{pqk}"), &cfg)?);
            // the K operand differs only in its head offset
            let last = cfg.len() - 1;
            cfg[last].1 = "1".into();
            prep_k = Some(c.get(gpu, pqk, &format!("h3_{pqk}"), &cfg)?);
            transpose = Some(c.get(
                gpu,
                "transpose_f16",
                "h3_transpose_f16",
                &vec![
                    ("h3.transpose_f16.width".into(), d.inner().to_string()),
                    ("h3.transpose_f16.row_capacity".into(), capacity.to_string()),
                ],
            )?);
        }

        let attention = {
            let mut stem = if d.causal {
                "attention_gqa8c_lds_f16_wmma".to_string()
            } else if d.head_dim == 64 {
                if waves == 8 {
                    "attention_mha648_lds_f16_wmma".into()
                } else {
                    "attention_mha64_lds_f16_wmma".into()
                }
            } else if waves == 8 {
                "attention_mha8_lds_f16_wmma".into()
            } else {
                "attention_mha_lds_f16_wmma".into()
            };
            if fused_qkv {
                stem = "attention_mha64hm32_lds_f16_wmma".into();
            } else if fast && waves == 4 {
                stem = "attention_mha64t32_lds_f16_wmma".into();
            }
            if qk_int {
                stem = if qk_head_major {
                    // Head-major operands; 64 shared keys amortise softmax and loop overhead.
                    "attention_i8qkhm_mha8_k64_lds_f16_wmma".into()
                } else if d.attn_i4 {
                    if tokens >= 20000 {
                        "attention_i4qkl_mha8_lds_f16_wmma".into()
                    } else if waves == 8 {
                        "attention_i4qk_mha8_lds_f16_wmma".into()
                    } else {
                        "attention_i4qk_mha_lds_f16_wmma".into()
                    }
                } else {
                    "attention_i8qk_mha_lds_f16_wmma".into()
                };
            }
            // H3_ATTN_SKIP_TAU selects the skip twin at that tau; the twins exist for int4 only.
            let tau = env_once("H3_ATTN_SKIP_TAU")
                .map(|v| v.to_ascii_lowercase())
                .filter(|v| !matches!(v.as_str(), "off" | "none" | "0" | "1e30" | ""))
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            let skip = d.attn_i4 && tau > 0.0;
            if skip {
                stem = stem.replace("i4qk", "i4qks");
            }
            let ns = format!("h3.{stem}.");
            let mut acfg: Cfg = vec![
                (format!("{ns}q_stride"), d.inner().to_string()),
                (format!("{ns}kv_stride"), d.kv_inner().to_string()),
                (format!("{ns}tokens"), tokens.to_string()),
                (format!("{ns}token_capacity"), capacity.to_string()),
                (format!("{ns}scale"), num(1.0 / (d.head_dim as f64).sqrt())),
                // the out projection reads this as its A operand
                (
                    format!("{ns}out_stride"),
                    if direct_attn {
                        pitch(d.inner())
                    } else {
                        d.inner()
                    }
                    .to_string(),
                ),
            ];
            if skip {
                acfg.push((format!("{ns}skip_tau"), num(tau)));
            }
            c.get(gpu, &stem, &format!("h3_{stem}"), &acfg)?
        };

        let t = capacity;
        let widest = pitch(d.ffn).max(pitch(d.hidden)).max(pitch(d.inner()));
        let a_q = gpu.alloc(t * widest * if quant { 1 } else { 2 })?;
        let a_s = gpu.alloc(t * 4)?;
        let fused = gpu.alloc(t * d.qkv() * 2)?;
        let qkv_split = if fused_qkv {
            None // q, k and v are views into `fused`
        } else {
            Some((
                gpu.alloc(t * d.inner() * 2)?,
                gpu.alloc(t * d.kv_inner() * 2)?,
                gpu.alloc(t * d.kv_inner() * 2)?,
            ))
        };
        let attn_width = if direct_attn {
            pitch(d.inner())
        } else {
            d.inner()
        };
        let attn = gpu.alloc(t * attn_width * 2)?;
        let gu = gpu.alloc(t * if direct_down { pitch(d.ffn) } else { d.ffn } * 2)?;
        gpu.memset(&fused, 0, t * d.qkv() * 2)?;
        gpu.memset(&attn, 0, t * attn_width * 2)?;
        if let Some((q, k, v)) = &qkv_split {
            gpu.memset(q, 0, t * d.inner() * 2)?;
            gpu.memset(k, 0, t * d.kv_inner() * 2)?;
            gpu.memset(v, 0, t * d.kv_inner() * 2)?;
        }

        let int_qk = if qk_int {
            let code_bytes = if d.attn_i4 { 64 } else { 128 };
            let qi = gpu.alloc(t * d.heads * code_bytes)?;
            let ki = gpu.alloc(t * d.heads * code_bytes)?;
            let qs = gpu.alloc(t * d.heads * 4)?;
            let ks = gpu.alloc(t * d.heads * 4)?;
            let kmean = gpu.alloc(d.inner() * 4)?;
            let zmean = gpu.alloc(d.inner() * 4)?;
            let vt = gpu.alloc(d.inner() * t * 2)?;
            gpu.memset(&zmean, 0, d.inner() * 4)?;
            gpu.memset(&vt, 0, d.inner() * t * 2)?;
            gpu.memset(&qi, 0, t * d.heads * code_bytes)?;
            gpu.memset(&ki, 0, t * d.heads * code_bytes)?;
            gpu.memset(&qs, 0, t * d.heads * 4)?;
            gpu.memset(&ks, 0, t * d.heads * 4)?;
            Some(IntQk {
                qi,
                ki,
                qs,
                ks,
                kmean,
                zmean,
                vt,
            })
        } else {
            None
        };

        Ok(Self {
            d,
            tokens,
            capacity,
            layers,
            tag: tag.into(),
            waves,
            calls: 0,
            direct_attn,
            direct_down,
            blocks,
            prep_norm,
            prep_attn,
            prep_down,
            gemm_qkv,
            gemm_gu,
            gemm_out,
            gemm_down,
            qkv_rope,
            qkv_group,
            rope,
            attention,
            colmean,
            prep_q,
            prep_k,
            transpose,
            a_q,
            a_s,
            fused,
            qkv_split,
            attn,
            gu,
            int_qk,
        })
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn tokens(&self) -> usize {
        self.tokens
    }
    pub fn layers(&self) -> usize {
        self.layers
    }
    pub fn block(&self, i: usize) -> &Block {
        &self.blocks[i]
    }

    /// Q, K and V, whether they are their own allocations or views into the fused one.
    fn qkv_views(&self) -> (BufferRef, BufferRef, BufferRef) {
        match &self.qkv_split {
            Some((q, k, v)) => (q.binding(), k.binding(), v.binding()),
            None => {
                // [Q/K/V][head][capacity][64] in one allocation
                let (t, inner, kv) = (self.capacity, self.d.inner(), self.d.kv_inner());
                (
                    self.fused.slice(0, t * inner * 2),
                    self.fused.slice(t * inner * 2, t * kv * 2),
                    self.fused.slice(t * inner * 2 + t * kv * 2, t * kv * 2),
                )
            }
        }
    }

    /// `x`: f32 `[capacity][hidden]`, rows past `tokens` untouched. `cls`: i32 `[tokens]`.
    /// `cos`/`sin`: f32 `[tokens][rope_dim/2]`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        gpu: &hrx::Gpu,
        prof: &mut Profile,
        x: BufferRef,
        cls: BufferRef,
        cos: BufferRef,
        sin: BufferRef,
        cond: &dyn Fn(usize) -> LayerCond,
        first: usize,
        last: Option<usize>,
    ) -> Result<()> {
        let t = self.tokens as u32;
        let last = last.unwrap_or(self.layers);
        let (q, k, v) = self.qkv_views();

        // H3_DUMP_BLOCKS=<dir>: this stack's x before the first block and after every one, as
        // [tokens][hidden] f32, on its H3_DUMP_CALL-th forward.
        let dump_dir = env_once("H3_DUMP_BLOCKS");
        let dump_call: usize = env_once("H3_DUMP_CALL")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        let dumping = dump_dir.is_some() && {
            let this = self.calls;
            self.calls += 1;
            this == dump_call && first == 0
        };
        if dumping {
            self.dump(gpu, dump_dir.unwrap(), "h_in", x)?;
        }

        for i in first..last {
            let lc = cond(i);
            let (norm1, norm2, qnorm, knorm) = {
                let b = &self.blocks[i];
                (
                    b.norm1.binding(),
                    b.norm2.binding(),
                    b.qnorm.binding(),
                    b.knorm.binding(),
                )
            };
            self.prep_norm.run(
                gpu,
                Some(prof),
                "prepare norm",
                t,
                x,
                Some((norm1, lc.table_msa, cls)),
                self.a_q.binding(),
                Some(self.a_s.binding()),
            )?;

            if let Some(kernel) = &self.qkv_rope {
                let b = &self.blocks[i];
                let bias = b
                    .qkv_b
                    .as_ref()
                    .expect("the fused projection is biased")
                    .binding();
                launch(
                    gpu,
                    kernel,
                    Some(prof),
                    "gemm qkv + rope",
                    [
                        (self.d.qkv() / 256) as u32,
                        gemm_grid_y(self.tokens, self.qkv_group, 128),
                        1,
                    ],
                    [THREADS, 1, 1],
                    &[t],
                    &[
                        self.a_q.binding(),
                        b.qkv_q.binding(),
                        self.fused.binding(),
                        bias,
                        qnorm,
                        knorm,
                        cos,
                        sin,
                    ],
                )?;
            } else {
                let b = &self.blocks[i];
                let scales = b.qkv_s.as_ref().map(|s| (s.binding(), self.a_s.binding()));
                self.gemm_qkv.as_ref().expect("built when not fused").run(
                    gpu,
                    Some(prof),
                    "gemm qkv",
                    t,
                    self.a_q.binding(),
                    b.qkv_q.binding(),
                    scales,
                    self.fused.binding(),
                    None,
                    b.qkv_b.as_ref().map(|x| x.binding()),
                )?;
                launch(
                    gpu,
                    self.rope.as_ref().expect("built when not fused"),
                    Some(prof),
                    "qk norm + rope",
                    [t, 1, 1],
                    [THREADS, 1, 1],
                    &[t],
                    &[self.fused.binding(), qnorm, knorm, cos, sin, q, k, v],
                )?;
            }

            self.attend(gpu, prof, t, q, k, v)?;

            let attn_operand = if self.direct_attn {
                self.attn.binding()
            } else {
                self.prep_attn
                    .as_ref()
                    .expect("built when not direct")
                    .run(
                        gpu,
                        Some(prof),
                        "prepare out input",
                        t,
                        self.attn.binding(),
                        None,
                        self.a_q.binding(),
                        Some(self.a_s.binding()),
                    )?;
                self.a_q.binding()
            };
            {
                let b = &self.blocks[i];
                let scales = b.out_s.as_ref().map(|s| (s.binding(), self.a_s.binding()));
                self.gemm_out.run(
                    gpu,
                    Some(prof),
                    "gemm out + residual",
                    t,
                    attn_operand,
                    b.out_q.binding(),
                    scales,
                    x,
                    Some((lc.gate_msa, cls)),
                    b.out_b.as_ref().map(|x| x.binding()),
                )?;
            }

            self.prep_norm.run(
                gpu,
                Some(prof),
                "prepare norm",
                t,
                x,
                Some((norm2, lc.table_mlp, cls)),
                self.a_q.binding(),
                Some(self.a_s.binding()),
            )?;
            {
                let b = &self.blocks[i];
                let scales = b.gu_s.as_ref().map(|s| (s.binding(), self.a_s.binding()));
                self.gemm_gu.run(
                    gpu,
                    Some(prof),
                    "gemm ff + swiglu",
                    t,
                    self.a_q.binding(),
                    b.gu_q.binding(),
                    scales,
                    self.gu.binding(),
                    None,
                    b.gu_b.as_ref().map(|x| x.binding()),
                )?;
            }
            let down_operand = if self.direct_down {
                self.gu.binding()
            } else {
                self.prep_down
                    .as_ref()
                    .expect("built when not direct")
                    .run(
                        gpu,
                        Some(prof),
                        "prepare down input",
                        t,
                        self.gu.binding(),
                        None,
                        self.a_q.binding(),
                        Some(self.a_s.binding()),
                    )?;
                self.a_q.binding()
            };
            {
                let b = &self.blocks[i];
                let scales = b.down_s.as_ref().map(|s| (s.binding(), self.a_s.binding()));
                self.gemm_down.run(
                    gpu,
                    Some(prof),
                    "gemm down + residual",
                    t,
                    down_operand,
                    b.down_q.binding(),
                    scales,
                    x,
                    Some((lc.gate_mlp, cls)),
                    b.down_b.as_ref().map(|x| x.binding()),
                )?;
            }
            if dumping {
                self.dump(gpu, dump_dir.unwrap(), &format!("blk_{i:02}"), x)?;
            }
        }
        Ok(())
    }

    /// Attention, on f16 Q/K/V or on the narrowed integer operands.
    fn attend(
        &self,
        gpu: &hrx::Gpu,
        prof: &mut Profile,
        t: u32,
        q: BufferRef,
        k: BufferRef,
        v: BufferRef,
    ) -> Result<()> {
        let query_block = 16 * self.waves as u32;
        let Some(int_qk) = &self.int_qk else {
            let grid = if self.d.causal {
                [t.div_ceil(16), self.d.kv_heads as u32, 1]
            } else {
                [t.div_ceil(query_block), self.d.heads as u32, 1]
            };
            let block = if self.d.causal {
                [THREADS, 1, 1]
            } else {
                [32 * self.waves as u32, 1, 1]
            };
            return launch(
                gpu,
                &self.attention,
                Some(prof),
                "attention",
                grid,
                block,
                &[t],
                &[q, k, v, self.attn.binding()],
            );
        };

        // K mean smoothing: off by default, having measured worse.
        let smooth = env_once("H3_KSMOOTH") == Some("1");
        if smooth {
            launch(
                gpu,
                self.colmean.as_ref().expect("built with integer QK"),
                Some(prof),
                "attention operands",
                [(self.d.inner() / 256) as u32, 1, 1],
                [THREADS, 1, 1],
                &[t],
                &[k, int_qk.kmean.binding()],
            )?;
        }
        launch(
            gpu,
            self.prep_q.as_ref().expect("built with integer QK"),
            Some(prof),
            "attention operands",
            [t, 1, 1],
            [THREADS, 1, 1],
            &[t],
            &[
                q,
                int_qk.zmean.binding(),
                int_qk.qi.binding(),
                int_qk.qs.binding(),
            ],
        )?;
        launch(
            gpu,
            self.prep_k.as_ref().expect("built with integer QK"),
            Some(prof),
            "attention operands",
            [t, 1, 1],
            [THREADS, 1, 1],
            &[t],
            &[
                k,
                if smooth {
                    int_qk.kmean.binding()
                } else {
                    int_qk.zmean.binding()
                },
                int_qk.ki.binding(),
                int_qk.ks.binding(),
            ],
        )?;
        launch(
            gpu,
            self.transpose.as_ref().expect("built with integer QK"),
            Some(prof),
            "attention operands",
            [t.div_ceil(32), (self.d.inner() / 32) as u32, 1],
            [THREADS, 1, 1],
            &[t],
            &[v, int_qk.vt.binding()],
        )?;
        launch(
            gpu,
            &self.attention,
            Some(prof),
            "attention",
            [t.div_ceil(query_block), self.d.heads as u32, 1],
            [32 * self.waves as u32, 1, 1],
            &[t],
            &[
                int_qk.qi.binding(),
                int_qk.qs.binding(),
                int_qk.ki.binding(),
                int_qk.ks.binding(),
                int_qk.vt.binding(),
                self.attn.binding(),
            ],
        )
    }

    fn dump(&self, gpu: &hrx::Gpu, dir: &str, name: &str, x: BufferRef) -> Result<()> {
        let bytes = self.tokens * self.d.hidden * 4;
        let mut host = vec![0u8; bytes];
        gpu.d2h_ref(x, &mut host)?;
        let path = std::path::Path::new(dir).join(format!("{}_{name}.f32", self.tag));
        let _ = std::fs::write(path, &host);
        Ok(())
    }
}
