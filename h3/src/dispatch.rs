//! The kernel builders the stacks dispatch through: the prepare family that produces a GEMM's A
//! operand, and the GEMM family itself.
//!
//! Each one decides its stem, its configuration and its launch geometry from the shape it is built
//! for. Those decisions are the ones `model.rs` holds, so the operand a plan laid out and the operand a
//! kernel reads agree on their pitch.
use crate::compile::{num, Cfg, Compiler};
use crate::model::*;
use hrx::View;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

pub type Result<T> = std::result::Result<T, crate::compile::Error>;

/// Per-stage wall time, printed after a run when `H3_PROFILE` is set. Timing a launch means
/// synchronising around it, so this is opt-in.
#[derive(Default)]
pub struct Profile {
    pub on: bool,
    pub micros: BTreeMap<String, f64>,
}

impl Profile {
    pub fn from_env() -> Self {
        Self {
            on: std::env::var_os("H3_PROFILE").is_some_and(|v| !v.is_empty() && v != "0"),
            micros: BTreeMap::new(),
        }
    }

    pub fn total_seconds(&self) -> f64 {
        self.micros.values().sum::<f64>() * 1e-6
    }

    /// The stages worth naming, largest first: any taking more than `floor` of the total.
    ///
    /// `H3_PROFILE` costs a synchronise around every launch, so a run that pays for it must get the
    /// report back — otherwise it is pure slowdown.
    pub fn report(&self, floor: f64) -> String {
        let total = self.total_seconds();
        if total <= 0.0 {
            return String::new();
        }
        let mut stages: Vec<(&String, &f64)> = self.micros.iter().collect();
        stages.sort_by(|a, b| b.1.total_cmp(a.1));
        let mut out = format!("{total:.1} s:");
        for (name, micros) in stages {
            if *micros > floor * total * 1e6 {
                out.push_str(&format!("  {name} {:.2}s", micros * 1e-6));
            }
        }
        out
    }

    /// Empties the counters, so the next phase is reported on its own.
    pub fn take(&mut self) -> BTreeMap<String, f64> {
        std::mem::take(&mut self.micros)
    }
}

/// One launch, timed into `profile` when it is on.
///
/// # Safety
///
/// This is the crate's one call to [`hrx::Gpu::dispatch`], and it inherits its contract: `grid`,
/// `block`, `scalars` and `bindings` must be what `kernel` was compiled for. It is not marked unsafe
/// because every caller is a builder in this module, which compiles a kernel and launches it from one
/// description — that pairing is the invariant, and it is why the builders exist rather than callers
/// assembling launches by hand.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    gpu: &hrx::Gpu,
    kernel: &hrx::Kernel,
    profile: Option<&mut Profile>,
    stage: &str,
    grid: [u32; 3],
    block: [u32; 3],
    scalars: &[u32],
    bindings: &[View<'_>],
) -> Result<()> {
    match profile {
        Some(p) if p.on => {
            gpu.sync()?;
            let started = Instant::now();
            unsafe { gpu.dispatch(kernel, grid, block, scalars, bindings)? };
            gpu.sync()?;
            *p.micros.entry(stage.to_string()).or_insert(0.0) +=
                started.elapsed().as_secs_f64() * 1e6;
        }
        _ => unsafe { gpu.dispatch(kernel, grid, block, scalars, bindings)? },
    }
    Ok(())
}

/// Produces a GEMM's A operand: `norm` and `lnorm` normalise and modulate the f32 residual stream,
/// `plain` narrows an existing f16 row. The int8 forms also write a per-token scale; the float ones
/// write rows and nothing else.
pub struct Prepare {
    kernel: Arc<hrx::Kernel>,
    lanes: usize,
    form: String,
    elem: String,
}

impl Prepare {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        c: &Compiler,
        gpu: &hrx::Gpu,
        form: &str,
        elem: &str,
        width: usize,
        eps: f32,
        classes: usize,
        out_stride: usize,
    ) -> Result<Self> {
        // rows past 64 KB of f32 stage as f16 instead
        let stem = if form == "plain" && quantised(elem) && width * 4 > 65536 {
            "prepare_plain16_i8".to_string()
        } else {
            format!("prepare_{form}_{elem}")
        };
        let lanes = lanes_for(width).ok_or_else(|| {
            crate::compile::Error::Io(format!("no prepare lane count for width {width}"))
        })?;
        let ns = format!("h3.{stem}.");
        let mut cfg: Cfg = vec![
            (format!("{ns}width"), width.to_string()),
            (format!("{ns}lanes"), lanes.to_string()),
        ];
        if form != "plain" {
            cfg.push((format!("{ns}eps"), num(f64::from(eps))));
            cfg.push((format!("{ns}classes"), classes.to_string()));
        }
        // the operand rows' pitch, which is the GEMM's k_stride
        cfg.push((
            format!("{ns}out_stride"),
            if out_stride != 0 { out_stride } else { width }.to_string(),
        ));
        let kernel = c.get(gpu, &stem, &format!("h3_{stem}"), &cfg)?;
        Ok(Self {
            kernel,
            lanes,
            form: form.into(),
            elem: elem.into(),
        })
    }

    /// norm forms take `(x, weight, table, cls)`; plain takes `(h)` alone. `a_s` is written only by
    /// the int8 forms.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        profile: Option<&mut Profile>,
        stage: &str,
        tokens: u32,
        x: View<'_>,
        norm: Option<(View<'_>, View<'_>, View<'_>)>,
        a_q: View<'_>,
        a_s: Option<View<'_>>,
    ) -> Result<()> {
        let mut bindings = vec![x];
        if self.form != "plain" {
            let (weight, table, cls) =
                norm.expect("a norm prepare needs its weight, table and class");
            bindings.extend([weight, table, cls]);
        }
        bindings.push(a_q);
        if quantised(&self.elem) {
            bindings.push(a_s.expect("an int8 prepare writes a token scale"));
        }
        launch(
            gpu,
            &self.kernel,
            profile,
            stage,
            [tokens, 1, 1],
            [self.lanes as u32, 1, 1],
            &[tokens],
            &bindings,
        )
    }
}

/// A GEMM of the int8, f16 or bf16 family for one (K, N, row group).
pub struct Gemm {
    kernel: Arc<hrx::Kernel>,
    n: usize,
    resid: bool,
    bias: bool,
    elem: String,
    m_group: u32,
    m_tile: usize,
    n_tile: usize,
    threads: u32,
}

/// Which workgroup tile the decoder uses. `Plain` is the general 256x128; the other two are the
/// measured decoder shapes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tile {
    Plain,
    Wide,
    Fast,
}

impl Gemm {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        c: &Compiler,
        gpu: &hrx::Gpu,
        mode: &str,
        elem: &str,
        bias: bool,
        gate_first: bool,
        k_size: usize,
        n_size: usize,
        tokens: usize,
        classes: usize,
        k_stride: usize,
        tile: Tile,
        out_stride: usize,
    ) -> Result<Self> {
        let (wide, fast) = (tile == Tile::Wide, tile == Tile::Fast);
        let resid = mode == "resid";
        if (wide || fast)
            && (elem != "f16"
                || !bias
                || !n_size.is_multiple_of(256)
                || (mode == "swiglu" && gate_first))
        {
            return Err(crate::compile::Error::Io(
                "decoder GEMM requires the biased f16 family and N divisible by 256".into(),
            ));
        }
        let m_tile = if fast { 128 } else { 256 };
        let n_tile = if wide || fast { 256 } else { 128 };
        let threads = if wide { 512 } else { THREADS };
        let m_group = if fast {
            vae_fast_m_group_for(tokens, k_size, n_size)
        } else {
            gemm_m_group_for(tokens, k_size, n_size, elem_bits(elem))
        };
        let stem = format!(
            "gemm_{elem}{}{}_256{}{}",
            if wide {
                "_wide"
            } else if fast {
                "_fast"
            } else {
                ""
            },
            if mode == "plain" {
                String::new()
            } else {
                format!("_{mode}")
            },
            if bias { "b" } else { "" },
            if mode == "swiglu" && !gate_first {
                "_gs"
            } else {
                ""
            },
        );
        let ns = format!("h3.{stem}.");
        let mut cfg: Cfg = vec![
            (format!("{ns}k_size"), k_size.to_string()),
            (format!("{ns}n_size"), n_size.to_string()),
            (format!("{ns}m_group"), m_group.to_string()),
        ];
        if resid {
            cfg.push((format!("{ns}classes"), classes.to_string()));
        }
        // the operand row pitch: see gemm_pitch
        cfg.push((
            format!("{ns}k_stride"),
            if k_stride != 0 { k_stride } else { k_size }.to_string(),
        ));
        if (wide || fast) && mode == "swiglu" {
            let stride = if out_stride != 0 {
                out_stride
            } else {
                n_size / 2
            };
            if stride < n_size / 2 || stride > 65536 || !stride.is_multiple_of(128) {
                return Err(crate::compile::Error::Io(
                    "invalid wide SwiGLU output stride".into(),
                ));
            }
            cfg.push((format!("{ns}out_stride"), stride.to_string()));
        }
        let kernel = c.get(gpu, &stem, &format!("h3_{stem}"), &cfg)?;
        Ok(Self {
            kernel,
            n: n_size,
            resid,
            bias,
            elem: elem.into(),
            m_group,
            m_tile,
            n_tile,
            threads,
        })
    }

    /// `(tokens, a_q, w_q[, w_s, a_s], out[, gate, cls][, bias])` — the float operands carry no scales.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        profile: Option<&mut Profile>,
        stage: &str,
        tokens: u32,
        a_q: View<'_>,
        w_q: View<'_>,
        scales: Option<(View<'_>, View<'_>)>,
        out: View<'_>,
        residual: Option<(View<'_>, View<'_>)>,
        bias: Option<View<'_>>,
    ) -> Result<()> {
        let mut bindings = vec![a_q, w_q];
        if quantised(&self.elem) {
            let (w_s, a_s) = scales.expect("an int8 GEMM needs its weight and token scales");
            bindings.extend([w_s, a_s]);
        }
        bindings.push(out);
        if self.resid {
            let (gate, cls) = residual.expect("a residual GEMM needs its gate and class rows");
            bindings.extend([gate, cls]);
        }
        if self.bias {
            bindings.push(bias.expect("a biased GEMM needs its bias"));
        }
        launch(
            gpu,
            &self.kernel,
            profile,
            stage,
            [
                (self.n / self.n_tile) as u32,
                gemm_grid_y(tokens as usize, self.m_group, self.m_tile),
                1,
            ],
            [self.threads, 1, 1],
            &[tokens],
            &bindings,
        )
    }

    pub fn m_group(&self) -> u32 {
        self.m_group
    }
}

/// A causal 3-D convolution as an implicit GEMM over channels-last f16.
///
/// The VAE encoder's whole stack is these: the taps are folded into the K dimension, so a 3x3x3
/// convolution over 128 channels is a GEMM with K = 27*128 rounded up to a multiple of 32. The `add`
/// form takes a residual as a fifth binding, which is how a ResNet block's second convolution and its
/// skip are one dispatch.
pub struct Conv3d {
    kernel: Arc<hrx::Kernel>,
    cout_pad: usize,
    /// the output extents, which the caller needs to drive the next layer
    pub tout: usize,
    pub ho: usize,
    pub wo: usize,
}

impl Conv3d {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        c: &Compiler,
        gpu: &hrx::Gpu,
        add: bool,
        frames: usize,
        h: usize,
        w: usize,
        stride: usize,
        tstride: usize,
        taps_t: usize,
        cin_pad: usize,
        cin_stride: usize,
        k_size: usize,
        cout_pad: usize,
    ) -> Result<Self> {
        let stem = if add {
            "conv3d_f16_wmma_add"
        } else {
            "conv3d_f16_wmma"
        };
        let ns = format!("h3.{stem}.");
        let rows_bound = (frames * h * w).div_ceil(64) * 64;
        let cfg: Cfg = vec![
            (format!("{ns}frames"), frames.to_string()),
            (format!("{ns}height"), h.to_string()),
            (format!("{ns}width"), w.to_string()),
            (format!("{ns}stride"), stride.to_string()),
            (format!("{ns}tstride"), tstride.to_string()),
            (format!("{ns}taps_t"), taps_t.to_string()),
            (format!("{ns}cin_pad"), cin_pad.to_string()),
            (format!("{ns}cin_stride"), cin_stride.to_string()),
            (format!("{ns}rows_bound"), rows_bound.to_string()),
            (format!("{ns}k_size"), k_size.to_string()),
            (format!("{ns}n_size"), cout_pad.to_string()),
        ];
        Ok(Self {
            kernel: c.get(gpu, stem, &format!("h3_{stem}"), &cfg)?,
            cout_pad,
            tout: (frames - 1) / tstride + 1,
            ho: h / stride,
            wo: w / stride,
        })
    }

    /// Output rows, which is the next layer's `frames * h * w`.
    pub fn rows(&self) -> usize {
        self.tout * self.ho * self.wo
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        profile: Option<&mut Profile>,
        stage: &str,
        a: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out: View<'_>,
        residual: Option<View<'_>>,
    ) -> Result<()> {
        let m = self.rows();
        let mut bindings = vec![a, w, b, out];
        bindings.extend(residual);
        launch(
            gpu,
            &self.kernel,
            profile,
            stage,
            [(self.cout_pad / 64) as u32, m.div_ceil(64) as u32, 1],
            [256, 1, 1],
            &[m as u32],
            &bindings,
        )
    }
}

/// GroupNorm over 32 groups followed by SiLU, in two dispatches: the statistics, then the scaling.
///
/// The split is not an optimisation detail — the statistics are over a whole (frame, group) plane, so
/// they have to land before any element is scaled.
pub struct GroupNormSilu {
    stats: Arc<hrx::Kernel>,
    silu: Arc<hrx::Kernel>,
    frames: usize,
    rows: usize,
    channels: usize,
}

impl GroupNormSilu {
    pub fn build(
        c: &Compiler,
        gpu: &hrx::Gpu,
        frames: usize,
        h: usize,
        w: usize,
        channels: usize,
    ) -> Result<Self> {
        let plane = h * w;
        let rows = frames * plane;
        let rows_bound = rows.div_ceil(64) * 64;
        let (ns, na) = ("h3.gn_stats_f16.", "h3.gn_silu_f16.");
        let stats_cfg: Cfg = vec![
            (format!("{ns}channels"), channels.to_string()),
            (format!("{ns}groups"), "32".into()),
            (format!("{ns}plane"), plane.to_string()),
            (format!("{ns}rows_bound"), rows_bound.to_string()),
        ];
        let silu_cfg: Cfg = vec![
            (format!("{na}channels"), channels.to_string()),
            (format!("{na}groups"), "32".into()),
            (format!("{na}plane"), plane.to_string()),
            (format!("{na}rows_bound"), rows_bound.to_string()),
            (format!("{na}eps"), num(1e-6)),
        ];
        Ok(Self {
            stats: c.get(gpu, "gn_stats_f16", "h3_gn_stats_f16", &stats_cfg)?,
            silu: c.get(gpu, "gn_silu_f16", "h3_gn_silu_f16", &silu_cfg)?,
            frames,
            rows,
            channels,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        mut profile: Option<&mut Profile>,
        stage: &str,
        x: View<'_>,
        gamma: View<'_>,
        beta: View<'_>,
        stats: View<'_>,
        out: View<'_>,
    ) -> Result<()> {
        launch(
            gpu,
            &self.stats,
            profile.as_deref_mut(),
            stage,
            [self.frames as u32, 32, 1],
            [32, 1, 1],
            &[self.frames as u32],
            &[x, stats],
        )?;
        launch(
            gpu,
            &self.silu,
            profile,
            stage,
            [(self.rows * self.channels).div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            &[self.frames as u32],
            &[x, stats, gamma, beta, out],
        )
    }
}

/// A biased f16 matmul with f16 in and out: the encoder's 1x1 shortcuts and its posterior head.
pub struct Matmul {
    kernel: Arc<hrx::Kernel>,
    n_size: usize,
}

impl Matmul {
    pub fn build(c: &Compiler, gpu: &hrx::Gpu, k_size: usize, n_size: usize) -> Result<Self> {
        let stem = "matmul_bias_f16_wmma_af16_cf16";
        let ns = format!("h3.{stem}.");
        let cfg: Cfg = vec![
            (format!("{ns}k_size"), k_size.to_string()),
            (format!("{ns}n_size"), n_size.to_string()),
        ];
        Ok(Self {
            kernel: c.get(gpu, stem, &format!("h3_{stem}"), &cfg)?,
            n_size,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        profile: Option<&mut Profile>,
        stage: &str,
        rows: usize,
        a: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out: View<'_>,
    ) -> Result<()> {
        launch(
            gpu,
            &self.kernel,
            profile,
            stage,
            [(self.n_size / 64) as u32, rows.div_ceil(64) as u32, 1],
            [256, 1, 1],
            &[rows as u32],
            &[a, w, b, out],
        )
    }
}

/// `out[m][n] = x[m][k] . w[n][k] + b`, one lane per output element.
///
/// The plain f32 matmul the heads and the patch projections use: no tiling, no quantisation, just the
/// arithmetic in the order the checkpoint stores it.
pub struct MatmulF32 {
    kernel: Arc<hrx::Kernel>,
    n: usize,
}

impl MatmulF32 {
    pub fn build(c: &Compiler, gpu: &hrx::Gpu, k: usize, n: usize) -> Result<Self> {
        let ns = "h3.matmul_f32.";
        let cfg: Cfg = vec![
            (format!("{ns}k"), k.to_string()),
            (format!("{ns}n"), n.to_string()),
        ];
        Ok(Self {
            kernel: c.get(gpu, "matmul_f32", "h3_matmul_f32", &cfg)?,
            n,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        profile: Option<&mut Profile>,
        stage: &str,
        m: usize,
        x: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out: View<'_>,
    ) -> Result<()> {
        launch(
            gpu,
            &self.kernel,
            profile,
            stage,
            [self.n.div_ceil(256) as u32, m as u32, 1],
            [THREADS, 1, 1],
            &[m as u32],
            &[x, w, b, out],
        )
    }
}

/// `y = a x + b y`, elementwise. The coefficients are compiled in, so each pair is its own kernel —
/// which is why they are spelled with the same `%.17g` every other float config uses.
#[allow(clippy::too_many_arguments)]
pub fn axpy(
    c: &Compiler,
    gpu: &hrx::Gpu,
    profile: Option<&mut Profile>,
    stage: &str,
    a: f32,
    b: f32,
    count: usize,
    x: View<'_>,
    y: View<'_>,
) -> Result<()> {
    let ns = "h3.axpy_f32.";
    let cfg: Cfg = vec![
        (format!("{ns}a"), num(f64::from(a))),
        (format!("{ns}b"), num(f64::from(b))),
    ];
    let kernel = c.get(gpu, "axpy_f32", "h3_axpy_f32", &cfg)?;
    launch(
        gpu,
        &kernel,
        profile,
        stage,
        [count.div_ceil(256) as u32, 1, 1],
        [THREADS, 1, 1],
        &[count as u32],
        &[x, y],
    )
}

/// The vision tower's GEMM family: `matmul_<kind>_bf16_wmma`, whose weight rows are bf16 as stored.
///
/// The kind decides what happens after the multiply — `bias` stops there, `gelu` and `gelu_erf` apply
/// their activation, and `resid` adds into an existing f16 stream. The two GELUs are not the same
/// function and are not interchangeable: the tower's MLP uses the tanh approximation and its mergers
/// the error function.
pub struct Matmul16 {
    kernel: Arc<hrx::Kernel>,
    n: usize,
    resid: bool,
}

impl Matmul16 {
    pub fn build(c: &Compiler, gpu: &hrx::Gpu, kind: &str, k: usize, n: usize) -> Result<Self> {
        let stem = format!("matmul_{kind}_bf16_wmma");
        let ns = format!("h3.{stem}.");
        let cfg: Cfg = vec![
            (format!("{ns}k_size"), k.to_string()),
            (format!("{ns}n_size"), n.to_string()),
        ];
        Ok(Self {
            kernel: c.get(gpu, &stem, &format!("h3_{stem}"), &cfg)?,
            n,
            resid: kind == "resid",
        })
    }

    /// `lambda` is the residual form's per-column scale, and is required by exactly that form.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        profile: Option<&mut Profile>,
        stage: &str,
        m: usize,
        a: View<'_>,
        w: View<'_>,
        bias: View<'_>,
        out: View<'_>,
        lambda: Option<View<'_>>,
    ) -> Result<()> {
        debug_assert_eq!(
            self.resid,
            lambda.is_some(),
            "only the resid form takes a lambda"
        );
        let mut bindings = vec![a, w, bias, out];
        bindings.extend(lambda);
        launch(
            gpu,
            &self.kernel,
            profile,
            stage,
            [(self.n / 64) as u32, m.div_ceil(64) as u32, 1],
            [256, 1, 1],
            &[m as u32],
            &bindings,
        )
    }
}

/// LayerNorm reading an f16 stream and writing f32, which is what the vision tower's blocks take.
pub struct LayerNorm16 {
    kernel: Arc<hrx::Kernel>,
}

impl LayerNorm16 {
    pub fn build(c: &Compiler, gpu: &hrx::Gpu, width: usize, eps: f64) -> Result<Self> {
        let ns = "h3.layernorm_f16_f32.";
        let cfg: Cfg = vec![
            (format!("{ns}width"), width.to_string()),
            (format!("{ns}eps"), num(eps)),
        ];
        Ok(Self {
            kernel: c.get(gpu, "layernorm_f16_f32", "h3_layernorm_f16_f32", &cfg)?,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        gpu: &hrx::Gpu,
        profile: Option<&mut Profile>,
        rows: usize,
        x16: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out32: View<'_>,
    ) -> Result<()> {
        launch(
            gpu,
            &self.kernel,
            profile,
            "vision layernorm",
            [rows as u32, 1, 1],
            [32, 1, 1],
            &[rows as u32],
            &[x16, w, b, out32],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stem names, which decide which kernel source is compiled. Built without a GPU by
    /// reproducing the same string the builder does.
    fn gemm_stem(elem: &str, mode: &str, bias: bool, gate_first: bool, tile: Tile) -> String {
        format!(
            "gemm_{elem}{}{}_256{}{}",
            match tile {
                Tile::Wide => "_wide",
                Tile::Fast => "_fast",
                Tile::Plain => "",
            },
            if mode == "plain" {
                String::new()
            } else {
                format!("_{mode}")
            },
            if bias { "b" } else { "" },
            if mode == "swiglu" && !gate_first {
                "_gs"
            } else {
                ""
            },
        )
    }

    #[test]
    fn gemm_stems_name_kernels_that_exist() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels");
        for (elem, mode, bias, gate_first, tile) in [
            ("i8", "plain", false, true, Tile::Plain),
            ("i8", "resid", false, true, Tile::Plain),
            ("i8", "swiglu", false, true, Tile::Plain),
            ("i8", "plain", true, true, Tile::Plain),
            ("i8", "resid", true, true, Tile::Plain),
            ("i8", "swiglu", true, false, Tile::Plain),
            ("f16", "plain", true, true, Tile::Plain),
            ("f16", "resid", true, true, Tile::Plain),
            ("f16", "swiglu", true, false, Tile::Plain),
            ("bf16", "plain", false, true, Tile::Plain),
            ("bf16", "resid", false, true, Tile::Plain),
            ("bf16", "swiglu", false, true, Tile::Plain),
            ("f16", "plain", true, true, Tile::Fast),
            ("f16", "resid", true, true, Tile::Fast),
            ("f16", "swiglu", true, false, Tile::Fast),
            ("f16", "plain", true, true, Tile::Wide),
        ] {
            let stem = gemm_stem(elem, mode, bias, gate_first, tile);
            assert!(
                root.join(format!("{stem}.loom")).exists(),
                "no kernel source for {stem}"
            );
        }
    }

    #[test]
    fn prepare_stems_name_kernels_that_exist() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../kernels");
        for elem in ["i8", "f16", "bf16"] {
            for form in ["norm", "lnorm", "plain"] {
                let stem = format!("prepare_{form}_{elem}");
                assert!(
                    root.join(format!("{stem}.loom")).exists(),
                    "no source for {stem}"
                );
            }
        }
        // the wide-row int8 plain form stages as f16
        assert!(root.join("prepare_plain16_i8.loom").exists());
    }

    #[test]
    fn the_decoder_tiles_refuse_what_they_cannot_do() {
        // These are compile-time refusals in build(); reproduced here as the same predicate, since
        // exercising build() needs a GPU.
        let bad = |elem: &str, bias: bool, n: usize, mode: &str, gate_first: bool| {
            elem != "f16" || !bias || !n.is_multiple_of(256) || (mode == "swiglu" && gate_first)
        };
        assert!(bad("i8", true, 2048, "plain", false)); // not f16
        assert!(bad("f16", false, 2048, "plain", false)); // no bias
        assert!(bad("f16", true, 2050, "plain", false)); // N not a multiple of 256
        assert!(bad("f16", true, 2048, "swiglu", true)); // gate-first SwiGLU
        assert!(!bad("f16", true, 2048, "swiglu", false)); // the shape the decoder uses
    }
}
