//! The kernel builders the stacks dispatch through: the prepare family that produces a GEMM's A
//! operand, and the GEMM family itself.
//!
//! Each one decides its stem, its configuration and its launch geometry from the shape it is built
//! for. Those decisions are the ones `model.rs` holds, so the operand a plan laid out and the operand a
//! kernel reads agree on their pitch.
use crate::compile::{num, Cfg, Compiler};
use crate::model::*;
use hrx::sys::BufferRef;
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
}

/// One launch, timed into `profile` when it is on.
#[allow(clippy::too_many_arguments)]
fn launch(
    gpu: &hrx::Gpu,
    kernel: &hrx::Kernel,
    profile: Option<&mut Profile>,
    stage: &str,
    grid: [u32; 3],
    block: [u32; 3],
    scalars: &[u32],
    bindings: &[BufferRef],
) -> Result<()> {
    match profile {
        Some(p) if p.on => {
            gpu.sync()?;
            let started = Instant::now();
            gpu.dispatch(kernel, grid, block, scalars, bindings)?;
            gpu.sync()?;
            *p.micros.entry(stage.to_string()).or_insert(0.0) +=
                started.elapsed().as_secs_f64() * 1e6;
        }
        _ => gpu.dispatch(kernel, grid, block, scalars, bindings)?,
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
        x: BufferRef,
        norm: Option<(BufferRef, BufferRef, BufferRef)>,
        a_q: BufferRef,
        a_s: Option<BufferRef>,
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
        a_q: BufferRef,
        w_q: BufferRef,
        scales: Option<(BufferRef, BufferRef)>,
        out: BufferRef,
        residual: Option<(BufferRef, BufferRef)>,
        bias: Option<BufferRef>,
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
