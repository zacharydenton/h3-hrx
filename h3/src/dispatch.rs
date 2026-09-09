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
/// Inherits [`hrx::Stream::dispatch`]'s contract: `grid`, `block`, `scalars` and `bindings` must be what
/// `kernel` was compiled for. Private to the crate, and every caller goes through [`checked`], which
/// discharges the binding half of that contract against the extents the kernel was configured with.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn launch(
    stream: &mut hrx::Stream,
    kernel: &hrx::Kernel,
    profile: Option<&mut Profile>,
    stage: &str,
    grid: [u32; 3],
    block: [u32; 3],
    scalars: &[u32],
    bindings: &[View<'_>],
) -> Result<()> {
    // Every H3 scalar is an unsigned Loom index. HRX handles the compiler's
    // checked 32/64-bit index lowering; arbitrary mixed scalars are never inferred.
    let constants = hrx::Constants::indices(kernel, scalars)?;
    match profile {
        Some(p) if p.on => {
            stream.synchronize()?;
            let started = Instant::now();
            unsafe { stream.dispatch(kernel, grid, block, &constants, bindings)? };
            stream.synchronize()?;
            *p.micros.entry(stage.to_string()).or_insert(0.0) +=
                started.elapsed().as_secs_f64() * 1e6;
        }
        _ => unsafe { stream.dispatch(kernel, grid, block, &constants, bindings)? },
    }
    Ok(())
}

/// Upload into `dst` at a byte offset.
///
/// `Stream::upload` takes the view it writes, so an offset upload is an upload into a slice — this
/// just names the length, which is the source's, rather than repeating it at every call.
pub fn upload_at(
    stream: &mut hrx::Stream,
    dst: &hrx::Buffer,
    offset: usize,
    bytes: &[u8],
) -> Result<()> {
    Ok(stream.upload(dst.binding().slice(offset, bytes.len())?, bytes)?)
}

/// A launch whose bindings have been checked against the extents its kernel addresses.
///
/// This is what lets everything above it be safe. A kernel is compiled from a set of extents — widths,
/// strides, token capacities — and `required` restates that same arithmetic in bytes per binding, so
/// the check is not a guess about what the kernel does: it is the numbers the kernel was built from. A
/// binding shorter than its entry is a caller's mistake, and it becomes an error here rather than an
/// out-of-bounds access on the device.
///
/// The bound is the extent the kernel *addresses*, which is not always the extent it fills: a GEMM
/// tiled 64 rows at a time reads its A operand at the compiled pitch, and an attention kernel indexes
/// to its token capacity rather than to the tokens of the call. Where the two differ the padded number
/// is the one that belongs here.
#[allow(clippy::too_many_arguments)]
pub(crate) fn checked(
    stream: &mut hrx::Stream,
    kernel: &crate::compile::Kernel,
    profile: Option<&mut Profile>,
    stage: &str,
    grid: [u32; 3],
    block: [u32; 3],
    scalars: &[u32],
    bindings: &[View<'_>],
    required: &[usize],
) -> Result<()> {
    debug_assert_eq!(
        bindings.len(),
        required.len(),
        "{stage}: a bound per binding"
    );
    for (i, (view, need)) in bindings.iter().zip(required).enumerate() {
        if view.len() < *need {
            return Err(crate::compile::Error::Io(format!(
                "{stage}: binding {i} is {} bytes, the kernel addresses {need}",
                view.len()
            )));
        }
    }
    // The kernel is built here if its batch has not been built already, which is what makes the
    // handle safe to hold: nothing can dispatch one that was never compiled.
    let kernel = kernel.resolve(stream)?;
    // Safety: every binding is at least as long as the extent this kernel was compiled to address,
    // checked immediately above, and the grid, block and scalars come from the same builder that
    // compiled it.
    unsafe {
        launch(
            stream, kernel, profile, stage, grid, block, scalars, bindings,
        )
    }
}

/// The most buffers any of these kernels binds.
const MAX_BINDINGS: usize = 8;

/// One launch's bindings and the extents they must cover, built on the stack.
///
/// The pair travels together because [`checked`] reads them in step, and building them as two
/// `Vec`s meant two allocations per launch — of which a denoising step makes thousands. Every
/// kernel binds at least one buffer, so the first one also fills the slots that stay unused.
struct Args<'v> {
    views: [View<'v>; MAX_BINDINGS],
    need: [usize; MAX_BINDINGS],
    n: usize,
}

impl<'v> Args<'v> {
    fn new(view: View<'v>, need: usize) -> Self {
        let mut args = Self {
            views: [view; MAX_BINDINGS],
            need: [0; MAX_BINDINGS],
            n: 1,
        };
        args.need[0] = need;
        args
    }

    fn push(&mut self, view: View<'v>, need: usize) {
        assert!(self.n < MAX_BINDINGS, "more than {MAX_BINDINGS} bindings");
        self.views[self.n] = view;
        self.need[self.n] = need;
        self.n += 1;
    }

    fn views(&self) -> &[View<'v>] {
        &self.views[..self.n]
    }

    fn need(&self) -> &[usize] {
        &self.need[..self.n]
    }
}

/// One `i32` per row, each of them a row of a modulation table.
///
/// The prepare, residual GEMM and `norm_mod_f32` kernels take that value as a table index under an
/// `index.assume` — `lt(%cls_idx0, %classes)` — so the device does not range-check it, and a class
/// past the table's rows reads memory the table does not own. Checking the values at the launch
/// would mean reading the buffer back from the device, which is a synchronise on every dispatch.
///
/// So the check happens where the values are written, and the bound they were checked against
/// travels with them in this type. A kernel takes these rows only when its own table is at least
/// that wide, which is what `ClassRows::against` decides. The buffer is only ever written here,
/// so a `Classes` that exists has been checked.
pub struct Classes {
    buf: hrx::Buffer,
    capacity: usize,
    /// how many leading rows hold checked values; the rest of the allocation is not offered
    written: usize,
    /// every written value is less than this
    bound: usize,
}

impl Classes {
    /// `rows` of class zero, which is a row of every table.
    pub fn zeroed(stream: &mut hrx::Stream, rows: usize) -> Result<Self> {
        let bytes = rows.max(1) * 4;
        let buf = stream.allocate(bytes)?;
        stream.fill(buf.slice(0, bytes), 0)?;
        Ok(Self {
            buf,
            capacity: rows,
            written: rows,
            bound: 1,
        })
    }

    /// Uploads `values` as the leading rows, refusing any that is not a row of a `classes`-row table.
    ///
    /// A refusal leaves nothing offered: the rows are marked unwritten first, so a caller that
    /// ignores the error cannot bind the stale ones.
    pub fn write(
        &mut self,
        stream: &mut hrx::Stream,
        values: &[i32],
        classes: usize,
    ) -> Result<()> {
        self.written = 0;
        checkable(values, classes, self.capacity)?;
        crate::dispatch::upload_at(stream, &self.buf, 0, bytemuck::cast_slice(values))?;
        self.written = values.len();
        self.bound = classes;
        Ok(())
    }

    /// Every row that has been checked.
    pub fn all(&self) -> ClassRows<'_> {
        self.slice(0, self.written)
    }

    /// `rows` of them from `from`, which is how a stage takes the classes of its own span.
    pub fn slice(&self, from: usize, rows: usize) -> ClassRows<'_> {
        assert!(
            from + rows <= self.written,
            "class rows past what was written"
        );
        ClassRows {
            view: self.buf.slice(from * 4, rows * 4),
            rows,
            bound: self.bound,
        }
    }
}

/// Whether every value is a row of a `classes`-row table, which is the half of [`Classes::write`]
/// that needs no device.
pub(crate) fn classes_fit(values: &[i32], classes: usize) -> Result<()> {
    if let Some(bad) = values.iter().find(|v| **v < 0 || **v as usize >= classes) {
        return Err(crate::compile::Error::Io(format!(
            "class {bad} is not one of the {classes} a table has"
        )));
    }
    Ok(())
}

/// What [`Classes::write`] accepts, without a device: rows that fit, each one a row of the table.
fn checkable(values: &[i32], classes: usize, capacity: usize) -> Result<()> {
    if values.len() > capacity {
        return Err(crate::compile::Error::Io(format!(
            "{} classes do not fit {capacity} rows",
            values.len()
        )));
    }
    classes_fit(values, classes)
}

/// A run of checked class rows, and the table width they were checked against.
#[derive(Clone, Copy)]
pub struct ClassRows<'a> {
    view: View<'a>,
    rows: usize,
    bound: usize,
}

impl<'a> ClassRows<'a> {
    /// The binding, once the kernel's own table has been found wide enough for these values and long
    /// enough for its launch.
    fn against(&self, stage: &str, tokens: usize, classes: usize) -> Result<View<'a>> {
        if self.rows < tokens {
            return Err(crate::compile::Error::Io(format!(
                "{stage}: {tokens} rows want a class each, {} given",
                self.rows
            )));
        }
        if self.bound > classes {
            return Err(crate::compile::Error::Io(format!(
                "{stage}: classes checked against {} rows, the table has {classes}",
                self.bound
            )));
        }
        Ok(self.view)
    }
}

/// Produces a GEMM's A operand: `norm` and `lnorm` normalise and modulate the f32 residual stream,
/// `plain` narrows an existing f16 row. The int8 forms also write a per-token scale; the float ones
/// write rows and nothing else.
pub struct Prepare {
    kernel: crate::compile::Kernel,
    lanes: usize,
    form: String,
    elem: String,
    /// the extents the kernel was compiled from, kept so `run` can bound its bindings
    width: usize,
    out_stride: usize,
    classes: usize,
}

impl Prepare {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
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
        let module = if stem == "prepare_plain16_i8" {
            stem.clone()
        } else {
            format!("prepare_{elem}_family")
        };
        let kernel = c.get(stream, &module, &format!("h3_{stem}"), &cfg)?;
        Ok(Self {
            kernel,
            lanes,
            form: form.into(),
            elem: elem.into(),
            width,
            out_stride: if out_stride != 0 { out_stride } else { width },
            classes,
        })
    }

    /// norm forms take `(x, weight, table, cls)`; plain takes `(h)` alone. `a_s` is written only by
    /// the int8 forms.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        profile: Option<&mut Profile>,
        stage: &str,
        tokens: u32,
        x: View<'_>,
        norm: Option<(View<'_>, View<'_>, ClassRows<'_>)>,
        a_q: View<'_>,
        a_s: Option<View<'_>>,
    ) -> Result<()> {
        let t = tokens as usize;
        // the norm forms read the f32 residual stream; `plain` narrows an f16 row that a GEMM
        // already wrote
        let in_bytes = if self.form == "plain" { 2 } else { 4 };
        let mut args = Args::new(x, t * self.width * in_bytes);
        if self.form != "plain" {
            let (weight, table, cls) =
                norm.expect("a norm prepare needs its weight, table and class");
            // the norm's weight, its (scale, shift) table per class, and a class per row
            args.push(weight, self.width * 4);
            args.push(table, 2 * self.classes * self.width * 4);
            args.push(cls.against(stage, t, self.classes)?, t * 4);
        }
        args.push(a_q, t * self.out_stride * elem_bits(&self.elem) / 8);
        if quantised(&self.elem) {
            args.push(a_s.expect("an int8 prepare writes a token scale"), t * 4);
        }
        checked(
            stream,
            &self.kernel,
            profile,
            stage,
            [tokens, 1, 1],
            [self.lanes as u32, 1, 1],
            &[tokens],
            args.views(),
            args.need(),
        )
    }
}

/// A GEMM of the int8, f16 or bf16 family for one (K, N, row group).
pub struct Gemm {
    kernel: crate::compile::Kernel,
    n: usize,
    resid: bool,
    bias: bool,
    elem: String,
    m_group: u32,
    m_tile: usize,
    n_tile: usize,
    threads: u32,
    /// the extents the kernel was compiled from, kept so `run` can bound its bindings. The output
    /// is narrower than N in the SwiGLU forms, where the gate consumes half the columns, and it is
    /// f32 only in the residual forms — every other form writes the f16 stream.
    k_stride: usize,
    classes: usize,
    out_width: usize,
    out_bytes: usize,
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
        stream: &mut hrx::Stream,
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
        let mut out_width = if mode == "swiglu" { n_size / 2 } else { n_size };
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
            out_width = stride;
        }
        let module = if tile == Tile::Plain && quantised(elem) {
            "gemm_packed_256".into()
        } else if tile == Tile::Plain {
            format!("gemm_{elem}_family")
        } else {
            stem.clone()
        };
        let kernel = c.get(stream, &module, &format!("h3_{stem}"), &cfg)?;
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
            k_stride: if k_stride != 0 { k_stride } else { k_size },
            classes,
            out_width,
            out_bytes: if resid { 4 } else { 2 },
        })
    }

    /// `(tokens, a_q, w_q[, w_s, a_s], out[, gate, cls][, bias])` — the float operands carry no scales.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        profile: Option<&mut Profile>,
        stage: &str,
        tokens: u32,
        a_q: View<'_>,
        w_q: View<'_>,
        scales: Option<(View<'_>, View<'_>)>,
        out: View<'_>,
        residual: Option<(View<'_>, ClassRows<'_>)>,
        bias: Option<View<'_>>,
    ) -> Result<()> {
        let t = tokens as usize;
        let bytes = elem_bits(&self.elem) / 8;
        // Both operands are pitched to `k_stride` in the operand's element type, and every access is
        // guarded to the token count, so the row extent is `t` rather than the grid's rounded-up
        // cover. The int8 forms read the same rows as packed quads, which is the same byte count.
        let mut args = Args::new(a_q, t * self.k_stride * bytes);
        args.push(w_q, self.n * self.k_stride * bytes);
        if quantised(&self.elem) {
            let (w_s, a_s) = scales.expect("an int8 GEMM needs its weight and token scales");
            args.push(w_s, self.n * 4);
            args.push(a_s, t * 4);
        }
        args.push(out, t * self.out_width * self.out_bytes);
        if self.resid {
            let (gate, cls) = residual.expect("a residual GEMM needs its gate and class rows");
            args.push(gate, self.classes * self.n * 4);
            args.push(cls.against(stage, t, self.classes)?, t * 4);
        }
        if self.bias {
            args.push(bias.expect("a biased GEMM needs its bias"), self.n * 4);
        }
        checked(
            stream,
            &self.kernel,
            profile,
            stage,
            [
                (self.n / self.n_tile) as u32,
                gemm_grid_y(t, self.m_group, self.m_tile),
                1,
            ],
            [self.threads, 1, 1],
            &[tokens],
            args.views(),
            args.need(),
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
    kernel: crate::compile::Kernel,
    cout_pad: usize,
    /// the extents the kernel was compiled from
    in_rows: usize,
    cin_stride: usize,
    k_size: usize,
    /// the output extents, which the caller needs to drive the next layer
    pub tout: usize,
    pub ho: usize,
    pub wo: usize,
}

impl Conv3d {
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
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
            kernel: c.get(stream, "conv3d_f16_family", &format!("h3_{stem}"), &cfg)?,
            cout_pad,
            in_rows: frames * h * w,
            cin_stride,
            k_size,
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
        stream: &mut hrx::Stream,
        profile: Option<&mut Profile>,
        stage: &str,
        a: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out: View<'_>,
        residual: Option<View<'_>>,
    ) -> Result<()> {
        let m = self.rows();
        // input rows at their channel pitch, the folded taps as [cout_pad][k_size], an f32 bias per
        // output channel, and the output rows — all f16 but the bias
        let plane = m * self.cout_pad * 2;
        let mut args = Args::new(a, self.in_rows * self.cin_stride * 2);
        args.push(w, self.cout_pad * self.k_size * 2);
        args.push(b, self.cout_pad * 4);
        args.push(out, plane);
        if let Some(r) = residual {
            args.push(r, plane);
        }
        checked(
            stream,
            &self.kernel,
            profile,
            stage,
            [(self.cout_pad / 64) as u32, m.div_ceil(64) as u32, 1],
            [256, 1, 1],
            &[m as u32],
            args.views(),
            args.need(),
        )
    }
}

/// GroupNorm over 32 groups followed by SiLU, in two dispatches: the statistics, then the scaling.
///
/// The split is not an optimisation detail — the statistics are over a whole (frame, group) plane, so
/// they have to land before any element is scaled.
pub struct GroupNormSilu {
    stats: crate::compile::Kernel,
    silu: crate::compile::Kernel,
    frames: usize,
    rows: usize,
    channels: usize,
}

impl GroupNormSilu {
    pub fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
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
            stats: c.get(stream, "gn_stats_f16", "h3_gn_stats_f16", &stats_cfg)?,
            silu: c.get(stream, "gn_silu_f16", "h3_gn_silu_f16", &silu_cfg)?,
            frames,
            rows,
            channels,
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        mut profile: Option<&mut Profile>,
        stage: &str,
        x: View<'_>,
        gamma: View<'_>,
        beta: View<'_>,
        stats: View<'_>,
        out: View<'_>,
    ) -> Result<()> {
        // f16 planes, an f32 gamma and beta per channel, and two f32 statistics per (frame, group)
        let plane = self.rows * self.channels * 2;
        let stats_bytes = self.frames * 32 * 2 * 4;
        checked(
            stream,
            &self.stats,
            profile.as_deref_mut(),
            stage,
            [self.frames as u32, 32, 1],
            [32, 1, 1],
            &[self.frames as u32],
            &[x, stats],
            &[plane, stats_bytes],
        )?;
        checked(
            stream,
            &self.silu,
            profile,
            stage,
            [(self.rows * self.channels).div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            &[self.frames as u32],
            &[x, stats, gamma, beta, out],
            &[
                plane,
                stats_bytes,
                self.channels * 4,
                self.channels * 4,
                plane,
            ],
        )
    }
}

/// A biased f16 matmul with f16 in and out: the encoder's 1x1 shortcuts and its posterior head.
pub struct Matmul {
    kernel: crate::compile::Kernel,
    k_size: usize,
    n_size: usize,
}

impl Matmul {
    pub fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
        k_size: usize,
        n_size: usize,
    ) -> Result<Self> {
        let stem = "matmul_bias_f16_wmma_af16_cf16";
        let ns = format!("h3.{stem}.");
        let cfg: Cfg = vec![
            (format!("{ns}k_size"), k_size.to_string()),
            (format!("{ns}n_size"), n_size.to_string()),
        ];
        Ok(Self {
            kernel: c.get(stream, stem, &format!("h3_{stem}"), &cfg)?,
            k_size,
            n_size,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        profile: Option<&mut Profile>,
        stage: &str,
        rows: usize,
        a: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out: View<'_>,
    ) -> Result<()> {
        checked(
            stream,
            &self.kernel,
            profile,
            stage,
            [(self.n_size / 64) as u32, rows.div_ceil(64) as u32, 1],
            [256, 1, 1],
            &[rows as u32],
            &[a, w, b, out],
            &[
                rows * self.k_size * 2,
                self.n_size * self.k_size * 2,
                self.n_size * 4,
                rows * self.n_size * 2,
            ],
        )
    }
}

/// `out[m][n] = x[m][k] . w[n][k] + b`, one lane per output element.
///
/// The plain f32 matmul the heads and the patch projections use: no tiling, no quantisation, just the
/// arithmetic in the order the checkpoint stores it.
pub struct MatmulF32 {
    kernel: crate::compile::Kernel,
    k: usize,
    n: usize,
}

impl MatmulF32 {
    pub fn build(c: &Compiler, stream: &mut hrx::Stream, k: usize, n: usize) -> Result<Self> {
        let ns = "h3.matmul_f32.";
        let cfg: Cfg = vec![
            (format!("{ns}k"), k.to_string()),
            (format!("{ns}n"), n.to_string()),
        ];
        Ok(Self {
            kernel: c.get(stream, "matmul_f32", "h3_matmul_f32", &cfg)?,
            k,
            n,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        profile: Option<&mut Profile>,
        stage: &str,
        m: usize,
        x: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out: View<'_>,
    ) -> Result<()> {
        checked(
            stream,
            &self.kernel,
            profile,
            stage,
            [self.n.div_ceil(256) as u32, m as u32, 1],
            [THREADS, 1, 1],
            &[m as u32],
            &[x, w, b, out],
            &[
                m * self.k * 4,
                self.n * self.k * 4,
                self.n * 4,
                m * self.n * 4,
            ],
        )
    }
}

/// `y = a x + b y`, elementwise. The coefficients are compiled in, so each pair is its own kernel —
/// which is why they are spelled with the same `%.17g` every other float config uses.
#[allow(clippy::too_many_arguments)]
pub fn axpy(
    c: &Compiler,
    stream: &mut hrx::Stream,
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
    let kernel = c.get(stream, "axpy_f32", "h3_axpy_f32", &cfg)?;
    checked(
        stream,
        &kernel,
        profile,
        stage,
        [count.div_ceil(256) as u32, 1, 1],
        [THREADS, 1, 1],
        &[count as u32],
        &[x, y],
        &[count * 4, count * 4],
    )
}

/// The vision tower's GEMM family: `matmul_<kind>_bf16_wmma`, whose weight rows are bf16 as stored.
///
/// The kind decides what happens after the multiply — `bias` stops there, `gelu` and `gelu_erf` apply
/// their activation, and `resid` adds into an existing f16 stream. The two GELUs are not the same
/// function and are not interchangeable: the tower's MLP uses the tanh approximation and its mergers
/// the error function.
pub struct Matmul16 {
    kernel: crate::compile::Kernel,
    k: usize,
    n: usize,
    resid: bool,
}

impl Matmul16 {
    pub fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
        kind: &str,
        k: usize,
        n: usize,
    ) -> Result<Self> {
        let stem = format!("matmul_{kind}_bf16_wmma");
        let ns = format!("h3.{stem}.");
        let cfg: Cfg = vec![
            (format!("{ns}k_size"), k.to_string()),
            (format!("{ns}n_size"), n.to_string()),
        ];
        Ok(Self {
            kernel: c.get(
                stream,
                if kind == "resid" {
                    &stem
                } else {
                    "matmul_bf16_family"
                },
                &format!("h3_{stem}"),
                &cfg,
            )?,
            k,
            n,
            resid: kind == "resid",
        })
    }

    /// `lambda` is the residual form's per-column scale, and is required by exactly that form.
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
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
        // the weight rows are bf16 as stored, and the bias and the residual form's lambda are one
        // f32 per output column. The element width the other operands carry is the kind's: `resid`
        // reads and writes the f16 stream, and the three that end a chain take f32 in and out.
        let elem = if self.resid { 2 } else { 4 };
        let mut args = Args::new(a, m * self.k * elem);
        args.push(w, self.n * self.k * 2);
        args.push(bias, self.n * 4);
        args.push(out, m * self.n * elem);
        if let Some(l) = lambda {
            args.push(l, self.n * 4);
        }
        checked(
            stream,
            &self.kernel,
            profile,
            stage,
            [(self.n / 64) as u32, m.div_ceil(64) as u32, 1],
            [256, 1, 1],
            &[m as u32],
            args.views(),
            args.need(),
        )
    }
}

/// The modulated RMS norm: `x[row] = rmsnorm(x[row]) * weight * (1 + scale[cls[row]]) + shift[cls[row]]`.
///
/// The DiT's final norm and the refiner's, which are the two places a stack's own blocks do not do
/// the modulating. Its table is `[2 * classes][width]` f32 — a scale row and a shift row per class.
pub struct NormMod {
    kernel: crate::compile::Kernel,
    width: usize,
    lanes: usize,
    classes: usize,
}

impl NormMod {
    pub fn build(
        c: &Compiler,
        stream: &mut hrx::Stream,
        width: usize,
        eps: f64,
        classes: usize,
    ) -> Result<Self> {
        let lanes = lanes_for(width).ok_or_else(|| {
            crate::compile::Error::Io(format!("no norm_mod lane count for width {width}"))
        })?;
        let ns = "h3.norm_mod_f32.";
        let cfg: Cfg = vec![
            (format!("{ns}width"), width.to_string()),
            (format!("{ns}lanes"), lanes.to_string()),
            (format!("{ns}eps"), num(eps)),
            (format!("{ns}classes"), classes.to_string()),
        ];
        Ok(Self {
            kernel: c.get(stream, "norm_mod_f32", "h3_norm_mod_f32", &cfg)?,
            width,
            lanes,
            classes,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        profile: Option<&mut Profile>,
        stage: &str,
        rows: usize,
        x: View<'_>,
        weight: View<'_>,
        table: View<'_>,
        cls: ClassRows<'_>,
    ) -> Result<()> {
        checked(
            stream,
            &self.kernel,
            profile,
            stage,
            [rows as u32, 1, 1],
            [self.lanes as u32, 1, 1],
            &[rows as u32],
            &[x, weight, table, cls.against(stage, rows, self.classes)?],
            &[
                rows * self.width * 4,
                self.width * 4,
                2 * self.classes * self.width * 4,
                rows * 4,
            ],
        )
    }
}

/// LayerNorm reading an f16 stream and writing f32, which is what the vision tower's blocks take.
pub struct LayerNorm16 {
    kernel: crate::compile::Kernel,
    width: usize,
}

impl LayerNorm16 {
    pub fn build(c: &Compiler, stream: &mut hrx::Stream, width: usize, eps: f64) -> Result<Self> {
        let ns = "h3.layernorm_f16_f32.";
        let cfg: Cfg = vec![
            (format!("{ns}width"), width.to_string()),
            (format!("{ns}eps"), num(eps)),
        ];
        Ok(Self {
            kernel: c.get(stream, "layernorm_f16_f32", "h3_layernorm_f16_f32", &cfg)?,
            width,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        profile: Option<&mut Profile>,
        rows: usize,
        x16: View<'_>,
        w: View<'_>,
        b: View<'_>,
        out32: View<'_>,
    ) -> Result<()> {
        checked(
            stream,
            &self.kernel,
            profile,
            "vision layernorm",
            [rows as u32, 1, 1],
            [32, 1, 1],
            &[rows as u32],
            &[x16, w, b, out32],
            &[
                rows * self.width * 2,
                self.width * 4,
                self.width * 4,
                rows * self.width * 4,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_class_is_a_row_of_the_table_it_indexes() {
        assert!(checkable(&[0, 1, 2], 3, 8).is_ok());
        // the kernel assumes the index is in range, so these are what stands between a caller's
        // mistake and a read outside the modulation table
        assert!(checkable(&[0, 3], 3, 8).is_err(), "a class past the table");
        assert!(checkable(&[-1], 3, 8).is_err(), "a negative class");
        assert!(checkable(&[0; 9], 3, 8).is_err(), "more rows than fit");
        assert!(checkable(&[0], 1, 8).is_ok(), "the single-class table");
    }

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
    fn gemm_modules_export_the_selected_kernels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
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
            let module = if tile == Tile::Plain && quantised(elem) {
                "gemm_packed_256".into()
            } else if tile == Tile::Plain {
                format!("gemm_{elem}_family")
            } else {
                stem.clone()
            };
            let source = std::fs::read_to_string(root.join(format!("{module}.loom"))).unwrap();
            assert!(
                source.contains(&format!("export(\"h3_{stem}\")")),
                "{module} does not export {stem}"
            );
        }
    }

    #[test]
    fn prepare_modules_export_the_selected_kernels() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels");
        for elem in ["i4", "i8", "f16", "bf16"] {
            let source =
                std::fs::read_to_string(root.join(format!("prepare_{elem}_family.loom"))).unwrap();
            for form in ["norm", "lnorm", "plain"] {
                if elem == "i4" && form == "lnorm" {
                    continue;
                }
                let stem = format!("prepare_{form}_{elem}");
                assert!(
                    source.contains(&format!("export(\"h3_{stem}\")")),
                    "prepare_{elem}_family does not export {stem}"
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
