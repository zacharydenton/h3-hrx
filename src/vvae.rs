//! The video VAE: 36 transformer blocks between a patch embedding and an unpatchify, run over spatial
//! tiles and temporal chunks the way the released model does.
//!
//! The decoder's shape is not free. A grid of `(ft, h, w)` latent voxels becomes `ft*h*w` tokens plus
//! four register tokens and a zero cls row, and the rotary table is built over coordinates normalised
//! to that grid — so a tile decoded on its own is not a crop of the whole frame decoded at once. That
//! is why the tiling and the blends exist, and why the stack is rebuilt whenever the grid changes.
use crate::compile::Compiler;
use crate::dispatch::{Conv3d, Gemm, GroupNormSilu, Matmul, Prepare, Profile, Tile};
use crate::error::{other, Result};
use crate::model::*;
use crate::stack::{Constants, LayerCond, Stack, StackDims};
use crate::tiles;
pub use crate::tiles::stitch_pixels;
use crate::weights::Weights;
use half::slice::HalfFloatSliceExt;

/// A latent grid: temporal length and the spatial extent in latent voxels.
///
/// Everything the decoder builds is a function of this — the token count, the rotary table, the two
/// projections' row groups — so it is one value rather than three loose integers that could disagree.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Grid {
    pub ft: usize,
    pub h: usize,
    pub w: usize,
}

impl Grid {
    /// Latent voxels, which is the token count before the registers and the cls row.
    pub fn voxels(&self) -> usize {
        self.ft * self.h * self.w
    }
    /// Every row the stack runs over.
    pub fn tokens(&self) -> usize {
        self.voxels() + VAE_REG + 1
    }
    /// The frames and pixels this grid decodes to.
    pub fn frames(&self) -> (usize, usize, usize) {
        (self.ft * VAE_PT, self.h * VAE_PS, self.w * VAE_PS)
    }

    /// Apply the decoder's 24×24 post-quant projection to channel-major latents.
    ///
    /// `weight` is row-major and `bias` has 24 entries. `out` holds one FP16 row
    /// of stride [`VAE_KIN`] per voxel. Padding after channel 24 is left untouched;
    /// the decoder initializes it to zero when allocating its staging buffer.
    pub fn prepare_decoder_input(
        &self,
        z: &[f32],
        weight: &[f32],
        bias: &[f32],
        out: &mut [half::f16],
    ) {
        let n = self.voxels();
        // Vectorize across voxels, never across the channel reduction: every lane keeps
        // the original FP32 addition order and narrows only after the final channel.
        const LANES: usize = 32;
        let full = n / LANES * LANES;
        for v in (0..full).step_by(LANES) {
            for o in 0..LATENT_CH {
                let mut acc = [bias[o]; LANES];
                for i in 0..LATENT_CH {
                    let input = &z[i * n + v..i * n + v + LANES];
                    let w = weight[o * LATENT_CH + i];
                    for lane in 0..LANES {
                        acc[lane] += w * input[lane];
                    }
                }
                let mut narrowed = [half::f16::ZERO; LANES];
                narrowed.convert_from_f32_slice(&acc);
                for lane in 0..LANES {
                    out[(v + lane) * VAE_KIN + o] = narrowed[lane];
                }
            }
        }
        for v in full..n {
            for o in 0..LATENT_CH {
                let mut acc = bias[o];
                for i in 0..LATENT_CH {
                    acc += weight[o * LATENT_CH + i] * z[i * n + v];
                }
                out[v * VAE_KIN + o] = half::f16::from_f32(acc);
            }
        }
    }

    /// Unpack decoder tokens `[ft][h][w][3][4][16][16]` into float pixels
    /// `[3][ft*4][h*16][w*16]`, resizing and overwriting `frames`.
    pub fn unpatchify(&self, patches: &[half::f16], frames: &mut Vec<f32>) {
        let (ftt, fh, fw) = self.frames();
        frames.resize(3 * ftt * fh * fw, 0.0);
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("f16c") {
            // SAFETY: the runtime check establishes the function's CPU feature requirement.
            unsafe { self.unpatchify_f16c(patches, frames) };
            return;
        }
        self.unpatchify_rows(patches, frames, |src, dst| src.convert_to_f32_slice(dst));
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "f16c")]
    unsafe fn unpatchify_f16c(&self, patches: &[half::f16], frames: &mut [f32]) {
        use std::arch::x86_64::{_mm256_cvtph_ps, _mm256_storeu_ps, _mm_loadu_si128};

        self.unpatchify_rows(
            patches,
            frames,
            |src: &[half::f16; 16], dst: &mut [f32; 16]| {
                // SAFETY: each array has 16 elements. Each unaligned load/store covers eight,
                // and the caller has established F16C support before entering this function.
                unsafe {
                    for i in [0, 8] {
                        let packed = _mm_loadu_si128(src.as_ptr().add(i).cast());
                        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(packed));
                    }
                }
            },
        );
    }

    // Keep row traversal shared while inlining the conversion into the selected CPU path.
    #[inline(always)]
    fn unpatchify_rows(
        &self,
        patches: &[half::f16],
        frames: &mut [f32],
        mut convert: impl FnMut(&[half::f16; VAE_PS], &mut [f32; VAE_PS]),
    ) {
        let (ftt, fh, fw) = self.frames();
        for t in 0..self.ft {
            for ch in 0..3 {
                for pt in 0..VAE_PT {
                    for y in 0..self.h {
                        for py in 0..VAE_PS {
                            for x in 0..self.w {
                                let tok = ((t * self.h + y) * self.w + x) * VAE_OUT;
                                let src = tok + ((ch * VAE_PT + pt) * VAE_PS + py) * VAE_PS;
                                let dst = ((ch * ftt + t * VAE_PT + pt) * fh + y * VAE_PS + py)
                                    * fw
                                    + x * VAE_PS;
                                convert(
                                    patches[src..src + VAE_PS].try_into().unwrap(),
                                    (&mut frames[dst..dst + VAE_PS]).try_into().unwrap(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The stack's dimensions. Fixed by the checkpoint, spelled out once.
fn dims() -> StackDims {
    StackDims {
        hidden: VAE_HID,
        heads: VAE_HEADS,
        kv_heads: VAE_HEADS,
        head_dim: VAE_D,
        ffn: VAE_FFN,
        rope_dim: 48,
        classes: 1,
        wbits: 16,
        eps: 1e-5,
        bias: true,
        gate_first: false,
        causal: false,
        attn_i4: false,
        attn_qk_bits: 16,
        bf16: false,
    }
}

/// What the stack and its buffers were last built for.
struct Built {
    stack: Stack,
    grid: Grid,
    proj_in: Gemm,
    norm_out: Prepare,
    proj_out: Gemm,
    x: hrx::Buffer,
    cos: hrx::Buffer,
    sin: hrx::Buffer,
    in16: hrx::Buffer,
    a_q: hrx::Buffer,
    a_s: hrx::Buffer,
    out16: hrx::Buffer,
    cls: crate::dispatch::Classes,
    norm_table: hrx::Buffer,
    /// the host side of the two transfers a clip makes, kept so a tiled decode does not allocate
    /// them again for every tile
    stage_in: Vec<half::f16>,
    stage_out: Vec<half::f16>,
    /// `H3_GRAPH=1`: the stack's blocks recorded once and replayed per tile. A clip is a hundred
    /// tiles of identical work over identical allocations, which is the shape a recording wants.
    graph: Option<hrx::GraphExec>,
}

/// Encoder tiles reuse their widest channels-last planes and padded host input.
struct EncoderScratch {
    rows: usize,
    frames: usize,
    input: Vec<half::f16>,
    x: hrx::Buffer,
    h: hrx::Buffer,
    y: hrx::Buffer,
    tmp: hrx::Buffer,
    sc: hrx::Buffer,
    stats: hrx::Buffer,
}

/// Resident video VAE state, used only on the stream passed to `open`.
pub struct VideoVae {
    stream_id: usize,
    weights: Weights,
    constants: Constants,
    built: Option<Built>,
    encoder: Option<EncoderScratch>,
    /// post_quant_conv, a 24x24 matrix and a bias applied per voxel on the host
    pq_w: Vec<f32>,
    pq_b: Vec<f32>,
    latents_mean: Vec<f32>,
    latents_std: Vec<f32>,
    /// the encoder's copies, which the checkpoint stores separately from the decoder's
    enc_mean: Vec<f32>,
    enc_std: Vec<f32>,
}

impl VideoVae {
    /// # Safety
    ///
    /// Maps the checkpoint; see [`crate::Session::new`].
    pub unsafe fn open(
        stream: &mut hrx::Stream,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let weights = unsafe { Weights::open(path, crate::plan::vvae::plan) }?;
        // the attention's per-head norms are ones for this stack, and its AdaLN tables are zero: it
        // modulates with a learned per-block scale alone
        Ok(Self {
            stream_id: stream.id(),
            pq_w: weights.host_f32("vae.post_quant_conv.w", LATENT_CH * LATENT_CH)?,
            pq_b: weights.host_f32("vae.post_quant_conv.b", LATENT_CH)?,
            latents_mean: weights.host_f32("vae.latents_mean", LATENT_CH)?,
            latents_std: weights.host_f32("vae.latents_std", LATENT_CH)?,
            enc_mean: weights.host_f32("venc.latents_mean", LATENT_CH)?,
            enc_std: weights.host_f32("venc.latents_std", LATENT_CH)?,
            weights,
            constants: Constants::new(stream)?,
            built: None,
            encoder: None,
        })
    }

    fn check_stream(&self, stream: &hrx::Stream) -> Result<()> {
        if stream.id() != self.stream_id {
            return crate::error::invalid("a video VAE must use the stream that created it");
        }
        Ok(())
    }

    /// Builds the stack, the two projections and the buffers for one latent grid, if the last call was
    /// for a different one.
    fn ensure(&mut self, stream: &mut hrx::Stream, c: &Compiler, grid: Grid) -> Result<()> {
        self.check_stream(stream)?;
        // Encoding finishes with a blocking read. Release its scratch before decoding,
        // including when the decoder already has a matching cached grid.
        self.encoder = None;
        if self.built.as_ref().is_some_and(|b| b.grid == grid) {
            return Ok(());
        }
        // dropping first frees the previous grid's device memory before the next is asked for
        self.built = None;
        let (n, nt) = (grid.voxels(), grid.tokens());
        let stack = Stack::new(
            c,
            stream,
            dims(),
            nt,
            VAE_BLOCKS,
            &self.weights,
            |i| format!("blocks.{i}."),
            false,
            self.constants.ones.clone(),
            "vae",
        )?;
        let t = stack.capacity();

        let x = stream.allocate_zeroed(t * VAE_HID * 4)?;
        let in16 = stream.allocate_zeroed(t * VAE_KIN * 2)?;
        let cls = crate::dispatch::Classes::zeroed(stream, t)?;

        // the rotary table over this grid; the register and cls rows keep the identity rotation
        let (mut cos_h, mut sin_h) = (
            vec![1.0f32; t * VAE_ROPE_HALF],
            vec![0.0f32; t * VAE_ROPE_HALF],
        );
        crate::rope::vae(grid.ft, grid.h, grid.w, &mut cos_h, &mut sin_h);
        let cos = stream.allocate(t * VAE_ROPE_HALF * 4)?;
        let sin = stream.allocate(t * VAE_ROPE_HALF * 4)?;
        stream.upload(cos.binding(), as_bytes(&cos_h))?;
        stream.upload(sin.binding(), as_bytes(&sin_h))?;

        // the output norm's (scale, shift) table: no scale, the checkpoint's bias as the shift
        let norm_table = stream.allocate(2 * VAE_HID * 4)?;
        crate::transfer::fill(stream, norm_table.slice(0, VAE_HID * 4), 0)?;
        let bias = self.weights.at(stream, "vae.norm_out.b", VAE_HID * 4)?;
        crate::transfer::copy(
            stream,
            norm_table.slice(VAE_HID * 4, VAE_HID * 4),
            bias.slice(0, VAE_HID * 4),
        )?;

        self.built = Some(Built {
            proj_in: Gemm::build(
                c,
                stream,
                "resid",
                "f16",
                true,
                true,
                VAE_KIN,
                VAE_HID,
                n,
                1,
                0,
                Tile::Plain,
                0,
            )?,
            norm_out: Prepare::build(c, stream, "lnorm", "f16", VAE_HID, 1e-5, 1, 0)?,
            proj_out: Gemm::build(
                c,
                stream,
                "plain",
                "f16",
                true,
                true,
                VAE_HID,
                VAE_OUT,
                nt,
                1,
                0,
                Tile::Plain,
                0,
            )?,
            a_q: stream.allocate(t * VAE_HID * 2)?,
            a_s: stream.allocate(t * 4)?,
            out16: stream.allocate(t * VAE_OUT * 2)?,
            stack,
            grid,
            x,
            cos,
            sin,
            in16,
            cls,
            norm_table,
            stage_in: vec![half::f16::ZERO; grid.voxels() * VAE_KIN],
            stage_out: vec![half::f16::ZERO; grid.voxels() * VAE_OUT],
            graph: None,
        });
        Ok(())
    }

    /// One clip: model-space latents `[24][ft][h][w]` to ImageNet-space frames
    /// `[3][ft*4][h*16][w*16]`.
    /// `frames` is resized to the clip and overwritten; it is an argument rather than a return so a
    /// tiled decode can hand back the same allocation for every tile.
    pub fn decode_clip(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        z: &[f32],
        grid: Grid,
        frames: &mut Vec<f32>,
    ) -> Result<()> {
        self.ensure(stream, c, grid)?;
        let (n, nt) = (grid.voxels(), grid.tokens());

        // post_quant_conv is a 24x24 matrix per voxel, small enough to stay on the host; the result
        // goes up as f16 padded to the GEMM's K of 64, the pad left at zero
        let Self {
            built,
            pq_w,
            pq_b,
            weights,
            constants,
            ..
        } = self;
        let b = built.as_mut().expect("built above");
        grid.prepare_decoder_input(z, pq_w, pq_b, &mut b.stage_in);

        stream.upload(b.in16.binding(), as_bytes_f16(&b.stage_in))?;
        crate::transfer::fill(stream, b.x.slice(0, nt * VAE_HID * 4), 0)?;

        let w_in = weights.at(stream, "vae.proj_in.w", VAE_HID * VAE_KIN * 2)?;
        let b_in = weights.at(stream, "vae.proj_in.b", VAE_HID * 4)?;
        b.proj_in.run(
            stream,
            Some(prof),
            "vae proj_in",
            n as u32,
            b.in16.binding(),
            w_in.binding(),
            None,
            b.x.binding(),
            Some((constants.ones.binding(), b.cls.all())),
            Some(b_in.binding()),
        )?;
        // the four register tokens follow the voxels, then a zero cls row
        let reg = weights.at(stream, "vae.register_tokens", VAE_REG * VAE_HID * 4)?;
        crate::transfer::copy(
            stream,
            b.x.slice(n * VAE_HID * 4, VAE_REG * VAE_HID * 4),
            reg.slice(0, VAE_REG * VAE_HID * 4),
        )?;

        // every block modulates with its own learned scale and no shift
        let zeros = constants.zeros.binding();
        // The buffers first, not views of them: a view borrows its allocation, and one borrowed from
        // the stack would conflict with the mutable borrow the forward pass takes.
        let scale_buffers: Vec<(std::sync::Arc<hrx::Buffer>, std::sync::Arc<hrx::Buffer>)> =
            (0..b.stack.layers())
                .map(|i| {
                    let blk = b.stack.block(i);
                    (
                        blk.scale1
                            .as_ref()
                            .expect("a decoder block has scale1")
                            .clone(),
                        blk.scale2
                            .as_ref()
                            .expect("a decoder block has scale2")
                            .clone(),
                    )
                })
                .collect();
        let scales: Vec<(hrx::View<'_>, hrx::View<'_>)> = scale_buffers
            .iter()
            .map(|(one, two)| (one.binding(), two.binding()))
            .collect();
        let cond = |i: usize| LayerCond {
            table_msa: zeros,
            gate_msa: scales[i].0,
            table_mlp: zeros,
            gate_mlp: scales[i].1,
        };
        let (x, cls, cos, sin) = (b.x.binding(), b.cls.all(), b.cos.binding(), b.sin.binding());
        b.stack
            .forward_cached(stream, prof, &mut b.graph, x, cls, cos, sin, &cond, 0, None)?;

        let w_norm = weights.at(stream, "vae.norm_out.w", VAE_HID * 4)?;
        b.norm_out.run(
            stream,
            Some(prof),
            "vae norm_out",
            nt as u32,
            x,
            Some((w_norm.binding(), b.norm_table.binding(), cls)),
            b.a_q.binding(),
            Some(b.a_s.binding()),
        )?;
        let w_out = weights.at(stream, "vae.proj_out.w", VAE_OUT * VAE_HID * 2)?;
        let b_out = weights.at(stream, "vae.proj_out.b", VAE_OUT * 4)?;
        b.proj_out.run(
            stream,
            Some(prof),
            "vae proj_out",
            nt as u32,
            b.a_q.binding(),
            w_out.binding(),
            None,
            b.out16.binding(),
            None,
            Some(b_out.binding()),
        )?;

        // read straight into the f16 buffer the unpatchify converts from, rather than into bytes and
        // then through a second allocation
        let out16 = &mut b.stage_out;
        stream.read_blocking(
            b.out16.slice(0, n * VAE_OUT * 2),
            bytes_mut(&mut out16[..n * VAE_OUT]),
        )?;

        grid.unpatchify(out16, frames);
        Ok(())
    }

    /// One clip in spatial tiles, blended in pixel space — the released VAE's `tiled_decode`.
    pub fn decode_spatial(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        z: &[f32],
        grid: Grid,
        frames: &mut Vec<f32>,
    ) -> Result<()> {
        let (ft, h, w) = (grid.ft, grid.h, grid.w);
        let (frames_n, height, width) = grid.frames();
        if height <= 256 && width <= 256 {
            return self.decode_clip(stream, c, prof, z, grid, frames);
        }
        let mut latent = Vec::new();
        stitch_pixels(frames_n, height, width, frames, |y0, x0, th, tw, tile| {
            let (lh, lw) = (th / VAE_PS, tw / VAE_PS);
            latent.resize(LATENT_CH * ft * lh * lw, 0.0);
            for ch in 0..LATENT_CH {
                for t in 0..ft {
                    for y in 0..lh {
                        let src = ((ch * ft + t) * h + y0 / VAE_PS + y) * w + x0 / VAE_PS;
                        let dst = ((ch * ft + t) * lh + y) * lw;
                        latent[dst..dst + lw].copy_from_slice(&z[src..src + lw]);
                    }
                }
            }
            self.decode_clip(stream, c, prof, &latent, Grid { ft, h: lh, w: lw }, tile)
        })
    }

    /// The whole clip: latents `[24][T][H][W]` in, RGB bytes `[frames][H*16][W*16][3]` out.
    pub fn decode_video(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        shape: &crate::layout::Shape,
        latents: &[f32],
        out: &mut [u8],
    ) -> Result<()> {
        self.decode_video_with(stream, c, prof, shape, latents, &mut |index, value, ch| {
            out[index] = crate::pixels::imagenet_denormalise(value, ch);
        })
    }

    /// Reference reconstruction retains float pixels before vision preprocessing.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_video_pixels(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        shape: &crate::layout::Shape,
        latents: &[f32],
        out: &mut [f32],
    ) -> Result<()> {
        self.decode_video_with(stream, c, prof, shape, latents, &mut |index, value, ch| {
            out[index] = (value * IMAGENET_STD[ch] + IMAGENET_MEAN[ch]).clamp(0.0, 1.0);
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_video_with(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        shape: &crate::layout::Shape,
        latents: &[f32],
        write: &mut impl FnMut(usize, f32, usize),
    ) -> Result<()> {
        let (t_len, h, w) = (
            shape.latent_t as usize,
            shape.lat_h as usize,
            shape.lat_w as usize,
        );
        let plan = tiles::chunk_plan(t_len);
        let tp = plan.padded_tokens;

        // undo the latent normalisation, repeating the last frame for the padding
        let mut zp = vec![0.0f32; LATENT_CH * tp * h * w];
        for ch in 0..LATENT_CH {
            for t in 0..tp {
                // the padding repeats the last latent frame
                let src = (ch * t_len + t.min(t_len - 1)) * h * w;
                let dst = (ch * tp + t) * h * w;
                for i in 0..h * w {
                    zp[dst + i] = latents[src + i] * self.latents_std[ch] + self.latents_mean[ch];
                }
            }
        }

        let mut z = Vec::new();
        decode_temporal(
            shape,
            |start, ft, clip| {
                z.clear();
                z.resize(LATENT_CH * ft * h * w, 0.0);
                for ch in 0..LATENT_CH {
                    let src = (ch * tp + start) * h * w;
                    let dst = ch * ft * h * w;
                    z[dst..dst + ft * h * w].copy_from_slice(&zp[src..src + ft * h * w]);
                }
                self.decode_spatial(stream, c, prof, &z, Grid { ft, h, w }, clip)
            },
            write,
        )
    }
}

/// Decode temporal windows and assemble the video VAE's channel-major float pixels.
///
/// The decoder callback receives a start token, a token count (including padded latents),
/// and a reusable output buffer for `[3][tokens*4][height][width]` pixels. The writer receives
/// an interleaved RGB output index, an ImageNet-normalized value, and its channel.
/// Trimming and cross-fades follow the released VAE's temporal chunk plan.
pub fn decode_temporal(
    shape: &crate::layout::Shape,
    mut decode: impl FnMut(usize, usize, &mut Vec<f32>) -> Result<()>,
    write: &mut impl FnMut(usize, f32, usize),
) -> Result<()> {
    let want_frames = shape.frames as usize;
    let plane = shape.lat_h as usize * shape.lat_w as usize * VAE_PS * VAE_PS;
    let plan = tiles::chunk_plan(shape.latent_t as usize);
    let mut overlap: Vec<f32> = Vec::new();
    let mut have_overlap = false;
    let mut decoded = 0usize;
    // the frames actually written, capped at what the caller asked for
    let mut append =
        |chunk: &[f32], frames: std::ops::Range<usize>, src_ft: usize, decoded: &mut usize| {
            let nf = frames.len();
            let take = if *decoded < want_frames {
                nf.min(want_frames - *decoded)
            } else {
                0
            };
            for f in 0..take {
                // Slice each source plane once per frame so the pixel loop needs
                // neither channel-offset arithmetic nor source bounds checks.
                let start = (frames.start + f) * plane;
                let stride = src_ft * plane;
                let red = &chunk[start..start + plane];
                let green = &chunk[stride + start..stride + start + plane];
                let blue = &chunk[2 * stride + start..2 * stride + start + plane];
                let base = (*decoded + f) * plane * 3;
                for (q, ((&r, &g), &b)) in red.iter().zip(green).zip(blue).enumerate() {
                    let index = base + q * 3;
                    write(index, r, 0);
                    write(index + 1, g, 1);
                    write(index + 2, b, 2);
                }
            }
            *decoded += nf;
        };

    let mut clip = Vec::new();
    for i in 0..plan.chunks {
        let start = i * VAE_CHUNK;
        let ft = (VAE_CHUNK + VAE_OVERLAP).min(plan.padded_tokens - start);
        decode(start, ft, &mut clip)?;
        let clip_frames = ft * VAE_TRATIO;
        for j in 0..2 {
            let f0 = j * plan.chunk_frames + plan.pre;
            let f1 = ((j + 1) * plan.chunk_frames).min(clip_frames);
            if f0 >= f1 {
                if j == 1 {
                    have_overlap = false;
                }
                continue;
            }
            let nf = f1 - f0;
            if j == 0 {
                if have_overlap {
                    tiles::crossfade(&mut clip, f0..f1, &overlap, plane, plan.overlap_frames);
                }
                append(&clip, f0..f1, clip_frames, &mut decoded);
            } else {
                // Retain only the tail needed by the next window. Emitted frames are
                // blended and read directly from clip, without a full-window scratch copy.
                overlap.resize(3 * nf * plane, 0.0);
                for ch in 0..3 {
                    let src = (ch * clip_frames + f0) * plane;
                    overlap[ch * nf * plane..(ch + 1) * nf * plane]
                        .copy_from_slice(&clip[src..src + nf * plane]);
                }
                have_overlap = true;
            }
        }
    }
    if have_overlap {
        let nf = overlap.len() / (3 * plane);
        append(&overlap, 0..nf, nf, &mut decoded);
    }

    let kept = decoded - plan.pad_frames;
    if kept != want_frames {
        return other(format!("decoded {kept} frames, expected {want_frames}"));
    }
    Ok(())
}

/// Pixels and the shape they are in: `[frames][height][width][3]` in `[0, 1]`.
///
/// One value rather than a slice and three integers, because passing a buffer that does not match its
/// dimensions is the mistake worth making impossible here.
#[derive(Clone, Copy)]
pub struct Clip<'a> {
    pub pixels: &'a [f32],
    pub frames: usize,
    pub height: usize,
    pub width: usize,
}

impl Clip<'_> {
    /// Checks the buffer against the dimensions it claims.
    ///
    /// A `Clip` is public and freely constructible, so its slice and its shape can disagree. Caught
    /// here, that is an error; carried inward it is an indexing panic somewhere in the encoder, or a
    /// temporal underflow for a clip of no frames.
    pub fn check(&self) -> crate::error::Result<()> {
        if self.frames == 0 || self.height == 0 || self.width == 0 {
            return crate::error::invalid(format!(
                "a clip of {} frames at {}x{} is empty",
                self.frames, self.height, self.width
            ));
        }
        if !self.height.is_multiple_of(VAE_PS) || !self.width.is_multiple_of(VAE_PS) {
            return crate::error::invalid(format!(
                "{}x{} is not a multiple of {VAE_PS}",
                self.height, self.width
            ));
        }
        // checked from the start: plane() multiplies height by width by three, and that product
        // overflows on its own for a large enough side — checking only the last multiply lets an
        // absurd height wrap to a small requirement that any buffer satisfies
        let need = [self.frames, self.height, self.width, 3]
            .into_iter()
            .try_fold(1usize, |a, b| a.checked_mul(b))
            .ok_or_else(|| crate::error::Error::Invalid("clip dimensions overflow".into()))?;
        if self.pixels.len() < need {
            return crate::error::invalid(format!(
                "a clip of {} frames at {}x{} needs {need} floats, {} given",
                self.frames,
                self.height,
                self.width,
                self.pixels.len()
            ));
        }
        Ok(())
    }

    fn plane(&self) -> usize {
        self.height * self.width * 3
    }
    /// The latent grid this clip encodes to, spatially.
    fn latent(&self) -> (usize, usize) {
        (self.height / VAE_PS, self.width / VAE_PS)
    }
}

/// The encoder's residual stack: channels per level, spatial stride, temporal stride.
const ENC_MID: [usize; 6] = [128, 256, 256, 512, 512, 1024];
const ENC_SDOWN: [usize; 6] = [2, 2, 2, 2, 1, 1];
const ENC_TDOWN: [usize; 6] = [1, 2, 2, 1, 1, 1];

impl VideoVae {
    fn ensure_encoder(
        &mut self,
        stream: &mut hrx::Stream,
        rows: usize,
        frames: usize,
    ) -> Result<()> {
        self.check_stream(stream)?;
        if self
            .encoder
            .as_ref()
            .is_some_and(|s| s.rows >= rows && s.frames >= frames)
        {
            return Ok(());
        }
        let (rows, frames) = self
            .encoder
            .as_ref()
            .map_or((rows, frames), |s| (rows.max(s.rows), frames.max(s.frames)));
        // Free the old planes first so growth does not temporarily double residency.
        self.encoder = None;
        let plane_bytes = rows * 128 * 2;
        self.encoder = Some(EncoderScratch {
            rows,
            frames,
            input: vec![half::f16::ZERO; rows * 8],
            x: stream.allocate(rows * 8 * 2)?,
            h: stream.allocate(plane_bytes)?,
            y: stream.allocate(plane_bytes)?,
            tmp: stream.allocate(plane_bytes)?,
            sc: stream.allocate(plane_bytes)?,
            stats: stream.allocate(frames * 32 * 2 * 4)?,
        });
        Ok(())
    }

    /// One tile of `frames` frames to moment rows, then to model-space latents `[24][T][h][w]`.
    ///
    /// A still image and a clip use different weights and a different tap count — `.w2` with one
    /// temporal tap against `.w3` with three — because the released encoder folds a single frame's
    /// causal padding into the kernel rather than replicating the frame.
    #[allow(clippy::too_many_arguments)]
    fn encode_tile(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        pixels: &[f32],
        frames: usize,
        height: usize,
        y0: usize,
        x0: usize,
        th: usize,
        tw: usize,
        full_w: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let image = frames == 1;
        let taps_t = if image { 1 } else { 3 };
        // K folds the taps in: 9 for an image, 27 for a clip, times the input channels, to a multiple
        // of 32
        let ksz = |cin_pad: usize| ((if image { 9 } else { 27 }) * cin_pad).div_ceil(32) * 32;
        let suffix = if image { ".w2" } else { ".w3" };

        // input rows [frames*th*tw][8] f16: ImageNet-normalised pixels, channels 3..7 left at zero
        let rows0 = frames * th * tw;
        self.ensure_encoder(stream, rows0, frames)?;
        let scratch = self.encoder.as_mut().expect("sized above");
        let inrows = &mut scratch.input[..rows0 * 8];
        for t in 0..frames {
            for y in 0..th {
                for x in 0..tw {
                    let src = ((t * height + y0 + y) * full_w + x0 + x) * 3;
                    let dst = ((t * th + y) * tw + x) * 8;
                    let n = crate::pixels::imagenet_normalise([
                        pixels[src],
                        pixels[src + 1],
                        pixels[src + 2],
                    ]);
                    for ch in 0..3 {
                        inrows[dst + ch] = half::f16::from_f32(n[ch]);
                    }
                }
            }
        }

        let x = &scratch.x;
        stream.upload(x.binding(), as_bytes_f16(inrows))?;
        let mut h = &scratch.h;
        let y = &scratch.y;
        let mut tmp = &scratch.tmp;
        let sc = &scratch.sc;
        let stats = &scratch.stats;

        let w3 = |stream: &mut hrx::Stream, nm: &str, cout_pad: usize, k: usize| {
            self.weights
                .at(stream, &format!("{nm}{suffix}"), cout_pad * k * 2)
        };

        let (mut t_len, mut hh, mut ww, mut chan) = (frames, th, tw, 128usize);
        let conv = Conv3d::build(
            c,
            stream,
            false,
            t_len,
            hh,
            ww,
            1,
            1,
            taps_t,
            8,
            8,
            ksz(8),
            128,
        )?;
        let hoisted_1 = w3(stream, "venc.conv_in", 128, ksz(8))?;
        let held_1 = self.weights.at(stream, "venc.conv_in.b", 128 * 4)?;
        conv.run(
            stream,
            Some(prof),
            "venc conv_in",
            x.binding(),
            hoisted_1.binding(),
            held_1.binding(),
            h.binding(),
            None,
        )?;

        for l in 0..6 {
            for r in 0..2 {
                let b = format!("venc.l{l}.r{r}.");
                let (cin, cout) = (chan, ENC_MID[l]);
                let held_1 = self.weights.at(stream, &format!("{b}norm1.g"), cin * 4)?;
                let held_2 = self.weights.at(stream, &format!("{b}norm1.b"), cin * 4)?;
                GroupNormSilu::build(c, stream, t_len, hh, ww, cin)?.run(
                    stream,
                    Some(prof),
                    "venc groupnorm",
                    h.binding(),
                    held_1.binding(),
                    held_2.binding(),
                    stats.binding(),
                    y.binding(),
                )?;
                let hoisted_1 = w3(stream, &format!("{b}conv1"), cout, ksz(cin))?;
                let held_1 = self.weights.at(stream, &format!("{b}conv1.b"), cout * 4)?;
                Conv3d::build(
                    c,
                    stream,
                    false,
                    t_len,
                    hh,
                    ww,
                    1,
                    1,
                    taps_t,
                    cin,
                    cin,
                    ksz(cin),
                    cout,
                )?
                .run(
                    stream,
                    Some(prof),
                    "venc conv",
                    y.binding(),
                    hoisted_1.binding(),
                    held_1.binding(),
                    tmp.binding(),
                    None,
                )?;
                let held_1 = self.weights.at(stream, &format!("{b}norm2.g"), cout * 4)?;
                let held_2 = self.weights.at(stream, &format!("{b}norm2.b"), cout * 4)?;
                GroupNormSilu::build(c, stream, t_len, hh, ww, cout)?.run(
                    stream,
                    Some(prof),
                    "venc groupnorm",
                    tmp.binding(),
                    held_1.binding(),
                    held_2.binding(),
                    stats.binding(),
                    y.binding(),
                )?;
                // a change of width needs the skip projected: a 1x1 convolution, which is a matmul
                let resid = if cin != cout {
                    let k = cin.div_ceil(32) * 32;
                    let held_1 = self
                        .weights
                        .at(stream, &format!("{b}nin.wm"), cout * k * 2)?;
                    let held_2 = self.weights.at(stream, &format!("{b}nin.b"), cout * 4)?;
                    Matmul::build(c, stream, k, cout)?.run(
                        stream,
                        Some(prof),
                        "venc shortcut",
                        t_len * hh * ww,
                        h.binding(),
                        held_1.binding(),
                        held_2.binding(),
                        sc.binding(),
                    )?;
                    sc.binding()
                } else {
                    h.binding()
                };
                let hoisted_1 = w3(stream, &format!("{b}conv2"), cout, ksz(cout))?;
                let held_1 = self.weights.at(stream, &format!("{b}conv2.b"), cout * 4)?;
                Conv3d::build(
                    c,
                    stream,
                    true,
                    t_len,
                    hh,
                    ww,
                    1,
                    1,
                    taps_t,
                    cout,
                    cout,
                    ksz(cout),
                    cout,
                )?
                .run(
                    stream,
                    Some(prof),
                    "venc conv",
                    y.binding(),
                    hoisted_1.binding(),
                    held_1.binding(),
                    tmp.binding(),
                    Some(resid),
                )?;
                std::mem::swap(&mut h, &mut tmp);
                chan = cout;
            }
            if ENC_SDOWN[l] * ENC_TDOWN[l] > 1 {
                let b = format!("venc.l{l}.down");
                let down = Conv3d::build(
                    c,
                    stream,
                    false,
                    t_len,
                    hh,
                    ww,
                    ENC_SDOWN[l],
                    if image { 1 } else { ENC_TDOWN[l] },
                    taps_t,
                    chan,
                    chan,
                    ksz(chan),
                    chan,
                )?;
                let hoisted_1 = w3(stream, &b, chan, ksz(chan))?;
                let held_1 = self.weights.at(stream, &format!("{b}.b"), chan * 4)?;
                down.run(
                    stream,
                    Some(prof),
                    "venc down",
                    h.binding(),
                    hoisted_1.binding(),
                    held_1.binding(),
                    tmp.binding(),
                    None,
                )?;
                std::mem::swap(&mut h, &mut tmp);
                t_len = down.tout;
                hh = down.ho;
                ww = down.wo;
            }
        }

        let held_1 = self.weights.at(stream, "venc.norm_out.g", chan * 4)?;
        let held_2 = self.weights.at(stream, "venc.norm_out.b", chan * 4)?;
        GroupNormSilu::build(c, stream, t_len, hh, ww, chan)?.run(
            stream,
            Some(prof),
            "venc groupnorm",
            h.binding(),
            held_1.binding(),
            held_2.binding(),
            stats.binding(),
            y.binding(),
        )?;
        let hoisted_1 = w3(stream, "venc.conv_out", 64, ksz(chan))?;
        let held_1 = self.weights.at(stream, "venc.conv_out.b", 64 * 4)?;
        Conv3d::build(
            c,
            stream,
            false,
            t_len,
            hh,
            ww,
            1,
            1,
            taps_t,
            chan,
            chan,
            ksz(chan),
            64,
        )?
        .run(
            stream,
            Some(prof),
            "venc conv_out",
            y.binding(),
            hoisted_1.binding(),
            held_1.binding(),
            tmp.binding(),
            None,
        )?;
        let m = t_len * hh * ww;
        let held_1 = self.weights.at(stream, "venc.quant.wm", 64 * 64 * 2)?;
        let held_2 = self.weights.at(stream, "venc.quant.b", 64 * 4)?;
        Matmul::build(c, stream, 64, 64)?.run(
            stream,
            Some(prof),
            "venc quant",
            m,
            tmp.binding(),
            held_1.binding(),
            held_2.binding(),
            y.binding(),
        )?;

        // the head emits 64 channels: the first 24 are the posterior's mean, the rest its log variance,
        // which sampling would use and this does not
        let mut mom = vec![half::f16::ZERO; m * 64];
        stream.read_blocking(y.slice(0, mom.len() * 2), bytes_mut(&mut mom))?;
        let mut latent = vec![0.0f32; LATENT_CH * m];
        for t in 0..t_len {
            for yy in 0..hh {
                for xx in 0..ww {
                    for ch in 0..LATENT_CH {
                        let v = mom[((t * hh + yy) * ww + xx) * 64 + ch].to_f32();
                        latent[((ch * t_len + t) * hh + yy) * ww + xx] =
                            (v - self.enc_mean[ch]) / self.enc_std[ch];
                    }
                }
            }
        }
        Ok((latent, t_len))
    }

    /// One clip, in 256-pixel tiles blended in latent space — ComfyUI's `tiled_encode`, in its order:
    /// blend the tile above in over the y overlap, then the tile to the left over the x overlap of the
    /// result, then crop the trailing overlaps and concatenate.
    pub fn encode_clip(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        clip: Clip<'_>,
    ) -> Result<(Vec<f32>, usize)> {
        self.check_stream(stream)?;
        let (frames, height, width) = (clip.frames, clip.height, clip.width);
        let (ys, yo) = tiles::split_tiles(height);
        let (xs, xo) = tiles::split_tiles(width);
        let (ny, nx) = (ys.len(), xs.len());
        let mut tile_of: Vec<Vec<f32>> = vec![Vec::new(); ny * nx];
        let (mut th_lat, mut tw_lat) = (vec![0usize; ny], vec![0usize; nx]);
        let mut t_lat = 0usize;
        for i in 0..ny {
            for j in 0..nx {
                let tile_h = if ny == 1 { height } else { 256 };
                let tile_w = if nx == 1 { width } else { 256 };
                th_lat[i] = tile_h / VAE_PS;
                tw_lat[j] = tile_w / VAE_PS;
                let (z, t) = self.encode_tile(
                    stream,
                    c,
                    prof,
                    clip.pixels,
                    frames,
                    height,
                    ys[i],
                    xs[j],
                    tile_h,
                    tile_w,
                    width,
                )?;
                tile_of[i * nx + j] = z;
                t_lat = t;
            }
        }

        let (lh, lw) = clip.latent();
        let mut out = vec![0.0f32; LATENT_CH * t_lat * lh * lw];
        let mut oy = 0usize;
        for i in 0..ny {
            let mut ox = 0usize;
            let h = th_lat[i];
            for j in 0..nx {
                let w = tw_lat[j];
                let mut tile = tile_of[i * nx + j].clone();
                if i > 0 {
                    tile = tiles::blend_latent(
                        &tile_of[(i - 1) * nx + j],
                        th_lat[i - 1],
                        w,
                        &tile,
                        h,
                        w,
                        yo[i - 1] / VAE_PS,
                        true,
                        t_lat,
                    );
                }
                if j > 0 {
                    tile = tiles::blend_latent(
                        &tile_of[i * nx + j - 1],
                        h,
                        tw_lat[j - 1],
                        &tile,
                        h,
                        w,
                        xo[j - 1] / VAE_PS,
                        false,
                        t_lat,
                    );
                }
                let keep_y = if i < ny - 1 { h - yo[i] / VAE_PS } else { h };
                let keep_x = if j < nx - 1 { w - xo[j] / VAE_PS } else { w };
                for ch in 0..LATENT_CH {
                    for t in 0..t_lat {
                        for y in 0..keep_y {
                            let dst = ((ch * t_lat + t) * lh + oy + y) * lw + ox;
                            let src = ((ch * t_lat + t) * h + y) * w;
                            out[dst..dst + keep_x].copy_from_slice(&tile[src..src + keep_x]);
                        }
                    }
                }
                ox += keep_x;
            }
            oy += if i < ny - 1 { h - yo[i] / VAE_PS } else { h };
        }
        Ok((out, t_lat))
    }

    /// Pixels `[frames][H][W][3]` in `[0, 1]` to model-space latents `[24][latent_t][H/16][W/16]`.
    ///
    /// A clip is encoded in seventeen-frame chunks, each giving five latent frames, and the last three
    /// of the concatenation are dropped — the decoder's token drop, undone.
    pub fn encode_video(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        clip: Clip<'_>,
    ) -> Result<(Vec<f32>, usize)> {
        clip.check()?;
        let (lh, lw) = clip.latent();
        let frames = clip.frames;
        if frames == 1 {
            let (z, t) = self.encode_clip(stream, c, prof, clip)?;
            if t != 1 {
                return other("image encode produced more than one latent frame");
            }
            return Ok((z, 1));
        }
        let chunks = frames.div_ceil(17);
        let tl = chunks * 5;
        let plane = clip.plane();
        let mut all = vec![0.0f32; LATENT_CH * tl * lh * lw];
        let mut chunk = vec![0.0f32; 17 * plane];
        for ch in 0..chunks {
            for f in 0..17 {
                // the tail repeats the last frame rather than padding with black
                let src = (ch * 17 + f).min(frames - 1) * plane;
                chunk[f * plane..(f + 1) * plane].copy_from_slice(&clip.pixels[src..src + plane]);
            }
            let (z, t) = self.encode_clip(
                stream,
                c,
                prof,
                Clip {
                    pixels: &chunk,
                    frames: 17,
                    ..clip
                },
            )?;
            if t != 5 {
                return other("a 17-frame chunk must give 5 latent frames");
            }
            for cc in 0..LATENT_CH {
                for t in 0..5 {
                    let dst = ((cc * tl + ch * 5 + t) * lh) * lw;
                    let src = ((cc * 5 + t) * lh) * lw;
                    all[dst..dst + lh * lw].copy_from_slice(&z[src..src + lh * lw]);
                }
            }
        }
        let latent_t = tl - VAE_TOKEN_DROP;
        let mut out = vec![0.0f32; LATENT_CH * latent_t * lh * lw];
        for cc in 0..LATENT_CH {
            for t in 0..latent_t {
                let dst = ((cc * latent_t + t) * lh) * lw;
                let src = ((cc * tl + t) * lh) * lw;
                out[dst..dst + lh * lw].copy_from_slice(&all[src..src + lh * lw]);
            }
        }
        Ok((out, latent_t))
    }
}

/// f32 and f16 have no padding and no invalid bit patterns, so their bytes are a plain
/// reinterpretation — which is what `bytemuck` proves rather than asserts.
pub fn as_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

pub fn as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    bytemuck::cast_slice_mut(v)
}

pub fn as_bytes_f16(v: &[half::f16]) -> &[u8] {
    bytemuck::cast_slice(v)
}

fn bytes_mut(v: &mut [half::f16]) -> &mut [u8] {
    bytemuck::cast_slice_mut(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_assembly_matches_scalar_frame_mapping() {
        fn pixel(start: usize, c: usize, frame: usize, q: usize) -> f32 {
            ((start * 100003 + c * 997 + frame * 29 + q * 7) % 4093) as f32 / 127.0 - 16.0
        }
        for frames in [5, 22, 39, 56, 73, 124, 362] {
            let shape = crate::shape_for(32, 32, frames).unwrap();
            let plane = 32 * 32;
            let windows = ((frames as usize - 5) / 17).max(1);
            let mut calls = 0;
            let mut written = 0;
            decode_temporal(
                &shape,
                |start, ft, clip| {
                    assert_eq!((start, ft), (calls * 5, 7));
                    calls += 1;
                    clip.clear();
                    for c in 0..3 {
                        for f in 0..ft * 4 {
                            for q in 0..plane {
                                clip.push(pixel(start, c, f, q));
                            }
                        }
                    }
                    Ok(())
                },
                &mut |index, value, c| {
                    assert_eq!(index, written);
                    assert_eq!(c, index % 3);
                    written += 1;
                    let f = index / (3 * plane);
                    let q = index / 3 % plane;
                    // Each full window emits raw frames 3..20. Its final 23..28 frames
                    // supply the next cross-fade, or the final five output frames.
                    let window = (f / 17).min(windows - 1);
                    let k = f - window * 17;
                    let raw_frame = if k < 17 { 3 + k } else { 23 + k - 17 };
                    let current = pixel(window * 5, c, raw_frame, q);
                    let expected = if window > 0 && k < 5 {
                        let previous = pixel((window - 1) * 5, c, 23 + k, q);
                        let weight = k as f32 / 5.0;
                        (1.0 - weight) * previous + weight * current
                    } else {
                        current
                    };
                    assert_eq!(
                        value.to_bits(),
                        expected.to_bits(),
                        "{frames} frames, index {index}"
                    );
                },
            )
            .unwrap();
            assert_eq!(calls, windows);
            assert_eq!(written, frames as usize * 3 * plane);
        }
    }

    #[test]
    fn temporal_assembly_propagates_decode_errors() {
        let shape = crate::shape_for(32, 32, 39).unwrap();
        let err = decode_temporal(
            &shape,
            |_, _, _| crate::error::invalid("decoder failed"),
            &mut |_, _, _| panic!("failed decode must not write pixels"),
        )
        .unwrap_err();
        assert!(err.to_string().contains("decoder failed"));
    }

    #[test]
    fn unpatchify_preserves_half_bits_and_overwrites_reused_frames() {
        let mut frames = Vec::new();
        for (ft, h, w) in [(1, 1, 1), (2, 3, 5), (7, 16, 16), (1, 2, 3), (0, 4, 4)] {
            let grid = Grid { ft, h, w };
            // Cycle through every half bit pattern, including signed zero, subnormals,
            // infinities and NaNs. The scalar conversion defines the expected float bits.
            let patches: Vec<_> = (0..grid.voxels() * VAE_OUT)
                .map(|i| half::f16::from_bits((i as u16).wrapping_mul(37) ^ (i >> 16) as u16))
                .collect();
            for pass in 0..2 {
                frames.fill(9.0);
                if pass == 0 {
                    grid.unpatchify(&patches, &mut frames);
                } else {
                    grid.unpatchify_rows(&patches, &mut frames, |src, dst| {
                        src.convert_to_f32_slice(dst)
                    });
                }
                let (nf, height, width) = grid.frames();
                assert_eq!(frames.len(), 3 * nf * height * width);
                for c in 0..3 {
                    for f in 0..nf {
                        for y in 0..height {
                            for x in 0..width {
                                let token =
                                    ((f / VAE_PT * h + y / VAE_PS) * w + x / VAE_PS) * VAE_OUT;
                                let offset = ((c * VAE_PT + f % VAE_PT) * VAE_PS + y % VAE_PS)
                                    * VAE_PS
                                    + x % VAE_PS;
                                let expected = patches[token + offset].to_f32();
                                let actual = frames[((c * nf + f) * height + y) * width + x];
                                assert_eq!(
                                    actual.to_bits(),
                                    expected.to_bits(),
                                    "{grid:?}, channel {c}, frame {f}, ({y}, {x})"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn decoder_input_matches_scalar_projection_and_preserves_padding() {
        let values = |n, seed| {
            (0..n)
                .map(|i| ((i * 37 + seed) % 257) as f32 / 113.0 - 1.0)
                .collect::<Vec<_>>()
        };
        let weight = values(LATENT_CH * LATENT_CH, 17);
        let bias = values(LATENT_CH, 29);
        for n in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127, 1792] {
            let grid = Grid { ft: 1, h: 1, w: n };
            let z = values(n * LATENT_CH, 43);
            let guard = half::f16::from_bits(0x3555);
            let mut actual = vec![guard; (n + 2) * VAE_KIN];
            let mut expected = actual.clone();
            for v in 0..n {
                for o in 0..LATENT_CH {
                    let mut sum = bias[o];
                    for i in 0..LATENT_CH {
                        sum += weight[o * LATENT_CH + i] * z[i * n + v];
                    }
                    expected[(v + 1) * VAE_KIN + o] = half::f16::from_f32(sum);
                }
            }
            // Include guards on both sides, nonzero row padding, and a second call over
            // reused storage. Compare half bits, including signed zero.
            for _ in 0..2 {
                grid.prepare_decoder_input(
                    &z,
                    &weight,
                    &bias,
                    &mut actual[VAE_KIN..(n + 1) * VAE_KIN],
                );
                assert_eq!(as_bytes_f16(&actual), as_bytes_f16(&expected), "{n} voxels");
            }
        }
    }

    #[test]
    fn decoder_input_keeps_fp32_accumulation_order_without_fma() {
        let grid = Grid { ft: 1, h: 1, w: 33 };
        let n = grid.voxels();
        let mut z = vec![0.0; LATENT_CH * n];
        for (i, value) in [4097.0, 16_777_216.0, 1.0, -16_777_216.0]
            .into_iter()
            .enumerate()
        {
            z[i * n..(i + 1) * n].fill(value);
        }
        let mut weight = vec![0.0; LATENT_CH * LATENT_CH];
        let mut bias = vec![0.0; LATENT_CH];
        // A fused multiply-add would produce one; the separate FP32 operations produce zero.
        weight[0] = 4097.0;
        bias[0] = -16_785_408.0;
        // Reordering this reduction can retain the one instead of rounding it away.
        weight[LATENT_CH + 1..LATENT_CH + 4].fill(1.0);
        let mut out = vec![half::f16::ZERO; n * VAE_KIN];
        grid.prepare_decoder_input(&z, &weight, &bias, &mut out);
        assert!(out.iter().all(|v| v.to_bits() == 0));
    }

    #[test]
    fn decoder_input_preserves_half_conversion_at_rounding_boundaries() {
        let grid = Grid { ft: 1, h: 1, w: 33 };
        let n = grid.voxels();
        let z = vec![0.0; LATENT_CH * n];
        let weight = vec![0.0; LATENT_CH * LATENT_CH];
        let bias = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1.0 + 1.0 / 2048.0,
            -1.0 - 1.0 / 2048.0,
            65504.0,
            -65504.0,
            65520.0,
            -65520.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            2.0f32.powi(-24),
            -2.0f32.powi(-24),
            2.0f32.powi(-25),
            -2.0f32.powi(-25),
            f32::MAX,
            -f32::MAX,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0xffc1_2345),
            1.0004882,
            1.0004884,
        ];
        let mut out = vec![half::f16::ZERO; n * VAE_KIN];
        grid.prepare_decoder_input(&z, &weight, &bias, &mut out);
        for v in 0..n {
            for o in 0..LATENT_CH {
                let mut sum = bias[o];
                for _ in 0..LATENT_CH {
                    sum += 0.0;
                }
                assert_eq!(
                    out[v * VAE_KIN + o].to_bits(),
                    half::f16::from_f32(sum).to_bits()
                );
            }
        }
    }
}
