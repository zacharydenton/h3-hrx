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
use crate::stack::{LayerCond, Stack, StackDims};
use crate::tiles;
use crate::weights::Weights;
use half::slice::HalfFloatSliceExt;
use std::sync::Arc;

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
    cls: hrx::Buffer,
    norm_table: hrx::Buffer,
}

pub struct VideoVae {
    weights: Weights,
    ones: Arc<hrx::Buffer>,
    zeros: hrx::Buffer,
    built: Option<Built>,
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
    pub fn open(gpu: &hrx::Gpu, path: impl AsRef<std::path::Path>) -> Result<Self> {
        let weights = Weights::open(path, crate::plan::vvae::plan)?;
        // the attention's per-head norms are ones for this stack, and the AdaLN tables are zero: it
        // modulates with a learned per-block scale alone
        let ones = Arc::new(gpu.alloc(VAE_HID * 4)?);
        let one_row: Vec<u8> = (0..VAE_HID).flat_map(|_| 1.0f32.to_le_bytes()).collect();
        gpu.h2d(&ones, &one_row)?;
        // A modulation table is (scale, shift) per class, so the norm kernel reads two rows of the
        // hidden width from it — not one. Half of that is an out-of-bounds read whose effect is a
        // garbage shift, small enough to look like rounding.
        let zeros = gpu.alloc(2 * VAE_HID * 4)?;
        gpu.memset(&zeros, 0, 2 * VAE_HID * 4)?;
        Ok(Self {
            pq_w: weights.host_f32("vae.post_quant_conv.w", LATENT_CH * LATENT_CH)?,
            pq_b: weights.host_f32("vae.post_quant_conv.b", LATENT_CH)?,
            latents_mean: weights.host_f32("vae.latents_mean", LATENT_CH)?,
            latents_std: weights.host_f32("vae.latents_std", LATENT_CH)?,
            enc_mean: weights.host_f32("venc.latents_mean", LATENT_CH)?,
            enc_std: weights.host_f32("venc.latents_std", LATENT_CH)?,
            weights,
            ones,
            zeros,
            built: None,
        })
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    /// Builds the stack, the two projections and the buffers for one latent grid, if the last call was
    /// for a different one.
    fn ensure(&mut self, gpu: &hrx::Gpu, c: &Compiler, grid: Grid) -> Result<()> {
        if self.built.as_ref().is_some_and(|b| b.grid == grid) {
            return Ok(());
        }
        // dropping first frees the previous grid's device memory before the next is asked for
        self.built = None;
        let (n, nt) = (grid.voxels(), grid.tokens());
        let stack = Stack::new(
            c,
            gpu,
            dims(),
            nt,
            VAE_BLOCKS,
            &self.weights,
            |i| format!("blocks.{i}."),
            false,
            self.ones.clone(),
            "vae",
        )?;
        let t = stack.capacity();

        let x = gpu.alloc(t * VAE_HID * 4)?;
        gpu.memset(&x, 0, t * VAE_HID * 4)?;
        let in16 = gpu.alloc(t * VAE_KIN * 2)?;
        gpu.memset(&in16, 0, t * VAE_KIN * 2)?;
        let cls = gpu.alloc(t * 4)?;
        gpu.memset(&cls, 0, t * 4)?;

        // the rotary table over this grid; the register and cls rows keep the identity rotation
        let (mut cos_h, mut sin_h) = (
            vec![1.0f32; t * VAE_ROPE_HALF],
            vec![0.0f32; t * VAE_ROPE_HALF],
        );
        crate::rope::vae(grid.ft, grid.h, grid.w, &mut cos_h, &mut sin_h);
        let cos = gpu.alloc(t * VAE_ROPE_HALF * 4)?;
        let sin = gpu.alloc(t * VAE_ROPE_HALF * 4)?;
        gpu.h2d(&cos, as_bytes(&cos_h))?;
        gpu.h2d(&sin, as_bytes(&sin_h))?;

        // the output norm's (scale, shift) table: no scale, the checkpoint's bias as the shift
        let norm_table = gpu.alloc(2 * VAE_HID * 4)?;
        gpu.memset(&norm_table, 0, VAE_HID * 4)?;
        let bias = self.weights.at(gpu, "vae.norm_out.b", VAE_HID * 4)?;
        gpu.d2d_at(&norm_table, VAE_HID * 4, &bias, 0, VAE_HID * 4)?;

        self.built = Some(Built {
            proj_in: Gemm::build(
                c,
                gpu,
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
            norm_out: Prepare::build(c, gpu, "lnorm", "f16", VAE_HID, 1e-5, 1, 0)?,
            proj_out: Gemm::build(
                c,
                gpu,
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
            a_q: gpu.alloc(t * VAE_HID * 2)?,
            a_s: gpu.alloc(t * 4)?,
            out16: gpu.alloc(t * VAE_OUT * 2)?,
            stack,
            grid,
            x,
            cos,
            sin,
            in16,
            cls,
            norm_table,
        });
        Ok(())
    }

    /// One clip: model-space latents `[24][ft][h][w]` to ImageNet-space frames
    /// `[3][ft*4][h*16][w*16]`.
    pub fn decode_clip(
        &mut self,
        gpu: &hrx::Gpu,
        c: &Compiler,
        prof: &mut Profile,
        z: &[f32],
        grid: Grid,
    ) -> Result<Vec<f32>> {
        self.ensure(gpu, c, grid)?;
        let (n, nt) = (grid.voxels(), grid.tokens());

        // post_quant_conv is a 24x24 matrix per voxel, small enough to stay on the host; the result
        // goes up as f16 padded to the GEMM's K of 64, the pad left at zero
        let mut in16 = vec![half::f16::ZERO; n * VAE_KIN];
        for v in 0..n {
            for o in 0..LATENT_CH {
                let mut acc = self.pq_b[o];
                for i in 0..LATENT_CH {
                    acc += self.pq_w[o * LATENT_CH + i] * z[i * n + v];
                }
                in16[v * VAE_KIN + o] = half::f16::from_f32(acc);
            }
        }

        let b = self.built.as_mut().expect("built above");
        gpu.h2d(&b.in16, as_bytes_f16(&in16))?;
        gpu.memset(&b.x, 0, nt * VAE_HID * 4)?;

        let w_in = self
            .weights
            .at(gpu, "vae.proj_in.w", VAE_HID * VAE_KIN * 2)?;
        let b_in = self.weights.at(gpu, "vae.proj_in.b", VAE_HID * 4)?;
        b.proj_in.run(
            gpu,
            Some(prof),
            "vae proj_in",
            n as u32,
            b.in16.binding(),
            w_in.binding(),
            None,
            b.x.binding(),
            Some((self.ones.binding(), b.cls.binding())),
            Some(b_in.binding()),
        )?;
        // the four register tokens follow the voxels, then a zero cls row
        let reg = self
            .weights
            .at(gpu, "vae.register_tokens", VAE_REG * VAE_HID * 4)?;
        gpu.d2d_at(&b.x, n * VAE_HID * 4, &reg, 0, VAE_REG * VAE_HID * 4)?;

        // every block modulates with its own learned scale and no shift
        let zeros = self.zeros.binding();
        let scales: Vec<(hrx::sys::BufferRef, hrx::sys::BufferRef)> = (0..b.stack.layers())
            .map(|i| {
                let blk = b.stack.block(i);
                (
                    blk.scale1
                        .as_ref()
                        .expect("a decoder block has scale1")
                        .binding(),
                    blk.scale2
                        .as_ref()
                        .expect("a decoder block has scale2")
                        .binding(),
                )
            })
            .collect();
        let cond = |i: usize| LayerCond {
            table_msa: zeros,
            gate_msa: scales[i].0,
            table_mlp: zeros,
            gate_mlp: scales[i].1,
        };
        let (x, cls, cos, sin) = (
            b.x.binding(),
            b.cls.binding(),
            b.cos.binding(),
            b.sin.binding(),
        );
        b.stack
            .forward(gpu, prof, x, cls, cos, sin, &cond, 0, None)?;

        let w_norm = self.weights.at(gpu, "vae.norm_out.w", VAE_HID * 4)?;
        b.norm_out.run(
            gpu,
            Some(prof),
            "vae norm_out",
            nt as u32,
            x,
            Some((w_norm.binding(), b.norm_table.binding(), cls)),
            b.a_q.binding(),
            Some(b.a_s.binding()),
        )?;
        let w_out = self
            .weights
            .at(gpu, "vae.proj_out.w", VAE_OUT * VAE_HID * 2)?;
        let b_out = self.weights.at(gpu, "vae.proj_out.b", VAE_OUT * 4)?;
        b.proj_out.run(
            gpu,
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
        let mut out16 = vec![half::f16::ZERO; n * VAE_OUT];
        gpu.sync()?;
        gpu.d2h_ref(b.out16.slice(0, out16.len() * 2), bytes_mut(&mut out16))?;

        // unpatchify: token (t, y, x) holds [3][4][16][16]
        let (ftt, fh, fw) = grid.frames();
        let mut frames = vec![0.0f32; 3 * ftt * fh * fw];
        for t in 0..grid.ft {
            for y in 0..grid.h {
                for x in 0..grid.w {
                    let tok = ((t * grid.h + y) * grid.w + x) * VAE_OUT;
                    for ch in 0..3 {
                        for pt in 0..VAE_PT {
                            for py in 0..VAE_PS {
                                let src = tok + ((ch * VAE_PT + pt) * VAE_PS + py) * VAE_PS;
                                let dst = ((ch * ftt + t * VAE_PT + pt) * fh + y * VAE_PS + py)
                                    * fw
                                    + x * VAE_PS;
                                out16[src..src + VAE_PS]
                                    .convert_to_f32_slice(&mut frames[dst..dst + VAE_PS]);
                            }
                        }
                    }
                }
            }
        }
        Ok(frames)
    }

    /// One clip in spatial tiles, blended in pixel space — the released VAE's `tiled_decode`.
    pub fn decode_spatial(
        &mut self,
        gpu: &hrx::Gpu,
        c: &Compiler,
        prof: &mut Profile,
        z: &[f32],
        grid: Grid,
    ) -> Result<Vec<f32>> {
        let (ft, h, w) = (grid.ft, grid.h, grid.w);
        let (frames_n, height, width) = grid.frames();
        let (ys, yo) = tiles::split_tiles(height);
        let (xs, xo) = tiles::split_tiles(width);
        if ys.len() == 1 && xs.len() == 1 {
            return self.decode_clip(gpu, c, prof, z, grid);
        }
        let mut frames = vec![0.0f32; 3 * frames_n * height * width];
        let (th, tw) = (height.min(256), width.min(256));
        let (lh, lw) = (th / VAE_PS, tw / VAE_PS);
        let mut above: Vec<Vec<f32>> = vec![Vec::new(); xs.len()];
        let mut row: Vec<Vec<f32>> = vec![Vec::new(); xs.len()];
        let mut latent = vec![0.0f32; LATENT_CH * ft * lh * lw];

        for iy in 0..ys.len() {
            for ix in 0..xs.len() {
                for ch in 0..LATENT_CH {
                    for t in 0..ft {
                        for y in 0..lh {
                            let src =
                                ((ch * ft + t) * h + ys[iy] / VAE_PS + y) * w + xs[ix] / VAE_PS;
                            let dst = ((ch * ft + t) * lh + y) * lw;
                            latent[dst..dst + lw].copy_from_slice(&z[src..src + lw]);
                        }
                    }
                }
                row[ix] = self.decode_clip(gpu, c, prof, &latent, Grid { ft, h: lh, w: lw })?;
                let mut tile = row[ix].clone();
                if iy > 0 {
                    tiles::blend_pixels(&mut tile, &above[ix], yo[iy - 1], true, frames_n, th, tw);
                }
                if ix > 0 {
                    tiles::blend_pixels(
                        &mut tile,
                        &row[ix - 1],
                        xo[ix - 1],
                        false,
                        frames_n,
                        th,
                        tw,
                    );
                }
                let keep_h = th - if iy + 1 < ys.len() { yo[iy] } else { 0 };
                let keep_w = tw - if ix + 1 < xs.len() { xo[ix] } else { 0 };
                for ch in 0..3 {
                    for t in 0..frames_n {
                        for y in 0..keep_h {
                            let dst = ((ch * frames_n + t) * height + ys[iy] + y) * width + xs[ix];
                            let src = ((ch * frames_n + t) * th + y) * tw;
                            frames[dst..dst + keep_w].copy_from_slice(&tile[src..src + keep_w]);
                        }
                    }
                }
            }
            std::mem::swap(&mut above, &mut row);
        }
        Ok(frames)
    }

    /// The whole clip: latents `[24][T][H][W]` in, RGB bytes `[frames][H*16][W*16][3]` out.
    pub fn decode_video(
        &mut self,
        gpu: &hrx::Gpu,
        c: &Compiler,
        prof: &mut Profile,
        shape: &crate::layout::Shape,
        latents: &[f32],
        out: &mut [u8],
    ) -> Result<()> {
        let (t_len, h, w) = (
            shape.latent_t as usize,
            shape.lat_h as usize,
            shape.lat_w as usize,
        );
        let want_frames = shape.frames as usize;
        let plane = h * w * VAE_PS * VAE_PS;
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

        let mut overlap: Vec<f32> = Vec::new();
        let mut have_overlap = false;
        let mut decoded = 0usize;
        // the frames actually written, capped at what the caller asked for
        let mut append = |chunk: &[f32], nf: usize, src_ft: usize, decoded: &mut usize| {
            let take = if *decoded < want_frames {
                nf.min(want_frames - *decoded)
            } else {
                0
            };
            for f in 0..take {
                for q in 0..plane {
                    for ch in 0..3 {
                        let v = chunk[(ch * src_ft + f) * plane + q];
                        out[((*decoded + f) * plane + q) * 3 + ch] =
                            crate::pixels::imagenet_denormalise(v, ch);
                    }
                }
            }
            *decoded += nf;
        };

        for i in 0..plan.chunks {
            let start = i * VAE_CHUNK;
            let ft = (VAE_CHUNK + VAE_OVERLAP).min(tp - start);
            let mut z = vec![0.0f32; LATENT_CH * ft * h * w];
            for ch in 0..LATENT_CH {
                let src = (ch * tp + start) * h * w;
                let dst = ch * ft * h * w;
                z[dst..dst + ft * h * w].copy_from_slice(&zp[src..src + ft * h * w]);
            }
            let clip = self.decode_spatial(gpu, c, prof, &z, Grid { ft, h, w })?;
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
                let mut chunk = vec![0.0f32; 3 * nf * plane];
                for ch in 0..3 {
                    let src = (ch * clip_frames + f0) * plane;
                    chunk[ch * nf * plane..(ch + 1) * nf * plane]
                        .copy_from_slice(&clip[src..src + nf * plane]);
                }
                if j == 0 {
                    if have_overlap {
                        tiles::crossfade(&mut chunk, nf, &overlap, plane, plan.overlap_frames);
                    }
                    append(&chunk, nf, nf, &mut decoded);
                } else {
                    overlap = chunk;
                    have_overlap = true;
                }
            }
        }
        if have_overlap {
            let nf = overlap.len() / (3 * plane);
            let tail = std::mem::take(&mut overlap);
            append(&tail, nf, nf, &mut decoded);
        }

        let kept = decoded - plan.pad_frames;
        if kept != want_frames {
            return other(format!("decoded {kept} frames, expected {want_frames}"));
        }
        Ok(())
    }
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
    /// One tile of `frames` frames to moment rows, then to model-space latents `[24][T][h][w]`.
    ///
    /// A still image and a clip use different weights and a different tap count — `.w2` with one
    /// temporal tap against `.w3` with three — because the released encoder folds a single frame's
    /// causal padding into the kernel rather than replicating the frame.
    #[allow(clippy::too_many_arguments)]
    fn encode_tile(
        &mut self,
        gpu: &hrx::Gpu,
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
        let mut inrows = vec![half::f16::ZERO; rows0 * 8];
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

        let x = gpu.alloc(rows0 * 8 * 2)?;
        gpu.h2d(&x, as_bytes_f16(&inrows))?;
        // the widest [rows][channels] plane of the stack: rows fall as fast as channels rise
        let plane_bytes = rows0 * 128 * 2;
        let mut h = gpu.alloc(plane_bytes)?;
        let y = gpu.alloc(plane_bytes)?;
        let mut tmp = gpu.alloc(plane_bytes)?;
        let sc = gpu.alloc(plane_bytes)?;
        let stats = gpu.alloc(frames * 32 * 2 * 4)?;

        let w3 = |gpu: &hrx::Gpu, nm: &str, cout_pad: usize, k: usize| {
            self.weights
                .at(gpu, &format!("{nm}{suffix}"), cout_pad * k * 2)
        };

        let (mut t_len, mut hh, mut ww, mut chan) = (frames, th, tw, 128usize);
        let conv = Conv3d::build(
            c,
            gpu,
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
        conv.run(
            gpu,
            Some(prof),
            "venc conv_in",
            x.binding(),
            w3(gpu, "venc.conv_in", 128, ksz(8))?.binding(),
            self.weights.at(gpu, "venc.conv_in.b", 128 * 4)?.binding(),
            h.binding(),
            None,
        )?;

        for l in 0..6 {
            for r in 0..2 {
                let b = format!("venc.l{l}.r{r}.");
                let (cin, cout) = (chan, ENC_MID[l]);
                GroupNormSilu::build(c, gpu, t_len, hh, ww, cin)?.run(
                    gpu,
                    Some(prof),
                    "venc groupnorm",
                    h.binding(),
                    self.weights
                        .at(gpu, &format!("{b}norm1.g"), cin * 4)?
                        .binding(),
                    self.weights
                        .at(gpu, &format!("{b}norm1.b"), cin * 4)?
                        .binding(),
                    stats.binding(),
                    y.binding(),
                )?;
                Conv3d::build(
                    c,
                    gpu,
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
                    gpu,
                    Some(prof),
                    "venc conv",
                    y.binding(),
                    w3(gpu, &format!("{b}conv1"), cout, ksz(cin))?.binding(),
                    self.weights
                        .at(gpu, &format!("{b}conv1.b"), cout * 4)?
                        .binding(),
                    tmp.binding(),
                    None,
                )?;
                GroupNormSilu::build(c, gpu, t_len, hh, ww, cout)?.run(
                    gpu,
                    Some(prof),
                    "venc groupnorm",
                    tmp.binding(),
                    self.weights
                        .at(gpu, &format!("{b}norm2.g"), cout * 4)?
                        .binding(),
                    self.weights
                        .at(gpu, &format!("{b}norm2.b"), cout * 4)?
                        .binding(),
                    stats.binding(),
                    y.binding(),
                )?;
                // a change of width needs the skip projected: a 1x1 convolution, which is a matmul
                let resid = if cin != cout {
                    let k = cin.div_ceil(32) * 32;
                    Matmul::build(c, gpu, k, cout)?.run(
                        gpu,
                        Some(prof),
                        "venc shortcut",
                        t_len * hh * ww,
                        h.binding(),
                        self.weights
                            .at(gpu, &format!("{b}nin.wm"), cout * k * 2)?
                            .binding(),
                        self.weights
                            .at(gpu, &format!("{b}nin.b"), cout * 4)?
                            .binding(),
                        sc.binding(),
                    )?;
                    sc.binding()
                } else {
                    h.binding()
                };
                Conv3d::build(
                    c,
                    gpu,
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
                    gpu,
                    Some(prof),
                    "venc conv",
                    y.binding(),
                    w3(gpu, &format!("{b}conv2"), cout, ksz(cout))?.binding(),
                    self.weights
                        .at(gpu, &format!("{b}conv2.b"), cout * 4)?
                        .binding(),
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
                    gpu,
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
                down.run(
                    gpu,
                    Some(prof),
                    "venc down",
                    h.binding(),
                    w3(gpu, &b, chan, ksz(chan))?.binding(),
                    self.weights.at(gpu, &format!("{b}.b"), chan * 4)?.binding(),
                    tmp.binding(),
                    None,
                )?;
                std::mem::swap(&mut h, &mut tmp);
                t_len = down.tout;
                hh = down.ho;
                ww = down.wo;
            }
        }

        GroupNormSilu::build(c, gpu, t_len, hh, ww, chan)?.run(
            gpu,
            Some(prof),
            "venc groupnorm",
            h.binding(),
            self.weights.at(gpu, "venc.norm_out.g", chan * 4)?.binding(),
            self.weights.at(gpu, "venc.norm_out.b", chan * 4)?.binding(),
            stats.binding(),
            y.binding(),
        )?;
        Conv3d::build(
            c,
            gpu,
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
            gpu,
            Some(prof),
            "venc conv_out",
            y.binding(),
            w3(gpu, "venc.conv_out", 64, ksz(chan))?.binding(),
            self.weights.at(gpu, "venc.conv_out.b", 64 * 4)?.binding(),
            tmp.binding(),
            None,
        )?;
        let m = t_len * hh * ww;
        Matmul::build(c, gpu, 64, 64)?.run(
            gpu,
            Some(prof),
            "venc quant",
            m,
            tmp.binding(),
            self.weights
                .at(gpu, "venc.quant.wm", 64 * 64 * 2)?
                .binding(),
            self.weights.at(gpu, "venc.quant.b", 64 * 4)?.binding(),
            y.binding(),
        )?;

        // the head emits 64 channels: the first 24 are the posterior's mean, the rest its log variance,
        // which sampling would use and this does not
        let mut mom = vec![half::f16::ZERO; m * 64];
        gpu.sync()?;
        gpu.d2h_ref(y.slice(0, mom.len() * 2), bytes_mut(&mut mom))?;
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
        gpu: &hrx::Gpu,
        c: &Compiler,
        prof: &mut Profile,
        clip: Clip<'_>,
    ) -> Result<(Vec<f32>, usize)> {
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
                    gpu,
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
        gpu: &hrx::Gpu,
        c: &Compiler,
        prof: &mut Profile,
        clip: Clip<'_>,
    ) -> Result<(Vec<f32>, usize)> {
        let (lh, lw) = clip.latent();
        let frames = clip.frames;
        if frames == 1 {
            let (z, t) = self.encode_clip(gpu, c, prof, clip)?;
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
                gpu,
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

/// f32 has no padding and no invalid bit patterns, so its bytes are a plain reinterpretation.
pub fn as_bytes(v: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

pub fn as_bytes_mut(v: &mut [f32]) -> &mut [u8] {
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

pub fn as_bytes_f16(v: &[half::f16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn bytes_mut(v: &mut [half::f16]) -> &mut [u8] {
    // f16 is a plain two-byte value with no padding and no invalid patterns, so writing its bytes is
    // the same as writing the values
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
