//! The video VAE: 36 transformer blocks between a patch embedding and an unpatchify, run over spatial
//! tiles and temporal chunks the way the released model does.
//!
//! The decoder's shape is not free. A grid of `(ft, h, w)` latent voxels becomes `ft*h*w` tokens plus
//! four register tokens and a zero cls row, and the rotary table is built over coordinates normalised
//! to that grid — so a tile decoded on its own is not a crop of the whole frame decoded at once. That
//! is why the tiling and the blends exist, and why the stack is rebuilt whenever the grid changes.
use crate::compile::Compiler;
use crate::dispatch::{Gemm, Prepare, Profile, Tile};
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
        gpu.h2d(&cos, bytemuck_f32(&cos_h))?;
        gpu.h2d(&sin, bytemuck_f32(&sin_h))?;

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
        gpu.h2d(&b.in16, bytemuck_f16(&in16))?;
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

fn bytemuck_f32(v: &[f32]) -> &[u8] {
    // f32 has no padding and no invalid bit patterns, so its bytes are a plain reinterpretation
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn bytemuck_f16(v: &[half::f16]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn bytes_mut(v: &mut [half::f16]) -> &mut [u8] {
    // f16 is a plain two-byte value with no padding and no invalid patterns, so writing its bytes is
    // the same as writing the values
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}
