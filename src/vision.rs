//! The vision tower: Qwen3-VL's ViT, 27 blocks on an f16 residual stream.
//!
//! It differs from every other stack here in enough ways to be worth its own module rather than a
//! configuration of `Stack`: LayerNorm with a bias instead of RMSNorm, GELU instead of SwiGLU, a
//! two-dimensional rope over the patch grid, heads of 72 padded to 128 for the WMMA attention, and
//! three DeepStack mergers tapping the stream mid-tower. Its weights live in the text encoder's
//! checkpoint, because that is where Qwen3-VL keeps them.
//!
//! The output is two things: `merged`, the tower's own embedding of every 2x2 patch block, and
//! `deepstack`, the same shape taken from blocks 8, 16 and 24 — the DiT consumes all four.
use crate::compile::{Cfg, Compiler};
use crate::dispatch::{checked, LayerNorm16, Matmul16, Profile};
use crate::error::{invalid, Result};
use crate::model::*;
pub use crate::rope::vision as rotary_tables;
use crate::weights::Weights;
use half::vec::HalfFloatVecExt;

/// What one image's tower run produces.
#[derive(Clone)]
pub struct Embedding {
    /// `[n/4][5120]`, one row per merged 2x2 patch block
    pub merged: Vec<f32>,
    /// `[3][n/4][5120]`, the same rows from blocks 8, 16 and 24
    pub deepstack: Vec<f32>,
    /// the merged row count
    pub tokens: usize,
}

/// The bilinear resample of the learned 48x48 position table onto a `gh` by `gw` grid, added to the
/// patch embedding in place.
///
/// The interpolation is f32 throughout and the grid coordinate is `hy * (G - 1) / (gh - 1)` — a single
/// row or column maps to zero rather than dividing by it. The row it lands on is the merge order, the
/// same permutation the patches themselves took.
pub fn add_positions(x: &mut [f32], pos: &[f32], gh: usize, gw: usize) {
    let g = VPOS_GRID;
    for hy in 0..gh {
        for wx in 0..gw {
            let fh = if gh == 1 {
                0.0
            } else {
                hy as f32 * (g - 1) as f32 / (gh - 1) as f32
            };
            let fw = if gw == 1 {
                0.0
            } else {
                wx as f32 * (g - 1) as f32 / (gw - 1) as f32
            };
            let (h0, w0) = (fh as usize, fw as usize);
            let (h1, w1) = ((h0 + 1).min(g - 1), (w0 + 1).min(g - 1));
            let (dh, dw) = (fh - h0 as f32, fw - w0 as f32);
            let pi = (((hy / 2) * (gw / 2) + wx / 2) * 2 + hy % 2) * 2 + wx % 2;
            let row = &mut x[pi * VHID..(pi + 1) * VHID];
            let at = |r: usize, c: usize| &pos[(r * g + c) * VHID..(r * g + c) * VHID + VHID];
            let (r00, r01, r10, r11) = (at(h0, w0), at(h0, w1), at(h1, w0), at(h1, w1));
            for c in 0..VHID {
                row[c] += (1.0 - dh) * (1.0 - dw) * r00[c]
                    + (1.0 - dh) * dw * r01[c]
                    + dh * (1.0 - dw) * r10[c]
                    + dh * dw * r11[c];
            }
        }
    }
}

/// Add interpolated positions to merge-ordered FP32 patch rows and narrow them
/// to the vision tower's FP16 residual stream. `x` holds `[gh*gw][1152]` values
/// and `pos` holds the learned `[48][48][1152]` table.
pub fn prepare_tokens(x: &mut [f32], pos: &[f32], gh: usize, gw: usize) -> Vec<half::f16> {
    add_positions(x, pos, gh, gw);
    Vec::from_f32_slice(x)
}

// Four position-table rows followed by four FP32 coefficient bit patterns per
// merge-ordered patch. Keep coordinate division on the host to match add_positions
// exactly; the device performs the channel-wise arithmetic without contraction.
fn position_map(gh: usize, gw: usize) -> Vec<u32> {
    let g = VPOS_GRID;
    let mut map = vec![0; gh * gw * 8];
    for hy in 0..gh {
        for wx in 0..gw {
            let fh = hy as f32 * (g - 1) as f32 / (gh - 1) as f32;
            let fw = wx as f32 * (g - 1) as f32 / (gw - 1) as f32;
            let (h0, w0) = (fh as usize, fw as usize);
            let (h1, w1) = ((h0 + 1).min(g - 1), (w0 + 1).min(g - 1));
            let (dh, dw) = (fh - h0 as f32, fw - w0 as f32);
            let pi = (((hy / 2) * (gw / 2) + wx / 2) * 2 + hy % 2) * 2 + wx % 2;
            map[pi * 8..(pi + 1) * 8].copy_from_slice(&[
                (h0 * g + w0) as u32,
                (h0 * g + w1) as u32,
                (h1 * g + w0) as u32,
                (h1 * g + w1) as u32,
                ((1.0 - dh) * (1.0 - dw)).to_bits(),
                ((1.0 - dh) * dw).to_bits(),
                (dh * (1.0 - dw)).to_bits(),
                (dh * dw).to_bits(),
            ]);
        }
    }
    map
}

fn prepare_tokens_device(
    stream: &mut hrx::Stream,
    c: &Compiler,
    prof: &mut Profile,
    x: hrx::View<'_>,
    pos: hrx::View<'_>,
    gh: usize,
    gw: usize,
) -> Result<hrx::Buffer> {
    let n = gh * gw;
    let count = n.checked_mul(VHID).filter(|&count| count <= 1 << 30);
    let Some(count) = count else {
        return invalid("vision patch grid exceeds the position kernel's extent");
    };
    let map = stream.allocate_from(bytemuck::cast_slice(&position_map(gh, gw)))?;
    let out = stream.allocate(n * VHID * 2)?;
    let kernel = c.get(
        stream,
        "vision_positions",
        "h3_vision_positions",
        &Cfg::new(),
    )?;
    checked(
        stream,
        &kernel,
        Some(prof),
        "vision positions",
        &[count as u32],
        &[count as u32],
        &[x, pos, map.binding(), out.binding()],
        &[
            n * VHID * 4,
            VPOS_GRID * VPOS_GRID * VHID * 4,
            n * 32,
            n * VHID * 2,
        ],
    )?;
    Ok(out)
}

/// Runs the tower over one image, reading its weights from the text encoder's checkpoint.
///
/// `pixels` is `[height][width][3]` in `[0, 1]`. Both extents must be multiples of 32: 16 for the patch
/// and another factor of two because the merger consumes 2x2 blocks.
pub fn embed(
    stream: &mut hrx::Stream,
    c: &Compiler,
    prof: &mut Profile,
    weights: &Weights,
    pixels: &[f32],
    height: usize,
    width: usize,
) -> Result<Embedding> {
    embed_frames(
        stream, c, prof, weights, pixels, pixels, height, width, false,
    )
}

/// Embed a real two-frame temporal patch, in upstream channel/time/space order.
#[allow(clippy::too_many_arguments)]
pub fn embed_pair(
    stream: &mut hrx::Stream,
    c: &Compiler,
    prof: &mut Profile,
    weights: &Weights,
    first: &[f32],
    second: &[f32],
    height: usize,
    width: usize,
) -> Result<Embedding> {
    embed_frames(stream, c, prof, weights, first, second, height, width, true)
}

#[allow(clippy::too_many_arguments)]
fn embed_frames(
    stream: &mut hrx::Stream,
    c: &Compiler,
    prof: &mut Profile,
    weights: &Weights,
    first: &[f32],
    second: &[f32],
    height: usize,
    width: usize,
    video: bool,
) -> Result<Embedding> {
    let need = height.checked_mul(width).and_then(|n| n.checked_mul(3));
    if need != Some(first.len()) || need != Some(second.len()) {
        return invalid("vision frame buffer does not match its dimensions");
    }
    if !height.is_multiple_of(32) || !width.is_multiple_of(32) || height < 32 || width < 32 {
        return invalid("vision images need height and width multiples of 32");
    }
    let (gh, gw) = (height / 16, width / 16);
    let n = gh * gw;
    let m = n / 4;
    // the attention's token capacity: sixteen rows of slack, rounded to 32
    let cap = (n + 16).div_ceil(32) * 32;

    // patches in merge order, CLIP-normalised, straight into the patch projection
    let patches = if video {
        crate::pixels::vision_pair_patches(first, second, gh, gw, width)
    } else {
        crate::pixels::vision_patches(first, gh, gw, width)
    };
    let pa = stream.allocate(patches.len() * 4)?;
    crate::transfer::upload(stream, pa.binding(), crate::vvae::as_bytes(&patches))?;
    let x32 = stream.allocate(n * VHID * 4)?;
    let held_1 = weights.at(stream, "vis.patch.w", VHID * VISION_PATCH * 2)?;
    let held_2 = weights.at(stream, "vis.patch.b", VHID * 4)?;
    Matmul16::build(c, stream, "bias", VISION_PATCH, VHID)?.run(
        stream,
        Some(prof),
        "vision patch",
        n,
        pa.binding(),
        held_1.binding(),
        held_2.binding(),
        x32.binding(),
        None,
    )?;

    let pos = weights.at(stream, "vis.pos", VPOS_GRID * VPOS_GRID * VHID * 4)?;
    let x16 = prepare_tokens_device(stream, c, prof, x32.binding(), pos.binding(), gh, gw)?;

    let (mut cosv, mut sinv) = (vec![0.0f32; n * 36], vec![0.0f32; n * 36]);
    rotary_tables(gh, gw, &mut cosv, &mut sinv);
    let cosb = stream.allocate(cosv.len() * 4)?;
    let sinb = stream.allocate(sinv.len() * 4)?;
    crate::transfer::upload(stream, cosb.binding(), crate::vvae::as_bytes(&cosv))?;
    crate::transfer::upload(stream, sinb.binding(), crate::vvae::as_bytes(&sinv))?;

    let qkv_width = 3 * VHEADS * VHD;
    let ln = stream.allocate(n * VHID * 4)?;
    let qkv = stream.allocate(n * qkv_width * 4)?;
    let q16 = stream.allocate(cap * VHEADS * VHDP * 2)?;
    let k16 = stream.allocate(cap * VHEADS * VHDP * 2)?;
    let v16 = stream.allocate(cap * VHEADS * VHDP * 2)?;
    let att16 = stream.allocate(cap * VHEADS * VHDP * 2)?;
    let hid16 = stream.allocate(n * VMLP * 2)?;
    let ln4 = stream.allocate(m * VMERGE * 4)?;
    let mid = stream.allocate(m * VMERGE * 4)?;
    let out5 = stream.allocate(m * VOUT * 4)?;
    // the residual GEMM's per-column scale, which this tower does not use: all ones
    let lam = stream.allocate(VMERGE * 4)?;
    let ones: Vec<f32> = vec![1.0; VMERGE];
    crate::transfer::upload(stream, lam.binding(), crate::vvae::as_bytes(&ones))?;
    // the rows past `n` are read by the attention and never written, so they start at zero
    for b in [&q16, &k16, &v16] {
        crate::transfer::fill(stream, b.slice(0, cap * VHEADS * VHDP * 2), 0)?;
    }

    let ans = "h3.attention_mha_lds_f16_wmma.";
    let attn_cfg: Cfg = vec![
        (format!("{ans}q_stride"), (VHEADS * VHDP).to_string()),
        (format!("{ans}kv_stride"), (VHEADS * VHDP).to_string()),
        (format!("{ans}tokens"), n.to_string()),
        (format!("{ans}token_capacity"), cap.to_string()),
        (
            format!("{ans}scale"),
            crate::compile::num(1.0 / (VHD as f64).sqrt()),
        ),
        (format!("{ans}out_stride"), (VHEADS * VHDP).to_string()),
    ];
    let attn = c.get(
        stream,
        "attention_mha_family",
        "h3_attention_mha_lds_f16_wmma",
        &attn_cfg,
    )?;
    let rns = "h3.rope2d_qkv_f16.";
    let rope_cfg: Cfg = vec![
        (format!("{rns}heads"), VHEADS.to_string()),
        (format!("{rns}hd"), VHD.to_string()),
        (format!("{rns}hd_pad"), VHDP.to_string()),
    ];
    let rope = c.get(stream, "rope2d_qkv_f16", "h3_rope2d_qkv_f16", &rope_cfg)?;

    let norm = LayerNorm16::build(c, stream, VHID, 1e-6)?;
    let norm4 = LayerNorm16::build(c, stream, VMERGE, 1e-6)?;
    let g_qkv = Matmul16::build(c, stream, "bias", VHID, qkv_width)?;
    let g_proj = Matmul16::build(c, stream, "resid", VHEADS * VHDP, VHID)?;
    let g_fc1 = Matmul16::build(c, stream, "gelu_f16", VHID, VMLP)?;
    let g_fc2 = Matmul16::build(c, stream, "resid", VMLP, VHID)?;
    let g_merge1 = Matmul16::build(c, stream, "gelu_erf", VMERGE, VMERGE)?;
    let g_merge2 = Matmul16::build(c, stream, "bias", VMERGE, VOUT)?;

    let mut deepstack = vec![0.0f32; 3 * m * VOUT];
    for i in 0..VBLOCKS {
        let b = format!("vis.b{i}.");
        let held_1 = weights.at(stream, &format!("{b}norm1.w"), VHID * 4)?;
        let held_2 = weights.at(stream, &format!("{b}norm1.b"), VHID * 4)?;
        norm.run(
            stream,
            Some(prof),
            n,
            x16.binding(),
            held_1.binding(),
            held_2.binding(),
            ln.binding(),
        )?;
        let held_1 = weights.at(stream, &format!("{b}qkv.w"), qkv_width * VHID * 2)?;
        let held_2 = weights.at(stream, &format!("{b}qkv.b"), qkv_width * 4)?;
        g_qkv.run(
            stream,
            Some(prof),
            "vision qkv",
            n,
            ln.binding(),
            held_1.binding(),
            held_2.binding(),
            qkv.binding(),
            None,
        )?;
        // rope reads the packed f32 q|k|v at its 72-deep heads and writes them out padded to 128
        checked(
            stream,
            &rope,
            Some(prof),
            "vision rope",
            &[n as u32],
            &[n as u32],
            &[
                qkv.binding(),
                cosb.binding(),
                sinb.binding(),
                q16.binding(),
                k16.binding(),
                v16.binding(),
            ],
            &[
                n * qkv_width * 4,
                n * (VHD / 2) * 4,
                n * (VHD / 2) * 4,
                n * VHEADS * VHDP * 2,
                n * VHEADS * VHDP * 2,
                n * VHEADS * VHDP * 2,
            ],
        )?;
        // the attention kernel was compiled to `cap` rows, which is what these hold
        let pad = cap * VHEADS * VHDP * 2;
        checked(
            stream,
            &attn,
            Some(prof),
            "vision attention",
            &[n as u32, VHEADS as u32],
            &[n as u32],
            &[q16.binding(), k16.binding(), v16.binding(), att16.binding()],
            &[pad, pad, pad, pad],
        )?;
        // the residual GEMM takes its A operand as f16, which is what the attention already wrote
        let held_1 = weights.at(stream, &format!("{b}proj.w"), VHID * VHEADS * VHDP * 2)?;
        let held_2 = weights.at(stream, &format!("{b}proj.b"), VHID * 4)?;
        g_proj.run(
            stream,
            Some(prof),
            "vision proj",
            n,
            att16.binding(),
            held_1.binding(),
            held_2.binding(),
            x16.binding(),
            Some(lam.binding()),
        )?;
        let held_1 = weights.at(stream, &format!("{b}norm2.w"), VHID * 4)?;
        let held_2 = weights.at(stream, &format!("{b}norm2.b"), VHID * 4)?;
        norm.run(
            stream,
            Some(prof),
            n,
            x16.binding(),
            held_1.binding(),
            held_2.binding(),
            ln.binding(),
        )?;
        let held_1 = weights.at(stream, &format!("{b}fc1.w"), VMLP * VHID * 2)?;
        let held_2 = weights.at(stream, &format!("{b}fc1.b"), VMLP * 4)?;
        g_fc1.run(
            stream,
            Some(prof),
            "vision fc1",
            n,
            ln.binding(),
            held_1.binding(),
            held_2.binding(),
            hid16.binding(),
            None,
        )?;
        let held_1 = weights.at(stream, &format!("{b}fc2.w"), VHID * VMLP * 2)?;
        let held_2 = weights.at(stream, &format!("{b}fc2.b"), VHID * 4)?;
        g_fc2.run(
            stream,
            Some(prof),
            "vision fc2",
            n,
            hid16.binding(),
            held_1.binding(),
            held_2.binding(),
            x16.binding(),
            Some(lam.binding()),
        )?;

        // a DeepStack merger reads the stream after its block, viewing [n][1152] as [m][4608]
        for (j, at) in VDEEPSTACK.iter().enumerate() {
            if i != *at {
                continue;
            }
            let d = format!("vis.ds{j}.");
            let held_1 = weights.at(stream, &format!("{d}norm.w"), VMERGE * 4)?;
            let held_2 = weights.at(stream, &format!("{d}norm.b"), VMERGE * 4)?;
            norm4.run(
                stream,
                Some(prof),
                m,
                x16.binding(),
                held_1.binding(),
                held_2.binding(),
                ln4.binding(),
            )?;
            let held_1 = weights.at(stream, &format!("{d}fc1.w"), VMERGE * VMERGE * 2)?;
            let held_2 = weights.at(stream, &format!("{d}fc1.b"), VMERGE * 4)?;
            g_merge1.run(
                stream,
                Some(prof),
                "vision deepstack",
                m,
                ln4.binding(),
                held_1.binding(),
                held_2.binding(),
                mid.binding(),
                None,
            )?;
            let held_1 = weights.at(stream, &format!("{d}fc2.w"), VOUT * VMERGE * 2)?;
            let held_2 = weights.at(stream, &format!("{d}fc2.b"), VOUT * 4)?;
            g_merge2.run(
                stream,
                Some(prof),
                "vision deepstack",
                m,
                mid.binding(),
                held_1.binding(),
                held_2.binding(),
                out5.binding(),
                None,
            )?;
            stream.read_blocking(
                out5.slice(0, m * VOUT * 4),
                crate::vvae::as_bytes_mut(&mut deepstack[j * m * VOUT..(j + 1) * m * VOUT]),
            )?;
        }
    }

    // The final merger normalises each 1152-wide patch row and only then lets the GEMM read the
    // result as [m][4608]. The DeepStack mergers above normalise the merged row itself. They look
    // interchangeable and are not — the statistics are over different sets of numbers.
    let held_1 = weights.at(stream, "vis.merger.norm.w", VHID * 4)?;
    let held_2 = weights.at(stream, "vis.merger.norm.b", VHID * 4)?;
    norm.run(
        stream,
        Some(prof),
        n,
        x16.binding(),
        held_1.binding(),
        held_2.binding(),
        ln.binding(),
    )?;
    let held_1 = weights.at(stream, "vis.merger.fc1.w", VMERGE * VMERGE * 2)?;
    let held_2 = weights.at(stream, "vis.merger.fc1.b", VMERGE * 4)?;
    g_merge1.run(
        stream,
        Some(prof),
        "vision merger",
        m,
        ln.binding(),
        held_1.binding(),
        held_2.binding(),
        mid.binding(),
        None,
    )?;
    let held_1 = weights.at(stream, "vis.merger.fc2.w", VOUT * VMERGE * 2)?;
    let held_2 = weights.at(stream, "vis.merger.fc2.b", VOUT * 4)?;
    g_merge2.run(
        stream,
        Some(prof),
        "vision merger",
        m,
        mid.binding(),
        held_1.binding(),
        held_2.binding(),
        out5.binding(),
        None,
    )?;
    let mut merged = vec![0.0f32; m * VOUT];
    stream.read_blocking(
        out5.slice(0, m * VOUT * 4),
        crate::vvae::as_bytes_mut(&mut merged),
    )?;
    Ok(Embedding {
        merged,
        deepstack,
        tokens: m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
    fn resident_positions_match_cpu_interpolation_and_half_rounding() -> Result<()> {
        for sanitized in [false, true] {
            let mut options = crate::compile::Options::default();
            let mut stream_options = hrx::StreamOptions::default();
            if sanitized {
                options.sanitizer = hrx::loom::SanitizerChecks {
                    access: true,
                    value: true,
                    operation: true,
                    race: true,
                };
                stream_options.compute_engine = hrx::execution::ComputeEngine::Aql {
                    maximum_private_bytes: 4096,
                };
                stream_options.sanitizer = Some(options.sanitizer_runtime.clone());
            }
            let mut stream = hrx::Device::open(0)?.stream_with_options(stream_options)?;
            let compiler = Compiler::with_options(None, "", options);
            let mut profile = Profile::default();
            for zero_positions in [false, true] {
                let positions: Vec<f32> = (0..VPOS_GRID * VPOS_GRID * VHID)
                    .map(|i| {
                        if zero_positions {
                            0.0
                        } else {
                            ((i * 37 % 1009) as f32 - 504.0) / 127.0
                        }
                    })
                    .collect();
                let pos = stream.allocate_from(bytemuck::cast_slice(&positions))?;
                for (gh, gw) in [(2, 2), (4, 6), (6, 4), (16, 18), (48, 84), (84, 48)] {
                    if sanitized && gh * gw > 16 * 18 {
                        continue;
                    }
                    let input: Vec<f32> = (0..gh * gw * VHID)
                        .map(|i| {
                            let bits = ((i / 2) as u16).wrapping_mul(37) % 0x7bff;
                            let lo = half::f16::from_bits(bits).to_f32();
                            let hi = half::f16::from_bits(bits + 1).to_f32();
                            let value = if i & 1 == 0 { lo } else { (lo + hi) * 0.5 };
                            if i & 2 == 0 {
                                value
                            } else {
                                -value
                            }
                        })
                        .collect();
                    let x = stream.allocate_from(bytemuck::cast_slice(&input))?;
                    let out = prepare_tokens_device(
                        &mut stream,
                        &compiler,
                        &mut profile,
                        x.binding(),
                        pos.binding(),
                        gh,
                        gw,
                    )?;
                    compiler.check_sanitizers(&mut stream)?;
                    let expected = prepare_tokens(&mut input.clone(), &positions, gh, gw);
                    let mut actual = vec![half::f16::ZERO; input.len()];
                    stream.read_blocking(out.binding(), bytemuck::cast_slice_mut(&mut actual))?;
                    for (i, (got, want)) in actual.iter().zip(&expected).enumerate() {
                        assert_eq!(
                            got.to_bits(),
                            want.to_bits(),
                            "element {i}, grid {gh}x{gw}, zero positions: {zero_positions}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn token_preparation_preserves_fp32_positions_and_scalar_half_bits() {
        let mut pos: Vec<_> = (0..VPOS_GRID * VPOS_GRID * VHID)
            .map(|i| ((i * 37 % 1009) as f32 - 504.0) / 127.0)
            .collect();
        for zero_positions in [false, true] {
            if zero_positions {
                pos.fill(0.0);
            }
            for (gh, gw) in [(2, 2), (4, 6), (6, 4), (16, 18)] {
                // Visit every half bit pattern and its next midpoint, including
                // subnormals, signed zeros, infinities and NaNs.
                let mut x: Vec<_> = (0..gh * gw * VHID)
                    .map(|i| {
                        let bits = ((i / 2) as u16).wrapping_mul(37);
                        let lo = half::f16::from_bits(bits).to_f32();
                        let hi = half::f16::from_bits(bits.wrapping_add(1)).to_f32();
                        if i & 1 == 0 {
                            lo
                        } else {
                            (lo + hi) * 0.5
                        }
                    })
                    .collect();
                let mut expected = x.clone();
                add_positions(&mut expected, &pos, gh, gw);
                let want: Vec<_> = expected.iter().map(|&v| half::f16::from_f32(v)).collect();
                let got = prepare_tokens(&mut x, &pos, gh, gw);
                assert_eq!(crate::vvae::as_bytes(&x), crate::vvae::as_bytes(&expected));
                assert_eq!(
                    crate::vvae::as_bytes_f16(&got),
                    crate::vvae::as_bytes_f16(&want),
                    "grid {gh}x{gw}, zero positions: {zero_positions}"
                );
            }
        }
    }

    /// A position table whose row `(r, c)` is the constant `r * 100 + c`, so an interpolated row reads
    /// back as its grid coordinate.
    fn ramp_table() -> Vec<f32> {
        let g = VPOS_GRID;
        let mut pos = vec![0.0f32; g * g * VHID];
        for r in 0..g {
            for c in 0..g {
                let v = (r * 100 + c) as f32;
                pos[(r * g + c) * VHID..(r * g + c) * VHID + VHID].fill(v);
            }
        }
        pos
    }

    #[test]
    fn the_corners_of_the_grid_land_on_the_corners_of_the_table() {
        let (gh, gw) = (4usize, 4usize);
        let pos = ramp_table();
        let mut x = vec![0.0f32; gh * gw * VHID];
        add_positions(&mut x, &pos, gh, gw);
        let row = |hy: usize, wx: usize| {
            let pi = (((hy / 2) * (gw / 2) + wx / 2) * 2 + hy % 2) * 2 + wx % 2;
            x[pi * VHID]
        };
        assert_eq!(row(0, 0), 0.0);
        let last = (VPOS_GRID - 1) as f32;
        assert!((row(gh - 1, gw - 1) - (last * 100.0 + last)).abs() < 0.01);
        // and a middle patch lands proportionally along
        assert!(
            (row(1, 0) - (47.0 / 3.0 * 100.0)).abs() < 1.0,
            "{}",
            row(1, 0)
        );
    }

    #[test]
    fn a_single_row_or_column_maps_to_zero_rather_than_dividing_by_it() {
        // `embed` cannot reach this — a multiple-of-32 extent gives at least two patches each way —
        // but the guard is what keeps the division from being by zero, so it is exercised directly.
        let pos = ramp_table();
        let mut x = vec![0.0f32; 2 * VHID];
        add_positions(&mut x, &pos, 1, 2);
        assert!(x.iter().all(|v| v.is_finite()));
        assert_eq!(x[0], 0.0, "a lone row reads the table's first row");
    }

    #[test]
    fn the_positions_are_added_not_assigned() {
        let pos = ramp_table();
        let (gh, gw) = (2usize, 2usize);
        let mut x = vec![7.0f32; gh * gw * VHID];
        add_positions(&mut x, &pos, gh, gw);
        assert_eq!(x[0], 7.0, "patch (0, 0) reads table row 0, which is zero");
        assert!(
            x[VHID] > 7.0,
            "and the others accumulate onto what was there"
        );
    }

    #[test]
    fn every_patch_gets_exactly_one_position() {
        // The merge permutation has to be a bijection, or some row would be written twice and another
        // not at all. It is one exactly when both extents are even — which `embed` guarantees, since a
        // multiple-of-32 image is a multiple-of-2 patch grid.
        let place = |hy: usize, wx: usize, gw: usize| {
            (((hy / 2) * (gw / 2) + wx / 2) * 2 + hy % 2) * 2 + wx % 2
        };
        for (gh, gw) in [(2usize, 2usize), (6, 8), (2, 64), (64, 2), (48, 84)] {
            let mut seen = vec![0u32; gh * gw];
            for hy in 0..gh {
                for wx in 0..gw {
                    seen[place(hy, wx, gw)] += 1;
                }
            }
            assert!(seen.iter().all(|c| *c == 1), "{gh}x{gw}: {seen:?}");
        }
        // one column scatters a patch past the end of the grid, which is why the extents are checked
        assert_eq!(place(1, 0, 1), 2);
    }
}
