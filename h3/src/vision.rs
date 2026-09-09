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
use crate::weights::Weights;

/// What one image's tower run produces.
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

/// Runs the tower over one image, reading its weights from the text encoder's checkpoint.
///
/// `pixels` is `[height][width][3]` in `[0, 1]`. Both extents must be multiples of 32: 16 for the patch
/// and another factor of two because the merger consumes 2x2 blocks.
pub fn embed(
    gpu: &hrx::Gpu,
    c: &Compiler,
    prof: &mut Profile,
    weights: &Weights,
    pixels: &[f32],
    height: usize,
    width: usize,
) -> Result<Embedding> {
    if !height.is_multiple_of(32) || !width.is_multiple_of(32) || height < 32 || width < 32 {
        return invalid("vision images need height and width multiples of 32");
    }
    let (gh, gw) = (height / 16, width / 16);
    let n = gh * gw;
    let m = n / 4;
    // the attention's token capacity: sixteen rows of slack, rounded to 32
    let cap = (n + 16).div_ceil(32) * 32;
    let v = |nm: &str, bytes: usize| weights.at(gpu, nm, bytes);

    // patches in merge order, CLIP-normalised, straight into the patch projection
    let patches = crate::pixels::vision_patches(pixels, gh, gw, width);
    let pa = gpu.alloc(patches.len() * 4)?;
    gpu.h2d(&pa, crate::vvae::as_bytes(&patches))?;
    let x32 = gpu.alloc(n * VHID * 4)?;
    Matmul16::build(c, gpu, "bias", VISION_PATCH, VHID)?.run(
        gpu,
        Some(prof),
        "vision patch",
        n,
        pa.binding(),
        v("vis.patch.w", VHID * VISION_PATCH * 2)?.binding(),
        v("vis.patch.b", VHID * 4)?.binding(),
        x32.binding(),
        None,
    )?;

    // the position table is resampled on the host, then the stream narrows to f16 for the blocks
    let pos = weights.host_f32("vis.pos", VPOS_GRID * VPOS_GRID * VHID)?;
    let mut x0 = vec![0.0f32; n * VHID];
    gpu.d2h_ref(x32.binding(), crate::vvae::as_bytes_mut(&mut x0))?;
    add_positions(&mut x0, &pos, gh, gw);
    let x16h: Vec<half::f16> = x0.iter().map(|v| half::f16::from_f32(*v)).collect();
    let x16 = gpu.alloc(x16h.len() * 2)?;
    gpu.h2d(&x16, crate::vvae::as_bytes_f16(&x16h))?;

    let (mut cosv, mut sinv) = (vec![0.0f32; n * 36], vec![0.0f32; n * 36]);
    crate::rope::vision(gh, gw, &mut cosv, &mut sinv);
    let cosb = gpu.alloc(cosv.len() * 4)?;
    let sinb = gpu.alloc(sinv.len() * 4)?;
    gpu.h2d(&cosb, crate::vvae::as_bytes(&cosv))?;
    gpu.h2d(&sinb, crate::vvae::as_bytes(&sinv))?;

    let qkv_width = 3 * VHEADS * VHD;
    let ln = gpu.alloc(n * VHID * 4)?;
    let qkv = gpu.alloc(n * qkv_width * 4)?;
    let q16 = gpu.alloc(cap * VHEADS * VHDP * 2)?;
    let k16 = gpu.alloc(cap * VHEADS * VHDP * 2)?;
    let v16 = gpu.alloc(cap * VHEADS * VHDP * 2)?;
    let att16 = gpu.alloc(cap * VHEADS * VHDP * 2)?;
    let hid = gpu.alloc(n * VMLP * 4)?;
    let hid16 = gpu.alloc(n * VMLP * 2)?;
    let ln4 = gpu.alloc(m * VMERGE * 4)?;
    let mid = gpu.alloc(m * VMERGE * 4)?;
    let out5 = gpu.alloc(m * VOUT * 4)?;
    // the residual GEMM's per-column scale, which this tower does not use: all ones
    let lam = gpu.alloc(VMERGE * 4)?;
    let ones: Vec<f32> = vec![1.0; VMERGE];
    gpu.h2d(&lam, crate::vvae::as_bytes(&ones))?;
    // the rows past `n` are read by the attention and never written, so they start at zero
    for b in [&q16, &k16, &v16] {
        gpu.memset(b, 0, cap * VHEADS * VHDP * 2)?;
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
        gpu,
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
    let rope = c.get(gpu, "rope2d_qkv_f16", "h3_rope2d_qkv_f16", &rope_cfg)?;
    let cast = c.get(gpu, "cast_f32_f16", "h3_cast_f32_f16", &Cfg::new())?;

    let norm = LayerNorm16::build(c, gpu, VHID, 1e-6)?;
    let norm4 = LayerNorm16::build(c, gpu, VMERGE, 1e-6)?;
    let g_qkv = Matmul16::build(c, gpu, "bias", VHID, qkv_width)?;
    let g_proj = Matmul16::build(c, gpu, "resid", VHEADS * VHDP, VHID)?;
    let g_fc1 = Matmul16::build(c, gpu, "gelu", VHID, VMLP)?;
    let g_fc2 = Matmul16::build(c, gpu, "resid", VMLP, VHID)?;
    let g_merge1 = Matmul16::build(c, gpu, "gelu_erf", VMERGE, VMERGE)?;
    let g_merge2 = Matmul16::build(c, gpu, "bias", VMERGE, VOUT)?;

    let mut deepstack = vec![0.0f32; 3 * m * VOUT];
    for i in 0..VBLOCKS {
        let b = format!("vis.b{i}.");
        norm.run(
            gpu,
            Some(prof),
            n,
            x16.binding(),
            v(&format!("{b}norm1.w"), VHID * 4)?.binding(),
            v(&format!("{b}norm1.b"), VHID * 4)?.binding(),
            ln.binding(),
        )?;
        g_qkv.run(
            gpu,
            Some(prof),
            "vision qkv",
            n,
            ln.binding(),
            v(&format!("{b}qkv.w"), qkv_width * VHID * 2)?.binding(),
            v(&format!("{b}qkv.b"), qkv_width * 4)?.binding(),
            qkv.binding(),
            None,
        )?;
        // rope reads the packed f32 q|k|v at its 72-deep heads and writes them out padded to 128
        checked(
            gpu,
            &rope,
            Some(prof),
            "vision rope",
            [n as u32, VHEADS as u32, 1],
            [VHDP as u32, 1, 1],
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
            gpu,
            &attn,
            Some(prof),
            "vision attention",
            [n.div_ceil(64) as u32, VHEADS as u32, 1],
            [128, 1, 1],
            &[n as u32],
            &[q16.binding(), k16.binding(), v16.binding(), att16.binding()],
            &[pad, pad, pad, pad],
        )?;
        // the residual GEMM takes its A operand as f16, which is what the attention already wrote
        g_proj.run(
            gpu,
            Some(prof),
            "vision proj",
            n,
            att16.binding(),
            v(&format!("{b}proj.w"), VHID * VHEADS * VHDP * 2)?.binding(),
            v(&format!("{b}proj.b"), VHID * 4)?.binding(),
            x16.binding(),
            Some(lam.binding()),
        )?;
        norm.run(
            gpu,
            Some(prof),
            n,
            x16.binding(),
            v(&format!("{b}norm2.w"), VHID * 4)?.binding(),
            v(&format!("{b}norm2.b"), VHID * 4)?.binding(),
            ln.binding(),
        )?;
        g_fc1.run(
            gpu,
            Some(prof),
            "vision fc1",
            n,
            ln.binding(),
            v(&format!("{b}fc1.w"), VMLP * VHID * 2)?.binding(),
            v(&format!("{b}fc1.b"), VMLP * 4)?.binding(),
            hid.binding(),
            None,
        )?;
        checked(
            gpu,
            &cast,
            Some(prof),
            "vision cast",
            [(n * VMLP).div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            &[(n * VMLP) as u32],
            &[hid.binding(), hid16.binding()],
            &[n * VMLP * 4, n * VMLP * 2],
        )?;
        g_fc2.run(
            gpu,
            Some(prof),
            "vision fc2",
            n,
            hid16.binding(),
            v(&format!("{b}fc2.w"), VHID * VMLP * 2)?.binding(),
            v(&format!("{b}fc2.b"), VHID * 4)?.binding(),
            x16.binding(),
            Some(lam.binding()),
        )?;

        // a DeepStack merger reads the stream after its block, viewing [n][1152] as [m][4608]
        for (j, at) in VDEEPSTACK.iter().enumerate() {
            if i != *at {
                continue;
            }
            let d = format!("vis.ds{j}.");
            norm4.run(
                gpu,
                Some(prof),
                m,
                x16.binding(),
                v(&format!("{d}norm.w"), VMERGE * 4)?.binding(),
                v(&format!("{d}norm.b"), VMERGE * 4)?.binding(),
                ln4.binding(),
            )?;
            g_merge1.run(
                gpu,
                Some(prof),
                "vision deepstack",
                m,
                ln4.binding(),
                v(&format!("{d}fc1.w"), VMERGE * VMERGE * 2)?.binding(),
                v(&format!("{d}fc1.b"), VMERGE * 4)?.binding(),
                mid.binding(),
                None,
            )?;
            g_merge2.run(
                gpu,
                Some(prof),
                "vision deepstack",
                m,
                mid.binding(),
                v(&format!("{d}fc2.w"), VOUT * VMERGE * 2)?.binding(),
                v(&format!("{d}fc2.b"), VOUT * 4)?.binding(),
                out5.binding(),
                None,
            )?;
            gpu.d2h_ref(
                out5.slice(0, m * VOUT * 4),
                crate::vvae::as_bytes_mut(&mut deepstack[j * m * VOUT..(j + 1) * m * VOUT]),
            )?;
        }
    }

    // The final merger normalises each 1152-wide patch row and only then lets the GEMM read the
    // result as [m][4608]. The DeepStack mergers above normalise the merged row itself. They look
    // interchangeable and are not — the statistics are over different sets of numbers.
    norm.run(
        gpu,
        Some(prof),
        n,
        x16.binding(),
        v("vis.merger.norm.w", VHID * 4)?.binding(),
        v("vis.merger.norm.b", VHID * 4)?.binding(),
        ln.binding(),
    )?;
    g_merge1.run(
        gpu,
        Some(prof),
        "vision merger",
        m,
        ln.binding(),
        v("vis.merger.fc1.w", VMERGE * VMERGE * 2)?.binding(),
        v("vis.merger.fc1.b", VMERGE * 4)?.binding(),
        mid.binding(),
        None,
    )?;
    g_merge2.run(
        gpu,
        Some(prof),
        "vision merger",
        m,
        mid.binding(),
        v("vis.merger.fc2.w", VOUT * VMERGE * 2)?.binding(),
        v("vis.merger.fc2.b", VOUT * 4)?.binding(),
        out5.binding(),
        None,
    )?;
    let mut merged = vec![0.0f32; m * VOUT];
    gpu.d2h_ref(
        out5.slice(0, m * VOUT * 4),
        crate::vvae::as_bytes_mut(&mut merged),
    )?;
    gpu.sync()?;
    Ok(Embedding {
        merged,
        deepstack,
        tokens: m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
