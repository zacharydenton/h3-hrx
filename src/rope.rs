//! The four rotary tables, each with its own arithmetic.
//!
//! They are not variations on one function and are deliberately not unified. The DiT's angle is a
//! `float` and goes through `cosf`; the other three accumulate in `f64` and narrow once at the end. The
//! frequency bases differ (5e6 for the text encoder, 1e4 for the vision tower, 1e2 for the VAE), as do
//! the layouts: three interleaved axes of sixteen for the DiT, Qwen3-VL's interleaved [24, 20, 20]
//! sections for the text encoder, two halves of eighteen for the vision tower, three axes of eight over
//! normalised coordinates for the VAE.
use crate::model::*;

/// Which of (t, h, w) pair `j` reads, for Qwen3-VL's interleaved mrope sections.
pub fn mrope_axis(pair: usize) -> usize {
    if pair < 60 {
        pair % 3
    } else {
        0
    }
}

/// The DiT's table: three axes of sixteen frequencies from the checkpoint's `rope.inv_freq`.
///
/// The angle is computed and passed as `float`, so this is `cosf`, not `cos` narrowed. That is the
/// difference between this and every other table here.
pub fn dit(pos: &[f64], inv_freq: &[f32], cos: &mut [f32], sin: &mut [f32]) {
    let rows = pos.len() / 3;
    for r in 0..rows {
        for ax in 0..3 {
            for j in 0..16 {
                let ang = pos[3 * r + ax] as f32 * inv_freq[j];
                cos[r * ROPE_HALF + ax * 16 + j] = ang.cos();
                sin[r * ROPE_HALF + ax * 16 + j] = ang.sin();
            }
        }
    }
}

/// One image's rows in the text encoder's presentation.
#[derive(Clone, Copy, Debug)]
pub struct VisionSpan {
    pub start: usize,
    pub count: usize,
    pub merged_h: usize,
    pub merged_w: usize,
}

/// Qwen2-VL's mrope position ids: text runs sequentially, each image span takes its grid, and the rest
/// of the sequence is offset by how much wider the grid was than the rows it occupied.
pub fn mrope_positions(n: usize, spans: &[VisionSpan]) -> Vec<f64> {
    let mut pos = vec![0.0f64; n * 3];
    let mut ordered = spans.to_vec();
    ordered.sort_by_key(|s| s.start);
    let (mut cursor, mut offset) = (0usize, 0.0f64);
    for sp in &ordered {
        for i in cursor..sp.start {
            for ax in 0..3 {
                pos[i * 3 + ax] = i as f64 + offset;
            }
        }
        for k in 0..sp.count {
            pos[(sp.start + k) * 3] = sp.start as f64 + offset;
            pos[(sp.start + k) * 3 + 1] = sp.start as f64 + offset + (k / sp.merged_w) as f64;
            pos[(sp.start + k) * 3 + 2] = sp.start as f64 + offset + (k % sp.merged_w) as f64;
        }
        let len_max = sp.merged_h.max(sp.merged_w) as f64;
        offset += len_max - sp.count as f64;
        cursor = sp.start + sp.count;
    }
    for i in cursor..n {
        for ax in 0..3 {
            pos[i * 3 + ax] = i as f64 + offset;
        }
    }
    pos
}

/// The text encoder's table: theta 5e6 over the head dimension, each pair reading its mrope axis.
pub fn te(pos: &[f64], cos: &mut [f32], sin: &mut [f32]) {
    let n = pos.len() / 3;
    for t in 0..n {
        for j in 0..TE_ROPE_HALF {
            let ax = mrope_axis(j);
            let inv = 5_000_000.0f64.powf(-((2 * j) as f64) / HEAD_DIM as f64);
            let ang = pos[t * 3 + ax] * inv;
            cos[t * TE_ROPE_HALF + j] = ang.cos() as f32;
            sin[t * TE_ROPE_HALF + j] = ang.sin() as f32;
        }
    }
}

/// The vision tower's 2-D table over a `gh` by `gw` patch grid.
///
/// Pairs below eighteen take the row coordinate and the rest the column, theta 1e4 over dimension 36.
/// The row a patch lands on is the merge order: 2x2 blocks, row-major within the block.
pub fn vision(gh: usize, gw: usize, cos: &mut [f32], sin: &mut [f32]) {
    for hy in 0..gh {
        for wx in 0..gw {
            let pi = (((hy / 2) * (gw / 2) + wx / 2) * 2 + hy % 2) * 2 + wx % 2;
            for j in 0..36 {
                let inv = 10_000.0f64.powf(-((2 * (j % 18)) as f64) / 36.0);
                let ang = if j < 18 { hy as f64 } else { wx as f64 } * inv;
                cos[pi * 36 + j] = ang.cos() as f32;
                sin[pi * 36 + j] = ang.sin() as f32;
            }
        }
    }
}

/// The VAE decoder's table over normalised coordinates.
///
/// Each axis maps its index to `2 (i + 0.5) / size - 1`, so a tile's angles depend on the tile's own
/// extent rather than on where it sits. Rows past the grid — the register and cls rows — keep the
/// identity rotation the caller filled in.
pub fn vae(ft: usize, h: usize, w: usize, cos: &mut [f32], sin: &mut [f32]) {
    let sizes = [ft, h, w];
    for t in 0..ft {
        for y in 0..h {
            for x in 0..w {
                let r = (t * h + y) * w + x;
                let idx = [t, y, x];
                for ax in 0..3 {
                    for j in 0..8 {
                        let pos = 2.0 * ((idx[ax] as f64 + 0.5) / sizes[ax] as f64) - 1.0;
                        let inv = 100.0f64.powf(-(j as f64) * 6.0 / 48.0);
                        let ang = 2.0 * std::f64::consts::PI * pos * inv;
                        cos[r * VAE_ROPE_HALF + ax * 8 + j] = ang.cos() as f32;
                        sin[r * VAE_ROPE_HALF + ax * 8 + j] = ang.sin() as f32;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_mrope_sections_are_interleaved_then_all_temporal() {
        // [24, 20, 20] interleaved: pairs 0..60 cycle t, h, w; the remaining four are temporal
        assert_eq!(mrope_axis(0), 0);
        assert_eq!(mrope_axis(1), 1);
        assert_eq!(mrope_axis(2), 2);
        assert_eq!(mrope_axis(59), 2);
        assert_eq!(mrope_axis(60), 0);
        assert_eq!(mrope_axis(63), 0);
        assert_eq!((0..60).filter(|j| mrope_axis(*j) == 0).count(), 20);
        assert_eq!(
            (0..TE_ROPE_HALF).filter(|j| mrope_axis(*j) == 0).count(),
            24
        );
    }

    #[test]
    fn positions_run_sequentially_without_images() {
        let pos = mrope_positions(5, &[]);
        for i in 0..5 {
            assert_eq!(&pos[i * 3..i * 3 + 3], &[i as f64; 3]);
        }
    }

    #[test]
    fn an_image_span_takes_its_grid_and_offsets_what_follows() {
        // two text rows, then a 2x3 grid over six rows, then two more text rows
        let span = VisionSpan {
            start: 2,
            count: 6,
            merged_h: 2,
            merged_w: 3,
        };
        let pos = mrope_positions(10, &[span]);
        assert_eq!(&pos[0..3], &[0.0, 0.0, 0.0]);
        assert_eq!(&pos[3..6], &[1.0, 1.0, 1.0]);
        // the span's rows share a temporal position and walk the grid
        assert_eq!(&pos[6..9], &[2.0, 2.0, 2.0]); // k = 0
        assert_eq!(&pos[9..12], &[2.0, 2.0, 3.0]); // k = 1
        assert_eq!(&pos[15..18], &[2.0, 3.0, 2.0]); // k = 3, second grid row
                                                    // max(2, 3) - 6 = -3, so the tail resumes three behind its row index
        assert_eq!(&pos[24..27], &[5.0, 5.0, 5.0]); // row 8
        assert_eq!(&pos[27..30], &[6.0, 6.0, 6.0]);
    }

    #[test]
    fn the_dit_table_is_cosf_of_a_float_angle() {
        // A position and frequency whose product differs between float and double multiplication makes
        // the two visible; the table must take the float path.
        let mut inv = [0.0f32; 16];
        inv[0] = 1.0f32 / 3.0;
        let pos = [1e7f64, 0.0, 0.0];
        let (mut c, mut s) = (vec![0.0f32; ROPE_HALF], vec![0.0f32; ROPE_HALF]);
        dit(&pos, &inv, &mut c, &mut s);
        let angle_f32 = 1e7f32 * (1.0f32 / 3.0);
        assert_eq!(c[0], angle_f32.cos());
        let angle_f64 = 1e7f64 * f64::from(1.0f32 / 3.0);
        assert_ne!(c[0], angle_f64.cos() as f32, "the angle must stay in f32");
    }

    #[test]
    fn every_table_starts_at_the_identity_rotation() {
        // position zero is angle zero on every axis, whatever the base
        let (mut c, mut s) = (vec![9.0f32; ROPE_HALF], vec![9.0f32; ROPE_HALF]);
        dit(&[0.0, 0.0, 0.0], &[0.5f32; 16], &mut c, &mut s);
        assert!(c.iter().all(|v| *v == 1.0) && s.iter().all(|v| *v == 0.0));

        let (mut c, mut s) = (vec![9.0f32; TE_ROPE_HALF], vec![9.0f32; TE_ROPE_HALF]);
        te(&[0.0, 0.0, 0.0], &mut c, &mut s);
        assert!(c.iter().all(|v| *v == 1.0) && s.iter().all(|v| *v == 0.0));

        let (mut c, mut s) = (vec![9.0f32; 36], vec![9.0f32; 36]);
        vision(1, 1, &mut c, &mut s);
        assert!(c.iter().all(|v| *v == 1.0) && s.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn the_vae_table_is_symmetric_about_the_centre_of_each_axis() {
        // coordinates are 2 (i + 0.5) / size - 1, so index i and size - 1 - i are negatives
        let (ft, h, w) = (4usize, 2usize, 2usize);
        let n = ft * h * w;
        let (mut c, mut s) = (
            vec![0.0f32; n * VAE_ROPE_HALF],
            vec![0.0f32; n * VAE_ROPE_HALF],
        );
        vae(ft, h, w, &mut c, &mut s);
        let row = |t: usize, y: usize, x: usize| ((t * h + y) * w + x) * VAE_ROPE_HALF;
        for j in 0..8 {
            let (a, b) = (row(0, 0, 0) + j, row(3, 0, 0) + j);
            assert!((c[a] - c[b]).abs() < 1e-6, "cos is even");
            assert!((s[a] + s[b]).abs() < 1e-6, "sin is odd");
        }
        // the first frequency of each axis is inv = 1, so the angle is 2 pi times the coordinate
        let want = (2.0 * std::f64::consts::PI * (2.0 * (0.5 / 4.0) - 1.0)).cos() as f32;
        assert_eq!(c[row(0, 0, 0)], want);
    }

    #[test]
    fn the_vision_grid_merges_in_two_by_two_blocks() {
        // a 2x4 grid: patch (0,2) is the first of the second block, so it lands on row 4
        let (gh, gw) = (2usize, 4usize);
        let (mut c, mut s) = (vec![0.0f32; gh * gw * 36], vec![0.0f32; gh * gw * 36]);
        vision(gh, gw, &mut c, &mut s);
        // pair 18 is the first column frequency with inv = 1, so it reads wx directly
        let at = |r: usize| c[r * 36 + 18];
        assert_eq!(at(0), 0.0f64.cos() as f32); // (hy, wx) = (0, 0)
        assert_eq!(at(1), 1.0f64.cos() as f32); // (0, 1)
        assert_eq!(at(4), 2.0f64.cos() as f32); // (0, 2) starts the next block
    }
}
