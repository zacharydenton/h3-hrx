//! The AdaLN tables the blocks modulate with, built on the host once per denoising step.
//!
//! A timestep becomes eight numbers by interpolating the checkpoint's curve, those eight drive each
//! layer's projection, and the projection's chunks are permuted into the row order the kernels index
//! by class. The accumulation is f32 in a fixed order and the permutation is semantic, so both are
//! carried over exactly rather than tidied.
use crate::model::*;

/// A timestep's eight-vector, interpolated from the checkpoint's 1025-point curve.
///
/// Clamped to `[0, 1]` and scaled by 1024, with the index floored and capped at 1023 so the second
/// sample is always in range; the interpolation itself is in `f64` and narrowed once.
pub fn temb(curve: &[f32], t: f32) -> [f32; 8] {
    let pos = f64::from(t).clamp(0.0, 1.0) * 1024.0;
    let i0 = (pos.floor() as usize).min(1023);
    let f = pos - i0 as f64;
    let mut out = [0.0f32; 8];
    for (j, o) in out.iter_mut().enumerate() {
        let a = f64::from(curve[i0 * 8 + j]);
        let b = f64::from(curve[(i0 + 1) * 8 + j]);
        *o = (a * (1.0 - f) + b * f) as f32;
    }
    out
}

/// One layer's projection: `bias + weight * te`, accumulated in f32 in the weights' own order.
fn project(w: &[f32], b: &[f32], te: &[f32; 8], out: &mut [f32]) {
    for (r, o) in out.iter_mut().enumerate() {
        let mut acc = b[r];
        for j in 0..8 {
            acc += w[r * 8 + j] * te[j];
        }
        *o = acc;
    }
}

/// The per-layer tables in the runtime's layout.
///
/// Rows `[0, 2C)` are (scale_msa, shift_msa) per class, `[2C, 3C)` gate_msa, `[3C, 5C)`
/// (scale_mlp, shift_mlp), `[5C, 6C)` gate_mlp, where a class is `timestep index * 3 + modality`. The
/// projection produces its chunks in a different order — shift, scale, gate for each of msa and mlp —
/// so the copy below swaps shift and scale. That swap is the layout, not a tidy-up.
pub fn mods_table(
    adaln_w: &[Vec<f32>],
    adaln_b: &[Vec<f32>],
    tv: &[f32; 8],
    ta: &[f32; 8],
    tcv: &[f32; 8],
    tca: &[f32; 8],
) -> Vec<f32> {
    let mut table = vec![0.0f32; BLOCKS * MODS_ROWS * HID];
    let mut proj = vec![0.0f32; MODALITIES * 6 * HID];
    for i in 0..BLOCKS {
        for m in 0..4 {
            let te = match m {
                0 => tv,
                1 => ta,
                2 => tcv,
                _ => tca,
            };
            project(&adaln_w[i], &adaln_b[i], te, &mut proj);
            for d in 0..MODALITIES {
                let cls = m * MODALITIES + d;
                let base = i * MODS_ROWS * HID;
                let ch = |j: usize| &proj[(d * 6 + j) * HID..(d * 6 + j) * HID + HID];
                let put = |table: &mut Vec<f32>, row: usize, src: &[f32]| {
                    table[base + row * HID..base + row * HID + HID].copy_from_slice(src);
                };
                put(&mut table, 2 * cls, ch(1)); // scale_msa
                put(&mut table, 2 * cls + 1, ch(0)); // shift_msa
                put(&mut table, 2 * CLASSES + cls, ch(2)); // gate_msa
                put(&mut table, 3 * CLASSES + 2 * cls, ch(4)); // scale_mlp
                put(&mut table, 3 * CLASSES + 2 * cls + 1, ch(3)); // shift_mlp
                put(&mut table, 5 * CLASSES + cls, ch(5)); // gate_mlp
            }
        }
    }
    table
}

/// The final layer's (scale, shift) per timestep class: row `2c` scale, `2c + 1` shift. The projection
/// hands back (shift | scale), so the halves land the other way round.
pub fn final_table(final_w: &[f32], final_b: &[f32], tv: &[f32; 8], ta: &[f32; 8]) -> Vec<f32> {
    let mut ft = vec![0.0f32; 4 * HID];
    for m in 0..2 {
        let te = if m == 0 { tv } else { ta };
        for r in 0..2 * HID {
            let mut acc = final_b[r];
            for j in 0..8 {
                acc += final_w[r * 8 + j] * te[j];
            }
            if r < HID {
                ft[(2 * m + 1) * HID + r] = acc;
            } else {
                ft[(2 * m) * HID + r - HID] = acc;
            }
        }
    }
    ft
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i % 97) as f32 * 0.01 - 0.5).collect()
    }

    #[test]
    fn temb_interpolates_and_clamps() {
        // a curve whose j-th component at sample i is i + j/10, so the answer is readable
        let curve: Vec<f32> = (0..1025 * 8)
            .map(|k| (k / 8) as f32 + (k % 8) as f32 / 10.0)
            .collect();
        // t = 0 is sample 0 exactly
        assert_eq!(temb(&curve, 0.0)[0], 0.0);
        assert!((temb(&curve, 0.0)[3] - 0.3).abs() < 1e-6);
        // t = 1 clamps to the last whole sample, with no second-sample overrun
        assert!((temb(&curve, 1.0)[0] - 1024.0).abs() < 1e-3);
        // halfway between samples 0 and 1
        let half = temb(&curve, 0.5 / 1024.0);
        assert!((half[0] - 0.5).abs() < 1e-5, "{}", half[0]);
        // out of range clamps rather than indexing past the curve
        assert_eq!(temb(&curve, -1.0)[0], temb(&curve, 0.0)[0]);
        assert_eq!(temb(&curve, 5.0)[0], temb(&curve, 1.0)[0]);
    }

    #[test]
    fn the_final_table_swaps_the_projection_halves() {
        // a weight of zero leaves the bias, so the permutation is all that is visible
        let w = vec![0.0f32; 2 * HID * 8];
        let mut b = vec![0.0f32; 2 * HID];
        for (r, v) in b.iter_mut().enumerate() {
            *v = if r < HID { 1.0 } else { 2.0 };
        }
        let te = [0.0f32; 8];
        let ft = final_table(&w, &b, &te, &te);
        // the projection's first half is the shift, which lands in row 2m + 1
        assert_eq!(ft[HID], 1.0); // row 1: shift for class 0
        assert_eq!(ft[0], 2.0); // row 0: scale for class 0
        assert_eq!(ft[3 * HID], 1.0); // row 3: shift for class 1
        assert_eq!(ft[2 * HID], 2.0); // row 2: scale for class 1
    }

    #[test]
    fn the_mods_table_places_every_class() {
        // one layer, weights zero so each chunk is its bias and its destination is identifiable
        let w = vec![vec![0.0f32; MODALITIES * 6 * HID * 8]; BLOCKS];
        let mut b = vec![vec![0.0f32; MODALITIES * 6 * HID]; BLOCKS];
        for d in 0..MODALITIES {
            for j in 0..6 {
                let value = (d * 6 + j) as f32;
                let at = (d * 6 + j) * HID;
                b[0][at..at + HID].fill(value);
            }
        }
        let te = [0.0f32; 8];
        let t = mods_table(&w, &b, &te, &te, &te, &te);
        assert_eq!(t.len(), BLOCKS * MODS_ROWS * HID);
        // modality 0, class 0: chunk 1 is the scale, chunk 0 the shift
        assert_eq!(t[0], 1.0);
        assert_eq!(t[HID], 0.0);
        assert_eq!(t[2 * CLASSES * HID], 2.0); // gate_msa
        assert_eq!(t[3 * CLASSES * HID], 4.0); // scale_mlp
        assert_eq!(t[(3 * CLASSES + 1) * HID], 3.0); // shift_mlp
        assert_eq!(t[5 * CLASSES * HID], 5.0); // gate_mlp
                                               // modality tag 1 of timestep 0 is class 1, and reads the second chunk group
        assert_eq!(t[2 * HID], 7.0); // scale_msa of class 1 is chunk (1, 1)
    }

    #[test]
    fn the_projection_accumulates_in_the_weights_order() {
        // Summing eight terms in a different order would change the last bits; this pins the order by
        // constructing values whose f32 sum is order-dependent.
        let te = [1e8f32, 1.0, -1e8, 1.0, 0.0, 0.0, 0.0, 0.0];
        let w: Vec<f32> = te.iter().map(|_| 1.0).collect();
        let b = vec![0.0f32];
        let mut out = [0.0f32; 1];
        project(&w, &b, &te, &mut out);
        // left to right: ((1e8 + 1) - 1e8) + 1 == 1, not 2
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn a_ramp_projects_deterministically() {
        let w = vec![ramp(MODALITIES * 6 * HID * 8); 1];
        let b = vec![ramp(MODALITIES * 6 * HID); 1];
        let te = [0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8];
        let mut a = vec![0.0f32; MODALITIES * 6 * HID];
        let mut b2 = vec![0.0f32; MODALITIES * 6 * HID];
        project(&w[0], &b[0], &te, &mut a);
        project(&w[0], &b[0], &te, &mut b2);
        assert_eq!(a, b2);
    }
}
