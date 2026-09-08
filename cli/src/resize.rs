//! PIL's bilinear resample (`ImagingResample`), ported so reference and keyframe conditioning keeps the
//! pixels the Python driver produced: a triangle filter whose support grows with the downscale factor,
//! weights normalised per output pixel, separable horizontal then vertical, accumulated in `f64`.
//!
//! This is deliberately not an image-crate filter. The reference path is compared against ComfyUI, and a
//! different resampler moves those numbers. The port was checked against the C implementation it replaces
//! (`host/h3_cli.cpp` at 4ff5639) on identical pseudo-random input: bit-identical f32 output at
//! 1920x1080 -> 864x480, 640x480 -> 864x480, 100x100 -> 32x32, 37x53 -> 64x96, 864x480 unchanged and
//! 3000x2000 -> 288x192.

/// The contributing input range `[lo, hi)` for one output pixel, and its normalised weights.
fn weights(in_len: i32, out_len: i32, x: i32) -> (i32, i32, Vec<f64>) {
    let scale = in_len as f64 / out_len as f64;
    let ss = if scale >= 1.0 { scale } else { 1.0 }; // upscaling keeps unit support
    let support = ss; // bilinear: support is 1.0 filter scales
    let center = (x as f64 + 0.5) * scale;
    let lo = 0.max((center - support + 0.5) as i32);
    let hi = in_len.min((center + support + 0.5) as i32);
    let mut w = vec![0.0f64; (hi - lo).max(0) as usize];
    let mut total = 0.0;
    for i in lo..hi {
        let t = ((i as f64 - center + 0.5) / ss).abs();
        let v = if t < 1.0 { 1.0 - t } else { 0.0 };
        w[(i - lo) as usize] = v;
        total += v;
    }
    if total > 0.0 {
        for v in &mut w {
            *v /= total;
        }
    }
    (lo, hi, w)
}

/// RGB8 `[sh][sw][3]` -> f32 `[dh][dw][3]` in `[0, 1]`.
pub fn pil_bilinear(rgb: &[u8], sw: i32, sh: i32, dw: i32, dh: i32) -> Vec<f32> {
    let (swz, shz, dwz, dhz) = (sw as usize, sh as usize, dw as usize, dh as usize);
    let mut horiz = vec![0.0f32; shz * dwz * 3]; // [sh][dw][3]
    let mut out = vec![0.0f32; dhz * dwz * 3]; // [dh][dw][3]

    for x in 0..dw {
        let (lo, hi, w) = weights(sw, dw, x);
        for y in 0..shz {
            for c in 0..3 {
                let mut acc = 0.0f64;
                for i in lo..hi {
                    acc += w[(i - lo) as usize] * rgb[(y * swz + i as usize) * 3 + c] as f64;
                }
                horiz[(y * dwz + x as usize) * 3 + c] = (acc / 255.0) as f32;
            }
        }
    }
    for y in 0..dh {
        let (lo, hi, w) = weights(sh, dh, y);
        for x in 0..dwz {
            for c in 0..3 {
                let mut acc = 0.0f64;
                for i in lo..hi {
                    acc += w[(i - lo) as usize] * horiz[(i as usize * dwz + x) * 3 + c] as f64;
                }
                out[(y as usize * dwz + x) * 3 + c] = (acc as f32).clamp(0.0, 1.0);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_and_stay_in_range() {
        for (input, output) in [(100, 33), (33, 100), (64, 64), (1920, 864), (7, 3)] {
            for x in 0..output {
                let (lo, hi, w) = weights(input, output, x);
                assert!(
                    lo >= 0 && hi <= input && lo < hi,
                    "{input}->{output} at {x}: {lo}..{hi}"
                );
                assert_eq!(w.len(), (hi - lo) as usize);
                assert!(
                    (w.iter().sum::<f64>() - 1.0).abs() < 1e-12,
                    "{input}->{output} at {x}"
                );
            }
        }
    }

    #[test]
    fn identity_size_returns_the_same_pixels() {
        let rgb: Vec<u8> = (0..4u32 * 3 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let out = pil_bilinear(&rgb, 3, 4, 3, 4);
        for (i, v) in out.iter().enumerate() {
            assert!((v - rgb[i] as f32 / 255.0).abs() < 1e-6, "pixel {i}");
        }
    }

    #[test]
    fn a_flat_image_stays_flat_through_any_scale() {
        let rgb = vec![200u8; 16 * 9 * 3];
        for (dw, dh) in [(4, 3), (32, 18), (16, 9)] {
            for v in pil_bilinear(&rgb, 16, 9, dw, dh) {
                assert!((v - 200.0 / 255.0).abs() < 1e-6, "{dw}x{dh}");
            }
        }
    }

    #[test]
    fn output_is_clamped_to_the_unit_range() {
        let rgb: Vec<u8> = (0..8u32 * 8 * 3)
            .map(|i| if i % 2 == 0 { 0 } else { 255 })
            .collect();
        for v in pil_bilinear(&rgb, 8, 8, 5, 5) {
            assert!((0.0..=1.0).contains(&v));
        }
    }
}
