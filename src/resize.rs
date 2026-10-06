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
    let mut out = vec![0.0f32; dhz * dwz * 3]; // [dh][dw][3]
    if swz == 0 || shz == 0 || dwz == 0 || dhz == 0 {
        return out;
    }
    if (sw, sh) == (dw, dh) {
        for (dest, source) in out.iter_mut().zip(&rgb[..dhz * dwz * 3]) {
            *dest = (*source as f64 / 255.0) as f32;
        }
        return out;
    }
    let mut horiz = vec![0.0f32; shz * dwz * 3]; // [sh][dw][3]

    let horizontal_weights: Vec<_> = (0..dw).map(|x| weights(sw, dw, x)).collect();
    // Visit whole rows so input and intermediate pixels stay local. Accumulate
    // RGB together, preserving each channel's tap order and f32 pass boundary.
    for (source, dest) in rgb[..shz * swz * 3]
        .chunks_exact(swz * 3)
        .zip(horiz.chunks_exact_mut(dwz * 3))
    {
        for ((lo, hi, w), pixel) in horizontal_weights
            .iter()
            .zip(dest.as_chunks_mut::<3>().0.iter_mut())
        {
            let mut acc = [0.0f64; 3];
            for (weight, input) in w.iter().zip(
                source[*lo as usize * 3..*hi as usize * 3]
                    .as_chunks::<3>()
                    .0
                    .iter(),
            ) {
                for c in 0..3 {
                    acc[c] += weight * input[c] as f64;
                }
            }
            for c in 0..3 {
                pixel[c] = (acc[c] / 255.0) as f32;
            }
        }
    }
    let mut acc = vec![0.0f64; dwz * 3];
    for y in 0..dh {
        let (lo, hi, w) = weights(sh, dh, y);
        acc.fill(0.0);
        // The tap remains the accumulation order for each component, while
        // contiguous row updates let the compiler vectorize across components.
        for (weight, source) in w
            .iter()
            .zip(horiz[lo as usize * dwz * 3..hi as usize * dwz * 3].chunks_exact(dwz * 3))
        {
            for (sum, value) in acc.iter_mut().zip(source) {
                *sum += weight * *value as f64;
            }
        }
        for (dest, sum) in out[y as usize * dwz * 3..][..dwz * 3].iter_mut().zip(&acc) {
            *dest = (*sum as f32).clamp(0.0, 1.0);
        }
    }
    out
}

/// The largest box with this aspect that fits a canvas' pixel count, on the
/// 32-pixel grid a vision span divides by.
///
/// References are scaled to at most the canvas' area rather than to the canvas:
/// a reference is conditioning and not a frame, and stretching one to the
/// output's aspect would change what it shows.
pub fn fit(width: i32, height: i32, canvas_width: i32, canvas_height: i32) -> (i32, i32) {
    let block = crate::presentation::VISION_BLOCK;
    let scale = (f64::from(canvas_width) * f64::from(canvas_height)
        / (f64::from(width) * f64::from(height)))
    .sqrt()
    .min(1.0);

    let round = |value: i32| {
        block.max(((f64::from(value) * scale / f64::from(block)).round() as i32) * block)
    };

    (round(width), round(height))
}

/// H3-World cover resize with Lanczos-3 followed by a centered crop.
/// Intermediate RGB8 rounding follows PIL's two-pass image resizing.
pub fn world_first_frame(rgb: &[u8], sw: i32, sh: i32, dw: i32, dh: i32) -> Vec<f32> {
    if dw == 0 || dh == 0 {
        return Vec::new();
    }
    let scale = (dw as f64 / sw as f64).max(dh as f64 / sh as f64);
    let rw = (sw as f64 * scale).round_ties_even() as usize;
    let rh = (sh as f64 * scale).round_ties_even() as usize;
    let coeff = |input: usize, output: usize, x: usize| {
        let scale = input as f64 / output as f64;
        let filter = scale.max(1.0);
        let center = (x as f64 + 0.5) * scale;
        let lo = ((center - 3.0 * filter + 0.5) as isize).max(0) as usize;
        let hi = ((center + 3.0 * filter + 0.5) as usize).min(input);
        let mut weights: Vec<f64> = (lo..hi)
            .map(|i| {
                let t = ((i as f64 - center + 0.5) / filter).abs();
                if t == 0.0 {
                    1.0
                } else if t >= 3.0 {
                    0.0
                } else {
                    let t = t * std::f64::consts::PI;
                    t.sin() / t * (t / 3.0).sin() / (t / 3.0)
                }
            })
            .collect();
        let total: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= total;
        }
        (lo, weights)
    };
    let left = (rw - dw as usize) / 2;
    let top = (rh - dh as usize) / 2;
    let columns: Vec<_> = (0..dw as usize)
        .map(|x| coeff(sw as usize, rw, x + left))
        .collect();
    let rows: Vec<_> = (0..dh as usize)
        .map(|y| coeff(sh as usize, rh, y + top))
        .collect();
    // Keep every source row touched by the vertical filter, including its
    // Lanczos halo, but omit resized columns and rows discarded by the crop.
    let first = rows.iter().map(|(lo, _)| *lo).min().unwrap();
    let end = rows.iter().map(|(lo, w)| lo + w.len()).max().unwrap();
    let stride = dw as usize * 3;
    let mut horizontal = vec![0u8; (end - first) * stride];
    for (x, (lo, weights)) in columns.iter().enumerate() {
        for y in first..end {
            let mut pixel = [0.0f64; 3];
            for (i, w) in weights.iter().enumerate() {
                let at = (y * sw as usize + lo + i) * 3;
                for c in 0..3 {
                    pixel[c] += w * rgb[at + c] as f64;
                }
            }
            for (c, v) in pixel.into_iter().enumerate() {
                horizontal[(y - first) * stride + x * 3 + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    let mut out = vec![0.0; dw as usize * dh as usize * 3];
    for (y, (lo, weights)) in rows.iter().enumerate() {
        for x in 0..dw as usize {
            let mut pixel = [0.0f64; 3];
            for (i, w) in weights.iter().enumerate() {
                let at = (lo + i - first) * stride + x * 3;
                for c in 0..3 {
                    pixel[c] += w * horizontal[at + c] as f64;
                }
            }
            for (c, v) in pixel.into_iter().enumerate() {
                out[(y * dw as usize + x) * 3 + c] = v.round().clamp(0.0, 255.0) as f32 / 255.0;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_wise_resize_preserves_scalar_output_bits() {
        for (sw, sh, dw, dh) in [
            (1920, 1080, 864, 480),
            (1920, 1080, 1344, 768),
            (640, 480, 864, 768),
            (864, 480, 864, 480),
            (3000, 2000, 288, 192),
            (37, 53, 64, 96),
            (100, 100, 32, 32),
            (1, 1, 7, 3),
            (17, 9, 1, 1),
            (0, 1, 2, 3),
            (1, 0, 2, 3),
            (2, 3, 0, 1),
            (2, 3, 1, 0),
        ] {
            let rgb: Vec<_> = (0..sw * sh * 3)
                .map(|i| ((i * 37 + i / 7) % 256) as u8)
                .collect();
            let mut horizontal = vec![0.0f32; (sh * dw * 3) as usize];
            // Original scalar two-pass order, including rounding between passes.
            for x in 0..dw {
                let (lo, hi, coeff) = weights(sw, dw, x);
                for y in 0..sh {
                    for c in 0..3 {
                        let mut sum = 0.0f64;
                        for i in lo..hi {
                            sum += coeff[(i - lo) as usize]
                                * rgb[((y * sw + i) * 3 + c) as usize] as f64;
                        }
                        horizontal[((y * dw + x) * 3 + c) as usize] = (sum / 255.0) as f32;
                    }
                }
            }
            let actual = pil_bilinear(&rgb, sw, sh, dw, dh);
            for y in 0..dh {
                let (lo, hi, coeff) = weights(sh, dh, y);
                for x in 0..dw {
                    for c in 0..3 {
                        let mut sum = 0.0f64;
                        for i in lo..hi {
                            sum += coeff[(i - lo) as usize]
                                * horizontal[((i * dw + x) * 3 + c) as usize] as f64;
                        }
                        let expected = (sum as f32).clamp(0.0, 1.0);
                        assert_eq!(
                            actual[((y * dw + x) * 3 + c) as usize].to_bits(),
                            expected.to_bits(),
                            "{sw}x{sh}->{dw}x{dh}, pixel {x},{y},{c}"
                        );
                    }
                }
            }
        }
    }

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
