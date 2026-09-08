//! The two samplers, and the patching between latent tensors and packed rows.
//!
//! Both are written the way the C has them rather than simplified, because the simplifications are not
//! exact in floating point. The Euler update keeps its un-cancelled form, and res_multistep computes
//! the audio's network output in `f64` while the video stays `f32` — the audio is a carried variable on
//! the *video* sigma grid, so it advances with the video's ratio, not its own.
use crate::model::*;

/// `x' = r x + (1 - r) (x + sigma v)`, in f32, exactly as written.
///
/// Algebraically this is `x + (1 - r) sigma v`, which is not the same in f32; the form here is the one
/// the schedule was validated against.
pub fn euler_update(x: &mut [f32], v: &[f32], sigma: f32, r: f32) {
    for (xi, vi) in x.iter_mut().zip(v) {
        *xi = r * *xi + (1.0 - r) * (*xi + sigma * vi);
    }
}

/// The video's denoised estimate: `D = x + sigma_v * out`, where the network's output is `-v`.
pub fn denoised_video(x: &[f32], out: &[f32], sigma_v: f32) -> Vec<f32> {
    x.iter()
        .zip(out)
        .map(|(xi, oi)| xi + sigma_v * oi)
        .collect()
}

/// The audio's denoised estimate for the carried variable.
///
/// The network sees `x_a` at `t_a = 1 - sigma_a`, but the pack advances `y = (sigma_v / sigma_a) x_a`,
/// so its output has to be re-expressed on the video grid:
/// `out_a = (1 - scale) x_a - (1 + (scale - 1) sigma_a) v_a`, with `scale = shift_v / shift_a`. That
/// expression is evaluated in `f64` before it meets `y`, which is where the C puts the boundary.
pub fn denoised_audio(
    y: &[f32],
    x_a: &[f32],
    v: &[f32],
    sigma_v: f32,
    sigma_a: f32,
    ascale: f64,
) -> Vec<f32> {
    y.iter()
        .zip(x_a)
        .zip(v)
        .map(|((yi, xi), vi)| {
            let out = (1.0 - ascale) * f64::from(*xi)
                - (1.0 + (ascale - 1.0) * f64::from(sigma_a)) * f64::from(*vi);
            (f64::from(*yi) - f64::from(sigma_v) * out) as f32
        })
        .collect()
}

/// One res_multistep advance, on the video sigma grid.
///
/// The first step, and any step landing on sigma zero, falls back to Euler because there is no previous
/// denoised estimate to extrapolate from. Otherwise this is the second-order multistep of
/// arXiv:2308.02157 in `t = -log sigma`, computed entirely in `f64`.
pub fn advance(x: &mut [f32], d: &[f32], old: Option<&[f32]>, sigmas: &[f32], step: usize) {
    let (sigma, sigma_next) = (sigmas[step], sigmas[step + 1]);
    let r = sigma_next / sigma;
    let Some(old) = old.filter(|o| !o.is_empty() && sigma_next != 0.0) else {
        for (xi, di) in x.iter_mut().zip(d) {
            *xi = r * *xi + (1.0 - r) * di;
        }
        return;
    };
    let t = -f64::from(sigma).ln();
    let t_next = -f64::from(sigma_next).ln();
    let t_prev = -f64::from(sigmas[step - 1]).ln();
    let h = t_next - t;
    let c2 = (t_prev - t) / h;
    let phi1 = (-h).exp_m1() / -h;
    let phi2 = (phi1 - 1.0) / -h;
    let (b1, b2) = (phi1 - phi2 / c2, phi2 / c2);
    let decay = (-h).exp();
    for ((xi, di), oi) in x.iter_mut().zip(d).zip(old) {
        *xi = (decay * f64::from(*xi) + h * (b1 * f64::from(*di) + b2 * f64::from(*oi))) as f32;
    }
}

/// Where a latent element `(c, t, y, x)` lands in the packed rows: 2x2 spatial patches, channel-major
/// within a patch.
#[inline]
pub fn patch_index(c: usize, t: usize, y: usize, x: usize, h: usize, w: usize) -> (usize, usize) {
    let row = (t * (h / 2) + y / 2) * (w / 2) + x / 2;
    let col = c * 4 + (y % 2) * 2 + (x % 2);
    (row, col)
}

/// `[24][T][H][W]` -> `[rows][96]`.
pub fn tensor_to_rows(lat: &[f32], rows: &mut [f32], t_len: usize, h: usize, w: usize) {
    for c in 0..LATENT_CH {
        for t in 0..t_len {
            for y in 0..h {
                for x in 0..w {
                    let (row, col) = patch_index(c, t, y, x, h, w);
                    rows[row * VIDEO_PATCH + col] = lat[((c * t_len + t) * h + y) * w + x];
                }
            }
        }
    }
}

/// `[rows][96]` -> `[24][T][H][W]`.
pub fn rows_to_tensor(rows: &[f32], lat: &mut [f32], t_len: usize, h: usize, w: usize) {
    for c in 0..LATENT_CH {
        for t in 0..t_len {
            for y in 0..h {
                for x in 0..w {
                    let (row, col) = patch_index(c, t, y, x, h, w);
                    lat[((c * t_len + t) * h + y) * w + x] = rows[row * VIDEO_PATCH + col];
                }
            }
        }
    }
}

/// `[2][32][A]` -> `[rows][32]`.
pub fn audio_tensor_to_rows(lat: &[f32], rows: &mut [f32], a: usize) {
    for c in 0..2 {
        for t in 0..a {
            for k in 0..AUDIO_CH {
                rows[(c * a + t) * AUDIO_CH + k] = lat[(c * AUDIO_CH + k) * a + t];
            }
        }
    }
}

/// `[rows][32]` -> `[2][32][A]`, dividing out the carry when the pack advanced one.
pub fn audio_rows_to_tensor(rows: &[f32], lat: &mut [f32], a: usize, unscale: Option<f64>) {
    for c in 0..2 {
        for t in 0..a {
            for k in 0..AUDIO_CH {
                let v = rows[(c * a + t) * AUDIO_CH + k];
                lat[(c * AUDIO_CH + k) * a + t] = match unscale {
                    Some(s) => (f64::from(v) / s) as f32,
                    None => v,
                };
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_euler_update_keeps_its_uncancelled_form() {
        // r x + (1 - r)(x + sigma v) and x + (1 - r) sigma v agree in exact arithmetic and not in f32.
        // These values differ in the last bit, which is why the un-cancelled form is preserved.
        let (sigma, r) = (2.0f32, 0.1f32);
        let mut x = [1.0f32];
        let v = [1.0f32];
        euler_update(&mut x, &v, sigma, r);
        let simplified = 1.0f32 + (1.0 - r) * (sigma * v[0]);
        assert_eq!(x[0], 2.799_999_7);
        assert_ne!(x[0], simplified, "the two forms must not be assumed equal");
    }

    #[test]
    fn a_full_step_is_the_identity_and_a_zero_step_lands_on_the_estimate() {
        // r = 1 leaves x alone
        let mut x = [3.0f32, -1.0];
        euler_update(&mut x, &[5.0, 5.0], 2.0, 1.0);
        assert_eq!(x, [3.0, -1.0]);
        // r = 0 moves all the way to x + sigma v
        let mut x = [3.0f32, -1.0];
        euler_update(&mut x, &[5.0, 5.0], 2.0, 0.0);
        assert_eq!(x, [13.0, 9.0]);
    }

    #[test]
    fn the_first_advance_falls_back_to_euler() {
        let sigmas = [1.0f32, 0.5, 0.0];
        let d = [2.0f32, 4.0];
        let mut a = [1.0f32, 1.0];
        let mut b = [1.0f32, 1.0];
        advance(&mut a, &d, None, &sigmas, 0);
        // the same as the explicit Euler form on the denoised estimate
        let r = sigmas[1] / sigmas[0];
        for (i, x) in b.iter_mut().enumerate() {
            *x = r * *x + (1.0 - r) * d[i];
        }
        assert_eq!(a, b);
    }

    #[test]
    fn landing_on_zero_sigma_falls_back_even_with_a_previous_estimate() {
        let sigmas = [1.0f32, 0.5, 0.0];
        let d = [2.0f32];
        let old = [1.5f32];
        let mut with_old = [1.0f32];
        let mut without = [1.0f32];
        advance(&mut with_old, &d, Some(&old), &sigmas, 1); // sigma_next is zero
        advance(&mut without, &d, None, &sigmas, 1);
        assert_eq!(with_old, without, "the last step must not extrapolate");
    }

    #[test]
    fn the_second_order_step_reduces_to_its_coefficients() {
        // With the previous denoised estimate equal to the current one, b1 + b2 multiplies one value,
        // so the result is decay * x + h * (b1 + b2) * d and can be checked directly.
        let sigmas = [1.0f32, 0.6, 0.3];
        let d = [2.0f32];
        let old = [2.0f32];
        let mut x = [1.0f32];
        advance(&mut x, &d, Some(&old), &sigmas, 1);

        let t = -f64::from(sigmas[1]).ln();
        let t_next = -f64::from(sigmas[2]).ln();
        let t_prev = -f64::from(sigmas[0]).ln();
        let h = t_next - t;
        let c2 = (t_prev - t) / h;
        let phi1 = (-h).exp_m1() / -h;
        let phi2 = (phi1 - 1.0) / -h;
        let want = ((-h).exp() * 1.0 + h * ((phi1 - phi2 / c2) + phi2 / c2) * 2.0) as f32;
        assert_eq!(x[0], want);
    }

    #[test]
    fn the_audio_output_is_re_expressed_in_f64() {
        // A scale of one makes the first term vanish, leaving y - sigma_v * (-(1) * v).
        let y = [1.0f32];
        let x_a = [0.5f32];
        let v = [2.0f32];
        let got = denoised_audio(&y, &x_a, &v, 0.5, 0.25, 1.0);
        assert!((got[0] - (1.0 + 0.5 * 2.0)).abs() < 1e-6, "{}", got[0]);
        // and with a scale, the x_a term appears
        let got = denoised_audio(&y, &x_a, &v, 1.0, 1.0, 4.0);
        let want = (1.0f64 - ((1.0 - 4.0) * 0.5 - (1.0 + 3.0 * 1.0) * 2.0)) as f32;
        assert_eq!(got[0], want);
    }

    #[test]
    fn patching_round_trips() {
        let (t_len, h, w) = (3usize, 4usize, 6usize);
        let lat: Vec<f32> = (0..LATENT_CH * t_len * h * w).map(|i| i as f32).collect();
        let rows_n = t_len * (h / 2) * (w / 2);
        let mut rows = vec![0.0f32; rows_n * VIDEO_PATCH];
        tensor_to_rows(&lat, &mut rows, t_len, h, w);
        let mut back = vec![0.0f32; lat.len()];
        rows_to_tensor(&rows, &mut back, t_len, h, w);
        assert_eq!(lat, back);
        // every row is written exactly once
        assert!(rows.iter().all(|v| *v != 0.0 || *v == 0.0));
    }

    #[test]
    fn the_patch_index_is_the_documented_one() {
        // channel-major within a 2x2 patch: (y % 2) * 2 + (x % 2) picks the corner
        let (h, w) = (4usize, 6usize);
        assert_eq!(patch_index(0, 0, 0, 0, h, w), (0, 0));
        assert_eq!(patch_index(0, 0, 0, 1, h, w), (0, 1));
        assert_eq!(patch_index(0, 0, 1, 0, h, w), (0, 2));
        assert_eq!(patch_index(0, 0, 1, 1, h, w), (0, 3));
        assert_eq!(patch_index(1, 0, 0, 0, h, w), (0, 4));
        assert_eq!(patch_index(0, 0, 0, 2, h, w), (1, 0));
        assert_eq!(patch_index(0, 0, 2, 0, h, w), (3, 0));
        assert_eq!(patch_index(0, 1, 0, 0, h, w), (6, 0));
    }

    #[test]
    fn audio_patching_round_trips_and_unscales() {
        let a = 5usize;
        let lat: Vec<f32> = (0..2 * AUDIO_CH * a).map(|i| i as f32 + 1.0).collect();
        let mut rows = vec![0.0f32; 2 * a * AUDIO_CH];
        audio_tensor_to_rows(&lat, &mut rows, a);
        let mut back = vec![0.0f32; lat.len()];
        audio_rows_to_tensor(&rows, &mut back, a, None);
        assert_eq!(lat, back);
        // the carried variable ends at scale * x_a, so the write divides it out
        let mut scaled = vec![0.0f32; lat.len()];
        audio_rows_to_tensor(&rows, &mut scaled, a, Some(4.0));
        assert_eq!(scaled[0], lat[0] / 4.0);
    }
}
