//! The seeded noise the sampler starts from.
//!
//! This is a deliberate break with the C, which had its own splitmix64 and a Box-Muller pair it threw
//! half of away. Neither was torch-compatible, so nothing was ever gated on the exact stream — but the
//! *ordering* of the three draw sites is load-bearing for a run being reproducible, so it is kept:
//! video noise in tensor order, audio noise from the same already-advanced stream, and a fresh
//! per-segment generator for reference condition augmentation.
//!
//! Every existing seed names a different clip than it did under the C.
use crate::model::*;
use crate::sampler::patch_index;
use rand::{Rng as _, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

/// A seeded stream of standard normals.
pub struct Noise(ChaCha12Rng);

impl Noise {
    pub fn new(seed: u64) -> Self {
        Self(ChaCha12Rng::seed_from_u64(seed))
    }

    pub fn normal(&mut self) -> f32 {
        self.0.sample(StandardNormal)
    }

    /// Fills a buffer in place, in its own order.
    pub fn fill(&mut self, out: &mut [f32]) {
        for x in out {
            *x = self.normal();
        }
    }
}

/// A reference's latent mixed with fresh noise, patched into rows.
///
/// ComfyUI's visual condition augmentation: `0.999 z + 0.001 n`, with a generator seeded per segment
/// from the run's seed so a reference's augmentation does not depend on how many latents preceded it.
pub fn augment_ref(lat: &[f32], rows: &mut [f32], t_len: usize, h: usize, w: usize, seed: u64) {
    let mut rng = Noise::new(seed);
    for c in 0..LATENT_CH {
        for t in 0..t_len {
            for y in 0..h {
                for x in 0..w {
                    let (row, col) = patch_index(c, t, y, x, h, w);
                    let z = lat[((c * t_len + t) * h + y) * w + x];
                    rows[row * VIDEO_PATCH + col] =
                        VISUAL_COND_AUG * z + (1.0 - VISUAL_COND_AUG) * rng.normal();
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_names_one_stream() {
        let a: Vec<f32> = (0..64).map(|_| Noise::new(7).normal()).collect();
        assert!(a.iter().all(|v| *v == a[0]), "a fresh generator restarts");
        let mut one = Noise::new(7);
        let mut two = Noise::new(7);
        let (mut x, mut y) = ([0.0f32; 4096], [0.0f32; 4096]);
        one.fill(&mut x);
        two.fill(&mut y);
        assert_eq!(x, y);
        let mut other = Noise::new(8);
        let mut z = [0.0f32; 4096];
        other.fill(&mut z);
        assert_ne!(x, z);
    }

    #[test]
    fn the_stream_is_standard_normal() {
        let mut rng = Noise::new(1);
        let mut v = vec![0.0f32; 1 << 16];
        rng.fill(&mut v);
        let n = v.len() as f64;
        let mean = v.iter().map(|x| f64::from(*x)).sum::<f64>() / n;
        let var = v
            .iter()
            .map(|x| (f64::from(*x) - mean).powi(2))
            .sum::<f64>()
            / n;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "variance {var}");
        // and it is not the truncated half of a Box-Muller pair: both tails are populated
        assert!(v.iter().filter(|x| **x > 3.0).count() > 50);
        assert!(v.iter().filter(|x| **x < -3.0).count() > 50);
    }

    #[test]
    fn augmentation_is_mostly_the_latent_and_per_segment_reproducible() {
        let (t_len, h, w) = (1usize, 4usize, 4usize);
        let lat: Vec<f32> = (0..LATENT_CH * t_len * h * w).map(|i| i as f32).collect();
        let rows_n = t_len * (h / 2) * (w / 2);
        let mut a = vec![0.0f32; rows_n * VIDEO_PATCH];
        let mut b = vec![0.0f32; rows_n * VIDEO_PATCH];
        augment_ref(&lat, &mut a, t_len, h, w, 42);
        augment_ref(&lat, &mut b, t_len, h, w, 42);
        assert_eq!(a, b, "the same seed augments the same way");
        // the noise contributes a thousandth, so each row is within a few sigma of the latent it patched
        let mut plain = vec![0.0f32; rows_n * VIDEO_PATCH];
        crate::sampler::tensor_to_rows(&lat, &mut plain, t_len, h, w);
        for (got, want) in a.iter().zip(&plain) {
            assert!(
                (got - VISUAL_COND_AUG * want).abs() < 0.02,
                "{got} vs {want}"
            );
        }
    }
}
