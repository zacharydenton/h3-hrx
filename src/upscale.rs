//! Learned spatial upscaling of normalized H3 video latents.
mod network;
use crate::{error::invalid, Result, Shape};
pub use network::{Upscaler, CHECKPOINT};

#[derive(Clone, Copy, Debug)]
pub enum UpscaleTarget {
    Scale(f64),
    Dimensions {
        width: i32,
        height: i32,
    },
    /// SeedHunter uses 1024² pixels per megapixel.
    Megapixels(f64),
}

#[derive(Clone, Copy, Debug)]
pub struct UpscaleSettings {
    pub target: UpscaleTarget,
    pub temporal_chunking: bool,
}
impl Default for UpscaleSettings {
    fn default() -> Self {
        Self {
            target: UpscaleTarget::Megapixels(1.2),
            temporal_chunking: true,
        }
    }
}
impl UpscaleSettings {
    pub fn output_shape(&self, input: &Shape) -> Result<Shape> {
        self.resolve(input).map(|v| v.0)
    }
    pub(crate) fn resolve(&self, input: &Shape) -> Result<(Shape, f32)> {
        let w = input
            .lat_w
            .checked_mul(16)
            .ok_or_else(|| crate::Error::Invalid("invalid upscale input width".into()))?;
        let h = input
            .lat_h
            .checked_mul(16)
            .ok_or_else(|| crate::Error::Invalid("invalid upscale input height".into()))?;
        if crate::shape_for(h, w, input.frames).as_ref() != Some(input) {
            return invalid("invalid upscaler input shape");
        }
        let (ow, oh, scale) = match self.target {
            UpscaleTarget::Scale(s) => (w as f64 * s, h as f64 * s, s),
            UpscaleTarget::Dimensions { width, height } => (
                width as f64,
                height as f64,
                (width as f64 / w as f64 + height as f64 / h as f64) / 2.0,
            ),
            UpscaleTarget::Megapixels(mp) => {
                let s = (mp * 1024.0 * 1024.0 / (w as f64 * h as f64)).sqrt();
                (w as f64 * s, h as f64 * s, s)
            }
        };
        if !scale.is_finite()
            || scale < 1.0
            || !ow.is_finite()
            || !oh.is_finite()
            || ow < w as f64
            || oh < h as f64
        {
            return invalid("upscale target must be finite and cannot shrink either dimension");
        }
        let width = (ow / 32.0).round_ties_even() * 32.0;
        let height = (oh / 32.0).round_ties_even() * 32.0;
        if width > crate::layout::MAX_SIDE as f64 || height > crate::layout::MAX_SIDE as f64 {
            return invalid("upscale target exceeds supported canvas");
        }
        let shape = crate::shape_for(height as i32, width as i32, input.frames)
            .ok_or_else(|| crate::Error::Invalid("invalid upscale target".into()))?;
        Ok((shape, scale as f32))
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RefinementSettings {
    /// Number of model evaluations, unlike the legacy generation schedule's point count.
    pub steps: usize,
    pub denoise: f64,
    /// None reuses the first pass seed.
    pub seed: Option<u64>,
}
impl Default for RefinementSettings {
    fn default() -> Self {
        Self {
            steps: 4,
            denoise: 0.4,
            seed: None,
        }
    }
}
impl RefinementSettings {
    pub fn validate(&self) -> Result<()> {
        self.schedule().map(|_| ())
    }
    pub(crate) fn schedule(&self) -> Result<crate::layout::Schedule> {
        if !(1..=1000).contains(&self.steps)
            || !self.denoise.is_finite()
            || !(0.0..=1.0).contains(&self.denoise)
        {
            return invalid("refinement requires 1..1000 steps and denoise in [0,1]");
        }
        if self.denoise == 0.0 {
            return Ok(crate::layout::Schedule {
                sigmas: vec![0.0],
                timesteps: vec![],
            });
        }
        let full = (self.steps as f64 / self.denoise).floor();
        if full > 1_000_000.0 {
            return invalid("refinement schedule exceeds one million points");
        }
        let full = full as usize;
        let linear = full / 2;
        let mut sigmas = Vec::with_capacity(self.steps + 1);
        for i in full - self.steps..full {
            let value = if full == 1 {
                1.0
            } else if i < linear {
                1.0 - i as f64 * 0.025 / linear as f64
            } else {
                let l = linear as f64;
                let q = (full - linear) as f64;
                let d = l - 0.025 * full as f64;
                let a = d / (l * q * q);
                let b = 0.025 / l - 2.0 * d / (q * q);
                1.0 - (a * (i * i) as f64 + b * i as f64 + a * l * l)
            };
            sigmas.push(value as f32);
        }
        sigmas.push(0.0);
        let timesteps = sigmas[..sigmas.len() - 1].iter().map(|s| 1.0 - s).collect();
        Ok(crate::layout::Schedule { sigmas, timesteps })
    }
}

pub struct UpscaledLatents {
    pub shape: Shape,
    pub latents: crate::Latents,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn targets_align_preserve_time_and_reject_invalid_sizes() {
        let s = crate::shape_for(64, 96, 22).unwrap();
        let settings = UpscaleSettings {
            target: UpscaleTarget::Scale(2.0),
            ..Default::default()
        };
        let out = settings.output_shape(&s).unwrap();
        assert_eq!(out.size(), (192, 128));
        assert_eq!((out.latent_t, out.audio_t), (s.latent_t, s.audio_t));
        for v in [0.5, f64::NAN, f64::INFINITY, 1e30] {
            assert!(UpscaleSettings {
                target: UpscaleTarget::Scale(v),
                ..settings
            }
            .output_shape(&s)
            .is_err());
        }
    }
    #[test]
    fn seedhunter_schedule_is_the_tail_of_ten_steps() {
        let s = RefinementSettings::default().schedule().unwrap();
        assert_eq!(s.sigmas.len(), 5);
        for (got, want) in s.sigmas.iter().zip([0.932, 0.813, 0.618, 0.347, 0.0]) {
            assert!((got - want).abs() < 1e-6, "{got} != {want}");
        }
        assert!(RefinementSettings {
            denoise: 0.0,
            ..Default::default()
        }
        .schedule()
        .unwrap()
        .timesteps
        .is_empty());
    }
}

#[cfg(test)]
mod validation {
    use super::*;
    #[test]
    fn malformed_shapes_and_refinement_settings_fail_without_overflow() {
        let mut shape = crate::shape_for(32, 32, 5).unwrap();
        shape.lat_w = i32::MAX;
        assert!(UpscaleSettings::default().output_shape(&shape).is_err());
        for denoise in [f64::NAN, f64::INFINITY, -1.0, 1.1, 1e-20] {
            assert!(RefinementSettings {
                denoise,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
        for steps in [0, 1001, usize::MAX] {
            assert!(RefinementSettings {
                steps,
                ..Default::default()
            }
            .validate()
            .is_err());
        }
        for steps in [1, 2, 4, 1000] {
            let s = RefinementSettings {
                steps,
                denoise: 1.0,
                seed: None,
            }
            .schedule()
            .unwrap();
            assert_eq!(s.sigmas.len(), steps + 1);
            assert_eq!(*s.sigmas.last().unwrap(), 0.0);
            assert!(s.sigmas.windows(2).all(|s| s[0] > s[1]));
        }
    }
}
