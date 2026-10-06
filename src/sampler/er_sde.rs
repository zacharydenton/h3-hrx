//! VP ER-SDE-3 on the flow-matching sigma grid (alpha = 1 - sigma).
//! Implements the solver equations in <https://arxiv.org/abs/2309.06169>.
pub(crate) struct ErSde {
    previous: Vec<f32>,
    derivative: Vec<f32>,
    noise: Vec<f32>,
    rng: crate::noise::Noise,
}
impl ErSde {
    pub fn new(len: usize, seed: u64) -> Self {
        Self {
            previous: vec![0.; len],
            derivative: vec![0.; len],
            noise: vec![0.; len],
            rng: crate::noise::Noise::new(seed),
        }
    }
    pub fn advance(&mut self, x: &mut [f32], denoised: &[f32], sigmas: &[f32], i: usize) {
        if sigmas[i + 1] == 0.0 {
            x.copy_from_slice(denoised);
            return;
        }
        self.rng.fill(&mut self.noise);
        self.advance_with_noise(x, denoised, sigmas, i);
    }
    fn advance_with_noise(&mut self, x: &mut [f32], denoised: &[f32], sigmas: &[f32], i: usize) {
        let lambda = |s: f32| s as f64 / (1.0 - s as f64);
        let f = |l: f64| l * (l.powf(0.3).exp() + 10.0);
        let (ls, lt) = (lambda(sigmas[i]), lambda(sigmas[i + 1]));
        let at = 1.0 - sigmas[i + 1] as f64;
        let ra = at / (1.0 - sigmas[i] as f64);
        let r = f(lt) / f(ls);
        let dt = lt - ls;
        let dl = -dt / 200.0;
        let (mut integral, mut moment) = (0.0, 0.0);
        if i > 0 {
            for j in 0..200 {
                let pos = lt + j as f64 * dl;
                integral += dl / f(pos);
                moment += (pos - ls) * dl / f(pos);
            }
        }
        let dweight = at * (dt + integral * f(lt));
        let uweight = at * (dt * dt / 2.0 + moment * f(lt));
        let noise_weight = at * (lt * lt - ls * ls * r * r).max(0.0).sqrt();
        for (j, value) in x.iter_mut().enumerate() {
            let mut next = ra * r * *value as f64 + at * (1.0 - r) * denoised[j] as f64;
            if i > 0 {
                let derivative =
                    (denoised[j] as f64 - self.previous[j] as f64) / (ls - lambda(sigmas[i - 1]));
                next += dweight * derivative;
                if i > 1 {
                    next += uweight * (derivative - self.derivative[j] as f64)
                        / ((ls - lambda(sigmas[i - 2])) / 2.0);
                }
                self.derivative[j] = derivative as f32;
            }
            *value = (next + noise_weight * self.noise[j] as f64) as f32;
        }
        self.previous.copy_from_slice(denoised);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn terminal_step_returns_clean_estimate() {
        let mut s = ErSde::new(2, 0);
        let mut x = vec![10., -4.];
        s.advance(&mut x, &[0.5, 2.], &[0.4, 0.0], 0);
        assert_eq!(x, [0.5, 2.]);
    }
    #[test]
    fn fixed_seed_is_repeatable_and_all_stages_stay_finite() {
        let sigmas = [0.932, 0.813, 0.618, 0.347, 0.0];
        let run = || {
            let mut s = ErSde::new(4, 42);
            let mut x = vec![1., -3., 0., 40.];
            for i in 0..4 {
                s.advance(&mut x, &[0.2, -0.5, 1., 20.], &sigmas, i);
                assert!(x.iter().all(|v| v.is_finite()));
            }
            x
        };
        assert_eq!(run(), run());
    }
}

#[cfg(test)]
mod upstream {
    #[test]
    fn all_three_stages_match_upstream_with_explicit_noise() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/upscale/er_sde.json")).unwrap();
        let floats = |v: &serde_json::Value| {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect::<Vec<_>>()
        };
        let sigmas = floats(&data["sigmas"]);
        let mut x = floats(&data["trace"][0]);
        let mut solver = super::ErSde::new(x.len(), 7);
        for i in 0..4 {
            let expected = floats(&data["trace"][i]);
            for (a, b) in x.iter().zip(expected) {
                assert!((a - b).abs() < 3e-5, "step {i}: {a} != {b}");
            }
            let denoised = x
                .iter()
                .zip([0.1, -0.4, 0.7, 0.2])
                .map(|(v, b)| v * 0.25 + b + sigmas[i] * 0.1)
                .collect::<Vec<_>>();
            if i == 3 {
                solver.advance(&mut x, &denoised, &sigmas, i);
            } else {
                solver.noise = floats(&data["noise"][i]);
                solver.advance_with_noise(&mut x, &denoised, &sigmas, i);
            }
        }
        for (a, b) in x.iter().zip(floats(&data["output"])) {
            assert!((a - b).abs() < 3e-5, "{a} != {b}");
        }
    }
}
