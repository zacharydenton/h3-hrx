//! First-block residual reuse with separate conditioning, audio and video gates.
//! The first and last two evaluations are full; two consecutive skips are forbidden.

/// Thresholds for independently accumulated relative L1 changes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CacheThresholds {
    pub conditioning: f32,
    pub audio: f32,
    pub video: f32,
}

/// Computation reuse remains opt-in. Observation runs every transformer block.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum CachePolicy {
    #[default]
    Off,
    Observe,
    Conservative(CacheThresholds),
}

impl CachePolicy {
    /// Reject invalid or ambiguous policies before a session opens any model.
    pub fn validate(self, legacy_threshold: f32) -> Result<(), &'static str> {
        if !legacy_threshold.is_finite() {
            return Err("cache threshold must be finite");
        }
        if self != Self::Off && legacy_threshold > 0.0 {
            return Err("explicit cache policy conflicts with cache_threshold");
        }
        if let Self::Conservative(t) = self {
            if [t.conditioning, t.audio, t.video]
                .iter()
                .any(|v| !v.is_finite() || *v <= 0.0)
            {
                return Err("each cache threshold must be finite and positive");
            }
        }
        Ok(())
    }
}

pub fn groups(n: usize) -> usize {
    n.div_ceil(2048)
}

pub struct StepCache {
    thresholds: [f64; 3],
    accumulated: [f64; 3],
    observe: bool,
    have: bool,
    skipped: u32,
    last_skipped: bool,
}

impl StepCache {
    pub fn configured(policy: CachePolicy, legacy: f32) -> Result<Option<Self>, &'static str> {
        policy.validate(legacy)?;
        let (thresholds, observe) = match policy {
            CachePolicy::Off if legacy <= 0.0 => return Ok(None),
            CachePolicy::Off => ([f64::from(legacy); 3], false),
            CachePolicy::Observe => ([0.0; 3], true),
            CachePolicy::Conservative(t) => (
                [
                    f64::from(t.conditioning),
                    f64::from(t.audio),
                    f64::from(t.video),
                ],
                false,
            ),
        };
        Ok(Some(Self {
            thresholds,
            accumulated: [0.0; 3],
            observe,
            have: false,
            skipped: 0,
            last_skipped: false,
        }))
    }

    /// Metrics are [sum absolute difference, sum previous magnitude], in packed modality order.
    pub fn consider(&mut self, step: usize, total: usize, metrics: [[f64; 2]; 3]) -> Change {
        let relative = if step == 0 {
            [0.0; 3]
        } else {
            metrics.map(|[d, m]| {
                if d.is_finite() && m.is_finite() && d >= 0.0 && m >= 0.0 {
                    d / m.max(1e-30)
                } else {
                    f64::INFINITY
                }
            })
        };
        for (acc, value) in self.accumulated.iter_mut().zip(relative) {
            *acc += value;
        }
        let interior = step >= 2 && step < total.saturating_sub(2);
        let skip = interior
            && self.have
            && !self.observe
            && !self.last_skipped
            && self
                .accumulated
                .iter()
                .zip(self.thresholds)
                .all(|(a, t)| *a < t);
        if skip {
            self.skipped += 1;
        }
        self.last_skipped = skip;
        Change {
            relative,
            accumulated: self.accumulated,
            skip,
        }
    }

    pub fn recorded(&mut self) {
        self.have = true;
        self.accumulated = [0.0; 3];
        self.last_skipped = false;
    }

    pub fn skipped(&self) -> u32 {
        self.skipped
    }
}

pub struct Change {
    pub relative: [f64; 3],
    pub accumulated: [f64; 3],
    pub skip: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cache() -> StepCache {
        StepCache::configured(CachePolicy::Off, 0.25)
            .unwrap()
            .unwrap()
    }
    fn small() -> [[f64; 2]; 3] {
        [[0.01, 1.0]; 3]
    }

    #[test]
    fn caller_recording_step_zero_does_not_bypass_warmup() {
        let mut c = cache();
        assert!(!c.consider(0, 20, small()).skip);
        c.recorded();
        assert!(!c.consider(1, 20, small()).skip);
        c.recorded();
        assert!(c.consider(2, 20, small()).skip);
        assert!(!c.consider(3, 20, small()).skip);
        c.recorded();
        assert!(c.consider(4, 20, small()).skip);
        c.recorded();
        assert!(!c.consider(18, 20, small()).skip);
        c.recorded();
        assert!(!c.consider(19, 20, small()).skip);
    }

    #[test]
    fn audio_change_cannot_hide_behind_video() {
        let mut c = cache();
        c.recorded();
        assert!(
            !c.consider(2, 20, [[0.001, 1.0], [0.3, 1.0], [0.001, 1.0]])
                .skip
        );
    }

    #[test]
    fn short_runs_and_observation_never_skip() {
        for total in 1..=4 {
            let mut c = cache();
            for step in 0..total {
                assert!(!c.consider(step, total, small()).skip);
                c.recorded();
            }
        }
        let mut c = StepCache::configured(CachePolicy::Observe, 0.0)
            .unwrap()
            .unwrap();
        for step in 0..20 {
            assert!(!c.consider(step, 20, small()).skip);
            c.recorded();
        }
        assert_eq!(c.skipped(), 0);
    }

    #[test]
    fn disabled_ambiguous_and_nonfinite_policies() {
        assert!(StepCache::configured(CachePolicy::Off, 0.0)
            .unwrap()
            .is_none());
        assert!(CachePolicy::Observe.validate(0.1).is_err());
        assert!(CachePolicy::Off.validate(f32::NAN).is_err());
        let mut c = cache();
        c.recorded();
        assert!(!c.consider(2, 20, [[f64::NAN, 1.0]; 3]).skip);
        c.recorded();
        assert!(c.consider(4, 20, [[0.0, 0.0]; 3]).skip);
    }
}
