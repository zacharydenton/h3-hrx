//! The first-block step cache (TeaCache / FBCache style).
//!
//! Block 0 runs every step. If its output has barely moved since the last full evaluation, blocks 1..49
//! are skipped and their recorded residual is added instead. The accounting below is the whole of the
//! host's part: the device measures the change, this decides what to do with it.
//!
//! The ordering is easy to get wrong, so it is spelled out: the first step never skips and never
//! accumulates, because there is nothing to compare against; the accumulator folds in every subsequent
//! step's relative change and is reset only by a full evaluation; and a skip requires a recorded
//! residual, so the step after the first is always full.

/// How many 2048-element groups the change metric reduces over.
pub fn groups(n: usize) -> usize {
    n.div_ceil(2048)
}

pub struct StepCache {
    threshold: f32,
    acc: f64,
    have: bool,
    skipped: u32,
}

impl StepCache {
    /// `None` when the threshold is off, which is what selects the plain every-block path.
    pub fn new(threshold: f32) -> Option<Self> {
        (threshold > 0.0).then_some(Self {
            threshold,
            acc: 0.0,
            have: false,
            skipped: 0,
        })
    }

    /// Folds one step's block-0 change into the accumulator and says whether to reuse the residual.
    ///
    /// `d` is the summed absolute difference and `m` the summed magnitude, both from the device's
    /// reduction; the floor on `m` is what keeps an all-zero state from dividing by zero.
    pub fn consider(&mut self, step: usize, d: f64, m: f64) -> Change {
        if step == 0 {
            return Change {
                relative: 0.0,
                accumulated: self.acc,
                skip: false,
            };
        }
        let relative = d / m.max(1e-30);
        self.acc += relative;
        let skip = self.acc < f64::from(self.threshold) && self.have;
        if skip {
            self.skipped += 1;
        }
        Change {
            relative,
            accumulated: self.acc,
            skip,
        }
    }

    /// A full evaluation ran, so its residual is now the cached one.
    pub fn recorded(&mut self) {
        self.have = true;
        self.acc = 0.0;
    }

    pub fn skipped(&self) -> u32 {
        self.skipped
    }
}

/// What one step's measurement came to, in the form the trace prints.
pub struct Change {
    pub relative: f64,
    pub accumulated: f64,
    pub skip: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_threshold_of_zero_or_less_turns_the_cache_off() {
        assert!(StepCache::new(0.0).is_none());
        assert!(StepCache::new(-1.0).is_none());
        assert!(StepCache::new(f32::MIN_POSITIVE).is_some());
    }

    #[test]
    fn the_first_step_neither_skips_nor_accumulates() {
        let mut c = StepCache::new(10.0).unwrap();
        let ch = c.consider(0, 5.0, 1.0);
        assert!(!ch.skip);
        assert_eq!(
            ch.accumulated, 0.0,
            "step zero has nothing to compare against"
        );
    }

    #[test]
    fn the_step_after_the_first_is_full_because_no_residual_is_recorded() {
        let mut c = StepCache::new(10.0).unwrap();
        c.consider(0, 0.0, 1.0);
        // a tiny change, well under the threshold, but there is no cached residual yet
        assert!(!c.consider(1, 1e-6, 1.0).skip);
    }

    #[test]
    fn a_recorded_residual_lets_small_changes_skip_until_they_add_up() {
        let mut c = StepCache::new(0.25).unwrap();
        c.consider(0, 0.0, 1.0);
        c.consider(1, 0.1, 1.0);
        c.recorded();
        assert!(c.consider(2, 0.1, 1.0).skip, "0.10 is under 0.25");
        assert!(c.consider(3, 0.1, 1.0).skip, "0.20 is still under");
        let ch = c.consider(4, 0.1, 1.0);
        assert!(!ch.skip, "0.30 is over");
        assert!((ch.accumulated - 0.3).abs() < 1e-12);
        assert_eq!(c.skipped(), 2);
        // and the full evaluation resets the accumulator
        c.recorded();
        assert!(c.consider(5, 0.1, 1.0).skip);
    }

    #[test]
    fn an_all_zero_state_does_not_divide_by_zero() {
        let mut c = StepCache::new(1.0).unwrap();
        c.consider(0, 0.0, 0.0);
        let ch = c.consider(1, 0.0, 0.0);
        assert_eq!(ch.relative, 0.0);
        assert!(ch.accumulated.is_finite());
    }

    #[test]
    fn the_metric_reduces_over_whole_groups() {
        assert_eq!(groups(0), 0);
        assert_eq!(groups(1), 1);
        assert_eq!(groups(2048), 1);
        assert_eq!(groups(2049), 2);
        // the DiT at a typical sequence length
        assert_eq!(groups(30000 * 5376), 78750);
    }
}
