//! H3-World's deterministic action interface and directed routing.
//!
//! Reference: Danzer1xxxxChan/H3-World at f0c7be2acbbde8256b2473f6e7b088f58272acae.
use crate::{error::invalid, DenoiseParams, Result, Sampler, Tokenizer};
use std::ops::Range;
mod rollout;
pub use rollout::{validate_rollout_parameters, WorldSegment, WorldState};

pub const UPSTREAM_REVISION: &str = "f0c7be2acbbde8256b2473f6e7b088f58272acae";
pub const ADAPTER_REVISION: &str = "cb1a1fe209415bd3c744a74b1058bb8bfd507268";
pub const ADAPTER_FILE: &str = "step-10000.safetensors";

/// Keyboard state: low-to-high bits are W,A,S,D,I,J,K,L,F.
/// I tilts down and K tilts up, matching the training recordings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Action(pub u16);
impl Action {
    pub fn from_keys(keys: &[impl AsRef<str>]) -> Result<Self> {
        let mut bits = 0;
        for key in keys {
            let i = ["W", "A", "S", "D", "I", "J", "K", "L", "F"]
                .iter()
                .position(|k| *k == key.as_ref())
                .ok_or_else(|| {
                    crate::Error::Invalid(format!("unknown world key {}", key.as_ref()))
                })?;
            bits |= 1 << i;
        }
        Ok(Self(bits))
    }
    pub fn sentence(self) -> String {
        let mut b = self.0;
        for (a, c) in [(0, 2), (1, 3), (4, 6), (5, 7)] {
            if b & (1 << a) != 0 && b & (1 << c) != 0 {
                b &= !((1 << a) | (1 << c));
            }
        }
        let on = |i| b & (1u16 << i) != 0u16;
        let motion: Vec<_> = [
            (0, "walks forward"),
            (2, "walks backward"),
            (1, "strafes left"),
            (3, "strafes right"),
        ]
        .into_iter()
        .filter_map(|(i, s)| on(i).then_some(s))
        .collect();
        let moving = !motion.is_empty();
        let motion = if moving {
            motion.join(" and ")
        } else {
            "stands still".into()
        };
        let mut camera = Vec::new();
        for (i, side) in [(5, "left"), (7, "right")] {
            if on(i) {
                camera.push(format!(
                    "pans {side} {}",
                    if on(8) { "sharply" } else { "slowly" }
                ));
            }
        }
        for (i, s) in [(4, "tilts down"), (6, "tilts up")] {
            if on(i) {
                camera.push(s.into());
            }
        }
        if camera.is_empty() {
            camera.push(
                if moving {
                    "follows him"
                } else {
                    "holds steady"
                }
                .into(),
            );
        }
        format!("the man {motion}, camera {}", camera.join(" and "))
    }
}

/// One key set per output RGB frame; intervals use the native 1,4,4,4,4 cadence.
#[derive(Clone, Debug)]
pub struct ActionSchedule(pub Vec<Action>);
impl ActionSchedule {
    pub fn preset(name: &str, frames: usize) -> Result<Self> {
        let keys: &[&str] = match name {
            "still" => &[],
            "forward" => &["W"],
            "back" => &["S"],
            "strafe-left" => &["A"],
            "strafe-right" => &["D"],
            "tilt-up" => &["K"],
            "tilt-down" => &["I"],
            "pan-left" => &["J"],
            "pan-right" => &["L"],
            "pan-left-fast" => &["J", "F"],
            "pan-right-fast" => &["L", "F"],
            _ => return invalid(format!("unknown action preset {name}")),
        };
        Ok(Self(vec![Action::from_keys(keys)?; frames]))
    }
    /// JSON is an array of frame key arrays, e.g. [["W"],["W","L"],...].
    pub fn from_json(json: &str) -> Result<Self> {
        let rows: Vec<Vec<String>> =
            serde_json::from_str(json).map_err(|e| crate::Error::Invalid(e.to_string()))?;
        Ok(Self(
            rows.iter()
                .map(|r| Action::from_keys(r))
                .collect::<Result<_>>()?,
        ))
    }
    pub fn sentences(&self, frames: usize) -> Result<Vec<String>> {
        if frames < 5
            || !(frames - 5).is_multiple_of(17)
            || self.0.len() != frames
            || self.0.iter().any(|a| a.0 > 511)
        {
            return invalid(
                "world schedule must have one valid key set per frame, with frames = 17k+5",
            );
        }
        let mut cursor = 0;
        let mut out = Vec::new();
        while cursor < frames {
            let width = [1, 4, 4, 4, 4][out.len() % 5];
            let end = (cursor + width).min(frames);
            let bits = self.0[cursor..end].iter().fold(0, |b, a| b | a.0);
            out.push(Action(bits).sentence());
            cursor = end;
        }
        Ok(out)
    }
}

/// Prepared actions, encoded independently from the static scene presentation.
#[derive(Clone, Debug)]
pub struct WorldRequest {
    pub(crate) tokens: Vec<Vec<i32>>,
    pub(crate) frames: usize,
    sentences: Vec<String>,
}
impl WorldRequest {
    pub fn new(tokenizer: &Tokenizer, schedule: &ActionSchedule, frames: usize) -> Result<Self> {
        let sentences = schedule.sentences(frames)?;
        let mut tokens = Vec::with_capacity(sentences.len());
        let mut count = 0;
        for sentence in &sentences {
            let ids = tokenizer.encode(sentence)?;
            count += ids.len();
            if count > 4096 {
                return invalid("actions exceed the H3 text token budget");
            }
            tokens.push(ids);
        }
        Ok(Self {
            tokens,
            frames,
            sentences,
        })
    }
    pub fn token_count(&self) -> usize {
        self.tokens.iter().map(Vec::len).sum()
    }

    pub fn sentences(&self) -> &[String] {
        &self.sentences
    }

    pub fn parameters() -> DenoiseParams {
        DenoiseParams {
            width: 832,
            height: 480,
            frames: 124,
            steps: 51,
            sampler: Sampler::Euler,
            ..Default::default()
        }
    }
    pub(crate) fn spans(&self, head: usize) -> Vec<Range<usize>> {
        let mut cursor = head;
        self.tokens
            .iter()
            .map(|t| {
                let start = cursor;
                cursor += t.len();
                start..cursor
            })
            .collect()
    }
}

/// Row codes: zero = static/conditioning/audio, positive = action k+1,
/// negative = video -(k+1). Padding is excluded by the kernel's token count.
#[cfg(test)]
pub(crate) fn allowed(q: i32, k: i32) -> bool {
    if k > 0 {
        q == k || q == -k
    } else {
        !(q > 0 && k < 0 && q != -k)
    }
}

pub(crate) fn configure(
    layout: &mut crate::layout::Layout,
    spans: &[Range<usize>],
) -> Result<Vec<i32>> {
    if spans.len() != layout.latent_t || spans.is_empty() {
        return invalid("action count differs from video latent count");
    }
    let video = layout.text_len + layout.ref_rows + layout.audio_rows;
    let frame_rows = layout.video_rows / layout.latent_t;
    let first = layout.pos[video * 3];
    let last = layout.pos[(video + (layout.latent_t - 1) * frame_rows) * 3] - first;
    let origin = layout.text_len as f64 - last - 1.0;
    if origin < spans[0].start as f64 {
        return invalid("action temporal positions overlap the static presentation");
    }
    let mut codes = vec![0; layout.seq_len];
    for (i, span) in spans.iter().enumerate() {
        let time = origin + layout.pos[(video + i * frame_rows) * 3] - first;
        for row in span.clone() {
            layout.pos[row * 3] = time;
            codes[row] = (i + 1) as i32;
        }
        codes[video + i * frame_rows..video + (i + 1) * frame_rows].fill(-((i + 1) as i32));
    }
    Ok(codes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mapping_and_pooling() {
        assert_eq!(
            Action::from_keys(&["W", "S", "J", "F"]).unwrap().sentence(),
            "the man stands still, camera pans left sharply"
        );
        assert_eq!(
            Action::from_keys(&["S", "A", "L"]).unwrap().sentence(),
            "the man walks backward and strafes left, camera pans right slowly"
        );
        let mut s = ActionSchedule::preset("still", 124).unwrap();
        s.0[1] = Action::from_keys(&["W"]).unwrap();
        s.0[4] = Action::from_keys(&["S"]).unwrap();
        let rows = s.sentences(124).unwrap();
        assert_eq!(rows.len(), 37);
        assert_eq!(rows[0], rows[1]);
        assert!(s.sentences(123).is_err());
    }
    #[test]
    fn directed_routing() {
        for q in -3..=3 {
            for k in -3..=3 {
                let expected = if k > 0 {
                    q == k || q == -k
                } else if q > 0 && k < 0 {
                    q == -k
                } else {
                    true
                };
                assert_eq!(allowed(q, k), expected);
            }
        }
        assert!(allowed(1, 0));
        assert!(!allowed(0, 1));
    }
    #[test]
    fn mirrored_positions_and_routing_cover_every_row() {
        let kfs = [crate::layout::Keyframe {
            frame_index: 0,
            audio_t: 0,
            has_audio: false,
        }];
        let mut layout = crate::layout::Layout::new(1000, 37, 4, 4, 207, &[], &kfs).unwrap();
        let spans: Vec<_> = (0..37).map(|i| 260 + i * 20..280 + i * 20).collect();
        let original = layout.pos.clone();
        let codes = configure(&mut layout, &spans).unwrap();
        let video = layout.text_len + layout.ref_rows + layout.audio_rows;
        let delta = layout.pos[video * 3] - layout.pos[spans[0].start * 3];
        for (i, span) in spans.iter().enumerate() {
            assert!(
                (layout.pos[(video + i * 4) * 3] - layout.pos[span.start * 3] - delta).abs()
                    < 1e-10
            );
            assert!(codes[span.clone()].iter().all(|c| *c == (i + 1) as i32));
        }
        assert_eq!(&layout.pos[1000 * 3..], &original[1000 * 3..]);
        assert_eq!(layout.pos[spans[36].start * 3], 999.0);
        assert!(codes[..260].iter().all(|c| *c == 0));
        assert!(codes[1000..video].iter().all(|c| *c == 0));
    }
}
