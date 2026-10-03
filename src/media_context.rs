//! A shared, ordered description of what the prompt processor and H3 are shown.
use crate::{Error, Presentation, Result, Shape, Tokenizer};
use std::sync::Arc;

/// RGB floats in [0,1], with dimensions aligned to H3's 32-pixel vision blocks.
#[derive(Clone, Debug)]
pub struct Frame {
    pub pixels: Arc<[f32]>,
    pub width: usize,
    pub height: usize,
}
impl Frame {
    /// ComfyUI Qwen3-VL resize: 32-pixel grid, 3136..12845056 pixels,
    /// bilinear sampling with align_corners=false and no antialiasing.
    pub fn for_vision(&self) -> Result<Self> {
        self.validate()?;
        let (h, w) = (self.height, self.width);
        let area = h * w;
        let (dh, dw) = if area < 3136 {
            let beta = (3136.0 / area as f64).sqrt();
            (
                ((h as f64 * beta / 32.0).ceil() as usize) * 32,
                ((w as f64 * beta / 32.0).ceil() as usize) * 32,
            )
        } else if area > 12845056 {
            let beta = (area as f64 / 12845056.0).sqrt();
            (
                ((h as f64 / beta / 32.0).floor() as usize).max(1) * 32,
                ((w as f64 / beta / 32.0).floor() as usize).max(1) * 32,
            )
        } else {
            return Ok(self.clone());
        };
        let mut pixels = vec![0.0; dh * dw * 3];
        for y in 0..dh {
            let fy = ((y as f32 + 0.5) * (h as f32 / dh as f32) - 0.5).max(0.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(h - 1);
            let ay = fy - y0 as f32;
            for x in 0..dw {
                let fx = ((x as f32 + 0.5) * (w as f32 / dw as f32) - 0.5).max(0.0);
                let x0 = fx.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let ax = fx - x0 as f32;
                for c in 0..3 {
                    let read = |y, x| self.pixels[(y * w + x) * 3 + c];
                    pixels[(y * dw + x) * 3 + c] = (read(y0, x0) * (1.0 - ax) + read(y0, x1) * ax)
                        * (1.0 - ay)
                        + (read(y1, x0) * (1.0 - ax) + read(y1, x1) * ax) * ay;
                }
            }
        }
        Ok(Self {
            pixels: pixels.into(),
            width: dw,
            height: dh,
        })
    }

    pub fn validate(&self) -> Result<()> {
        let need = self
            .width
            .checked_mul(self.height)
            .and_then(|n| n.checked_mul(3));
        if self.width == 0
            || self.height == 0
            || !self.width.is_multiple_of(32)
            || !self.height.is_multiple_of(32)
            || need != Some(self.pixels.len())
            || self
                .pixels
                .iter()
                .any(|x| !x.is_finite() || !(0.0..=1.0).contains(x))
        {
            return Err(Error::Invalid("invalid presentation frame".into()));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub enum Media {
    Picture(Frame),
    /// Frames sampled at 2 fps; timestamps remain explicit for synthetic RefMod timelines.
    Video(Vec<(f64, Frame)>),
    /// Planar stereo at 32 kHz. H3 receives a label; only the LLM receives the waveform.
    Audio(Arc<[f32]>),
}

#[derive(Clone, Debug)]
pub struct MediaEntry {
    pub media: Media,
    /// `first_frame`, `last_frame`, or `reference`.
    pub role: String,
    /// Descriptive provenance, never an instruction or implicit speaker assignment.
    pub metadata: serde_json::Value,
}

/// A real temporal patch. Stills use the same frame in both slots.
#[derive(Clone, Debug)]
pub struct VisualBlock {
    pub video: bool,
    pub first: Frame,
    pub second: Frame,
}

#[derive(Clone, Debug)]
pub struct ReferenceLabel {
    pub label: String,
    pub role: String,
    pub metadata: serde_json::Value,
}

/// Immutable token presentation and its ordered visual blocks. Build from the same manifest
/// used for prompt generation; latent references are still passed separately to the sampler.
pub struct PreparedPresentation {
    ids: Vec<i32>,
    blocks: Vec<VisualBlock>,
    labels: Vec<ReferenceLabel>,
    prefix_len: usize,
}
impl PreparedPresentation {
    pub fn new(
        tokenizer: &Tokenizer,
        entries: &[MediaEntry],
        prompt: &str,
        shape: &Shape,
    ) -> Result<Self> {
        let mut p = Presentation::new(tokenizer);
        let mut counts = [0; 3];
        let mut blocks = Vec::new();
        let mut labels = Vec::new();
        let mut keyframes = [false; 2];
        for entry in entries {
            match entry.role.as_str() {
                "reference" => {}
                "first_frame" | "last_frame" if matches!(entry.media, Media::Picture(_)) => {
                    let slot = usize::from(entry.role == "last_frame");
                    if keyframes[slot] {
                        return Err(Error::Invalid("duplicate keyframe role".into()));
                    }
                    keyframes[slot] = true;
                }
                _ => return Err(Error::Invalid("invalid media role".into())),
            }
            let kind = match &entry.media {
                Media::Picture(_) => 0,
                Media::Video(_) => 1,
                Media::Audio(_) => 2,
            };
            counts[kind] += 1;
            labels.push(ReferenceLabel {
                label: format!("<{} {}>", ["Picture", "Video", "Audio"][kind], counts[kind]),
                role: entry.role.clone(),
                metadata: entry.metadata.clone(),
            });
            match &entry.media {
                Media::Picture(frame) => {
                    let frame = frame.for_vision()?;
                    p.picture(
                        i32::try_from(frame.width)
                            .map_err(|_| Error::Invalid("frame too wide".into()))?,
                        i32::try_from(frame.height)
                            .map_err(|_| Error::Invalid("frame too high".into()))?,
                    )?;
                    blocks.push(VisualBlock {
                        video: false,
                        first: frame.clone(),
                        second: frame.clone(),
                    });
                }
                Media::Video(frames) => {
                    if frames.is_empty() {
                        return Err(Error::Invalid("empty video presentation".into()));
                    }
                    p.video()?;
                    let mut previous = None;
                    for (time, frame) in frames {
                        frame.validate()?;
                        if !time.is_finite() || *time < 0.0 || previous.is_some_and(|v| *time <= v)
                        {
                            return Err(Error::Invalid("video timestamps must increase".into()));
                        }
                        previous = Some(*time);
                    }
                    for pair in frames.chunks(2) {
                        let (t0, first) = &pair[0];
                        let (t1, second) = pair.get(1).unwrap_or(&pair[0]);
                        if (first.width, first.height) != (second.width, second.height) {
                            return Err(Error::Invalid("video frame dimensions differ".into()));
                        }
                        let first = first.for_vision()?;
                        let second = if pair.len() == 1 {
                            first.clone()
                        } else {
                            second.for_vision()?
                        };
                        let w = i32::try_from(first.width)
                            .map_err(|_| Error::Invalid("frame too wide".into()))?;
                        let h = i32::try_from(first.height)
                            .map_err(|_| Error::Invalid("frame too high".into()))?;
                        p.video_block(w, h, (t0 + t1) / 2.0)?;
                        blocks.push(VisualBlock {
                            video: true,
                            first: first.clone(),
                            second: second.clone(),
                        });
                    }
                }
                Media::Audio(samples) => {
                    if samples.is_empty()
                        || !samples.len().is_multiple_of(2)
                        || samples.iter().any(|x| !x.is_finite())
                    {
                        return Err(Error::Invalid("invalid planar stereo reference".into()));
                    }
                    p.audio()?;
                }
            }
        }
        let prefix_len = p.len();
        let ids = p.finish(prompt, shape)?;
        Ok(Self {
            ids,
            blocks,
            labels,
            prefix_len,
        })
    }
    pub fn ids(&self) -> &[i32] {
        &self.ids
    }
    pub fn blocks(&self) -> &[VisualBlock] {
        &self.blocks
    }
    pub fn labels(&self) -> &[ReferenceLabel] {
        &self.labels
    }
    pub fn prefix_len(&self) -> usize {
        self.prefix_len
    }
}

/// Upstream RefMod frame sampling: timestamps rather than accumulated rounded frame strides.
pub fn video_sample_indices(frames: usize, fps: f64) -> Result<Vec<(f64, usize)>> {
    if frames == 0 || !fps.is_finite() || !(1.0..=120.0).contains(&fps) {
        return Err(Error::Invalid(
            "reference fps must be 1..120 and video nonempty".into(),
        ));
    }
    Ok((0..(frames as f64 * 2.0 / fps).ceil() as usize)
        .map(|i| {
            let t = i as f64 / 2.0;
            (
                t,
                (t * fps).round_ties_even().min((frames - 1) as f64) as usize,
            )
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimum_pixel_resize_matches_upstream_video_preprocessing() {
        let data = safetensors::SafeTensors::deserialize(include_bytes!(
            "../tests/fixtures/presentation/video_resize.safetensors"
        ))
        .unwrap();
        let floats = |name| {
            data.tensor(name)
                .unwrap()
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let frames = floats("frames");
        let stride = 32 * 64 * 3;
        let first = Frame {
            pixels: frames[..stride].to_vec().into(),
            height: 32,
            width: 64,
        }
        .for_vision()
        .unwrap();
        let second = Frame {
            pixels: frames[stride..].to_vec().into(),
            height: 32,
            width: 64,
        }
        .for_vision()
        .unwrap();
        assert_eq!((first.height, first.width), (64, 96));
        let got = crate::pixels::vision_pair_patches(
            &first.pixels,
            &second.pixels,
            first.height / 16,
            first.width / 16,
            first.width,
        );
        let expected = floats("patches");
        assert_eq!(got.len(), expected.len());
        let error = got
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(
            error < 2e-6,
            "maximum normalized interpolation error {error}"
        );
    }

    #[test]
    fn tokens_and_modality_tags_match_pinned_comfyui() {
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/presentation/upstream.json"))
                .unwrap();
        let frame = || Frame {
            pixels: vec![0.5; 64 * 96 * 3].into(),
            width: 96,
            height: 64,
        };
        let entry = |media| MediaEntry {
            media,
            role: "reference".into(),
            metadata: serde_json::Value::Null,
        };
        let entries = [
            entry(Media::Picture(frame())),
            entry(Media::Audio(vec![0.0; 10].into())),
            entry(Media::Video(vec![
                (0.0, frame()),
                (0.5, frame()),
                (1.0, frame()),
            ])),
        ];
        let p = PreparedPresentation::new(
            &Tokenizer::new().unwrap(),
            &entries,
            "a ball",
            &crate::shape_for(64, 96, 124).unwrap(),
        )
        .unwrap();
        let ids: Vec<i32> = serde_json::from_value(oracle["ids"].clone()).unwrap();
        assert_eq!(p.ids(), ids);
        let mut layout = crate::layout::Layout::new(ids.len(), 2, 4, 6, 8, &[], &[]).unwrap();
        let mut at = 0;
        while at < ids.len() {
            if ids[at] < 0 {
                let start = at;
                while at < ids.len() && ids[at] < 0 {
                    at += 1;
                }
                layout.mark_vision(start, at - start).unwrap();
            } else {
                at += 1;
            }
        }
        let tags: Vec<i32> = serde_json::from_value(oracle["tags"].clone()).unwrap();
        assert_eq!(&layout.adaln_rows[..ids.len()], tags);
    }

    #[test]
    fn pairs_keep_distinct_frames_pad_the_tail_and_count_kinds_independently() {
        let frame = |v| Frame {
            pixels: vec![v; 32 * 32 * 3].into(),
            width: 32,
            height: 32,
        };
        let entries = vec![
            MediaEntry {
                media: Media::Picture(frame(0.0)),
                role: "first_frame".into(),
                metadata: serde_json::Value::Null,
            },
            MediaEntry {
                media: Media::Audio(vec![0.0; 10].into()),
                role: "reference".into(),
                metadata: serde_json::Value::Null,
            },
            MediaEntry {
                media: Media::Video(vec![
                    (0.0, frame(0.1)),
                    (0.5, frame(0.2)),
                    (1.0, frame(0.3)),
                ]),
                role: "reference".into(),
                metadata: serde_json::Value::Null,
            },
        ];
        let p = PreparedPresentation::new(
            &Tokenizer::new().unwrap(),
            &entries,
            "x",
            &crate::shape_for(32, 32, 124).unwrap(),
        )
        .unwrap();
        assert_eq!(
            p.labels
                .iter()
                .map(|x| x.label.as_str())
                .collect::<Vec<_>>(),
            ["<Picture 1>", "<Audio 1>", "<Video 1>"]
        );
        assert_eq!(p.blocks.len(), 3);
        assert_eq!(p.blocks[1].first.pixels[0], 0.1);
        assert_eq!(p.blocks[1].second.pixels[0], 0.2);
        assert!(Arc::ptr_eq(
            &p.blocks[2].first.pixels,
            &p.blocks[2].second.pixels
        ));
        assert_eq!(
            video_sample_indices(25, 24.0).unwrap(),
            vec![(0.0, 0), (0.5, 12), (1.0, 24)]
        );
    }
}
