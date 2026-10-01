use super::*;
use crate::{Clip, Session};

/// Decoded RGB8 image; images are independently encoded on a common canvas.
#[derive(Clone, Copy)]
pub struct ImageInput<'a> {
    pub pixels: &'a [u8],
    pub width: i32,
    pub height: i32,
}
/// Planar stereo `[2,samples_per_channel]`, already resampled to 32 kHz.
#[derive(Clone, Copy)]
pub struct AudioInput<'a> {
    pub samples: &'a [f32],
    pub samples_per_channel: usize,
}
#[derive(Clone, Debug)]
pub struct CreateOptions {
    pub name: String,
    pub description: String,
    pub concept_type: String,
    pub resolution: i32,
    pub max_tokens: usize,
    pub audio_max_seconds: f32,
    pub audio_max_tokens: usize,
    pub truncate_audio: bool,
}
impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            name: "refmod".into(),
            description: String::new(),
            concept_type: "identity".into(),
            resolution: 1024,
            max_tokens: 8192,
            audio_max_seconds: 30.0,
            audio_max_tokens: 5120,
            truncate_audio: false,
        }
    }
}
impl CreateOptions {
    pub fn check(&self) -> Result<()> {
        if self.name.is_empty()
            || self.resolution < 32
            || self.resolution > 8192
            || self.resolution % 32 != 0
            || self.max_tokens == 0
            || self.audio_max_tokens < 2
            || !self.audio_max_seconds.is_finite()
            || self.audio_max_seconds <= 0.0
            || self.audio_max_seconds > 3600.0
        {
            return invalid("refmod: invalid name, resolution (32..8192, multiple of 32), budget, or audio duration (0..3600 seconds)");
        }
        Ok(())
    }
    /// Canvas for the first image. Rounding follows Python's ties-to-even rule.
    pub fn canvas(&self, width: i32, height: i32) -> Result<(i32, i32)> {
        self.check()?;
        if width <= 0 || height <= 0 {
            return invalid("refmod: empty image");
        }
        let scale = (f64::from(self.resolution) / f64::from(width.min(height))).min(1.0);
        let dim = |d: i32| (f64::from(d) * scale / 32.0).round_ties_even().max(1.0) * 32.0;
        let (w, h) = (dim(width), dim(height));
        if w > 8192.0 || h > 8192.0 {
            return invalid("refmod: image aspect ratio requires a canvas larger than 8192");
        }
        let (w, h) = (w as i32, h as i32);
        if (w as usize / 32) * (h as usize / 32) > self.max_tokens {
            return invalid(
                "refmod: one image exceeds the visual token budget; lower --resolution",
            );
        }
        Ok((w, h))
    }
}

impl Session {
    /// Encode an image set and optional audio. Use a retaining session to reuse
    /// encoder weights across inputs. Only the requested VAEs are opened.
    pub fn create_refmod(
        &mut self,
        images: &[ImageInput<'_>],
        audio: Option<AudioInput<'_>>,
        options: &CreateOptions,
    ) -> Result<RefMod> {
        options.check()?;
        if images.is_empty() && audio.is_none() {
            return invalid("refmod: no images or audio");
        }
        for image in images {
            let count = usize::try_from(image.width)
                .ok()
                .zip(usize::try_from(image.height).ok())
                .and_then(|(w, h)| w.checked_mul(h))
                .and_then(|n| n.checked_mul(3));
            if image.width <= 0 || image.height <= 0 || count != Some(image.pixels.len()) {
                return invalid("refmod: invalid RGB image buffer");
            }
        }
        if let Some(a) = audio {
            if a.samples_per_channel == 0
                || a.samples_per_channel.checked_mul(2) != Some(a.samples.len())
                || a.samples.iter().any(|x| !x.is_finite())
            {
                return invalid("refmod: invalid stereo audio buffer");
            }
            let frames = audio_samples(a.samples_per_channel, options).div_ceil(800);
            if !options.truncate_audio && frames * 2 > options.audio_max_tokens {
                return invalid(
                    "refmod: audio token budget exceeded; shorten duration or enable truncation",
                );
            }
        }
        let mut members = Vec::new();
        if let Some(first) = images.first() {
            let (w, h) = options.canvas(first.width, first.height)?;
            let mut frames = Vec::new();
            for image in images {
                let pixels = preprocess(*image, w, h);
                let (z, t) = self.encode_video(Clip {
                    pixels: &pixels,
                    frames: 1,
                    height: h as usize,
                    width: w as usize,
                })?;
                if t != 1 {
                    return invalid("refmod: image encoder returned multiple latent frames");
                }
                // Match the portable visual representation before token fitting.
                frames.push(
                    z.into_iter()
                        .map(|v| f16::from_f32(v).to_f32())
                        .collect::<Vec<_>>(),
                );
            }
            let (lh, lw) = (h as usize / 16, w as usize / 16);
            let selected = select_frames(&frames, (lh / 2) * (lw / 2), options.max_tokens)?;
            let values = stack_frames(&frames, &selected, lh * lw);
            let mut member = RefModMember::visual(
                &options.name,
                values,
                LatentGrid {
                    frames: selected.len(),
                    height: lh,
                    width: lw,
                },
            )?;
            member.metadata["description"] = json!(options.description);
            member.metadata["concept_type"] = json!(options.concept_type);
            member.metadata["source"] = json!(if images.len() == 1 { "image" } else { "stack" });
            member.metadata["source_shape"] = json!(format!("{}x{}x{}", images.len(), lh, lw));
            member.metadata["pool"] = json!(format!("full-res {w}x{h}px"));
            member.metadata["h3_hrx_preprocessing"] = json!({"filter":"bilinear","crop":"center","resolution":options.resolution,
                "source_images":images.len(),"selected_indices":selected});
            members.push(member);
        }
        if let Some(a) = audio {
            let n = audio_samples(a.samples_per_channel, options);
            let mut chunks = Vec::new();
            let mut frames = 0;
            for start in (0..n).step_by(320000) {
                let end = (start + 320000).min(n);
                let mut piece = a.samples[start..end].to_vec();
                piece.extend_from_slice(
                    &a.samples[a.samples_per_channel + start..a.samples_per_channel + end],
                );
                let (z, t) = self.encode_audio(&piece, end - start)?;
                frames += t;
                chunks.push((z, t));
            }
            let keep = frames.min(options.audio_max_tokens / 2);
            let mut values = vec![0.0; 64 * keep];
            let mut at = 0;
            for (z, t) in chunks {
                let take = t.min(keep - at);
                for c in 0..64 {
                    values[c * keep + at..c * keep + at + take]
                        .copy_from_slice(&z[c * t..c * t + take]);
                }
                at += take;
                if at == keep {
                    break;
                }
            }
            let audio_name = if images.is_empty() {
                options.name.clone()
            } else {
                format!("{}_audio", options.name)
            };
            let mut member = RefModMember::audio(&audio_name, values, keep)?;
            member.metadata["description"] = json!(options.description);
            member.metadata["h3_hrx_preprocessing"] = json!({"sample_rate":32000,"chunk_seconds":10,"max_seconds":options.audio_max_seconds});
            members.push(member);
        }
        RefMod::new(&options.name, members)
    }
}
fn audio_samples(n: usize, options: &CreateOptions) -> usize {
    n.min(
        (f64::from(options.audio_max_seconds) * 32000.0)
            .round_ties_even()
            .max(1.0) as usize,
    )
}
fn preprocess(image: ImageInput<'_>, w: i32, h: i32) -> Vec<f32> {
    let (sw, sh) = (image.width, image.height);
    let ratio = f64::from(w) / f64::from(h);
    let (cw, ch) = if f64::from(sw) / f64::from(sh) > ratio {
        ((f64::from(sh) * ratio).round().max(1.0) as i32, sh)
    } else {
        (sw, (f64::from(sw) / ratio).round().max(1.0) as i32)
    };
    let (x, y) = ((sw - cw) / 2, (sh - ch) / 2);
    let mut cropped = Vec::with_capacity(cw as usize * ch as usize * 3);
    for row in y..y + ch {
        let offset = (row as usize * sw as usize + x as usize) * 3;
        cropped.extend_from_slice(&image.pixels[offset..offset + cw as usize * 3]);
    }
    crate::resize::pil_bilinear(&cropped, cw, ch, w, h)
}
fn select_frames(frames: &[Vec<f32>], cost: usize, budget: usize) -> Result<Vec<usize>> {
    if cost == 0 || cost > budget {
        return invalid("refmod: one frame exceeds token budget");
    }
    if frames.len() <= budget / cost {
        return Ok((0..frames.len()).collect());
    }
    let mut kept = vec![0];
    for i in 1..frames.len() {
        let previous = &frames[*kept.last().unwrap()];
        let current = &frames[i];
        let mut diff = 0.0f32;
        let mut magnitude = 0.0f32;
        for (&a, &b) in previous.iter().zip(current) {
            diff += (a - b).abs();
            magnitude += (a.abs() + b.abs()) / 2.0;
        }
        if diff / (magnitude + current.len() as f32 * 1e-6) >= 0.02 {
            kept.push(i);
        }
    }
    let fit = budget / cost;
    if kept.len() <= fit {
        return Ok(kept);
    }
    if fit == 1 {
        return Ok(vec![kept[0]]);
    }
    Ok((0..fit)
        .map(|i| {
            kept[(i as f64 * (kept.len() - 1) as f64 / (fit - 1) as f64).round_ties_even() as usize]
        })
        .collect())
}
fn stack_frames(frames: &[Vec<f32>], indices: &[usize], plane: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(24 * indices.len() * plane);
    for c in 0..24 {
        for &i in indices {
            values.extend_from_slice(&frames[i][c * plane..(c + 1) * plane]);
        }
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn temporal_selection_preserves_space_and_rounds_ties_even() {
        let frames = (0..6).map(|i| vec![i as f32; 96]).collect::<Vec<_>>();
        assert_eq!(select_frames(&frames, 1, 3).unwrap(), vec![0, 2, 5]);
        let frames = vec![vec![1.0; 96]; 8];
        assert_eq!(select_frames(&frames, 4, 8).unwrap(), vec![0]);
        assert!(select_frames(&frames, 4, 3).is_err());
    }
    #[test]
    fn stacking_interleaves_frames_within_channels() {
        let frames = (0..3)
            .map(|f| {
                (0..24)
                    .flat_map(|c| [100.0 * f as f32 + c as f32; 4])
                    .collect()
            })
            .collect::<Vec<_>>();
        let stack = stack_frames(&frames, &[0, 2], 4);
        assert_eq!(
            &stack[..12],
            &[0., 0., 0., 0., 200., 200., 200., 200., 1., 1., 1., 1.]
        );
    }
    #[test]
    fn canvas_preserves_aspect_and_rejects_impossible_budgets() {
        let options = CreateOptions::default();
        assert_eq!(options.canvas(2048, 1024).unwrap(), (2048, 1024));
        assert_eq!(options.canvas(512, 512).unwrap(), (512, 512));
        assert!(options.canvas(0, 100).is_err());
        assert!(CreateOptions {
            max_tokens: 1,
            ..options
        }
        .canvas(512, 512)
        .is_err());
    }
}
