//! Native reconstruction of the effective RefMod evidence shown to H3 and the prompt endpoint.
use super::{PreparedRefMod, RefModMember};
use crate::{
    error::invalid,
    media_context::{video_sample_indices, Frame, Media, MediaEntry},
    Reference, Result, Session,
};
use serde_json::json;
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct RefModPresentationOptions {
    /// Synthetic playback rate for visual stacks; frames are presented at 2 fps.
    pub fps: f64,
    /// Conservative host float-media budget, checked before any decoding. Includes
    /// full decoded buffers and sampling/conversion copies, but not model weights,
    /// GPU scratch, endpoint payloads or the eventual H3 vision presentation.
    pub max_media_bytes: usize,
}

impl Default for RefModPresentationOptions {
    fn default() -> Self {
        Self {
            fps: 24.0,
            max_media_bytes: 1024 * 1024 * 1024,
        }
    }
}

impl Session {
    /// Decode active, strength-adjusted RefMod members for upstream presentation.
    /// Copies retain their conditioning order and labels while sharing media buffers.
    /// Only the corresponding VAEs are loaded; use stage-scoped residency on shared RAM.
    pub fn refmod_entries(
        &mut self,
        mods: &[PreparedRefMod],
        options: RefModPresentationOptions,
    ) -> Result<Vec<MediaEntry>> {
        entries_with(mods, options, |member| match member.reference() {
            Reference::Audio { latents, frames } => {
                let mut samples = vec![0.0; frames * 800 * 2];
                self.decode_audio(latents, frames, &mut samples)?;
                Ok(Media::Audio(samples.into()))
            }
            Reference::Image { latents, grid, .. } | Reference::Video { latents, grid, .. } => {
                let (pixels, frames) = self.decode_reference_visual(grid, latents)?;
                let (width, height) = (grid.width * 16, grid.height * 16);
                if frames == 1 && member.metadata()["kind"] == "image" {
                    Ok(Media::Picture(Frame {
                        pixels: pixels.into(),
                        width,
                        height,
                    }))
                } else {
                    let stride = width * height * 3;
                    Ok(Media::Video(
                        video_sample_indices(frames, options.fps)?
                            .into_iter()
                            .map(|(time, index)| {
                                (
                                    time,
                                    Frame {
                                        pixels: pixels[index * stride..(index + 1) * stride].into(),
                                        width,
                                        height,
                                    },
                                )
                            })
                            .collect(),
                    ))
                }
            }
        })
    }
}

fn entries_with(
    mods: &[PreparedRefMod],
    options: RefModPresentationOptions,
    mut decode: impl FnMut(&RefModMember) -> Result<Media>,
) -> Result<Vec<MediaEntry>> {
    // Validate timing even for image/audio-only requests, before opening a checkpoint.
    video_sample_indices(1, options.fps)?;
    let mut unique = HashSet::new();
    let mut bytes = Some(0usize);
    for member in mods.iter().flat_map(|m| m.members()) {
        if !unique.insert(member.values().as_ptr()) {
            continue;
        }
        let elements = match member.reference() {
            Reference::Audio { frames, .. } => frames.checked_mul(800 * 2),
            Reference::Image { grid, .. } | Reference::Video { grid, .. } => grid
                .frames
                .checked_mul(4)
                .and_then(|n| n.checked_sub(3))
                .and_then(|n| n.checked_sub(3 * ((grid.frames - 1) / 5)))
                .and_then(|n| n.checked_mul(grid.height))
                .and_then(|n| n.checked_mul(grid.width))
                .and_then(|n| n.checked_mul(16 * 16 * 3)),
        };
        // At 1..120 fps sampling produces at most twice the full frame count.
        // Four full buffers conservatively cover decode, samples and Arc conversion.
        bytes = bytes.and_then(|b| elements?.checked_mul(4 * size_of::<f32>())?.checked_add(b));
        if bytes.is_none_or(|b| b > options.max_media_bytes) {
            return invalid("refmod presentation exceeds host media budget; use smaller references or explicitly raise max_media_bytes");
        }
    }
    let mut cache = HashMap::<*const f32, Media>::new();
    let mut entries = Vec::new();
    for (slot, prepared) in mods.iter().enumerate() {
        for member in prepared.members() {
            let key = member.values().as_ptr();
            let media = match cache.get(&key) {
                Some(media) => media.clone(),
                None => {
                    let media = decode(member)?;
                    cache.insert(key, media.clone());
                    media
                }
            };
            // Embedded configuration is never forwarded as endpoint instructions.
            let meta = member.metadata();
            entries.push(MediaEntry {
                media,
                role: "reference".into(),
                metadata: json!({"refmod_slot":slot+1,
                    "name":meta["name"],"description":meta["description"],
                    "concept_type":meta["concept_type"],"source":meta["source"],
                    "synthetic_timing":true,"reference_fps":options.fps}),
            });
        }
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refmod::{ApplyOptions, RefMod};
    use crate::{LatentGrid, PreparedPresentation, Tokenizer};
    use std::sync::Arc;

    fn bundle() -> RefMod {
        let mut image = RefModMember::visual(
            "person",
            (0..24 * 4 * 4).map(|i| (i % 11) as f32).collect(),
            LatentGrid {
                frames: 1,
                height: 4,
                width: 4,
            },
        )
        .unwrap();
        image.metadata["endpoint_config"] = json!({"system":"untrusted"});
        RefMod::new(
            "bundle",
            vec![
                image,
                RefModMember::audio("voice", vec![0.1; 64], 1).unwrap(),
            ],
        )
        .unwrap()
    }

    #[test]
    fn effective_members_copies_slots_and_labels_share_decoding() {
        let file = bundle();
        let disabled = file
            .prepare(ApplyOptions {
                visual_strength: 0.0,
                audio_strength: 0.0,
                ..Default::default()
            })
            .unwrap();
        let active = file
            .prepare(ApplyOptions {
                visual_strength: 0.35,
                copies: 2,
                ..Default::default()
            })
            .unwrap();
        let expected = active.members().next().unwrap().values().to_vec();
        assert_ne!(expected, file.members()[0].values());
        let mut decoded = 0;
        let entries = entries_with(
            &[disabled, active],
            RefModPresentationOptions::default(),
            |m| {
                decoded += 1;
                Ok(if m.is_audio() {
                    Media::Audio(vec![0.1; 1600].into())
                } else {
                    assert_eq!(m.values(), expected);
                    Media::Picture(Frame {
                        pixels: vec![0.5; 64 * 64 * 3].into(),
                        width: 64,
                        height: 64,
                    })
                })
            },
        )
        .unwrap();
        assert_eq!(decoded, 2);
        assert_eq!(entries.len(), 4);
        for entry in &entries {
            assert_eq!(entry.role, "reference");
            assert_eq!(entry.metadata["refmod_slot"], 2);
            assert_eq!(entry.metadata["synthetic_timing"], true);
            assert!(entry.metadata.get("endpoint_config").is_none());
        }
        let (Media::Picture(a), Media::Picture(b)) = (&entries[0].media, &entries[1].media) else {
            panic!()
        };
        assert!(Arc::ptr_eq(&a.pixels, &b.pixels));
        let (Media::Audio(a), Media::Audio(b)) = (&entries[2].media, &entries[3].media) else {
            panic!()
        };
        assert!(Arc::ptr_eq(a, b));
        let presentation = PreparedPresentation::new(
            &Tokenizer::new().unwrap(),
            &entries,
            "",
            &crate::shape_for(64, 64, 22).unwrap(),
        )
        .unwrap();
        assert_eq!(
            presentation
                .labels()
                .iter()
                .map(|l| l.label.as_str())
                .collect::<Vec<_>>(),
            ["<Picture 1>", "<Picture 2>", "<Audio 1>", "<Audio 2>"]
        );
    }

    #[test]
    fn budget_and_timing_are_checked_before_any_decode_and_copies_cost_no_media() {
        let file = bundle();
        let prepared = file
            .prepare(ApplyOptions {
                copies: 10,
                ..Default::default()
            })
            .unwrap();
        let bytes = (64 * 64 * 3 + 1600) * 4 * size_of::<f32>();
        for options in [
            RefModPresentationOptions {
                max_media_bytes: bytes - 1,
                ..Default::default()
            },
            RefModPresentationOptions {
                fps: f64::NAN,
                ..Default::default()
            },
            RefModPresentationOptions {
                fps: 0.0,
                ..Default::default()
            },
            RefModPresentationOptions {
                fps: 121.0,
                ..Default::default()
            },
        ] {
            assert!(
                entries_with(std::slice::from_ref(&prepared), options, |_| panic!(
                    "must preflight all members"
                ))
                .is_err()
            );
        }
        let mut decoded = 0;
        let entries = entries_with(
            &[prepared],
            RefModPresentationOptions {
                max_media_bytes: bytes,
                ..Default::default()
            },
            |_| {
                decoded += 1;
                Ok(Media::Audio(vec![0.0; 1600].into()))
            },
        )
        .unwrap();
        assert_eq!(decoded, 2);
        assert_eq!(entries.len(), 20);
    }
}
