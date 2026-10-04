//! Native reconstruction of the effective RefMod evidence shown to H3 and the prompt endpoint.
use super::{PreparedRefMod, RefModMember};
use crate::{
    error::invalid,
    media_context::{video_sample_indices, Frame, Media, MediaEntry},
    Reference, Result, Session,
};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};

#[derive(Clone, Copy, Debug)]
pub struct RefModPresentationOptions {
    /// Synthetic playback rate for visual stacks; frames are presented at 2 fps.
    pub fps: f64,
    /// Conservative host float-media budget, checked before any decoding. Includes
    /// full decoded buffers and sampling/conversion copies, but not model weights,
    /// GPU scratch, codec subprocesses, endpoint payloads or the eventual H3 vision presentation.
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

/// Original decoded media supplied in place of reconstructing one RefMod member.
/// Only presentation changes: the prepared latent references remain untouched.
#[derive(Clone)]
pub struct RefModSource {
    /// One-based position in the RefMod list.
    pub slot: usize,
    /// One-based member position in the original file, before filtering or copies.
    pub member: usize,
    pub media: Media,
    /// Source provenance, recorded separately from the embedded RefMod metadata.
    pub provenance: Value,
    pub synthetic_timing: bool,
}

/// Assemble presentation without a session or VAEs. Every active member must
/// have a supplied source; copies share its decoded buffers.
pub fn entries_from_sources(
    mods: &[PreparedRefMod],
    options: RefModPresentationOptions,
    sources: &[RefModSource],
) -> Result<Vec<MediaEntry>> {
    entries_with_sources(mods, options, sources, |_| {
        invalid("refmod member has no original source; reconstruction requires a Session")
    })
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
        self.refmod_entries_with_sources(mods, options, &[])
    }

    /// Use original media where supplied and reconstruct only the remaining members.
    pub fn refmod_entries_with_sources(
        &mut self,
        mods: &[PreparedRefMod],
        options: RefModPresentationOptions,
        sources: &[RefModSource],
    ) -> Result<Vec<MediaEntry>> {
        entries_with_sources(mods, options, sources, |member| match member.reference() {
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

#[cfg(test)]
fn entries_with(
    mods: &[PreparedRefMod],
    options: RefModPresentationOptions,
    decode: impl FnMut(&RefModMember) -> Result<Media>,
) -> Result<Vec<MediaEntry>> {
    entries_with_sources(mods, options, &[], decode)
}

fn entries_with_sources(
    mods: &[PreparedRefMod],
    options: RefModPresentationOptions,
    sources: &[RefModSource],
    mut decode: impl FnMut(&RefModMember) -> Result<Media>,
) -> Result<Vec<MediaEntry>> {
    let mut overrides = HashMap::new();
    for source in sources {
        let member = mods
            .get(source.slot.wrapping_sub(1))
            .and_then(|m| m.indexed_members().find(|(i, _)| *i == source.member))
            .map(|(_, m)| m);
        let Some(member) = member else {
            return invalid("refmod source targets an unknown or disabled member");
        };
        let compatible = matches!(
            (member.reference(), &source.media),
            (Reference::Audio { .. }, Media::Audio(_))
                | (Reference::Image { .. }, Media::Picture(_))
                | (Reference::Video { .. }, Media::Video(_))
        );
        if !compatible {
            return invalid("refmod source media type does not match member");
        }
        if overrides
            .insert((source.slot, source.member), source)
            .is_some()
        {
            return invalid("duplicate refmod source member");
        }
    }
    // Validate timing even for image/audio-only requests, before opening a checkpoint.
    video_sample_indices(1, options.fps)?;
    let mut unique = HashSet::new();
    let mut bytes = Some(0usize);
    for (slot, index, member) in mods.iter().enumerate().flat_map(|(slot, m)| {
        m.indexed_members()
            .map(move |(index, member)| (slot + 1, index, member))
    }) {
        if !unique.insert((slot, index)) {
            continue;
        }
        let elements = if let Some(source) = overrides.get(&(slot, index)) {
            match &source.media {
                Media::Picture(frame) => Some(frame.pixels.len()),
                Media::Video(frames) => frames
                    .iter()
                    .try_fold(0usize, |n, (_, f)| n.checked_add(f.pixels.len())),
                Media::Audio(samples) => Some(samples.len()),
            }
        } else {
            match member.reference() {
                Reference::Audio { frames, .. } => frames.checked_mul(800 * 2),
                Reference::Image { grid, .. } | Reference::Video { grid, .. } => grid
                    .frames
                    .checked_mul(4)
                    .and_then(|n| n.checked_sub(3))
                    .and_then(|n| n.checked_sub(3 * ((grid.frames - 1) / 5)))
                    .and_then(|n| n.checked_mul(grid.height))
                    .and_then(|n| n.checked_mul(grid.width))
                    .and_then(|n| n.checked_mul(16 * 16 * 3)),
            }
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
        for (index, member) in prepared.indexed_members() {
            let source = overrides.get(&(slot + 1, index));
            let key = member.values().as_ptr();
            let media = if let Some(source) = source {
                source.media.clone()
            } else {
                match cache.get(&key) {
                    Some(media) => media.clone(),
                    None => {
                        let media = decode(member)?;
                        cache.insert(key, media.clone());
                        media
                    }
                }
            };
            // Embedded configuration is never forwarded as endpoint instructions.
            let meta = member.metadata();
            entries.push(MediaEntry {
                media,
                role: "reference".into(),
                metadata: json!({"refmod_slot":slot+1, "refmod_member":index,
                    "presentation_source":if source.is_some() {"original_file"} else {"reconstructed_latents"},
                    "original_source":source.map(|s| &s.provenance),
                    "name":meta["name"],"description":meta["description"],
                    "concept_type":meta["concept_type"],"source":meta["source"],
                    "synthetic_timing":source.is_none_or(|s| s.synthetic_timing),
                    "reference_fps":source.map_or(Some(options.fps), |s| s.synthetic_timing.then_some(2.0))}),
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
    fn original_sources_skip_only_matching_decodes_and_preserve_latents_and_copies() {
        let prepared = bundle()
            .prepare(ApplyOptions {
                visual_strength: 0.35,
                copies: 2,
                ..Default::default()
            })
            .unwrap();
        let before: Vec<_> = prepared.members().map(|m| m.values().to_vec()).collect();
        let sources = [RefModSource {
            slot: 1,
            member: 1,
            media: Media::Picture(Frame {
                pixels: vec![0.8; 64 * 64 * 3].into(),
                width: 64,
                height: 64,
            }),
            provenance: json!({"paths":["original.png"], "strength_applied":false}),
            synthetic_timing: false,
        }];
        let mut decoded = 0;
        let entries = entries_with_sources(
            std::slice::from_ref(&prepared),
            RefModPresentationOptions::default(),
            &sources,
            |m| {
                assert!(m.is_audio());
                decoded += 1;
                Ok(Media::Audio(vec![0.0; 1600].into()))
            },
        )
        .unwrap();
        assert_eq!(decoded, 1);
        assert_eq!(entries.len(), 4);
        let (Media::Picture(a), Media::Picture(b), Media::Picture(original)) =
            (&entries[0].media, &entries[1].media, &sources[0].media)
        else {
            panic!()
        };
        assert!(Arc::ptr_eq(&a.pixels, &b.pixels));
        assert!(Arc::ptr_eq(&a.pixels, &original.pixels));
        assert_eq!(entries[0].metadata["presentation_source"], "original_file");
        assert_eq!(
            entries[0].metadata["original_source"]["strength_applied"],
            false
        );
        assert_eq!(
            entries[2].metadata["presentation_source"],
            "reconstructed_latents"
        );
        assert_eq!(
            before,
            prepared
                .members()
                .map(|m| m.values().to_vec())
                .collect::<Vec<_>>()
        );
        // Sources and reconstructed media share the same aggregate budget.
        assert!(entries_with_sources(
            &[prepared],
            RefModPresentationOptions {
                max_media_bytes: (64 * 64 * 3 + 1600) * 16 - 1,
                ..Default::default()
            },
            &sources,
            |_| panic!("budget must fail first")
        )
        .is_err());
    }

    #[test]
    fn original_member_numbers_survive_filtering_and_invalid_sources_fail() {
        let prepared = bundle()
            .prepare(ApplyOptions {
                visual_strength: 0.0,
                copies: 3,
                ..Default::default()
            })
            .unwrap();
        assert_eq!(
            prepared
                .indexed_members()
                .map(|(i, _)| i)
                .collect::<Vec<_>>(),
            [2, 2, 2]
        );
        let mut source = RefModSource {
            slot: 1,
            member: 2,
            media: Media::Audio(vec![0.0; 1600].into()),
            provenance: json!({}),
            synthetic_timing: false,
        };
        let mods = [prepared];
        let options = RefModPresentationOptions::default();
        assert_eq!(
            entries_from_sources(&mods, options, &[source.clone()])
                .unwrap()
                .len(),
            3
        );
        assert!(entries_from_sources(&mods, options, &[source.clone(), source.clone()]).is_err());
        assert!(entries_from_sources(&mods, options, &[]).is_err());
        source.member = 1; // disabled, not renumbered
        assert!(entries_from_sources(&mods, options, &[source.clone()]).is_err());
        source.member = 2;
        source.slot = 0;
        assert!(entries_from_sources(&mods, options, &[source.clone()]).is_err());
        source.slot = 1;
        source.media = Media::Video(vec![]);
        assert!(entries_from_sources(&mods, options, &[source]).is_err());
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
