//! Original-file presentation overrides. These files are never VAE-encoded.
use super::{is_audio, is_image, lower_ext, media, Cli};
use anyhow::{bail, Context, Result};
use h3_hrx::{
    media_context::{Frame, Media},
    refmod::{PreparedRefMod, RefModSource},
    Reference,
};
use serde_json::json;
use std::{collections::BTreeMap, path::PathBuf};

pub type Sources = BTreeMap<(usize, usize), Vec<PathBuf>>;

pub fn parse(cli: &Cli, mods: &[PreparedRefMod]) -> Result<Sources> {
    let mut sources = Sources::new();
    for spec in &cli.refmod_source {
        let (target, path) = spec
            .split_once('=')
            .context("--refmod-source needs SLOT:MEMBER=PATH")?;
        let (slot, member) = target
            .split_once(':')
            .context("--refmod-source needs SLOT:MEMBER=PATH")?;
        let (slot, member): (usize, usize) = (slot.parse()?, member.parse()?);
        if slot == 0 || member == 0 || path.is_empty() {
            bail!("--refmod-source needs positive one-based indices and a nonempty path");
        }
        let active = mods
            .get(slot - 1)
            .and_then(|m| m.indexed_members().find(|(i, _)| *i == member))
            .with_context(|| format!("--refmod-source {target}: unknown or disabled member"))?
            .1;
        let path = PathBuf::from(path);
        let ext = lower_ext(&path);
        let valid = match active.reference() {
            Reference::Image { .. } => is_image(&ext),
            Reference::Audio { .. } => is_audio(&ext),
            Reference::Video { .. } => {
                is_image(&ext) || matches!(ext.as_str(), "mp4" | "mov" | "mkv" | "webm")
            }
        };
        if !valid {
            bail!("--refmod-source {target}: file type does not match member");
        }
        let paths = sources.entry((slot, member)).or_default();
        if !(paths.is_empty()
            || matches!(active.reference(), Reference::Video { .. })
                && is_image(&ext)
                && paths.iter().all(|p| is_image(&lower_ext(p))))
        {
            bail!("--refmod-source {target}: repeat only for an ordered image stack");
        }
        paths.push(path);
    }
    Ok(sources)
}

pub fn missing_modalities(mods: &[PreparedRefMod], sources: &Sources) -> (bool, bool) {
    let (mut visual, mut audio) = (false, false);
    for (slot, prepared) in mods.iter().enumerate() {
        for (index, member) in prepared.indexed_members() {
            if !sources.contains_key(&(slot + 1, index)) {
                if member.is_audio() {
                    audio = true;
                } else {
                    visual = true;
                }
            }
        }
    }
    (visual, audio)
}

pub fn load(
    cli: &Cli,
    mods: &[PreparedRefMod],
    sources: &Sources,
    max_media_bytes: usize,
) -> Result<Vec<RefModSource>> {
    let mut remaining = max_media_bytes;
    let mut result = Vec::new();
    for (&(slot, index), paths) in sources {
        let member = mods[slot - 1]
            .indexed_members()
            .find(|(i, _)| *i == index)
            .context("source member disappeared")?
            .1;
        let image_stack = matches!(member.reference(), Reference::Video { .. })
            && is_image(&lower_ext(&paths[0]));
        let mut frames = Vec::new();
        let mut size = None;
        let media = if member.is_audio() {
            let samples = media::decode_source_audio(&paths[0], remaining)?;
            remaining -= samples.len() * 16;
            Media::Audio(samples.into())
        } else {
            for path in paths {
                let still = is_image(&lower_ext(path));
                let (rgb, width, height) = media::decode_source_visual(
                    path,
                    still,
                    size,
                    (cli.width.unwrap_or(864), cli.height.unwrap_or(480)),
                    remaining,
                )?;
                size = Some((width, height));
                remaining -= rgb.len() * 16;
                let stride = width as usize * height as usize * 3;
                for pixels in rgb.chunks_exact(stride) {
                    // Image stacks preserve all explicitly supplied views. Their ordering
                    // is synthetic; a video file is sampled on its real 2 fps timeline.
                    frames.push((
                        frames.len() as f64 / 2.0,
                        Frame {
                            pixels: pixels.iter().map(|b| *b as f32 / 255.0).collect(),
                            width: width as usize,
                            height: height as usize,
                        },
                    ));
                }
            }
            if matches!(member.reference(), Reference::Image { .. }) {
                Media::Picture(frames.remove(0).1)
            } else {
                Media::Video(frames)
            }
        };
        result.push(RefModSource {
            slot,
            member: index,
            media,
            provenance: json!({"paths":paths, "strength_applied":false}),
            synthetic_timing: image_stack,
        });
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use h3_hrx::{
        refmod::{ApplyOptions, RefMod, RefModMember},
        LatentGrid,
    };

    fn prepared() -> Vec<PreparedRefMod> {
        vec![RefMod::new(
            "bundle",
            vec![
                RefModMember::visual(
                    "views",
                    vec![0.0; 24 * 2 * 4 * 4],
                    LatentGrid {
                        frames: 2,
                        width: 4,
                        height: 4,
                    },
                )
                .unwrap(),
                RefModMember::audio("voice", vec![0.0; 64], 1).unwrap(),
            ],
        )
        .unwrap()
        .prepare(ApplyOptions::default())
        .unwrap()]
    }

    #[test]
    fn mapping_is_unambiguous_and_reports_only_missing_modalities() {
        let mods = prepared();
        let cli = Cli::try_parse_from([
            "h3",
            "--refmod-source",
            "1:1=first image.png",
            "--refmod-source",
            "1:1=second=image.png",
        ])
        .unwrap();
        let paths = parse(&cli, &mods).unwrap();
        assert_eq!(
            paths[&(1, 1)],
            [
                PathBuf::from("first image.png"),
                PathBuf::from("second=image.png")
            ]
        );
        assert_eq!(missing_modalities(&mods, &paths), (false, true));
        for specs in [
            vec!["0:1=image.png"],
            vec!["1:0=image.png"],
            vec!["2:1=image.png"],
            vec!["1:3=image.png"],
            vec!["1:2=image.png"],
            vec!["1:1="],
            vec!["1=image.png"],
            vec!["1:1=clip.mp4", "1:1=image.png"],
            vec!["1:2=voice.wav", "1:2=voice.wav"],
        ] {
            let args =
                std::iter::once("h3").chain(specs.iter().flat_map(|s| ["--refmod-source", *s]));
            assert!(
                parse(&Cli::try_parse_from(args).unwrap(), &mods).is_err(),
                "{specs:?}"
            );
        }
    }

    #[test]
    fn audio_and_ordered_views_are_bounded_and_keep_their_media_kinds() {
        let dir = tempfile::tempdir().unwrap();
        let audio = dir.path().join("voice.wav");
        std::fs::write(&audio, media::wav_bytes(&vec![0.25; 1600], 800)).unwrap();
        let first = dir.path().join("first.bmp");
        let second = dir.path().join("second.bmp");
        for (path, color) in [(&first, "red"), (&second, "blue")] {
            let output = std::process::Command::new("ffmpeg")
                .args([
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    &format!("color=c={color}:s=64x64"),
                    "-frames:v",
                    "1",
                ])
                .arg(path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        let cli = Cli::try_parse_from([
            "h3",
            "--refmod-source",
            &format!("1:1={}", first.display()),
            "--refmod-source",
            &format!("1:1={}", second.display()),
            "--refmod-source",
            &format!("1:2={}", audio.display()),
        ])
        .unwrap();
        let mods = prepared();
        let paths = parse(&cli, &mods).unwrap();
        assert_eq!(missing_modalities(&mods, &paths), (false, false));
        let sources = load(&cli, &mods, &paths, 1024 * 1024).unwrap();
        let Media::Video(frames) = &sources[0].media else {
            panic!()
        };
        assert_eq!(frames.len(), 2);
        assert_eq!((frames[0].0, frames[1].0), (0.0, 0.5));
        assert!(frames[0].1.pixels[0] > 0.9 && frames[1].1.pixels[2] > 0.9);
        assert!(sources[0].synthetic_timing);
        let Media::Audio(samples) = &sources[1].media else {
            panic!()
        };
        assert_eq!(samples.len(), 1600);
        assert!((samples[0] - 0.25).abs() < 0.001);
        assert!(load(&cli, &mods, &paths, 64 * 64 * 3 * 16).is_err());
        assert!(media::decode_source_audio(&audio, 100)
            .unwrap_err()
            .to_string()
            .contains("host media budget"));
        let video = dir.path().join("clip.mkv");
        let output = std::process::Command::new("ffmpeg")
            .args([
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=64x64:r=24:d=1",
                "-c:v",
                "ffv1",
            ])
            .arg(&video)
            .output()
            .unwrap();
        assert!(output.status.success());
        let (pixels, w, h) =
            media::decode_source_visual(&video, false, None, (64, 64), 1024 * 1024).unwrap();
        assert_eq!((w, h, pixels.len()), (64, 64, 2 * 64 * 64 * 3));
        assert!(
            media::decode_source_visual(&video, false, None, (64, 64), 64 * 64 * 3 * 16)
                .unwrap_err()
                .to_string()
                .contains("host media budget")
        );
    }
}
