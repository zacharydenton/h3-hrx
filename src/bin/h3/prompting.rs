//! CLI media preparation; inference and prompt generation consume the same manifest.
use super::{media, Cli, Image};
#[cfg(feature = "prompt-generation")]
use anyhow::Context;
use anyhow::Result;
use h3_hrx::{
    media_context::{video_sample_indices, Frame, Media, MediaEntry},
    refmod::{PreparedRefMod, RefModPresentationOptions},
    Config, Session, Shape,
};
use serde_json::{json, Value};
use std::path::Path;

#[cfg(feature = "prompt-generation")]
pub type Generator = h3_hrx::prompt::PromptGenerator;
#[cfg(not(feature = "prompt-generation"))]
pub struct Generator;

pub fn generator(cli: &Cli) -> Result<Option<Generator>> {
    endpoint_generator(
        cli.generate_prompt,
        cli.prompt_base_url.as_deref(),
        cli.prompt_model.as_deref(),
        cli.prompt_images,
        cli.prompt_audio,
    )
}

pub fn endpoint_generator(
    enabled: bool,
    base_url: Option<&str>,
    model: Option<&str>,
    images: bool,
    audio: bool,
) -> Result<Option<Generator>> {
    if !enabled {
        return Ok(None);
    }
    #[cfg(feature = "prompt-generation")]
    {
        let url = base_url
            .map(str::to_owned)
            .or_else(|| std::env::var("H3_PROMPT_BASE_URL").ok())
            .context("set --prompt-base-url or H3_PROMPT_BASE_URL")?;
        let model = model
            .map(str::to_owned)
            .or_else(|| std::env::var("H3_PROMPT_MODEL").ok())
            .context("set --prompt-model or H3_PROMPT_MODEL")?;
        let mut config = h3_hrx::prompt::EndpointConfig::new(url, model);
        config.api_key = std::env::var("H3_PROMPT_API_KEY")
            .ok()
            .filter(|s| !s.is_empty());
        config.images = images;
        config.audio = audio;
        Ok(Some(Generator::new(config)?))
    }
    #[cfg(not(feature = "prompt-generation"))]
    {
        let _ = (base_url, model, images, audio);
        anyhow::bail!("rebuild h3 with the prompt-generation feature");
    }
}

pub fn generate(
    generator: Generator,
    cli: &Cli,
    entries: &[MediaEntry],
    shape: &Shape,
    instruction: &str,
    world: bool,
) -> Result<String> {
    #[cfg(feature = "prompt-generation")]
    {
        eprintln!("prompt: analyzing references and rewriting instruction");
        let request = h3_hrx::prompt::PromptRequest {
            instruction,
            entries,
            shape,
        };
        let result = if world {
            generator.generate_world_scene(request)?
        } else {
            generator.generate(request)?
        };
        save(cli, &result.text, result.record)?;
        Ok(result.text)
    }
    #[cfg(not(feature = "prompt-generation"))]
    {
        let _ = (generator, cli, entries, shape, instruction, world);
        anyhow::bail!("prompt-generation feature is disabled");
    }
}

pub fn save(cli: &Cli, text: &str, record: Value) -> Result<()> {
    if let Some(path) = &cli.save_prompt {
        let mut record_name = path.as_os_str().to_owned();
        record_name.push(".json");
        let record_path = std::path::PathBuf::from(record_name);
        for (path, bytes) in [
            (path, text.as_bytes().to_vec()),
            (&record_path, serde_json::to_vec_pretty(&record)?),
        ] {
            use std::io::Write;
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let mut f = tempfile::NamedTempFile::new_in(parent)?;
            f.write_all(&bytes)?;
            f.as_file().sync_all()?;
            f.persist(path).map_err(|e| e.error)?;
        }
    }
    Ok(())
}

pub struct RawVideo {
    pub pixels: Vec<f32>,
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub audio: Option<(Vec<f32>, i32)>,
}
pub fn load_video(path: &Path, width: i32, height: i32, audio: bool) -> Result<RawVideo> {
    let (pixels, frames, w, h) = media::decode_reference_video(path, width, height)?;
    let audio = audio.then(|| media::decode_audio(path)).transpose()?;
    Ok(RawVideo {
        pixels,
        frames,
        width: w as usize,
        height: h as usize,
        audio,
    })
}
fn frame(image: &Image) -> Frame {
    Frame {
        pixels: image.pixels.clone().into(),
        width: image.w as usize,
        height: image.h as usize,
    }
}
fn video_frames(
    pixels: &[f32],
    frames: usize,
    width: usize,
    height: usize,
    fps: f64,
) -> Result<Media> {
    let stride = width * height * 3;
    Ok(Media::Video(
        video_sample_indices(frames, fps)?
            .into_iter()
            .map(|(t, i)| {
                (
                    t,
                    Frame {
                        pixels: pixels[i * stride..(i + 1) * stride].to_vec().into(),
                        width,
                        height,
                    },
                )
            })
            .collect(),
    ))
}
pub fn raw_entries(
    keyframes: &[(i32, Image)],
    images: &[Image],
    audio: &[(Vec<f32>, i32)],
    videos: &[RawVideo],
) -> Vec<MediaEntry> {
    let mut out = Vec::new();
    for (index, image) in keyframes {
        out.push(MediaEntry {
            media: Media::Picture(frame(image)),
            role: if *index == 0 {
                "first_frame"
            } else {
                "last_frame"
            }
            .into(),
            metadata: json!({"frame_index":index}),
        });
    }
    for image in images {
        out.push(MediaEntry {
            media: Media::Picture(frame(image)),
            role: "reference".into(),
            metadata: Value::Null,
        });
    }
    for (samples, _) in audio {
        out.push(MediaEntry {
            media: Media::Audio(samples.clone().into()),
            role: "reference".into(),
            metadata: Value::Null,
        });
    }
    for (i, video) in videos.iter().enumerate() {
        // Decoding guarantees a nonempty clip and uses the fixed valid 24 fps.
        out.push(MediaEntry {
            media: video_frames(&video.pixels, video.frames, video.width, video.height, 24.0)
                .expect("decoded video"),
            role: "reference".into(),
            metadata: json!({"source_video":i+1,"synthetic_timing":false}),
        });
        if let Some((samples, _)) = &video.audio {
            out.push(MediaEntry {
                media: Media::Audio(samples.clone().into()),
                role: "reference".into(),
                metadata: json!({"source_video":i+1,"synchronized_track":true}),
            });
        }
    }
    out
}

pub fn refmod_entries(
    cli: &Cli,
    mods: &[PreparedRefMod],
    paths: &super::refmod_sources::Sources,
) -> Result<Vec<MediaEntry>> {
    let max_media_bytes = cli
        .refmod_media_budget_mib
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--refmod-media-budget-mib overflows"))?;
    let sources = super::refmod_sources::load(cli, mods, paths, max_media_bytes)?;
    let (visual, audio) = super::refmod_sources::missing_modalities(mods, paths);
    let options = RefModPresentationOptions {
        fps: cli.reference_fps,
        max_media_bytes,
    };
    if !visual && !audio {
        return Ok(h3_hrx::refmod::entries_from_sources(
            mods, options, &sources,
        )?);
    }
    let resolver = h3_hrx::models::Resolver::new().offline(cli.offline);
    let resolve = |explicit: &Option<std::path::PathBuf>, name: &str| -> Result<_> {
        Ok(match explicit {
            Some(p) => p.clone(),
            None => resolver.find(name)?,
        })
    };
    let config = Config {
        compiler: cli.runtime.compiler(),
        weight_io: cli.runtime.weight_io(),
        video_vae: if visual {
            Some(resolve(&cli.video_vae, h3_hrx::models::VIDEO_VAE)?)
        } else {
            None
        },
        audio_vae: if audio {
            Some(resolve(&cli.audio_vae, h3_hrx::models::AUDIO_VAE)?)
        } else {
            None
        },
        kernel_sources: cli
            .root
            .as_ref()
            .map(|r| r.join("kernels"))
            .unwrap_or_default(),
        loom_library: std::env::var_os("HRX_LOOM_LIBRARY").map(Into::into),
        ..Default::default()
    };
    let budget = super::memory_budget_bytes(cli)?
        .map(hrx::residency::ResidencyManager::new)
        .transpose()?;
    let context = hrx::inference::ModelContext::new(
        cli.runtime.options(budget.as_ref().map(|b| b.budget()))?,
    )?;
    // The CLI never modifies the VAE files during this session.
    let mut session = unsafe {
        Session::new_in(
            config,
            h3_hrx::SessionOptions {
                residency: h3_hrx::ResidencyPolicy::StageScoped,
                ..Default::default()
            },
            &context,
        )
    }?;
    Ok(session.refmod_entries_with_sources(mods, options, &sources)?)
}
