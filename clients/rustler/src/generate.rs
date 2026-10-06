use h3_hrx::{DenoiseParams, LatentGrid, Reference, Session};

/// Owned latent references. Video soundtracks can be supplied as an Audio entry.
#[derive(rustler::NifTaggedEnum)]
pub enum LatentReference {
    Image(Vec<f32>, usize, usize),
    Audio(Vec<f32>, usize),
    Video(Vec<f32>, usize, usize, usize),
}
impl LatentReference {
    fn borrow(&self) -> Reference<'_> {
        match self {
            Self::Image(latents, height, width) => Reference::Image {
                latents,
                grid: LatentGrid {
                    frames: 1,
                    height: *height,
                    width: *width,
                },
                presented: None,
            },
            Self::Audio(latents, frames) => Reference::Audio {
                latents,
                frames: *frames,
            },
            Self::Video(latents, frames, height, width) => Reference::Video {
                latents,
                grid: LatentGrid {
                    frames: *frames,
                    height: *height,
                    width: *width,
                },
                audio: None,
            },
        }
    }
}
#[derive(rustler::NifMap)]
pub struct GenerateRequest {
    pub prompt: String,
    pub width: i32,
    pub height: i32,
    pub frames: i32,
    pub steps: usize,
    pub seed: u64,
    pub references: Vec<LatentReference>,
    /// File, visual strength, audio strength, copies, in conditioning order.
    pub refmods: Vec<(String, f32, f32, usize)>,
    pub upscale_width: i32,
    pub upscale_height: i32,
    pub upscale_steps: usize,
    pub upscale_denoise: f64,
    pub upscale_seed: Option<u64>,
}

pub fn run(
    session: &mut Session,
    request: GenerateRequest,
) -> h3_hrx::Result<h3_hrx::UpscaledLatents> {
    let mut p = DenoiseParams {
        width: request.width,
        height: request.height,
        frames: request.frames,
        steps: request.steps,
        seed: request.seed,
        ..Default::default()
    };
    let shape = h3_hrx::shape_for(p.height, p.width, p.frames)
        .ok_or_else(|| h3_hrx::Error::Invalid("invalid generation dimensions".into()))?;
    let upscale = h3_hrx::UpscaleSettings {
        target: h3_hrx::UpscaleTarget::Dimensions {
            width: request.upscale_width,
            height: request.upscale_height,
        },
        ..Default::default()
    };
    upscale.output_shape(&shape)?;
    let refine = h3_hrx::RefinementSettings {
        steps: request.upscale_steps,
        denoise: request.upscale_denoise,
        seed: request.upscale_seed,
    };
    refine.validate()?;
    let prepared = request
        .refmods
        .iter()
        .map(|(path, visual_strength, audio_strength, copies)| {
            h3_hrx::refmod::RefMod::load(path)?.prepare(h3_hrx::refmod::ApplyOptions {
                visual_strength: *visual_strength,
                audio_strength: *audio_strength,
                copies: *copies,
                ..Default::default()
            })
        })
        .collect::<h3_hrx::Result<Vec<_>>>()?;
    let mut refs = request
        .references
        .iter()
        .map(LatentReference::borrow)
        .collect::<Vec<_>>();
    for r in &prepared {
        refs.extend(r.references());
    }
    let ids = h3_hrx::Tokenizer::new()?.encode(&request.prompt)?;
    let initial = session.denoise(&ids, &p, h3_hrx::Noise::default(), &refs, &[], None)?;
    let mut out = session.upscale_latents(&initial, &shape, &upscale, None)?;
    (p.width, p.height) = out.shape.size();
    out.latents = session.refine(&ids, None, &p, &out.latents, &refine, &refs, &[], None)?;
    Ok(out)
}
