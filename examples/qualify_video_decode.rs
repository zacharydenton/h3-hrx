//! Decode a fixed latent fixture independently of sampling.
use h3_hrx::{compile::Compiler, dispatch::Profile, models, shape_for, vvae::VideoVae};
use std::{io::Write, path::Path};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: qualify_video_decode LATENTS_F32 OUTPUT_RGB".into());
    }
    let bytes = std::fs::read(&args[1])?;
    let shape = shape_for(256, 256, 5).ok_or("shape")?;
    let count = 24 * shape.latent_t as usize * shape.lat_h as usize * shape.lat_w as usize;
    if bytes.len() != count * 4 {
        return Err("incorrect latent byte count".into());
    }
    let latents: Vec<_> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    if !latents.iter().all(|x| x.is_finite()) {
        return Err("nonfinite latents".into());
    }
    let mut stream = hrx::Stream::open()?;
    let compiler = Compiler::new(
        std::env::var_os("HRX_LOOM_LIBRARY").map(Into::into),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels"),
    );
    let path = models::Resolver::new()
        .offline(true)
        .find(models::VIDEO_VAE)?;
    // The checksum-verified checkpoint remains immutable while mapped.
    let mut vae = unsafe { VideoVae::open(&mut stream, &path) }?;
    let mut output = vec![0; 256 * 256 * 5 * 3];
    vae.decode_video(
        &mut stream,
        &compiler,
        &mut Profile::default(),
        &shape,
        &latents,
        &mut output,
    )?;
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?
        .write_all(&output)?;
    Ok(())
}
