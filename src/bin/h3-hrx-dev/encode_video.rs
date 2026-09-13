//! Encodes pixels to model-space video latents with the Rust encoder.
//!
//!   encode_video <video_vae.safetensors> <pixels.f32> <out.f32> <frames> <height> <width>
//!
//! Pixels are `[frames][H][W][3]` f32 in `[0, 1]`; the output is `[24][latent_t][H/16][W/16]` f32, the
//! same array `h3_encode_video` writes. The companion script runs the C on the same pixels.
use h3_hrx::compile::Compiler;
use h3_hrx::dispatch::Profile;
use h3_hrx::vvae::{Clip, VideoVae};
use std::io::Write;

pub fn run(args: Vec<String>) {
    let a = args;
    let (frames, height, width): (usize, usize, usize) = (
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
        a[5].parse().unwrap(),
    );

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("kernels"));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let mut vae = unsafe { VideoVae::open(&mut stream, &a[0]) }.expect("video VAE checkpoint");

    let bytes = std::fs::read(&a[1]).expect("pixels");
    let pixels: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    assert_eq!(
        pixels.len(),
        frames * height * width * 3,
        "wrong pixel count"
    );

    let mut prof = Profile::from_env();
    let start = std::time::Instant::now();
    let (z, t) = vae
        .encode_video(
            &mut stream,
            &compiler,
            &mut prof,
            Clip {
                pixels: &pixels,
                frames,
                height,
                width,
            },
        )
        .expect("encode");
    eprintln!(
        "encoded {frames} frames at {height}x{width} to {t} latent frames in {:.2}s",
        start.elapsed().as_secs_f64()
    );
    let mut f = std::io::BufWriter::new(std::fs::File::create(&a[2]).expect("output"));
    for v in &z {
        f.write_all(&v.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}
