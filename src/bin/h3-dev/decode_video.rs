//! Decodes model-space video latents to RGB bytes with the Rust decoder.
//!
//!   decode_video <video_vae.safetensors> <latents.f32> <out.rgb> <height> <width> <frames>
//!
//! Latents are `[24][latent_t][H/16][W/16]` f32; the output is `[frames][H][W][3]` u8, the same bytes
//! `h3_decode_video` writes. The companion script runs the C implementation on the same latents
//! through its Python binding and compares.
use h3_hrx::compile::Compiler;
use h3_hrx::dispatch::Profile;
use h3_hrx::layout::shape_for;
use h3_hrx::vvae::VideoVae;

pub fn run(args: Vec<String>) {
    let a = args;
    let (height, width, frames): (i32, i32, i32) = (
        a[3].parse().unwrap(),
        a[4].parse().unwrap(),
        a[5].parse().unwrap(),
    );
    let shape = shape_for(height, width, frames).expect("a shape this model can serve");

    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("kernels"));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let mut vae = unsafe { VideoVae::open(&mut stream, &a[0]) }.expect("video VAE checkpoint");

    let bytes = std::fs::read(&a[1]).expect("latents");
    let latents: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect();
    let want = 24 * shape.latent_t as usize * shape.lat_h as usize * shape.lat_w as usize;
    assert_eq!(latents.len(), want, "latents are the wrong length");

    let mut out = vec![0u8; shape.frames as usize * height as usize * width as usize * 3];
    let mut prof = Profile::from_env();
    // H3_REPEAT=n decodes n times and reports each: the first call builds the stack and loads its
    // kernels, so only a later one is comparable with a warm session.
    let repeat: usize = std::env::var("H3_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    for _ in 0..repeat {
        let start = std::time::Instant::now();
        vae.decode_video(
            &mut stream,
            &compiler,
            &mut prof,
            &shape,
            &latents,
            &mut out,
        )
        .expect("decode");
        eprintln!(
            "decoded {} frames at {height}x{width} in {:.2}s",
            shape.frames,
            start.elapsed().as_secs_f64()
        );
    }
    std::fs::write(&a[2], &out).expect("output");
}
