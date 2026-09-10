//! Runs the audio VAE either way with the Rust implementation.
//!
//!   audio_vae decode <audio_vae.safetensors> <latents.f32> <out.f32> <audio_t>
//!   audio_vae encode <audio_vae.safetensors> <samples.f32> <out.f32> <n>
//!
//! Latents are `[2][32][audio_t]` f32, samples `[2][n]` f32 at 32 kHz — the same arrays
//! `h3_decode_audio` and `h3_encode_audio` take and write. The companion script runs the C on
//! the same inputs.
use h3::avae::{AudioVae, AUDIO_CH, HOP};
use h3::compile::Compiler;
use h3::dispatch::Profile;
use std::io::Write;

fn read(path: &str) -> Vec<f32> {
    std::fs::read(path)
        .expect("input")
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn write(path: &str, v: &[f32]) {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).expect("output"));
    for x in v {
        f.write_all(&x.to_le_bytes()).unwrap();
    }
    f.flush().unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let n: usize = a[4].parse().expect("a count");

    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let exe = std::env::var_os("HRX_LOOM_LIBRARY").map(std::path::PathBuf::from);
    let mut stream = hrx::Stream::open().expect("stream");
    let compiler = Compiler::new(exe, root.join("h3/kernels"));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let mut vae = unsafe { AudioVae::open(&mut stream, &a[1]) }.expect("audio VAE checkpoint");
    let mut prof = Profile::from_env();
    let start = std::time::Instant::now();

    match a[0].as_str() {
        "decode" => {
            let latents = read(&a[2]);
            assert_eq!(latents.len(), 2 * AUDIO_CH * n, "wrong latent count");
            let mut samples = vec![0.0f32; 2 * n * HOP];
            vae.decode(&mut stream, &compiler, &mut prof, &latents, n, &mut samples)
                .expect("decode");
            eprintln!(
                "decoded {n} latent frames in {:.2}s",
                start.elapsed().as_secs_f64()
            );
            write(&a[3], &samples);
        }
        "encode" => {
            let samples = read(&a[2]);
            assert_eq!(samples.len(), 2 * n, "wrong sample count");
            let (z, t) = vae
                .encode(&mut stream, &compiler, &mut prof, &samples, n)
                .expect("encode");
            eprintln!(
                "encoded {n} samples to {t} latent frames in {:.2}s",
                start.elapsed().as_secs_f64()
            );
            write(&a[3], &z);
        }
        other => panic!("unknown direction {other}"),
    }
}
