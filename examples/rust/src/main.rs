//! The `h3` crate from Rust: prompt -> frames + samples, written as <out>.rgb and <out>.wav.
//!   cargo run --release -- "A red fox ..." [frames] [steps] [out]     (from the repository root, or set H3_ROOT)
//!
//! A Rust caller does not go through the C ABI. `Session` is the library's own API — slices, borrows
//! and `Result` — and this is what it looks like used directly. `examples/c` and `examples/go` are the
//! C-ABI clients.
use h3::{Config, DenoiseParams, Noise, Session, Tokenizer};
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: minimal \"prompt\" [frames] [steps] [out]");
        std::process::exit(64);
    }
    let root = std::env::var("H3_ROOT").unwrap_or_else(|_| ".".into());
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
    let models = std::env::var("H3_MODELS").unwrap_or(format!("{home}/comfy-models"));
    let out = args.get(4).cloned().unwrap_or_else(|| "minimal".into());

    // the vocabulary compiled into the crate; H3_TOKENIZER names a file instead
    let ids = Tokenizer::new()
        .expect("tokenizer")
        .encode(&args[1])
        .expect("cannot tokenize the prompt");

    let file = |p: &str| Some(std::path::PathBuf::from(format!("{models}/{p}")));
    // Safety: a diagnostic run over checkpoints the operator named and is not writing to.
    let mut session = unsafe { Session::new(Config {
        dit: file("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
        te: file("text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
        video_vae: file("vae/minimax_h3_video_vae_fp16.safetensors"),
        audio_vae: file("vae/minimax_h3_audio_vae_fp32.safetensors"),
        kernel_sources: format!("{root}/kernels").into(),
        cache_dir: format!("{root}/build/kernel_cache").into(),
        loom_compile: std::env::var("LOOM_COMPILE").unwrap_or_else(|_| "loom-compile".into()),
        attention: h3::Attention::I8,
    })
    .expect("session") };

    let p = DenoiseParams {
        height: 480,
        width: 864,
        frames: args.get(2).map_or(124, |v| v.parse().unwrap()),
        steps: args.get(3).map_or(31, |v| v.parse().unwrap()),
        ..DenoiseParams::default()
    };
    let sh = Session::shape_for(p.height, p.width, p.frames).expect("invalid parameters");
    eprintln!(
        "{} frames, {}x{}x{} latents, {} audio latents, {} prompt tokens",
        sh.frames,
        sh.latent_t,
        sh.lat_h,
        sh.lat_w,
        sh.audio_t,
        ids.len()
    );

    let mut show = |step: usize, steps: usize, seconds: f64| {
        eprintln!("  step {step}/{steps}  {seconds:.1} s");
        false // true cancels
    };
    let latents = session
        .denoise(&ids, &p, Noise::default(), &[], &[], Some(&mut show))
        .expect("denoise");

    let mut frames =
        vec![0u8; sh.frames as usize * p.height as usize * p.width as usize * 3];
    session
        .decode_video(&sh, &latents.video, &mut frames)
        .expect("decode video");
    let audio_t = sh.audio_t as usize;
    let mut samples = vec![0f32; 2 * audio_t * 800];
    session
        .decode_audio(&latents.audio, audio_t, &mut samples)
        .expect("decode audio");

    std::fs::write(format!("{out}.rgb"), &frames).unwrap();
    write_wav(&format!("{out}.wav"), &samples, audio_t * 800);
    eprintln!(
        "wrote {out}.rgb ({} x {}x{} rgb24) and {out}.wav",
        sh.frames, p.width, p.height
    );
}

/// 16-bit PCM, interleaved stereo, 32 kHz.
fn write_wav(path: &str, samples: &[f32], n: usize) {
    let bytes = (n * 4) as u32;
    let mut w = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    w.write_all(b"RIFF").unwrap();
    w.write_all(&(36 + bytes).to_le_bytes()).unwrap();
    w.write_all(b"WAVEfmt ").unwrap();
    // each field at its own width: packing format and channels into one u32 wrote format 2,
    // channels 1, which is neither what the payload is nor readable
    w.write_all(&16u32.to_le_bytes()).unwrap(); // fmt chunk size
    w.write_all(&1u16.to_le_bytes()).unwrap(); // PCM
    w.write_all(&2u16.to_le_bytes()).unwrap(); // stereo
    w.write_all(&32000u32.to_le_bytes()).unwrap(); // sample rate
    w.write_all(&128_000u32.to_le_bytes()).unwrap(); // bytes per second
    w.write_all(&4u16.to_le_bytes()).unwrap(); // block align
    w.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
    w.write_all(b"data").unwrap();
    w.write_all(&bytes.to_le_bytes()).unwrap();
    for i in 0..n {
        for ch in 0..2 {
            let v = samples[ch * n + i].clamp(-1.0, 1.0);
            w.write_all(&((v * 32767.0) as i16).to_le_bytes()).unwrap();
        }
    }
}
