//! Separate warm audio encode/decode and intrusive per-kernel timing.
use h3_hrx::{avae::AudioVae, compile::Compiler, dispatch::Profile};
use std::{path::Path, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let checkpoint = args
        .get(1)
        .ok_or("expected audio checkpoint and output path")?;
    let output = args.get(2).ok_or("expected output path")?;
    let residency = hrx::residency::ResidencyManager::new(64 << 30)?;
    let mut stream = hrx::Stream::open()?;
    if std::env::var("H3_PROFILE_BUDGET").as_deref() != Ok("0") {
        stream = stream.with_memory_budget(residency.budget());
    }
    let compiler = Compiler::new(
        std::env::var_os("HRX_LOOM_LIBRARY").map(Into::into),
        Path::new(env!("CARGO_MANIFEST_DIR")).join("kernels"),
    );
    // SAFETY: the supplied local checkpoint remains immutable during this diagnostic.
    let mut vae = unsafe { AudioVae::open(&mut stream, checkpoint) }?;
    let input: Vec<f32> = (0..6400).map(|i| ((i % 97) as f32 - 48.) / 97.).collect();
    let mut profile = Profile::from_env();
    let mut expected = None;
    for round in 0..4 {
        let start = Instant::now();
        let (encoded, t) = vae.encode(&mut stream, &compiler, &mut profile, &input, 3200)?;
        let encode_ms = start.elapsed().as_secs_f64() * 1000.;
        let encode_stages_us = profile.take();
        let mut decoded = vec![0.; 2 * t * h3_hrx::avae::HOP];
        let start = Instant::now();
        vae.decode(
            &mut stream,
            &compiler,
            &mut profile,
            &encoded,
            t,
            &mut decoded,
        )?;
        let decode_ms = start.elapsed().as_secs_f64() * 1000.;
        let decode_stages_us = profile.take();
        assert!(decoded.iter().all(|v| v.is_finite()));
        if let Some(expected) = &expected {
            assert_eq!(expected, &decoded, "replay changed");
        } else {
            std::fs::write(output, bytemuck::cast_slice(&decoded))?;
            expected = Some(decoded);
        }
        if round > 0 {
            println!("{{\"round\":{round},\"intrusive_profile\":{},\"encode_ms\":{encode_ms},\"decode_ms\":{decode_ms},\"encode_stages_us\":{encode_stages_us:?},\"decode_stages_us\":{decode_stages_us:?}}}",profile.on);
        }
    }
    Ok(())
}
