//! Warm Session audio VAE roundtrip with cached production weights.
use h3_hrx::{Config, Session, SessionOptions};
use std::{path::Path, time::Instant};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let snapshot = Path::new(&args[1]);
    let config = Config {
        dit: Some(snapshot.join(h3_hrx::models::DIT_FL2VA)),
        te: Some(snapshot.join(h3_hrx::models::TE)),
        video_vae: Some(snapshot.join(h3_hrx::models::VIDEO_VAE)),
        audio_vae: Some(snapshot.join(h3_hrx::models::AUDIO_VAE)),
        ..Config::default()
    };
    let context = hrx::inference::ModelContext::new(Default::default())?;
    // SAFETY: the caller supplies an immutable local model snapshot.
    let mut session = unsafe { Session::new_in(config, SessionOptions::default(), &context) }?;
    let input: Vec<f32> = (0..6400).map(|i| ((i % 97) as f32 - 48.) / 97.).collect();
    let mut run = || -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let (audio, t) = session.encode_audio(&input, 3200)?;
        let mut decoded = vec![0.; 2 * t * h3_hrx::avae::HOP];
        session.decode_audio(&audio, t, &mut decoded)?;
        Ok(decoded)
    };
    let expected = run()?;
    assert!(expected.iter().all(|v| v.is_finite()));
    std::fs::write(&args[2], bytemuck::cast_slice(&expected))?;
    let mut samples = Vec::new();
    for _ in 0..11 {
        let start = Instant::now();
        let actual = run()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.);
        assert_eq!(actual, expected, "audio replay changed");
    }
    samples.sort_by(f64::total_cmp);
    println!("{{\"scope\":\"warm Session audio encode+decode, 3200 samples/channel\", \"median_ms\":{}, \"samples_ms\":{:?}}}", samples[5], samples);
    Ok(())
}
