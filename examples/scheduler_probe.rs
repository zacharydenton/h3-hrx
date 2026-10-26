//! Compare warm H3 output and latency with small jobs in the same ModelContext.
use h3_hrx::{Config, DenoiseParams, Noise, Session, SessionOptions, Tokenizer};
use hrx::execution::MemoryPlacement;
use std::{
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

struct Stop<'a>(&'a AtomicBool);
impl Drop for Stop<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let decode = std::env::args().any(|arg| arg == "--decode");
    let context = hrx::inference::ModelContext::new(Default::default())?;
    let ids =
        Tokenizer::new()?.encode("a red fox trotting through a snowy forest at dawn, cinematic")?;
    let params = DenoiseParams {
        height: 256,
        width: 256,
        frames: 5,
        steps: 2,
        seed: 7,
        ..Default::default()
    };
    let run = |session: &mut Session| -> h3_hrx::Result<(Vec<u8>, Vec<u8>)> {
        if decode {
            let shape = Session::shape_for(256, 256, 5).unwrap();
            let latents: Vec<f32> =
                (0..24 * shape.latent_t as usize * shape.lat_h as usize * shape.lat_w as usize)
                    .map(|i| ((i % 251) as f32 - 125.0) / 125.0)
                    .collect();
            let mut pixels = vec![0; shape.frames as usize * 256 * 256 * 3];
            session.decode_video(&shape, &latents, &mut pixels)?;
            Ok((pixels, Vec::new()))
        } else {
            let latents = session.denoise(&ids, &params, Noise::default(), &[], &[], None)?;
            Ok((
                bytemuck::cast_slice::<f32, u8>(&latents.video).to_vec(),
                bytemuck::cast_slice::<f32, u8>(&latents.audio).to_vec(),
            ))
        }
    };
    // SAFETY: checkpoints remain immutable throughout this diagnostic run.
    let mut session =
        unsafe { Session::new_in(Config::default(), SessionOptions::default(), &context) }?;
    eprintln!("warming H3 and its recorded graphs");
    let expected = run(&mut session)?;
    let start = Instant::now();
    let baseline = run(&mut session)?;
    let baseline_seconds = start.elapsed().as_secs_f64();
    assert_eq!(expected.0, baseline.0);
    assert_eq!(expected.1, baseline.1);
    let output = context
        .runtime()
        .allocate(64, MemoryPlacement::HostVisible)?;
    let mut graph = context.runtime().graph();
    graph.fill(output.view(), 0x5a)?;
    let graph = graph.prepare()?;
    graph.submit()?.wait()?;
    let stop = AtomicBool::new(false);
    let start = Instant::now();
    let (actual, latencies) = std::thread::scope(|scope| {
        let worker = scope.spawn(|| -> hrx::Result<Vec<f64>> {
            let mut latencies = Vec::new();
            while !stop.load(Ordering::Acquire) {
                let start = Instant::now();
                let done = graph.submit()?;
                if !done.wait_timeout(Duration::from_secs(30))? {
                    return Err(hrx::Error::Message(
                        "short graph exceeded 30 seconds".into(),
                    ));
                }
                latencies.push(start.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(&*output.map_read()?, &[0x5a; 64]);
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(latencies)
        });
        let stop_on_unwind = Stop(&stop);
        let actual = run(&mut session);
        drop(stop_on_unwind);
        (actual, worker.join().expect("probe worker panicked"))
    });
    let contended_seconds = start.elapsed().as_secs_f64();
    let actual = actual?;
    let mut latencies = latencies?;
    assert!(!latencies.is_empty());
    assert_eq!(expected.0, actual.0);
    assert_eq!(expected.1, actual.1);
    latencies.sort_by(f64::total_cmp);
    println!(
        "{}",
        serde_json::json!({
            "workload": if decode { "video_decode" } else { "denoise" },
            "baseline_seconds": baseline_seconds,
            "contended_seconds": contended_seconds,
            "short_jobs": latencies.len(),
            "latency_p50_ms": latencies[latencies.len() / 2],
            "latency_p95_ms": latencies[(latencies.len() - 1) * 95 / 100],
            "latency_max_ms": latencies.last().unwrap(),
            "video_sha256": hrx::bundle::digest(&actual.0),
            "audio_sha256": hrx::bundle::digest(&actual.1),
            "byte_identical": true,
        })
    );
    Ok(())
}
