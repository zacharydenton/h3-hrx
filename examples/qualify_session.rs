//! Capture a pre-migration session fixture, then verify without replacing it.
use h3_hrx::{
    Clip, Config, DenoiseParams, Noise, ResidencyPolicy, Session, SessionOptions, Tokenizer,
};
use std::{io::Write, path::Path, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(4..=5).contains(&args.len()) || !matches!(args[1].as_str(), "record" | "compare") {
        return Err("usage: qualify_session record|compare FIXTURE_DIR PINNED_SNAPSHOT [stage-scoped|budgeted]".into());
    }
    let policy = match args.get(4).map(String::as_str).unwrap_or("stage-scoped") {
        "stage-scoped" => ResidencyPolicy::StageScoped,
        "budgeted" => ResidencyPolicy::Budgeted,
        _ => return Err("unknown residency policy".into()),
    };
    let record = args[1] == "record";
    let directory = Path::new(&args[2]);
    let snapshot = Path::new(&args[3]);
    let check = |name: &str, bytes: &[u8]| -> Result<(), Box<dyn std::error::Error>> {
        let path = directory.join(name);
        if record {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)?
                .write_all(bytes)?;
        } else if std::fs::read(&path)? != bytes {
            return Err(format!("{} differs from the recorded session", path.display()).into());
        }
        eprintln!(
            "{} {name}: {} bytes",
            if record { "recorded" } else { "matched" },
            bytes.len()
        );
        Ok(())
    };
    let floats = |name: &str, values: &[f32]| -> Result<(), Box<dyn std::error::Error>> {
        if values.iter().any(|v| !v.is_finite()) {
            return Err(format!("nonfinite {name}").into());
        }
        check(name, bytemuck::cast_slice(values))
    };
    let config = Config {
        dit: Some(snapshot.join(h3_hrx::models::DIT_FL2VA)),
        te: Some(snapshot.join(h3_hrx::models::TE)),
        video_vae: Some(snapshot.join(h3_hrx::models::VIDEO_VAE)),
        audio_vae: Some(snapshot.join(h3_hrx::models::AUDIO_VAE)),
        ..Config::default()
    };
    let residency = hrx::residency::ResidencyManager::new(80 << 30)?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(residency.budget()),
        ..Default::default()
    })?;
    context.runtime().start_trace(64)?;
    // SAFETY: Qualification uses immutable, checksum-verified Hub snapshots.
    let mut session = unsafe {
        Session::new_in(
            config,
            SessionOptions {
                residency: policy,
                ..SessionOptions::default()
            },
            &context,
        )
    }?;
    let samples: Vec<f32> = (0..6400).map(|i| ((i % 97) as f32 - 48.) / 97.).collect();
    let start = Instant::now();
    let (audio, audio_t) = session.encode_audio(&samples, 3200)?;
    floats("encoded-audio.f32", &audio)?;
    let mut decoded = vec![0.; 2 * audio_t * h3_hrx::avae::HOP];
    session.decode_audio(&audio, audio_t, &mut decoded)?;
    floats("decoded-audio.f32", &decoded)?;
    eprintln!("audio boundaries: {:.3}s", start.elapsed().as_secs_f64());

    let pixels: Vec<f32> = (0..256 * 256 * 3)
        .map(|i| (i % 251) as f32 / 250.)
        .collect();
    let start = Instant::now();
    let (video, _) = session.encode_video(Clip {
        pixels: &pixels,
        frames: 1,
        height: 256,
        width: 256,
    })?;
    floats("encoded-video.f32", &video)?;
    eprintln!("video encode: {:.3}s", start.elapsed().as_secs_f64());

    let ids = Tokenizer::new()?.encode("a red fox in snow")?;
    let params = DenoiseParams {
        width: 256,
        height: 256,
        frames: 5,
        steps: 2,
        sampler: h3_hrx::Sampler::Euler,
        seed: 7,
        ..DenoiseParams::default()
    };
    let start = Instant::now();
    if policy == ResidencyPolicy::Budgeted {
        // A callback may enqueue other compute, but that work must not execute
        // while this native stage owns the shared lane. It must not wait here.
        let probe = context
            .runtime()
            .allocate(16, hrx::execution::MemoryPlacement::GpuLocal)?;
        let mut graph = context.runtime().graph();
        graph.fill(probe.view(), 7)?;
        let graph = graph.prepare()?;
        let caller = std::thread::current().id();
        let mut competing = None;
        let cancelled = session.denoise(
            &ids,
            &params,
            Noise::default(),
            &[],
            &[],
            Some(&mut |_, _, _| {
                assert_eq!(std::thread::current().id(), caller);
                let completion = graph.submit().unwrap();
                assert!(!completion.is_complete());
                competing = Some(completion);
                true
            }),
        );
        assert!(matches!(cancelled, Err(h3_hrx::Error::Cancelled)));
        competing.expect("progress callback ran").wait()?;
        eprintln!("budgeted denoise cancellation drained; retrying retained units");
    }
    let latents = session.denoise(&ids, &params, Noise::default(), &[], &[], None)?;
    floats("generated-video.f32", &latents.video)?;
    floats("generated-audio.f32", &latents.audio)?;
    eprintln!("denoise: {:.3}s", start.elapsed().as_secs_f64());
    let shape = Session::shape_for(256, 256, 5).ok_or("invalid fixture shape")?;
    let mut rgb = vec![0; 5 * 256 * 256 * 3];
    let start = Instant::now();
    session.decode_video(&shape, &latents.video, &mut rgb)?;
    check("generated.rgb", &rgb)?;
    eprintln!("video decode: {:.3}s", start.elapsed().as_secs_f64());
    eprintln!(
        "post-stage reserved bytes: {}",
        residency.statistics().reserved_bytes
    );
    if policy == ResidencyPolicy::Budgeted {
        let before = residency.statistics();
        let pressure = residency
            .budget()
            .reserve(before.budget_bytes - (256 << 20))?;
        assert!(residency.statistics().evictions > before.evictions);
        drop(pressure);
        assert!(residency.statistics().reserved_bytes < 256 << 20);
        // Audio is the oldest idle unit, so pressure must evict it. Reopening
        // and replaying verifies native ownership after eviction, not just counters.
        let mut replay = vec![0.; decoded.len()];
        session.decode_audio(&audio, audio_t, &mut replay)?;
        assert_eq!(
            bytemuck::cast_slice::<f32, u8>(&replay),
            bytemuck::cast_slice::<f32, u8>(&decoded)
        );
        eprintln!(
            "budgeted eviction/reload matches; {} evictions",
            residency.statistics().evictions
        );
    }
    drop(session);
    assert_eq!(residency.statistics().reserved_bytes, 0);
    assert_eq!(residency.statistics().resources, 0);
    let trace = context.runtime().finish_trace().expect("trace started");
    // Encode/decode audio, encode video, denoise, and decode video: five stages.
    assert!(trace.events.len() >= 5);
    assert!(trace
        .events
        .iter()
        .all(|event| event.lane == 1 && !event.failed));
    eprintln!(
        "shared compute trace: {} completed regions",
        trace.events.len()
    );
    Ok(())
}
