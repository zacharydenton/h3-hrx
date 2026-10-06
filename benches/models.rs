//! Warm checkpoint-backed workloads using the standard model cache.
mod support;
use criterion::{criterion_group, criterion_main, Bencher, Criterion};
use h3_hrx::{
    compile::Compiler,
    dispatch::{Classes, Profile, Sink},
    model::*,
    stack::{Constants, LayerCond, Stack, StackDims},
    weights::Weights,
    Config, ResidencyPolicy, Session, SessionOptions,
};
use std::{
    hint::black_box,
    path::PathBuf,
    time::{Duration, Instant},
};
use support::checkpoint;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn audio(b: &mut Bencher) -> Result<()> {
    let config = Config {
        audio_vae: Some(checkpoint(h3_hrx::models::AUDIO_VAE)),
        kernel_sources: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"),
        ..Config::default()
    };
    let residency = hrx::residency::ResidencyManager::new(support::budget_bytes_or(32))?;
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(residency.budget()),
        ..Default::default()
    })?;
    // SAFETY: the caller supplies an immutable local model snapshot.
    let mut session = unsafe {
        Session::new_in(
            config,
            SessionOptions {
                residency: ResidencyPolicy::Budgeted,
                ..Default::default()
            },
            &context,
        )
    }?;
    let input: Vec<f32> = (0..6400).map(|i| ((i % 97) as f32 - 48.) / 97.).collect();
    let mut run = || -> Result<Vec<f32>> {
        let (audio, t) = session.encode_audio(&input, 3200)?;
        let mut decoded = vec![0.; 2 * t * h3_hrx::avae::HOP];
        session.decode_audio(&audio, t, &mut decoded)?;
        Ok(decoded)
    };
    let expected = run()?;
    assert!(expected.iter().all(|v| v.is_finite()));
    // Session methods finish their GPU work and read back before returning.
    // Include allocation, transfer and output disposal in this API-level measurement.
    b.iter(|| {
        black_box(run().unwrap());
    });
    assert_eq!(run()?, expected, "audio replay changed");
    Ok(())
}

fn stack(b: &mut Bencher, tokens: usize, layers: usize, graph_mode: bool) -> Result<()> {
    let residency = hrx::residency::ResidencyManager::new(support::budget_bytes_or(32))?;
    let mut stream = hrx::Stream::open()?.with_memory_budget(residency.budget());
    let compiler = Compiler::new(
        None,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("kernels"),
    );
    // SAFETY: The caller supplies an immutable checkpoint for this process.
    let weights = unsafe {
        Weights::open(
            checkpoint(h3_hrx::models::DIT_FL2VA),
            h3_hrx::plan::dit::plan,
        )
    }?;
    let constants = Constants::new(&mut stream)?;
    let dims = StackDims {
        hidden: HID,
        heads: HEADS,
        kv_heads: HEADS,
        head_dim: HEAD_DIM,
        ffn: FFN,
        rope_dim: ROPE_DIM,
        classes: CLASSES,
        wbits: 8,
        eps: 1e-5,
        bias: false,
        gate_first: true,
        causal: false,
        attn_i4: false,
        attn_qk_bits: 8,
        bf16: false,
    };
    let mut stack = Stack::new(
        &compiler,
        &mut stream,
        dims,
        tokens,
        layers,
        &weights,
        |i| format!("blocks.{i}."),
        true,
        constants.ones.clone(),
        "bench",
    )?;
    let x = stream.allocate_zeroed(stack.capacity() * HID * 4)?;
    let mut classes = Classes::zeroed(&mut stream, tokens)?;
    classes.write(
        &mut stream,
        &(0..tokens)
            .map(|r| (r % CLASSES) as i32)
            .collect::<Vec<_>>(),
        CLASSES,
    )?;
    let upload = |stream: &mut hrx::Stream, data: Vec<f32>| -> hrx::Result<hrx::Buffer> {
        let b = stream.allocate(data.len() * 4)?;
        stream.upload_blocking(b.binding(), bytemuck::cast_slice(&data))?;
        Ok(b)
    };
    let cos = upload(
        &mut stream,
        (0..tokens * ROPE_HALF)
            .map(|i| ((i % 271) as f32 / 271.).cos())
            .collect(),
    )?;
    let sin = upload(
        &mut stream,
        (0..tokens * ROPE_HALF)
            .map(|i| ((i % 271) as f32 / 271.).sin())
            .collect(),
    )?;
    let table = upload(
        &mut stream,
        (0..CLASSES * 2 * HID)
            .map(|i| (i % 13) as f32 / 100.)
            .collect(),
    )?;
    let gate = upload(
        &mut stream,
        (0..CLASSES * HID)
            .map(|i| 0.1 + (i % 7) as f32 / 20.)
            .collect(),
    )?;
    let cond = |_| LayerCond {
        table_msa: table.binding(),
        gate_msa: gate.binding(),
        table_mlp: table.binding(),
        gate_mlp: gate.binding(),
    };
    let input: Vec<f32> = (0..tokens * HID)
        .map(|i| ((i * 17) % 127) as f32 / 64. - 1.)
        .collect();

    // Compile and warm the eager path before recording or timing.
    let mut profile = Profile::from_env();
    stream.upload_blocking(x.binding(), bytemuck::cast_slice(&input))?;
    stack.forward(
        &mut stream,
        &mut profile,
        x.binding(),
        classes.all(),
        cos.binding(),
        sin.binding(),
        &cond,
        0,
        None,
    )?;
    let mut expected = vec![0u8; input.len() * 4];
    stream.read_blocking(x.binding(), &mut expected)?;
    assert!(expected
        .as_chunks::<4>()
        .0
        .iter()
        .all(|v| f32::from_le_bytes(*v).is_finite()));
    let mut graph = if graph_mode {
        let mut recording = stream.graph()?;
        stack.emit(
            &mut Sink::Graph {
                graph: &mut recording,
                after: Default::default(),
            },
            &mut profile,
            x.binding(),
            classes.all(),
            cos.binding(),
            sin.binding(),
            &cond,
            0,
            None,
            false,
        )?;
        Some(recording.finish()?)
    } else {
        None
    };
    b.iter_custom(|iterations| {
        let mut elapsed = Duration::ZERO;
        for _ in 0..iterations {
            // Each forward mutates the residual. Restore identical input outside timing.
            stream
                .upload_blocking(x.binding(), bytemuck::cast_slice(&input))
                .unwrap();
            stream.synchronize().unwrap();
            let start = Instant::now();
            if let Some(recorded) = &mut graph {
                stream.launch(recorded).unwrap();
            } else {
                stack
                    .forward(
                        &mut stream,
                        &mut profile,
                        x.binding(),
                        classes.all(),
                        cos.binding(),
                        sin.binding(),
                        &cond,
                        0,
                        None,
                    )
                    .unwrap();
            }
            stream.synchronize().unwrap();
            elapsed += start.elapsed();
        }
        let mut actual = vec![0u8; expected.len()];
        stream.read_blocking(x.binding(), &mut actual).unwrap();
        assert_eq!(actual, expected, "stack replay changed");
        elapsed
    });
    Ok(())
}

fn models(c: &mut Criterion) {
    c.bench_function("audio/roundtrip/3200", |b| audio(b).unwrap());
    let mut group = c.benchmark_group("dit_stack");
    // 4096 also covers the head-major integer attention path.
    for tokens in [256, 2048, 4096] {
        for graph in [false, true] {
            let mode = if graph { "graph" } else { "eager" };
            group.bench_function(format!("{mode}/{tokens}/1_layer"), |b| {
                stack(b, tokens, 1, graph).unwrap()
            });
        }
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = models
}
criterion_main!(benches);
