mod support;
use criterion::{criterion_group, criterion_main, Criterion};
use h3_hrx::{plan, weights::Weights};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint");
    type Plan = fn(
        &h3_hrx::checkpoint::Checkpoint,
        &mut std::collections::BTreeMap<String, h3_hrx::weights::Recipe>,
    ) -> h3_hrx::weights::Result<()>;
    for (name, relative, plan, tensor) in [
        (
            "dit",
            h3_hrx::models::DIT_FL2VA,
            plan::dit::plan as Plan,
            "blocks.0.gu.q",
        ),
        (
            "text",
            h3_hrx::models::TE,
            plan::te::plan as Plan,
            "blocks.0.gu.q",
        ),
        (
            "video",
            h3_hrx::models::VIDEO_VAE,
            plan::vvae::plan as Plan,
            "blocks.0.qkv.q",
        ),
        (
            "audio",
            h3_hrx::models::AUDIO_VAE,
            plan::avae::plan as Plan,
            "audio.conv_pre.w",
        ),
    ] {
        group.bench_function(format!("{name}/map_and_plan"), |b| {
            let path = support::checkpoint(relative);
            // SAFETY: the caller keeps the local checkpoints immutable.
            b.iter(|| black_box(unsafe { Weights::open(&path, plan) }.unwrap()));
        });
        let mut weights = None;
        group.bench_function(format!("{name}/pack/{tensor}"), |b| {
            let weights = weights.get_or_insert_with(|| {
                unsafe { Weights::open(support::checkpoint(relative), plan) }.unwrap()
            });
            let expected = weights.assemble(tensor).unwrap();
            support::measure(
                b,
                || weights.assemble(tensor).unwrap(),
                |actual| assert_eq!(*actual, expected),
            );
        });
        group.bench_function(format!("{name}/pack_and_upload/{tensor}"), |b| {
            let path = support::checkpoint(relative);
            let mut stream = hrx::Stream::open().unwrap();
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    // A fresh owner prevents Weights::at's memoized device buffer from turning
                    // this into a lookup benchmark. Mapping/plan construction is excluded here.
                    let weights = unsafe { Weights::open(&path, plan) }.unwrap();
                    let size = weights.recipe(tensor).unwrap().device_bytes();
                    let start = Instant::now();
                    let buffer = weights.at(&mut stream, tensor, size).unwrap();
                    stream.synchronize().unwrap();
                    elapsed += start.elapsed();
                    let actual = stream
                        .read(buffer.binding())
                        .unwrap()
                        .wait(&mut stream)
                        .unwrap();
                    assert_eq!(actual, weights.assemble(tensor).unwrap());
                }
                elapsed
            });
        });
    }
    group.finish();
}

fn eviction(c: &mut Criterion) {
    let mut state = None;
    c.bench_function("residency/audio_evict_and_reload", |b| {
        let (runtime, session, latents, expected) = state.get_or_insert_with(|| {
            let runtime = support::Runtime::new();
            let mut session = runtime.session(
                support::config(false),
                h3_hrx::SessionOptions {
                    residency: h3_hrx::ResidencyPolicy::Budgeted,
                    ..Default::default()
                },
            );
            let latents = support::values(64 * 5, 0.5);
            let mut out = vec![0.; 2 * 5 * h3_hrx::avae::HOP];
            session.decode_audio(&latents, 5, &mut out).unwrap();
            (runtime, session, latents, out)
        });
        b.iter_custom(|iterations| {
            let mut total = Duration::ZERO;
            for _ in 0..iterations {
                let mut out = vec![0.; expected.len()];
                let before = runtime.manager.statistics().evictions;
                let start = Instant::now();
                // The stream retains bounded staging outside the evictable model unit.
                let pressure = runtime
                    .manager
                    .budget()
                    .reserve(support::budget_bytes() - (256 << 20))
                    .unwrap();
                drop(pressure);
                session.decode_audio(latents, 5, &mut out).unwrap();
                total += start.elapsed();
                assert!(
                    runtime.manager.statistics().evictions > before,
                    "no eviction occurred"
                );
                assert_eq!(out, *expected);
            }
            total
        });
    });
}
criterion_group! { name = benches; config = support::criterion(); targets = loading, eviction }
criterion_main!(benches);
