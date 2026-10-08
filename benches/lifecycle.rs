mod support;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use h3_hrx::{model::*, plan, weights::Weights};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

type Plan = fn(
    &h3_hrx::checkpoint::Checkpoint,
    &mut std::collections::BTreeMap<String, h3_hrx::weights::Recipe>,
) -> h3_hrx::weights::Result<()>;

fn loading(c: &mut Criterion) {
    let mut group = c.benchmark_group("checkpoint");
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
            let manager = hrx::residency::ResidencyManager::new(support::budget_bytes()).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
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
                    let actual = support::read(&mut stream, buffer.binding());
                    assert_eq!(actual, weights.assemble(tensor).unwrap());
                }
                elapsed
            });
        });
    }
    group.finish();
}

/// The four large projections dominate initial transformer loading. Keep one block resident,
/// including allocation, packing and completed uploads, without loading a whole model.
fn block_loading(c: &mut Criterion) {
    for (name, relative, plan, hid, qkv, inner, ffn, bits) in [
        (
            "dit",
            h3_hrx::models::DIT_FL2VA,
            plan::dit::plan as Plan,
            HID,
            QKV,
            INNER,
            FFN,
            8,
        ),
        (
            "text",
            h3_hrx::models::TE,
            plan::te::plan as Plan,
            TE_HID,
            (TE_HEADS + 2 * TE_KV) * HEAD_DIM,
            TE_HEADS * HEAD_DIM,
            TE_FFN,
            8,
        ),
        (
            "video",
            h3_hrx::models::VIDEO_VAE,
            plan::vvae::plan as Plan,
            VAE_HID,
            3 * VAE_HID,
            VAE_HID,
            VAE_FFN,
            16,
        ),
    ] {
        let elem = bits / 8;
        let tensors = [
            ("blocks.0.qkv.q", qkv * gemm_pitch(hid, bits) * elem),
            ("blocks.0.out.q", hid * gemm_pitch(inner, bits) * elem),
            ("blocks.0.gu.q", 2 * ffn * gemm_pitch(hid, bits) * elem),
            ("blocks.0.down.q", hid * gemm_pitch(ffn, bits) * elem),
        ];
        let mut group = c.benchmark_group(format!("checkpoint/{name}"));
        group.throughput(Throughput::Bytes(
            tensors.iter().map(|(_, bytes)| *bytes as u64).sum(),
        ));
        group.bench_function("load_block", |b| {
            let path = support::checkpoint(relative);
            let manager = hrx::residency::ResidencyManager::new(support::budget_bytes()).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    // SAFETY: benchmarks require immutable checkpoint files.
                    let weights = unsafe { Weights::open(&path, plan) }.unwrap();
                    let start = Instant::now();
                    let buffers: Vec<_> = tensors
                        .iter()
                        .map(|(tensor, bytes)| weights.at(&mut stream, tensor, *bytes).unwrap())
                        .collect();
                    stream.synchronize().unwrap();
                    elapsed += start.elapsed();
                    for ((tensor, _), buffer) in tensors.iter().zip(buffers) {
                        let actual = support::read(&mut stream, buffer.binding());
                        assert_eq!(actual, weights.assemble(tensor).unwrap());
                    }
                }
                elapsed
            });
        });
        group.finish();
    }
}

/// Evict only a private disk fixture, never the shared checkpoint or global page cache.
#[cfg(target_os = "linux")]
fn cold_loading(c: &mut Criterion) {
    use h3_hrx::weights::Recipe;
    use std::{io::Write, os::fd::AsRawFd};

    let mut group = c.benchmark_group("checkpoint/cold_file");
    for (name, relative, plan, tensor, size) in [
        (
            "dit",
            h3_hrx::models::DIT_FL2VA,
            plan::dit::plan as Plan,
            "blocks.0.qkv.q",
            QKV * HID,
        ),
        (
            "dit",
            h3_hrx::models::DIT_FL2VA,
            plan::dit::plan as Plan,
            "blocks.0.gu.q",
            2 * FFN * HID,
        ),
        (
            "text",
            h3_hrx::models::TE,
            plan::te::plan as Plan,
            "blocks.0.qkv.q",
            (TE_HEADS + 2 * TE_KV) * HEAD_DIM * gemm_pitch(TE_HID, 8),
        ),
        (
            "text",
            h3_hrx::models::TE,
            plan::te::plan as Plan,
            "blocks.0.gu.q",
            2 * TE_FFN * gemm_pitch(TE_HID, 8),
        ),
    ] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_function(format!("{name}/{tensor}"), |b| {
            // /tmp may be tmpfs. Use the workspace filesystem for the disk fixture.
            let dir = tempfile::tempdir_in(concat!(env!("CARGO_MANIFEST_DIR"), "/target")).unwrap();
            let path = dir.path().join("weights.safetensors");
            let (rows, row_bytes, pitch_bytes, segments) = {
                // SAFETY: benchmark checkpoints remain immutable while mapped.
                let source = unsafe { Weights::open(support::checkpoint(relative), plan) }.unwrap();
                let Recipe::Rows {
                    rows,
                    row_bytes,
                    pitch_bytes,
                    segments,
                } = source.recipe(tensor).unwrap()
                else {
                    panic!("expected a row recipe");
                };
                // Text QKV and gate/up gather several separate source tensors.
                // Keep those source tensors separate and in checkpoint order.
                let names: std::collections::BTreeSet<_> =
                    segments.iter().map(|s| &s.tensor).collect();
                let mut entries: Vec<_> = names
                    .into_iter()
                    .map(|name| (name, source.file().at(name).unwrap()))
                    .collect();
                entries.sort_by_key(|(_, entry)| entry.offset);
                let mut header = serde_json::Map::new();
                let mut offset = 0;
                for (name, entry) in &entries {
                    header.insert(
                        (*name).clone(),
                        serde_json::json!({
                            "dtype":"I8", "shape":entry.shape,
                            "data_offsets":[offset, offset + entry.bytes]
                        }),
                    );
                    offset += entry.bytes;
                }
                let header = serde_json::to_vec(&header).unwrap();
                let mut file = std::fs::File::create(&path).unwrap();
                file.write_all(&(header.len() as u64).to_le_bytes())
                    .unwrap();
                file.write_all(&header).unwrap();
                for (_, entry) in entries {
                    file.write_all(source.file().bytes(entry)).unwrap();
                }
                file.sync_all().unwrap();
                (*rows, *row_bytes, *pitch_bytes, segments.clone())
            };
            let file = std::fs::File::open(&path).unwrap();
            let manager = hrx::residency::ResidencyManager::new(support::budget_bytes()).unwrap();
            let mut stream = support::stream(Some(manager.budget()));
            b.iter_custom(|iterations| {
                let mut elapsed = Duration::ZERO;
                for _ in 0..iterations {
                    // No mapping survives the prior iteration, and sync_all made the file clean.
                    // SAFETY: this fd is live; advice touches only this private fixture.
                    let status = unsafe {
                        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
                    };
                    assert_eq!(status, 0, "fixture eviction failed");
                    // SAFETY: the fixture remains immutable until its mapping is dropped.
                    let weights = unsafe {
                        Weights::open(&path, |_, out| {
                            out.insert(
                                tensor.into(),
                                Recipe::Rows {
                                    rows,
                                    row_bytes,
                                    pitch_bytes,
                                    segments: segments.clone(),
                                },
                            );
                            Ok(())
                        })
                    }
                    .unwrap();
                    let start = Instant::now();
                    let buffer = weights.at(&mut stream, tensor, size).unwrap();
                    stream.synchronize().unwrap();
                    elapsed += start.elapsed();
                    let actual = support::read(&mut stream, buffer.binding());
                    assert_eq!(actual, weights.assemble(tensor).unwrap());
                }
                elapsed
            });
        });
    }
    group.finish();
}

#[cfg(not(target_os = "linux"))]
fn cold_loading(_: &mut Criterion) {}

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
criterion_group! { name = benches; config = support::criterion(); targets = loading, block_loading, cold_loading, eviction }
criterion_main!(benches);
