//! Full and padded-head attention with an independent sampled FP64 oracle.
mod support;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use h3_hrx::{
    compile::{num, Cfg},
    dispatch::{emit, Profile, Sink},
};
use half::f16;
use std::time::{Duration, Instant};

fn reference(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    tokens: usize,
    heads: usize,
    active: usize,
) -> Vec<(usize, f64)> {
    let stride = heads * 128;
    let mut expected = Vec::new();
    for row in [0, tokens / 2, tokens - 1] {
        for head in [0, heads - 1] {
            let mut scores = Vec::with_capacity(tokens);
            for key in 0..tokens {
                let score = (0..128)
                    .map(|c| {
                        q[row * stride + head * 128 + c].to_f64()
                            * k[key * stride + head * 128 + c].to_f64()
                    })
                    .sum::<f64>()
                    / (active as f64).sqrt();
                scores.push(score);
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores
                .into_iter()
                .map(|score| (score - maximum).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            for channel in [0, 31, 64, 127] {
                let result = weights
                    .iter()
                    .enumerate()
                    .map(|(key, w)| w * v[key * stride + head * 128 + channel].to_f64())
                    .sum::<f64>()
                    / total;
                expected.push((row * stride + head * 128 + channel, result));
            }
        }
    }
    expected
}
fn attention(c: &mut Criterion) {
    attention_layout(c, false);
    attention_layout(c, true);
}
fn attention_layout(c: &mut Criterion, vision: bool) {
    let mut group = c.benchmark_group(if vision {
        "attention_vision"
    } else {
        "attention_f16_tiles"
    });
    let token_counts = if vision {
        [16usize, 65, 257, 1620, 4032, 8193]
    } else {
        [257usize, 2048, 4095, 4096, 4097, 8193]
    };
    for tokens in token_counts {
        let heads = if vision { 16 } else { 4 };
        let active = if vision { 72 } else { 128 };
        let stride = heads * 128;
        let capacity = (tokens + 16).div_ceil(128) * 128;
        let values = |seed: usize| {
            (0..capacity * stride)
                .map(|i| {
                    if i < tokens * stride && i % 128 < active {
                        f16::from_f32((((i * 73 + seed) % 997) as f32 - 498.) / 512.)
                    } else {
                        f16::ZERO
                    }
                })
                .collect::<Vec<_>>()
        };
        let q = values(3);
        let k = values(17);
        let v = values(37);
        let expected = reference(&q, &k, &v, tokens, heads, active);
        let mut variants = if vision {
            [
                ("full128", "attention_mha_lds_f16_wmma"),
                ("padded80", "attention_mha80_lds_f16_wmma"),
            ]
        } else {
            [
                ("waves4", "attention_mha_lds_f16_wmma"),
                ("waves8", "attention_mha8_lds_f16_wmma"),
            ]
        };
        if std::env::var_os("H3_BENCH_ATTENTION_REVERSE").is_some() {
            variants.reverse();
        }
        for (variant, stem) in variants {
            group.bench_function(BenchmarkId::new(variant, tokens), |b| {
                let manager = hrx::residency::ResidencyManager::new(512 << 20).unwrap();
                let mut stream = support::stream(Some(manager.budget()));
                let compiler = support::compiler();
                let cfg: Cfg = [
                    ("q_stride", stride.to_string()),
                    ("kv_stride", stride.to_string()),
                    ("out_stride", stride.to_string()),
                    ("tokens", tokens.to_string()),
                    ("token_capacity", capacity.to_string()),
                    ("scale", num(1. / (active as f64).sqrt())),
                ]
                .into_iter()
                .map(|(k, v)| (format!("h3.{stem}.{k}"), v))
                .collect();
                let kernel = compiler
                    .get(
                        &mut stream,
                        "attention_mha_family",
                        &format!("h3_{stem}"),
                        &cfg,
                    )
                    .unwrap();
                compiler.flush(&mut stream).unwrap();
                let q = stream.allocate_from(bytemuck::cast_slice(&q)).unwrap();
                let k = stream.allocate_from(bytemuck::cast_slice(&k)).unwrap();
                let v = stream.allocate_from(bytemuck::cast_slice(&v)).unwrap();
                let output = stream.allocate(tokens * stride * 2).unwrap();
                let bindings = [q.binding(), k.binding(), v.binding(), output.binding()];
                let required = bindings.map(|view| view.len());
                let mut profile = Profile::from_env();
                let mut run = |stream: &mut hrx::Stream| {
                    emit(
                        &mut Sink::Stream(stream),
                        &kernel,
                        Some(&mut profile),
                        if vision {
                            "vision attention"
                        } else {
                            "head128 attention"
                        },
                        &[tokens as u32, heads as u32],
                        &[tokens as u32],
                        &bindings,
                        &required,
                    )
                    .unwrap();
                    stream.synchronize().unwrap();
                };
                run(&mut stream);
                let actual = support::read(&mut stream, output.binding());
                for &(index, want) in &expected {
                    let got =
                        f16::from_le_bytes([actual[index * 2], actual[index * 2 + 1]]).to_f64();
                    assert!(
                        got.is_finite() && (got - want).abs() <= 0.003 + 0.01 * want.abs(),
                        "{tokens}/{variant}/{index}: {got} vs {want}"
                    );
                }
                support::report_digest(&format!("attention_f16/{tokens}/{variant}"), &actual);
                b.iter_custom(|iterations| {
                    let mut elapsed = Duration::ZERO;
                    for _ in 0..iterations {
                        let start = Instant::now();
                        run(&mut stream);
                        elapsed += start.elapsed();
                    }
                    if std::env::var_os("H3_BENCH_DETAILS").is_some() {
                        eprintln!(
                            "attention-tile {tokens}/{variant}: {:.6} ms",
                            elapsed.as_secs_f64() * 1000. / iterations as f64
                        );
                    }
                    elapsed
                });
                assert_eq!(support::read(&mut stream, output.binding()), actual);
                let report = profile.report(0.0);
                if !report.is_empty() {
                    eprintln!("attention profile {tokens}/{variant}: {report}");
                }
            });
        }
    }
    group.finish();
}
criterion_group! {name=benches;config=support::criterion();targets=attention}
criterion_main!(benches);
