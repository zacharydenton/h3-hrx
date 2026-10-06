mod support;
use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use h3_hrx::{
    media_context::{Frame, Media, MediaEntry},
    refmod::{ApplyOptions, RefMod},
    PreparedPresentation, Tokenizer,
};
use std::{hint::black_box, path::PathBuf};

fn host(c: &mut Criterion) {
    let tokenizer = Tokenizer::new().unwrap();
    let mut group = c.benchmark_group("tokenize");
    for repeats in [1, 32, 128] {
        let text = support::PROMPT.repeat(repeats);
        group.throughput(Throughput::Bytes(text.len() as u64));
        group.bench_function(format!("prompt_x{repeats}"), |b| {
            b.iter(|| tokenizer.encode(black_box(&text)).unwrap())
        });
    }
    group.finish();
    let frame = Frame {
        pixels: support::pixels(256, 256, 1).into(),
        width: 256,
        height: 256,
    };
    let entries = vec![
        MediaEntry {
            media: Media::Picture(frame.clone()),
            role: "first_frame".into(),
            metadata: Default::default(),
        },
        MediaEntry {
            media: Media::Video(vec![(0., frame.clone()), (0.5, frame)]),
            role: "reference".into(),
            metadata: Default::default(),
        },
        MediaEntry {
            media: Media::Audio(support::values(64000, 0.2).into()),
            role: "reference".into(),
            metadata: Default::default(),
        },
    ];
    let shape = h3_hrx::shape_for(480, 864, 124).unwrap();
    c.bench_function("presentation/mixed_media", |b| {
        b.iter(|| {
            PreparedPresentation::new(&tokenizer, black_box(&entries), support::PROMPT, &shape)
                .unwrap()
        })
    });
    for (name, sw, sh, dw, dh) in [
        ("1080p_to_480p", 1920, 1080, 864, 480),
        ("1080p_to_768p", 1920, 1080, 1344, 768),
        ("upscale", 640, 480, 864, 768),
        ("identity", 864, 480, 864, 480),
        ("large_downscale", 3000, 2000, 288, 192),
    ] {
        let rgb: Vec<_> = (0..sw * sh * 3).map(|i| (i % 251) as u8).collect();
        c.bench_function(&format!("resize/{name}"), |b| {
            b.iter(|| h3_hrx::resize::pil_bilinear(black_box(&rgb), sw, sh, dw, dh))
        });
    }
    for (name, sw, sh, dw, dh) in [
        ("1080p_to_480p", 1920, 1080, 864, 480),
        ("portrait_to_landscape", 1080, 1920, 864, 480),
        ("landscape_to_portrait", 1920, 1080, 480, 864),
        ("upscale", 320, 240, 864, 480),
    ] {
        c.bench_function(&format!("world_resize/{name}"), |b| {
            let rgb: Vec<_> = (0..sw * sh * 3).map(|i| (i % 251) as u8).collect();
            b.iter(|| h3_hrx::resize::world_first_frame(black_box(&rgb), sw, sh, dw, dh))
        });
    }
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/refmod/combined.safetensors");
    c.bench_function("refmod/load", |b| {
        b.iter(|| RefMod::load(black_box(&path)).unwrap())
    });
    let refmod = RefMod::load(path).unwrap();
    c.bench_function("refmod/prepare_strength_and_copies", |b| {
        b.iter(|| {
            refmod
                .prepare(ApplyOptions {
                    visual_strength: 0.35,
                    audio_strength: 0.35,
                    copies: 2,
                    ..Default::default()
                })
                .unwrap()
        })
    });
    c.bench_function("layout/768p", |b| {
        b.iter(|| {
            let shape = h3_hrx::shape_for(768, 1344, 124).unwrap();
            black_box(
                h3_hrx::layout::Layout::new(
                    512,
                    shape.latent_t as usize,
                    shape.lat_h as usize,
                    shape.lat_w as usize,
                    shape.audio_t as usize,
                    &[],
                    &[],
                )
                .unwrap(),
            )
        })
    });
}
fn rotary(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotary/dit");
    let inv: Vec<_> = (0..16)
        .map(|j| 10_000.0f32.powf(-(j as f32) / 16.0))
        .collect();
    for (name, height, width, frames, with_refs) in [
        ("smoke", 64, 64, 5, false),
        ("480p_5s", 480, 864, 124, false),
        ("768p_5s", 768, 1344, 124, false),
        ("768p_5s_references", 768, 1344, 124, true),
        ("768p_15s", 768, 1344, 362, false),
        ("1536p_15s", 1536, 2688, 362, false),
    ] {
        let sh = h3_hrx::shape_for(height, width, frames).unwrap();
        let refs = [h3_hrx::layout::Ref {
            kind: 2,
            latent_t: 7,
            lat_h: 30,
            lat_w: 54,
            audio_t: 40,
            has_audio: true,
        }];
        let kfs = [h3_hrx::layout::Keyframe {
            frame_index: 0,
            audio_t: 0,
            has_audio: false,
        }];
        let layout = h3_hrx::layout::Layout::new(
            512,
            sh.latent_t as usize,
            sh.lat_h as usize,
            sh.lat_w as usize,
            sh.audio_t as usize,
            if with_refs { &refs } else { &[] },
            if with_refs { &kfs } else { &[] },
        )
        .unwrap();
        group.throughput(Throughput::Elements(layout.seq_len as u64));
        group.bench_function(name, |b| {
            b.iter(|| black_box(&layout).rotary_tables(black_box(&inv)))
        });
    }
    group.finish();
}
fn video_decoder_input(c: &mut Criterion) {
    use h3_hrx::{
        model::{LATENT_CH, VAE_KIN},
        vvae::Grid,
    };

    let weight = support::values(LATENT_CH * LATENT_CH, 0.2);
    let bias = support::values(LATENT_CH, 0.1);
    let mut group = c.benchmark_group("video_decoder_input");
    for (ft, h, w) in [(1, 1, 1), (1, 1, 7), (2, 4, 4), (7, 8, 16), (7, 16, 16)] {
        let grid = Grid { ft, h, w };
        let z = support::values(LATENT_CH * grid.voxels(), 0.5);
        let mut out = vec![half::f16::ZERO; grid.voxels() * VAE_KIN];
        group.throughput(Throughput::Elements(grid.voxels() as u64));
        group.bench_function(format!("{ft}x{h}x{w}"), |b| {
            b.iter(|| {
                grid.prepare_decoder_input(
                    black_box(&z),
                    black_box(&weight),
                    black_box(&bias),
                    black_box(&mut out),
                );
            })
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = support::criterion(); targets = host, rotary, video_decoder_input }
criterion_main!(benches);
