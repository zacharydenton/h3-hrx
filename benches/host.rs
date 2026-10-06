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
    let rgb: Vec<_> = (0..1920 * 1080 * 3).map(|i| (i % 251) as u8).collect();
    c.bench_function("resize/1080p_to_480p", |b| {
        b.iter(|| h3_hrx::resize::pil_bilinear(black_box(&rgb), 1920, 1080, 864, 480))
    });
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
criterion_group! { name = benches; config = support::criterion(); targets = host }
criterion_main!(benches);
