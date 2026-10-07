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
fn video_decoder_output(c: &mut Criterion) {
    use h3_hrx::{model::VAE_OUT, vvae::Grid};

    let mut group = c.benchmark_group("video_decoder_output");
    for (ft, h, w) in [(1, 1, 1), (2, 4, 4), (7, 8, 16), (7, 16, 16)] {
        let grid = Grid { ft, h, w };
        let patches: Vec<_> = support::values(grid.voxels() * VAE_OUT, 0.5)
            .into_iter()
            .map(half::f16::from_f32)
            .collect();
        group.throughput(Throughput::Elements(patches.len() as u64));
        group.bench_function(format!("{ft}x{h}x{w}/reused"), |b| {
            let mut frames = Vec::new();
            grid.unpatchify(&patches, &mut frames);
            b.iter(|| grid.unpatchify(black_box(&patches), black_box(&mut frames)))
        });
        group.bench_function(format!("{ft}x{h}x{w}/fresh"), |b| {
            b.iter(|| {
                let mut frames = Vec::new();
                grid.unpatchify(black_box(&patches), &mut frames);
                frames
            })
        });
    }
    group.finish();
}
fn video_temporal(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_temporal");
    for (height, width, frames) in [(64, 64, 5), (64, 64, 124), (480, 864, 39), (768, 1344, 39)] {
        let shape = h3_hrx::shape_for(height, width, frames).unwrap();
        let plane = height as usize * width as usize;
        let fixture = support::values(3 * 28 * plane, 0.5);
        let mut out = vec![0.0f32; 3 * shape.frames as usize * plane];
        group.throughput(Throughput::Elements(out.len() as u64));
        group.bench_function(format!("{height}x{width}x{frames}"), |b| {
            b.iter(|| {
                h3_hrx::vvae::decode_temporal(
                    black_box(&shape),
                    |_, ft, clip| {
                        let count = 3 * ft * 4 * plane;
                        clip.resize(count, 0.0);
                        clip.copy_from_slice(&fixture[..count]);
                        Ok(())
                    },
                    &mut |index, value, _| out[index] = value,
                )
                .unwrap();
                black_box(&out);
            });
        });
    }
    group.finish();
}
fn video_spatial(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_spatial");
    for (frames, height, width) in [
        (4, 272, 256),
        (4, 256, 272),
        (28, 480, 864),
        (28, 768, 1344),
    ] {
        let fixture = support::values(3 * frames * 256 * 256, 0.5);
        let run = |out: &mut Vec<f32>| {
            h3_hrx::vvae::stitch_pixels(frames, height, width, out, |_, _, th, tw, tile| {
                let count = 3 * frames * th * tw;
                tile.resize(count, 0.0);
                tile.copy_from_slice(&fixture[..count]);
                Ok::<_, std::convert::Infallible>(())
            })
            .unwrap();
        };
        group.throughput(Throughput::Elements((3 * frames * height * width) as u64));
        group.bench_function(format!("{height}x{width}x{frames}/reused"), |b| {
            let mut out = Vec::new();
            run(&mut out);
            b.iter(|| run(black_box(&mut out)));
        });
        group.bench_function(format!("{height}x{width}x{frames}/fresh"), |b| {
            b.iter(|| {
                let mut out = Vec::new();
                run(&mut out);
                black_box(out)
            });
        });
    }
    group.finish();
}
fn video_encoder_input(c: &mut Criterion) {
    let mut group = c.benchmark_group("video_encoder_input");
    for (name, frames, height, width, y0, x0, th, tw) in [
        ("tiny", 1, 16, 16, 0, 0, 16, 16),
        ("image", 1, 256, 256, 0, 0, 256, 256),
        ("clip", 17, 256, 256, 0, 0, 256, 256),
        ("cropped_clip", 17, 480, 864, 192, 176, 256, 256),
        ("narrow_clip", 17, 64, 48, 0, 16, 64, 16),
    ] {
        let pixels = support::values(frames * height * width * 3, 0.5);
        let clip = h3_hrx::vvae::Clip {
            pixels: &pixels,
            frames,
            height,
            width,
        };
        let mut out = vec![half::f16::ZERO; frames * th * tw * 8];
        group.throughput(Throughput::Elements((frames * th * tw) as u64));
        group.bench_function(name, |b| {
            b.iter(|| clip.prepare_encoder_input(y0, x0, th, tw, black_box(&mut out)));
        });
    }
    group.finish();
}
fn vision_tokens(c: &mut Criterion) {
    use h3_hrx::model::{VHID, VPOS_GRID};

    let positions = support::values(VPOS_GRID * VPOS_GRID * VHID, 0.1);
    let mut group = c.benchmark_group("vision_tokens");
    for (height, width) in [(32, 32), (256, 256), (480, 864), (768, 1344), (1344, 768)] {
        let (gh, gw) = (height / 16, width / 16);
        let projected = support::values(gh * gw * VHID, 0.5);
        let mut input = projected.clone();
        group.throughput(Throughput::Elements(projected.len() as u64));
        group.bench_function(format!("{height}x{width}"), |b| {
            b.iter(|| {
                input.copy_from_slice(black_box(&projected));
                h3_hrx::vision::prepare_tokens(black_box(&mut input), black_box(&positions), gh, gw)
            });
        });
    }
    group.finish();
}
fn text_rotary(c: &mut Criterion) {
    use h3_hrx::te::{mrope_positions, rotary_tables, VisionSpan};
    let mut group = c.benchmark_group("rotary/text");
    let mut cases = vec![
        ("one_token", 1, vec![]),
        ("prompt_128", 128, vec![]),
        ("prompt_4096", 4096, vec![]),
    ];
    for (name, images, h, w) in [
        ("image_768p", 1, 24, 42),
        ("two_portraits", 2, 42, 24),
        ("video_16_pairs", 16, 24, 42),
    ] {
        let spans = (0..images)
            .map(|i| VisionSpan {
                start: 64 + i * (h * w + 8),
                count: h * w,
                merged_h: h,
                merged_w: w,
            })
            .collect();
        cases.push((name, 128 + images * (h * w + 8), spans));
    }
    for (name, tokens, spans) in cases {
        let pos = mrope_positions(tokens, &spans);
        let mut cos = vec![0.0; tokens * h3_hrx::model::TE_ROPE_HALF];
        let mut sin = cos.clone();
        group.throughput(Throughput::Elements(tokens as u64));
        group.bench_function(name, |b| {
            b.iter(|| rotary_tables(black_box(&pos), black_box(&mut cos), black_box(&mut sin)));
        });
    }
    group.finish();
}

fn vision_rotary(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotary/vision");
    for (height, width) in [
        (32, 32),
        (256, 256),
        (480, 864),
        (768, 1344),
        (1344, 768),
        (3584, 3584),
    ] {
        let (gh, gw) = (height / 16, width / 16);
        group.throughput(Throughput::Elements((gh * gw) as u64));
        group.bench_function(format!("{height}x{width}"), |b| {
            let mut cos = vec![0.0; gh * gw * 36];
            let mut sin = vec![0.0; gh * gw * 36];
            b.iter(|| {
                h3_hrx::vision::rotary_tables(
                    black_box(gh),
                    black_box(gw),
                    black_box(&mut cos),
                    black_box(&mut sin),
                )
            });
        });
    }
    group.finish();
}
criterion_group! { name = benches; config = support::criterion(); targets = host, rotary, video_decoder_input, video_decoder_output, video_temporal, video_spatial, video_encoder_input, vision_tokens, text_rotary, vision_rotary }
criterion_main!(benches);
