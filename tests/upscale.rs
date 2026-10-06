use h3_hrx::{
    compile::Compiler, dispatch::Profile, upscale::Upscaler, UpscaleSettings, UpscaleTarget,
};
use half::f16;
use std::collections::BTreeMap;

fn fixture(path: &std::path::Path) {
    let c = 64;
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<u8>)>::new();
    let mut add = |name: String, shape: Vec<usize>, value: f32| {
        let n = shape.iter().product::<usize>();
        tensors.insert(
            name,
            (
                shape,
                (0..n)
                    .flat_map(|_| f16::from_f32(value).to_le_bytes())
                    .collect(),
            ),
        );
    };
    for (name, dims) in [
        ("conv_in", vec![c, 24, 3, 3, 3]),
        ("conv_out", vec![24, c, 3, 3, 3]),
        ("embed.0", vec![64, 1]),
        ("embed.2", vec![64, 64]),
    ] {
        add(
            format!("{name}.bias"),
            vec![dims[0]],
            if name == "conv_out" { 0.25 } else { 0.0 },
        );
        add(format!("{name}.weight"), dims, 0.0);
    }
    for suffix in ["weight", "bias"] {
        add(
            format!("norm_out.{suffix}"),
            vec![c],
            if suffix == "weight" { 1.0 } else { 0.0 },
        );
    }
    for side in ["in_blocks", "out_blocks"] {
        let name = format!("{side}.0");
        for norm in ["in_layers.0", "out_norm"] {
            for suffix in ["weight", "bias"] {
                add(
                    format!("{name}.{norm}.{suffix}"),
                    vec![c],
                    if suffix == "weight" { 1.0 } else { 0.0 },
                );
            }
        }
        for conv in ["in_layers.2", "out_layers.2"] {
            add(format!("{name}.{conv}.weight"), vec![c, c, 3, 3, 3], 0.0);
            add(format!("{name}.{conv}.bias"), vec![c], 0.0);
        }
        add(format!("{name}.emb_layers.1.weight"), vec![2 * c, 64], 0.0);
        add(format!("{name}.emb_layers.1.bias"), vec![2 * c], 0.0);
        let name = format!("{side}.1");
        add(format!("{name}.dwconv.weight"), vec![c, 1, 5, 1, 1], 0.0);
        add(format!("{name}.dwconv.bias"), vec![c], 0.0);
        add(format!("{name}.pwconv.weight"), vec![c, c, 1, 1, 1], 0.0);
        add(format!("{name}.pwconv.bias"), vec![c], 0.0);
        for suffix in ["weight", "bias"] {
            add(
                format!("{name}.norm.{suffix}"),
                vec![c],
                if suffix == "weight" { 1.0 } else { 0.0 },
            );
        }
    }
    let views = tensors.iter().map(|(name, (shape, data))| {
        (
            name,
            safetensors::tensor::TensorView::new(safetensors::Dtype::F16, shape.clone(), data)
                .unwrap(),
        )
    });
    let bytes = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(path, bytes).unwrap();
}
#[test]
fn checkpoint_validation_rejects_unexpected_tensors_and_bad_dimensions() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("upscale.safetensors");
    fixture(&path);
    // SAFETY: test owns the file and does not modify it while mapped.
    let model = unsafe { Upscaler::open(&path) }.unwrap();
    assert!(model.device_bytes() > 0);
    drop(model);
    let bytes = std::fs::read(&path).unwrap();
    let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let mut views = tensors.tensors();
    let extra = [0u8; 2];
    views.push((
        "unexpected.weight".into(),
        safetensors::tensor::TensorView::new(safetensors::Dtype::F16, vec![1], &extra).unwrap(),
    ));
    std::fs::write(&path, safetensors::tensor::serialize(views, None).unwrap()).unwrap();
    assert!(unsafe { Upscaler::open(&path) }.is_err());
}
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn learned_network_runs_all_layers_preserves_identity_and_blends_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("upscale.safetensors");
    fixture(&path);
    let model = unsafe { Upscaler::open(&path) }.unwrap();
    let mut stream = hrx::Stream::open().unwrap();
    let compiler = Compiler::new(None, std::path::PathBuf::new());
    let mut profile = Profile::default();
    for frames in [5, 277] {
        let shape = h3_hrx::shape_for(32, 32, frames).unwrap();
        let input = vec![0.5; 24 * shape.latent_t as usize * 4];
        let identity = UpscaleSettings {
            target: UpscaleTarget::Scale(1.0),
            ..Default::default()
        };
        assert_eq!(
            model
                .run(
                    &mut stream,
                    &compiler,
                    &mut profile,
                    &input,
                    &shape,
                    &identity,
                    None
                )
                .unwrap()
                .1,
            input
        );
        let settings = UpscaleSettings {
            target: UpscaleTarget::Scale(2.0),
            ..Default::default()
        };
        let (out, values) = model
            .run(
                &mut stream,
                &compiler,
                &mut profile,
                &input,
                &shape,
                &settings,
                None,
            )
            .unwrap();
        assert_eq!(out.size(), (64, 64));
        assert_eq!(out.latent_t, shape.latent_t);
        assert!(values.iter().all(|v| (*v - 0.25).abs() < 1e-6));
    }
    stream.synchronize().unwrap();
}
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn released_checkpoint_upscales_a_bounded_clip() {
    let path = h3_hrx::models::Resolver::new()
        .repository("LBH-123-AI", "Minimax_h3_latent_Upscaler")
        .find(h3_hrx::upscale::CHECKPOINT)
        .unwrap();
    let mut session = unsafe {
        h3_hrx::Session::new_with_options(
            h3_hrx::Config {
                latent_upscaler: Some(path),
                ..Default::default()
            },
            h3_hrx::SessionOptions {
                residency: h3_hrx::ResidencyPolicy::StageScoped,
                ..Default::default()
            },
        )
    }
    .unwrap();
    let shape = h3_hrx::shape_for(32, 32, 5).unwrap();
    let input = h3_hrx::Latents {
        video: include_bytes!("fixtures/upscale/input.f32")
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        audio: vec![0.125; 64 * shape.audio_t as usize],
    };
    let settings = UpscaleSettings {
        target: UpscaleTarget::Scale(2.0),
        ..Default::default()
    };
    let output = session
        .upscale_latents(&input, &shape, &settings, None)
        .unwrap();
    assert_eq!(output.shape.size(), (64, 64));
    assert_eq!(output.latents.audio, input.audio);
    assert!(output.latents.video.iter().all(|v| v.is_finite()));
    let expected = include_bytes!("fixtures/upscale/output.f32")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect::<Vec<_>>();
    let rmse = (output
        .latents
        .video
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).powi(2) as f64)
        .sum::<f64>()
        / expected.len() as f64)
        .sqrt();
    let max = output
        .latents
        .video
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("upscaler upstream parity: rmse={rmse}, max={max}");
    assert!(rmse < 0.005 && max < 0.03, "rmse={rmse} max={max}");
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires provisioned HRX and H3 checkpoints"
)]
fn two_pass_generation_retains_refmods_audio_and_target_keyframes() {
    use h3_hrx::{
        Config, Control, DenoiseParams, Keyframe, Noise, RefinementSettings, ResidencyPolicy,
        Session, SessionOptions,
    };
    let manager = hrx::residency::ResidencyManager::new(64usize << 30).unwrap();
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })
    .unwrap();
    let resolver = h3_hrx::models::Resolver::new();
    let config = Config {
        dit: Some(resolver.find(h3_hrx::models::DIT_REF2VA).unwrap()),
        ..Default::default()
    };
    let mut session = unsafe {
        Session::new_in(
            config,
            SessionOptions {
                residency: ResidencyPolicy::Budgeted,
                ..Default::default()
            },
            &context,
        )
    }
    .unwrap();
    let mods = h3_hrx::refmod::RefMod::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/refmod/combined.safetensors"
    ))
    .unwrap();
    let prepared = mods
        .prepare(h3_hrx::refmod::ApplyOptions {
            visual_strength: 0.75,
            audio_strength: 0.5,
            copies: 2,
            ..Default::default()
        })
        .unwrap();
    let refs = prepared.references();
    let ids = h3_hrx::Tokenizer::new()
        .unwrap()
        .encode("A red fox walks through snow.")
        .unwrap();
    let mut p = DenoiseParams {
        width: 32,
        height: 32,
        frames: 5,
        steps: 2,
        seed: 42,
        ..Default::default()
    };
    let shape = h3_hrx::shape_for(p.height, p.width, p.frames).unwrap();
    eprintln!("first pass");
    let low = session
        .denoise(&ids, &p, Noise::default(), &refs, &[], None)
        .unwrap();
    let up = session
        .upscale_latents(
            &low,
            &shape,
            &UpscaleSettings {
                target: UpscaleTarget::Scale(2.0),
                ..Default::default()
            },
            None,
        )
        .unwrap();
    (p.width, p.height) = up.shape.size();
    let settings = RefinementSettings::default();
    let key = vec![0.0; 24 * 4 * 4];
    let kfs = [Keyframe {
        frame_index: 0,
        latents: &key,
        presented: None,
        audio: None,
    }];
    let mut completed = 0;
    let mut progress = |step, _, _| {
        completed = step;
        Control::Continue
    };
    eprintln!("refinement with RefMods");
    let refined = session
        .refine(
            &ids,
            None,
            &p,
            &up.latents,
            &settings,
            &refs,
            &kfs,
            Some(&mut progress),
        )
        .unwrap();
    assert_eq!(completed, 4);
    assert_eq!(refined.audio, low.audio);
    assert_eq!(refined.video.len(), 24 * up.shape.latent_t as usize * 16);
    eprintln!("refinement without references");
    let without = session
        .refine(&ids, None, &p, &up.latents, &settings, &[], &kfs, None)
        .unwrap();
    assert!(
        refined
            .video
            .iter()
            .zip(&without.video)
            .any(|(a, b)| (a - b).abs() > 1e-5),
        "references must affect refinement"
    );
    let mut cancel = |_, _, _| Control::Cancel;
    assert!(matches!(
        session.refine(
            &ids,
            None,
            &p,
            &up.latents,
            &settings,
            &refs,
            &kfs,
            Some(&mut cancel)
        ),
        Err(h3_hrx::Error::Cancelled)
    ));
    eprintln!("refinement after cancellation");
    let again = session
        .refine(&ids, None, &p, &up.latents, &settings, &refs, &kfs, None)
        .unwrap();
    assert_eq!(again.video, refined.video);
    assert_eq!(again.audio, low.audio);
    let mut rgb = vec![0u8; up.shape.video_bytes()];
    session
        .decode_video(&up.shape, &refined.video, &mut rgb)
        .unwrap();
    let mut audio = vec![0f32; up.shape.audio_samples()];
    session
        .decode_audio(&refined.audio, up.shape.audio_t as usize, &mut audio)
        .unwrap();
    assert!(audio.iter().all(|v| v.is_finite()));
    drop(session);
    assert_eq!(manager.statistics().reserved_bytes, 0);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires provisioned HRX")]
fn upscaler_budget_eviction_and_cancel_release_native_owners() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("upscale.safetensors");
    fixture(&path);
    let manager = hrx::residency::ResidencyManager::new(16 << 20).unwrap();
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })
    .unwrap();
    let mut session = unsafe {
        h3_hrx::Session::new_in(
            h3_hrx::Config {
                latent_upscaler: Some(path),
                ..Default::default()
            },
            h3_hrx::SessionOptions {
                residency: h3_hrx::ResidencyPolicy::Budgeted,
                ..Default::default()
            },
            &context,
        )
    }
    .unwrap();
    let shape = h3_hrx::shape_for(32, 32, 5).unwrap();
    let input = h3_hrx::Latents {
        video: vec![0.5; 24 * shape.latent_t as usize * 4],
        audio: vec![0.25; 64 * shape.audio_t as usize],
    };
    let settings = UpscaleSettings {
        target: UpscaleTarget::Scale(2.0),
        ..Default::default()
    };
    let out = session
        .upscale_latents(&input, &shape, &settings, None)
        .unwrap();
    let evictions = manager.statistics().evictions;
    let pressure = manager.budget().reserve(15 << 20).unwrap();
    drop(pressure);
    assert!(manager.statistics().evictions > evictions);
    let again = session
        .upscale_latents(&input, &shape, &settings, None)
        .unwrap();
    assert_eq!(again.latents.video, out.latents.video);
    let mut cancel = |_, _, _| h3_hrx::Control::Cancel;
    assert!(matches!(
        session.upscale_latents(&input, &shape, &settings, Some(&mut cancel)),
        Err(h3_hrx::Error::Cancelled)
    ));
    let again = session
        .upscale_latents(&input, &shape, &settings, None)
        .unwrap();
    assert_eq!(again.latents.video, out.latents.video);
    let ids = h3_hrx::Tokenizer::new().unwrap().encode("fox").unwrap();
    let p = h3_hrx::DenoiseParams {
        width: 64,
        height: 64,
        frames: 5,
        ..Default::default()
    };
    let untouched = session
        .refine(
            &ids,
            None,
            &p,
            &out.latents,
            &h3_hrx::RefinementSettings {
                denoise: 0.0,
                ..Default::default()
            },
            &[],
            &[],
            None,
        )
        .unwrap();
    assert_eq!(untouched.video, out.latents.video);
    assert_eq!(untouched.audio, input.audio);
    drop(session);
    assert_eq!(manager.statistics().reserved_bytes, 0);
}
