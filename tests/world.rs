use h3_hrx::{Action, ActionSchedule};

#[test]
fn all_keyboard_states_match_pinned_upstream() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/world/actions.json")).unwrap();
    for bits in 0..512 {
        assert_eq!(
            Action(bits).sentence(),
            fixture["sentences"][bits as usize].as_str().unwrap(),
            "bits={bits}"
        );
    }
    assert!(ActionSchedule::from_json(r#"[["Q"]]"#).is_err());
    assert!(ActionSchedule::from_json(r#"[["W"],null]"#).is_err());
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and cached base/world checkpoints"
)]
fn world_changes_actions_and_reuses_a_session() {
    use h3_hrx::{
        media_context::{Frame, Media, MediaEntry},
        Config, Keyframe, Noise, PreparedPresentation, ResidencyPolicy, Session, SessionOptions,
        Tokenizer, WorldRequest,
    };
    let base = h3_hrx::models::Resolver::new();
    let adapter = h3_hrx::models::Resolver::new()
        .repository("DANNY621", "H3-World")
        .revision(Some(h3_hrx::world::ADAPTER_REVISION.into()))
        .find(h3_hrx::world::ADAPTER_FILE)
        .unwrap();
    let config = Config {
        dit: Some(base.find(h3_hrx::models::DIT_FL2VA).unwrap()),
        te: Some(base.find(h3_hrx::models::TE).unwrap()),
        video_vae: Some("/absent/vae".into()),
        audio_vae: Some("/absent/audio".into()),
        loras: vec![h3_hrx::adapter::Lora::new(adapter, 1.0)],
        ..Default::default()
    };
    let manager = hrx::residency::ResidencyManager::new(28usize << 30).unwrap();
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })
    .unwrap();
    let mut session = unsafe {
        Session::new_in(
            config,
            SessionOptions {
                residency: ResidencyPolicy::StageScoped,
                ..Default::default()
            },
            &context,
        )
    }
    .unwrap();
    let mut p = WorldRequest::parameters();
    p.width = 64;
    p.height = 64;
    p.frames = 5;
    p.steps = 2;
    p.seed = 7;
    let shape = Session::shape_for(64, 64, 5).unwrap();
    let tok = Tokenizer::new().unwrap();
    let entries = [MediaEntry {
        media: Media::Picture(Frame {
            pixels: vec![0.4; 64 * 64 * 3].into(),
            width: 64,
            height: 64,
        }),
        role: "first_frame".into(),
        metadata: serde_json::Value::Null,
    }];
    let presentation = PreparedPresentation::new(
        &tok,
        &entries,
        "A third-person view of a man standing in a parking garage.",
        &shape,
    )
    .unwrap();
    let latent = vec![0.0; 24 * 4 * 4];
    let key = Keyframe {
        frame_index: 0,
        latents: &latent,
        audio: None,
        presented: None,
    };
    let mut run = |preset| {
        eprintln!("world regression: {preset}");
        let request =
            WorldRequest::new(&tok, &ActionSchedule::preset(preset, 5).unwrap(), 5).unwrap();
        session
            .denoise_world(&presentation, &request, &p, Noise::default(), &key, None)
            .unwrap()
    };
    let left = run("pan-left");
    let right = run("pan-right");
    let repeat = run("pan-left");
    assert!(left
        .video
        .iter()
        .chain(&left.audio)
        .chain(&right.video)
        .chain(&right.audio)
        .all(|x| x.is_finite()));
    assert_ne!(
        left.video, right.video,
        "action schedule must affect generated latents"
    );
    assert_eq!(
        left.video, repeat.video,
        "routing must reset on the third request"
    );
    assert_eq!(left.audio, repeat.audio);
}

#[test]
fn action_tokens_match_upstream_presentation() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/world/tokens.json")).unwrap();
    let tok = h3_hrx::Tokenizer::new().unwrap();
    for (sentence, ids) in fixture["tokens"].as_object().unwrap() {
        let want: Vec<i32> = serde_json::from_value(ids.clone()).unwrap();
        assert_eq!(tok.encode(sentence).unwrap(), want);
    }
}

#[test]
fn first_frame_matches_pillow_cover_resize() {
    let fixtures: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/world/resize.json")).unwrap();
    for f in fixtures.as_array().unwrap() {
        let sw = f["input"][0].as_u64().unwrap() as i32;
        let sh = f["input"][1].as_u64().unwrap() as i32;
        let dw = f["output"][0].as_u64().unwrap() as i32;
        let dh = f["output"][1].as_u64().unwrap() as i32;
        let input: Vec<u8> = (0..sw as usize * sh as usize * 3)
            .map(|i| ((i * 37 + i / 7) % 256) as u8)
            .collect();
        let want = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/world")
                .join(f["file"].as_str().unwrap()),
        )
        .unwrap();
        let got = h3_hrx::resize::world_first_frame(&input, sw, sh, dw, dh);
        let max = got
            .iter()
            .zip(want)
            .map(|(a, b)| ((a * 255.0).round() as i32 - b as i32).abs())
            .max()
            .unwrap();
        assert!(
            max <= 1,
            "PIL fixed-point rounding tolerance: {sw}x{sh} -> {dw}x{dh}, max={max}"
        );
    }
}

#[test]
fn cropped_first_frame_preserves_original_output_bits() {
    // SHA-256 of little-endian f32 output from the full-intermediate resizer at
    // 6e5fd55. Keep the Pillow fixture check above as the independent reference.
    for (sw, sh, dw, dh, digest) in [
        (
            1920,
            1080,
            864,
            480,
            "817a013c07804d1251a02beee3bee8de1b1147c9d8449258bf1af766408c6d88",
        ),
        (
            1080,
            1920,
            864,
            480,
            "cfb3f03b3860b7572b4a7920c6dfba72f46a479d3b8b98b735a1d1ee85365cd8",
        ),
        (
            1920,
            1080,
            480,
            864,
            "1c8dd2ff92ab9b2a1e27c6cb1b1b3e50b858a4bdc77320854e326f138796d2d3",
        ),
        (
            320,
            240,
            864,
            480,
            "4e43189c1777f032f54224e765bbdc2597e44d415578e75324467d74e2d086ee",
        ),
        (
            37,
            53,
            64,
            32,
            "46d11f416b4d43772e2710c3d402423cb03fd9760bab2964ca6b71147a32782f",
        ),
        (
            64,
            32,
            64,
            32,
            "639a7182e105bb318e5a768ad0f84c118e1078d55f92340525f028850ff941a6",
        ),
        (
            1,
            1,
            7,
            3,
            "91c5ecb35447065e657e788040834cfa79c673982763c3f12a232e204af9bbda",
        ),
        (
            17,
            9,
            1,
            1,
            "ec9c7696d467e355478810ede56bf36398f7bdaf446aaaeaa24679db37b95218",
        ),
    ] {
        let input: Vec<_> = (0..sw * sh * 3)
            .map(|i| ((i * 37 + i / 7) % 256) as u8)
            .collect();
        let output = h3_hrx::resize::world_first_frame(&input, sw, sh, dw, dh);
        let bytes: Vec<_> = output.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(hrx::bundle::digest(&bytes), digest, "{sw}x{sh}->{dw}x{dh}");
    }
}

#[test]
#[cfg(feature = "cli")]
fn world_cli_rejects_invalid_requests_before_loading_models() {
    let binary = env!("CARGO_BIN_EXE_h3");
    for extra in [
        vec![],
        vec!["--last-frame", "absent.png"],
        vec!["--frames", "6"],
        vec!["--lora", "absent.safetensors"],
        vec!["--sampler", "res_multistep"],
    ] {
        let mut cmd = std::process::Command::new(binary);
        cmd.args(["world", "--action-preset", "forward", "--offline"]);
        if !extra.is_empty() {
            cmd.args(["--first-frame", "absent.png", "-p", "A man in a garage."]);
        }
        let out = cmd.args(extra).output().unwrap();
        assert!(!out.status.success());
        let error = String::from_utf8_lossy(&out.stderr);
        assert!(!error.contains("session in"), "{error}");
        assert!(
            !error.contains("not found in"),
            "validation should precede cache lookup: {error}"
        );
    }
}

#[test]
#[cfg(feature = "cli")]
fn world_session_rejects_invalid_canvas_and_resume_overrides_before_models() {
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("world.h3world");
    let binary = env!("CARGO_BIN_EXE_h3");
    let out = std::process::Command::new(binary)
        .args([
            "world-session",
            "--first-frame",
            "/absent/frame.png",
            "-p",
            "scene",
            "--state",
        ])
        .arg(&state)
        .arg("--out")
        .arg(dir.path().join("outputs"))
        .args(["--frames", "6", "--action-preset", "forward", "--offline"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("frames = 17k+5"));
    assert!(!state.exists());
    // Lock files remain, but their advisory lock must be released on error.
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(dir.path().join("world.h3world.lock"))
        .unwrap();
    lock.try_lock().unwrap();
    let out = std::process::Command::new(binary)
        .args(["world-session", "--resume"])
        .arg(&state)
        .args(["--frames", "22", "--interactive"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot be used with"));
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires gfx1151 and cached base/world checkpoints; runs serially under a 28 GiB cap"
)]
fn world_rollout_decodes_and_resumes() {
    use h3_hrx::{
        Config, ResidencyPolicy, Session, SessionOptions, Tokenizer, WorldRequest, WorldState,
    };
    let resolver = h3_hrx::models::Resolver::new();
    let adapter = h3_hrx::models::Resolver::new()
        .repository("DANNY621", "H3-World")
        .revision(Some(h3_hrx::world::ADAPTER_REVISION.into()))
        .find(h3_hrx::world::ADAPTER_FILE)
        .unwrap();
    let config = Config {
        dit: Some(resolver.find(h3_hrx::models::DIT_FL2VA).unwrap()),
        te: Some(resolver.find(h3_hrx::models::TE).unwrap()),
        video_vae: Some(resolver.find(h3_hrx::models::VIDEO_VAE).unwrap()),
        audio_vae: Some(resolver.find(h3_hrx::models::AUDIO_VAE).unwrap()),
        loras: vec![h3_hrx::adapter::Lora::new(adapter, 1.)],
        ..Default::default()
    };
    let manager = hrx::residency::ResidencyManager::new(28usize << 30).unwrap();
    let context = hrx::inference::ModelContext::new(hrx::execution::RuntimeOptions {
        memory_budget: Some(manager.budget()),
        ..Default::default()
    })
    .unwrap();
    let mut session = unsafe {
        Session::new_in(
            config,
            SessionOptions {
                residency: ResidencyPolicy::StageScoped,
                ..Default::default()
            },
            &context,
        )
    }
    .unwrap();
    let p = h3_hrx::DenoiseParams {
        width: 64,
        height: 64,
        frames: 5,
        steps: 2,
        seed: 9,
        ..WorldRequest::parameters()
    };
    let mut state = WorldState::new("A man in a garage.", p, vec![128; 64 * 64 * 3]).unwrap();
    let tokenizer = Tokenizer::new().unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state.h3world");
    let left = session
        .step_world(
            &mut state,
            &tokenizer,
            &ActionSchedule::preset("pan-left", 5).unwrap(),
            None,
        )
        .unwrap();
    assert_eq!(
        state.observation(),
        &left.video[left.video.len() - 64 * 64 * 3..]
    );
    assert_eq!(
        (state.segments(), state.frame(), state.parameters().seed),
        (1, 4, 10)
    );
    state.save(&path).unwrap();
    let mut resumed = WorldState::load(&path).unwrap();
    assert_eq!(state.observation(), resumed.observation());
    let right = session
        .step_world(
            &mut resumed,
            &tokenizer,
            &ActionSchedule::preset("pan-right", 5).unwrap(),
            None,
        )
        .unwrap();
    assert_eq!((right.index, right.start_frame, right.seed), (1, 4, 10));
    assert_eq!(
        resumed.observation(),
        &right.video[right.video.len() - 64 * 64 * 3..]
    );
    assert_eq!(
        (
            resumed.segments(),
            resumed.frame(),
            resumed.parameters().seed
        ),
        (2, 8, 11)
    );
    assert_eq!(
        state.frame(),
        4,
        "resumed branch must not mutate original state"
    );
    assert!(left.audio.iter().chain(&right.audio).all(|v| v.is_finite()));
}
