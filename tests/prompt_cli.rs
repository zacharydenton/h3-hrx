#![cfg(all(feature = "prompt-generation", feature = "cli"))]
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::TcpListener,
    process::Command,
    time::Duration,
};

fn endpoint(final_prompt: &'static str) -> (String, std::thread::JoinHandle<Vec<Value>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let worker = std::thread::spawn(move || {
        let mut requests = Vec::new();
        for content in ["A red ball.", final_prompt] {
            let start = std::time::Instant::now();
            let (mut stream, _) = loop {
                match listener.accept() {
                    Ok(s) => break s,
                    Err(e)
                        if e.kind() == std::io::ErrorKind::WouldBlock
                            && start.elapsed() < Duration::from_secs(15) =>
                    {
                        std::thread::sleep(Duration::from_millis(5))
                    }
                    Err(e) => panic!("mock endpoint accept: {e}"),
                }
            };
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut header = Vec::new();
            let mut byte = [0];
            while !header.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                header.push(byte[0]);
            }
            let header = String::from_utf8(header).unwrap();
            let length = header
                .lines()
                .find_map(|line| {
                    line.to_lowercase()
                        .strip_prefix("content-length: ")
                        .map(|v| v.parse::<usize>().unwrap())
                })
                .unwrap();
            let mut bytes = vec![0; length];
            stream.read_exact(&mut bytes).unwrap();
            requests.push(serde_json::from_slice::<Value>(&bytes).unwrap());
            let reply = json!({"choices":[{"finish_reason":"stop","message":{"content":content}}]})
                .to_string();
            write!(stream,"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{reply}",reply.len()).unwrap();
        }
        requests
    });
    (url, worker)
}

#[test]
fn prompt_command_uses_shared_arguments_and_needs_no_models_or_runtime() {
    let (url, worker) = endpoint("integrated_multimodal_description: [Shot 1] Cinematic, a red ball rolls.\noverall_soundscape: Soft rolling.\nnon_diegetic_music: N/A");
    let dir = tempfile::tempdir().unwrap();
    let saved = dir.path().join("prompt.txt");
    let result = Command::new(env!("CARGO_BIN_EXE_h3"))
        .args([
            "prompt",
            "-p",
            "a red ball",
            "--frames",
            "120",
            "--width",
            "64",
            "--height",
            "64",
            "--offline",
            "--prompt-base-url",
            &url,
            "--prompt-model",
            "mock",
            "--save-prompt",
        ])
        .arg(&saved)
        .env("HF_HUB_CACHE", dir.path().join("empty-cache"))
        .env("HF_HUB_OFFLINE", "1")
        .env("HRX_OFFLINE", "1")
        .env_remove("H3_PROMPT_API_KEY")
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let stdout = String::from_utf8(result.stdout).unwrap();
    assert!(stdout.starts_with("integrated_multimodal_description:"));
    assert_eq!(stdout.trim_end(), std::fs::read_to_string(&saved).unwrap());
    let record: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("prompt.txt.json")).unwrap())
            .unwrap();
    assert_eq!(record["duration_seconds"], 124.0 / 24.0);
    assert!(!dir.path().join("empty-cache").exists());
    assert!(!dir.path().join("h3_out.mp4").exists());
    assert_eq!(worker.join().unwrap().len(), 2);
}

#[test]
fn generated_prompts_reject_legacy_refmod_presentation_before_media_access() {
    let result = Command::new(env!("CARGO_BIN_EXE_h3"))
        .args([
            "prompt",
            "-p",
            "x",
            "--refmod",
            "/absent/refmod",
            "--refmod-presentation",
            "latent-only",
            "--prompt-base-url",
            "http://127.0.0.1:1/v1",
            "--prompt-model",
            "mock",
        ])
        .output()
        .unwrap();
    assert!(!result.status.success());
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("require --refmod-presentation upstream")
    );
}

#[test]
fn removed_local_prompt_flags_are_rejected_before_loading_media() {
    for (flag, value) in [
        ("--prompt-backend", "local"),
        ("--prompt-local-model-dir", "/absent/model"),
        ("--prompt-context-tokens", "4096"),
        ("--prompt-max-tokens", "128"),
        ("--prompt-audio-note", "1=voice"),
    ] {
        let output = Command::new(env!("CARGO_BIN_EXE_h3"))
            .args([
                "prompt",
                flag,
                value,
                "--refmod",
                "/absent/refmod",
                "-p",
                "A ball rolls",
            ])
            .output()
            .unwrap();
        assert!(!output.status.success());
        let error = String::from_utf8_lossy(&output.stderr);
        assert!(
            error.contains("unexpected argument") && error.contains(flag),
            "{error}"
        );
    }
}

#[test]
fn original_refmod_images_prompt_without_models_and_keep_copy_labels() {
    use h3_hrx::{
        refmod::{RefMod, RefModMember},
        LatentGrid,
    };
    let dir = tempfile::tempdir().unwrap();
    let original = dir.path().join("original image=red.png");
    let mut encoder = png::Encoder::new(std::fs::File::create(&original).unwrap(), 64, 64);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(&[255, 0, 0].repeat(64 * 64))
        .unwrap();
    let refmod = dir.path().join("reference.safetensors");
    RefMod::new(
        "red",
        vec![RefModMember::visual(
            "red",
            vec![0.0; 24 * 4 * 4],
            LatentGrid {
                frames: 1,
                width: 4,
                height: 4,
            },
        )
        .unwrap()],
    )
    .unwrap()
    .save(&refmod, false)
    .unwrap();
    let (url, worker) = endpoint("subject_definitions: <Subject 1> is the ball in <Picture 1> and <Picture 2>.\nsummary: [reference generation] A ball rolls.\nretention_analysis: <Subject 1>: fully_preserved - color and shape.\ndetailed_description: Cinematic. [Shot 1] The red ball rolls.\noverall_soundscape: Soft rolling.\nnon_diegetic_music: N/A");
    let saved = dir.path().join("prompt.txt");
    let output = Command::new(env!("CARGO_BIN_EXE_h3"))
        .args([
            "prompt",
            "--offline",
            "--prompt-images",
            "--prompt-base-url",
            &url,
            "--prompt-model",
            "mock",
            "--width",
            "64",
            "--height",
            "64",
            "--refmod",
        ])
        .arg(&refmod)
        .args([
            "--refmod-source",
            &format!("1:1={}", original.display()),
            "--refmod-copies",
            "1=2",
            "--refmod-visual-strength",
            "1=0.4",
            "--video-vae",
            "/absent/video-vae",
            "--audio-vae",
            "/absent/audio-vae",
            "--te",
            "/absent/te",
            "--dit",
            "/absent/dit",
            "-p",
            "Make the ball roll",
            "--save-prompt",
        ])
        .arg(&saved)
        .env("HF_HUB_CACHE", dir.path().join("empty-cache"))
        .env("HF_HUB_OFFLINE", "1")
        .env("HRX_OFFLINE", "1")
        .env_remove("H3_PROMPT_API_KEY")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let requests = worker.join().unwrap();
    assert_eq!(requests.len(), 2);
    let images: Vec<_> = requests[0]["messages"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["content"].as_array())
        .flatten()
        .filter(|p| p["type"] == "image_url")
        .collect();
    assert_eq!(images.len(), 2);
    assert_eq!(images[0], images[1]);
    // Verify the actual original pixels reach the endpoint, not just the labels.
    use base64::Engine;
    let url = images[0]["image_url"]["url"].as_str().unwrap();
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(url.split_once(',').unwrap().1)
        .unwrap();
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes))
        .read_info()
        .unwrap();
    let mut pixels = vec![0; decoder.output_buffer_size().unwrap()];
    let info = decoder.next_frame(&mut pixels).unwrap();
    assert!(pixels[..info.buffer_size()]
        .as_chunks::<3>()
        .0
        .iter()
        .all(|p| *p == [255, 0, 0]));
    let record: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("prompt.txt.json")).unwrap())
            .unwrap();
    let record_text = record.to_string();
    assert!(record_text.contains("original_file"));
    assert!(record_text.contains("<Picture 2>"));
    assert!(!record_text.contains("<Picture 3>"));
    assert!(!dir.path().join("empty-cache").exists());
}
