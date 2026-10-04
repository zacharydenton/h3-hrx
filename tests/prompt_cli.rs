#![cfg(all(feature = "prompt-generation", feature = "cli"))]
use serde_json::{json, Value};
use std::{
    io::{Read, Write},
    net::TcpListener,
    process::Command,
    time::Duration,
};

#[test]
fn prompt_command_uses_shared_arguments_and_needs_no_models_or_runtime() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let worker = std::thread::spawn(move || {
        let final_prompt="integrated_multimodal_description: [Shot 1] Cinematic, a red ball rolls.\noverall_soundscape: Soft rolling.\nnon_diegetic_music: N/A";
        let mut requests = Vec::new();
        for content in ["A red ball, no reference assets.", final_prompt] {
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

#[cfg(not(feature = "local-prompt-generation"))]
#[test]
fn local_backend_reports_its_optional_build_feature() {
    let output = Command::new(env!("CARGO_BIN_EXE_h3"))
        .args(["prompt", "--prompt-backend", "local", "-p", "A ball rolls"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("local-prompt-generation"));
}

#[cfg(feature = "local-prompt-generation")]
#[test]
fn local_limits_fail_before_media_or_model_access() {
    let output = Command::new(env!("CARGO_BIN_EXE_h3"))
        .args([
            "prompt",
            "--prompt-backend",
            "local",
            "--prompt-context-tokens",
            "0",
            "--te",
            "/absent/encoder",
            "--refmod",
            "/absent/reference",
            "-p",
            "A ball rolls",
        ])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("local context must be"));
}

#[cfg(feature = "local-prompt-generation")]
#[test]
fn local_audio_notes_fail_before_any_checkpoint_or_runtime_is_opened() {
    let output = Command::new(env!("CARGO_BIN_EXE_h3"))
        .args([
            "prompt",
            "--prompt-backend",
            "local",
            "--offline",
            "--te",
            "/absent/encoder",
            "--refmod",
            "tests/fixtures/refmod/combined.safetensors",
            "-p",
            "Greet the viewer",
        ])
        .env("HRX_OFFLINE", "1")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("audio note"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
