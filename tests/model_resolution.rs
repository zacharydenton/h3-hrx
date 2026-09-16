//! Model selection and offline behavior, using a local Hub stub and no GPU or weights.
#![cfg(feature = "cli")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::process::{Command, Output};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

fn run(binary: &str, args: &[&str], offline: &str) -> (Output, Vec<String>) {
    let cache = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    listener.set_nonblocking(true).unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let stop = Arc::new(AtomicBool::new(false));
    let server_requests = requests.clone();
    let server_stop = stop.clone();
    let server = std::thread::spawn(move || {
        while !server_stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_secs(5)))
                        .unwrap();
                    let mut request = Vec::new();
                    let mut buffer = [0; 1024];
                    while !request.windows(4).any(|part| part == b"\r\n\r\n") {
                        let count = stream.read(&mut buffer).unwrap();
                        if count == 0 {
                            break;
                        }
                        request.extend_from_slice(&buffer[..count]);
                    }
                    server_requests.lock().unwrap().push(
                        String::from_utf8_lossy(&request)
                            .lines()
                            .next()
                            .unwrap_or_default()
                            .to_string(),
                    );
                    stream
                        .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                        .unwrap();
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(error) => panic!("Hub stub: {error}"),
            }
        }
    });
    let output = Command::new(binary)
        .args(args)
        .env_remove("H3_MODELS")
        .env("HF_HUB_CACHE", cache.path())
        .env("HF_ENDPOINT", endpoint)
        .env("HF_HUB_OFFLINE", offline)
        .env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
        .env("HRX_OFFLINE", "1")
        .output()
        .unwrap();
    stop.store(true, Ordering::Relaxed);
    server.join().unwrap();
    let requests = requests.lock().unwrap().clone();
    (output, requests)
}

#[test]
fn hub_offline_environment_prevents_requests_even_without_the_cli_flag() {
    for value in ["1", "on", "Yes", "TRUE"] {
        let (output, requests) = run(
            env!("CARGO_BIN_EXE_h3-dev"),
            &["resolve", "vae/missing.safetensors"],
            value,
        );
        assert!(!output.status.success());
        assert!(requests.is_empty(), "{value}: {requests:?}");
        let message = String::from_utf8_lossy(&output.stderr);
        assert!(
            message.contains("not in the Hugging Face cache"),
            "{message}"
        );
        assert!(message.contains("downloading is off"), "{message}");
    }
    let (_, requests) = run(
        env!("CARGO_BIN_EXE_h3-dev"),
        &["resolve", "vae/missing.safetensors"],
        "0",
    );
    assert_eq!(requests.len(), 1, "online resolution must reach the stub");
    let (_, requests) = run(
        env!("CARGO_BIN_EXE_h3-dev"),
        &["resolve", "--offline", "vae/missing.safetensors"],
        "0",
    );
    assert!(requests.is_empty());
}

#[test]
fn reference_generation_fetches_ref2va_and_explicit_base_mode_fetches_fl2va() {
    for (extra, expected) in [
        (vec!["reference.jpg"], "minimax_h3_ref2va_"),
        (vec!["reference.wav"], "minimax_h3_ref2va_"),
        (vec!["reference.jpg", "--base-weights"], "minimax_h3_fl2va_"),
        (vec!["--first-frame", "first.png"], "minimax_h3_fl2va_"),
    ] {
        let mut args = vec!["-p", "test prompt", "--no-decode"];
        args.extend(extra);
        let (output, requests) = run(env!("CARGO_BIN_EXE_h3"), &args, "0");
        assert!(!output.status.success()); // The stub deliberately has no checkpoint.
        assert_eq!(requests.len(), 1, "{requests:?}");
        assert!(requests[0].contains(expected), "{requests:?}");
        assert_eq!(output.status.code(), Some(1)); // Download errors are runtime failures.
    }
}

#[test]
fn removed_model_directory_flag_is_rejected() {
    let (output, requests) = run(
        env!("CARGO_BIN_EXE_h3"),
        &["--models", "legacy-models", "-p", "test prompt"],
        "0",
    );
    assert_eq!(output.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&output.stderr).contains("unexpected argument '--models'"));
    assert!(requests.is_empty());
}
