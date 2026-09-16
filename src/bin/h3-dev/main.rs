//! Diagnostics for individual model stages and kernels.
//!
//! Run with `cargo run --release --bin h3-dev -- <command> [args]`.

mod audio_vae;
mod compare_decode;
mod compare_gemm;
mod compare_vision;
mod compile_probes;
mod decode_video;
mod denoise_cases;
mod dispatch_cost;
mod encode_video;
mod inspect_adapter;
mod parity_dump;
mod resolve;
mod text_in;
mod tune_gemm;
mod two_sessions;
mod verify_gemm;
mod verify_send;
mod verify_stack_kernels;
mod verify_upload;
mod vision_embed;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let command = if args.is_empty() {
        String::new()
    } else {
        args.remove(0)
    };
    match command.as_str() {
        "audio-vae" => audio_vae::run(args),
        "compare-decode" => compare_decode::run(args),
        "compare-gemm" => compare_gemm::run(args),
        "compile-probes" => compile_probes::run(args),
        "compare-vision" => compare_vision::run(args),
        "decode-video" => decode_video::run(args),
        "denoise-cases" => denoise_cases::run(args),
        "dispatch-cost" => dispatch_cost::run(args),
        "encode-video" => encode_video::run(args),
        "inspect-adapter" => inspect_adapter::run(args),
        "parity-dump" => parity_dump::run(args),
        "resolve" => resolve::run(args),
        "text-in" => text_in::run(args),
        "tune-gemm" => tune_gemm::run(args),
        "two-sessions" => two_sessions::run(args),
        "verify-gemm" => verify_gemm::run(args),
        "verify-send" => verify_send::run(args),
        "verify-stack-kernels" => verify_stack_kernels::run(args),
        "verify-upload" => verify_upload::run(args),
        "vision-embed" => vision_embed::run(args),
        other => {
            eprintln!("h3-dev: unknown command {other:?}\n\ncommands:");
            for name in [
                "audio-vae",
                "compare-decode",
                "compare-gemm",
                "compile-probes",
                "compare-vision",
                "decode-video",
                "denoise-cases",
                "dispatch-cost",
                "encode-video",
                "inspect-adapter",
                "parity-dump",
                "resolve",
                "text-in",
                "tune-gemm",
                "two-sessions",
                "verify-gemm",
                "verify-send",
                "verify-stack-kernels",
                "verify-upload",
                "vision-embed",
            ] {
                eprintln!("  {name}");
            }
            std::process::exit(2);
        }
    }
}
