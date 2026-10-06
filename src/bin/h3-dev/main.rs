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
    let mut args = std::env::args().skip(1);
    let command = args.next().unwrap_or_default();
    let commands = [
        ("audio-vae", audio_vae::run as fn(Vec<String>)),
        ("compare-decode", compare_decode::run),
        ("compare-gemm", compare_gemm::run),
        ("compare-vision", compare_vision::run),
        ("compile-probes", compile_probes::run),
        ("decode-video", decode_video::run),
        ("denoise-cases", denoise_cases::run),
        ("dispatch-cost", dispatch_cost::run),
        ("encode-video", encode_video::run),
        ("inspect-adapter", inspect_adapter::run),
        ("parity-dump", parity_dump::run),
        ("resolve", resolve::run),
        ("text-in", text_in::run),
        ("tune-gemm", tune_gemm::run),
        ("two-sessions", two_sessions::run),
        ("verify-gemm", verify_gemm::run),
        ("verify-send", verify_send::run),
        ("verify-stack-kernels", verify_stack_kernels::run),
        ("verify-upload", verify_upload::run),
        ("vision-embed", vision_embed::run),
    ];
    if let Some((_, run)) = commands.iter().find(|(name, _)| *name == command) {
        run(args.collect());
    } else {
        eprintln!("h3-dev: unknown command {command:?}\n\ncommands:");
        for (name, _) in commands {
            eprintln!("  {name}");
        }
        std::process::exit(2);
    }
}
