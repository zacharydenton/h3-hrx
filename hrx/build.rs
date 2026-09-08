// Where libhrx lives, matching scripts/env.sh (HRX_SYSTEM/HRX_BUILD).
//
// Only the link *search* path propagates from a dependency's build script to a dependent binary;
// `cargo:rustc-link-arg` does not. Each binary crate that links this one therefore emits its own
// `-rpath` — see loomrun/build.rs, which calls the same helper by duplicating these four lines.
fn main() {
    println!("cargo:rustc-link-search=native={}", hrx_lib_dir());
    println!("cargo:rerun-if-env-changed=HRX_SYSTEM");
    println!("cargo:rerun-if-env-changed=HRX_BUILD");
    println!("cargo:rerun-if-changed=build.rs");
}

fn hrx_lib_dir() -> String {
    let system = std::env::var("HRX_SYSTEM")
        .unwrap_or_else(|_| format!("{}/code/hrx-system", std::env::var("HOME").unwrap_or_default()));
    let build = std::env::var("HRX_BUILD").unwrap_or_else(|_| format!("{system}/build-cuda"));
    format!("{build}/libhrx/src/libhrx")
}
