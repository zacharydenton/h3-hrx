// Bake libhrx's directory into this binary's rpath, so `build/h3` runs from anywhere without
// LD_LIBRARY_PATH. The h3 crate links libhrx, but `cargo:rustc-link-arg` does not propagate across
// crates, so every package that produces an executable emits its own.
fn main() {
    let system = std::env::var("HRX_SYSTEM").unwrap_or_else(|_| {
        format!(
            "{}/code/hrx-system",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let build = std::env::var("HRX_BUILD").unwrap_or_else(|_| format!("{system}/build-cuda"));
    println!("cargo:rustc-link-arg=-Wl,-rpath,{build}/libhrx/src/libhrx");
    println!("cargo:rerun-if-env-changed=HRX_SYSTEM");
    println!("cargo:rerun-if-env-changed=HRX_BUILD");
    println!("cargo:rerun-if-changed=build.rs");
}
