// Bake libhrx's directory into the binary's rpath so it runs without LD_LIBRARY_PATH. The hrx crate
// contributes the link search path; link args do not propagate across crates, so this is here.
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
