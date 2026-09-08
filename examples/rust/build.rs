// Bake libhrx's directory into this example's rpath: it links the h3 crate, which links libhrx.

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
