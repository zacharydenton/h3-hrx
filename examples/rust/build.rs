// Link ../../build/libh3pipe.so and bake its directory into the binary's rpath.
fn main() {
    let lib = std::fs::canonicalize(concat!(env!("CARGO_MANIFEST_DIR"), "/../../build")).expect("build the host first: scripts/build_host.sh");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rerun-if-changed=build.rs");
}
