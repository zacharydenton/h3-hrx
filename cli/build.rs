// Link ../build/libh3pipe.so and bake its directory into the binary's rpath, so `build/h3` runs from
// anywhere without LD_LIBRARY_PATH. `scripts/build_host.sh` builds the library first.
fn main() {
    let lib = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../build"));
    let lib = std::fs::canonicalize(lib).expect("build the host first: scripts/build_host.sh");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
    println!("cargo:rerun-if-changed=build.rs");
}
