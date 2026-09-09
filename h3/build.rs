//! The embedded Loom sources; no headers, no native linker paths, no build downloads.
use std::{fs, path::PathBuf};
fn main() {
    let root = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
    let kernels = root.join("kernels");
    println!("cargo:rerun-if-changed={}", kernels.display());
    let mut entries: Vec<_> = fs::read_dir(&kernels)
        .expect("packaged H3 kernel sources")
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "loom"))
        .collect();
    entries.sort();
    let mut code =
        String::from("fn embedded_source(stem: &str) -> Option<&'static str> { match stem {\n");
    for path in entries {
        code.push_str(&format!(
            "{:?} => Some(include_str!({:?})),\n",
            path.file_stem().unwrap().to_str().unwrap(),
            path
        ));
    }
    code.push_str("_ => None, } }\n");
    fs::write(out.join("kernel_sources.rs"), code).unwrap();
}
