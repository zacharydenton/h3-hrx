//! Verify artifacts in the shared Rust kernel cache. Legacy Python/C cache
//! names use a different scheme and are not inputs to the shared Rust compiler.
//! Usage: verify_cache_tags <cache-dir> (or the former kernels/cache/compiler args).
use std::path::PathBuf;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let root = PathBuf::from(if args.len() >= 3 {
        &args[1]
    } else {
        args.first().ok_or("usage: verify_cache_tags <cache-dir>")?
    });
    let root = if root.join("hrx-v1").is_dir() {
        root.join("hrx-v1")
    } else {
        root
    };
    let mut verified = 0;
    for entry in std::fs::read_dir(&root)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        let artifact = path.join("kernel.hsaco");
        if !artifact.is_file() {
            continue;
        }
        let recorded = std::fs::read_to_string(path.join("kernel.sha256"))?;
        if hrx::bundle::file_digest(&artifact)? != recorded {
            return Err(format!("corrupt artifact {}", artifact.display()).into());
        }
        verified += 1;
    }
    println!(
        "verified {verified} shared cache artifacts in {}",
        root.display()
    );
    Ok(())
}
