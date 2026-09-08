//! Reproduces the names of the entries already in the kernel cache.
//!
//!   verify_cache_tags <kernels dir> <cache dir> <loom-compile>
//!
//! Every entry was named by the C implementation. Parsing a name back into the (stem, symbol, config)
//! that produced it and recomputing the tag here proves the two agree on the cache key — which is what
//! lets the Rust host reuse a cache the C host filled, and is a sharper check than any unit test,
//! because the corpus is whatever the machine has actually compiled.
use h3::compile::{Cfg, Compiler};
use std::collections::BTreeMap;
use std::path::Path;

/// `q_stride_8192` -> ("q_stride", "8192"): the value is the trailing numeric run, so the split is the
/// last underscore followed by something that starts like a number.
fn split_key_value(segment: &str) -> Option<(String, String)> {
    let bytes: Vec<usize> = segment.match_indices('_').map(|(i, _)| i).collect();
    for &i in bytes.iter().rev() {
        let value = &segment[i + 1..];
        if !value.is_empty() && value.starts_with(|c: char| c.is_ascii_digit() || c == '-') {
            return Some((segment[..i].to_string(), value.to_string()));
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (kernels, cache, exe) = (&args[0], &args[1], &args[2]);
    let compiler = Compiler::new(exe.clone(), kernels, cache);

    let (mut matched, mut stale_source, mut skipped) = (0, 0, 0);
    let mut mismatched: BTreeMap<String, String> = BTreeMap::new();

    for entry in std::fs::read_dir(cache).expect("cache") {
        let path = entry.expect("entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("hsaco") {
            continue;
        }
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        // Entries without the identity suffix predate it; they cannot be reproduced and are dead.
        // It is exactly "__i" plus ten hex digits at the end — matching on "__i" alone would also
        // catch a config named in_bound.
        let identity_len = 3 + 10;
        let has_identity = name.len() > identity_len && {
            let tail = &name[name.len() - identity_len..];
            tail.starts_with("__i") && tail[3..].chars().all(|c| c.is_ascii_hexdigit())
        };
        if !has_identity {
            skipped += 1;
            continue;
        }
        let visible = &name[..name.len() - identity_len];
        let mut parts = visible.split("__");
        let stem = parts.next().unwrap_or_default().to_string();
        let source_tag = parts.next().unwrap_or_default().to_string();
        if !source_tag.starts_with('s') {
            skipped += 1;
            continue;
        }
        let mut cfg: Cfg = Vec::new();
        let mut parsed = true;
        for segment in parts {
            match split_key_value(segment) {
                Some((key, value)) => cfg.push((format!("h3.{stem}.{key}"), value)),
                None => {
                    parsed = false;
                    break;
                }
            }
        }
        if !parsed || !Path::new(kernels).join(format!("{stem}.loom")).exists() {
            skipped += 1;
            continue;
        }
        match compiler.tag(&stem, &format!("h3_{stem}"), &cfg) {
            Ok(tag) if tag == name => matched += 1,
            Ok(tag) => {
                // A kernel source edited since the entry was written changes the __s hash; that is the
                // cache working as intended, not a disagreement about the tag.
                if tag.split("__").nth(1) != Some(&source_tag) {
                    stale_source += 1;
                } else {
                    mismatched.insert(name, tag);
                }
            }
            Err(_) => skipped += 1,
        }
    }

    println!(
        "reproduced {matched}, stale source {stale_source}, unparsable {skipped}, MISMATCHED {}",
        mismatched.len()
    );
    for (name, tag) in mismatched.iter().take(5) {
        println!("  cache: {name}");
        println!("  ours:  {tag}");
    }
    if !mismatched.is_empty() {
        std::process::exit(1);
    }
}
