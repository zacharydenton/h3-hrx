//! Uploads every recipe a plan builds and reads it back, comparing with the host-assembled bytes.
//!
//!   verify_upload <plan> <checkpoint>
//!
//! This is what proves the device path, as distinct from the layout: the three upload routes — a built
//! array, one straight run of the mapping, and rows gathered at a wider pitch — must all land exactly
//! what `assemble` produces, pad included.
use h3::checkpoint::Checkpoint;
use h3::weights::{Recipe, Weights};
use std::collections::BTreeMap;

fn plan_for(
    which: &str,
) -> fn(&Checkpoint, &mut BTreeMap<String, Recipe>) -> h3::weights::Result<()> {
    match which {
        "dit" => h3::plan::dit::plan,
        "te" => h3::plan::te::plan,
        "vvae" => h3::plan::vvae::plan,
        "avae" => h3::plan::avae::plan,
        other => panic!("unknown plan {other}"),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let which = args
        .next()
        .expect("usage: verify_upload <plan> <checkpoint>");
    let path = args
        .next()
        .expect("usage: verify_upload <plan> <checkpoint>");
    let limit: usize = std::env::var("H3_VERIFY_MAX_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(usize::MAX);

    let gpu = hrx::Gpu::open().expect("gpu");
    let weights = Weights::open(&path, plan_for(&which)).expect("plan");

    let names: Vec<String> = weights.names().cloned().collect();
    let (mut checked, mut skipped, mut bytes) = (0usize, 0usize, 0usize);
    let mut kinds = BTreeMap::new();
    for name in &names {
        let recipe = weights.recipe(name).expect("recipe");
        let size = recipe.device_bytes();
        if size > limit {
            skipped += 1;
            continue;
        }
        // which of the three routes this one takes, so the report shows all were exercised
        let kind = match recipe {
            Recipe::Built { .. } => "built",
            Recipe::Rows {
                row_bytes,
                pitch_bytes,
                segments,
                ..
            } if segments.len() == 1 && row_bytes == pitch_bytes => "run",
            Recipe::Rows { .. } => "gathered",
        };
        *kinds.entry(kind).or_insert(0usize) += 1;

        let want = recipe.assemble(weights.file()).expect("assemble");
        let buffer = weights.at(&gpu, name, size).expect("upload");
        let mut got = vec![0u8; size];
        gpu.d2h(&buffer, &mut got).expect("read back");
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap_or(0);
            println!("MISMATCH {name} ({kind}, {size} bytes) first differs at {at}");
            std::process::exit(1);
        }
        checked += 1;
        bytes += size;
    }
    println!(
        "{which}: {checked} tensors identical after a round trip ({:.2} GB), {skipped} over the limit",
        bytes as f64 / 1e9
    );
    for (kind, count) in kinds {
        println!("  {kind}: {count}");
    }
}
