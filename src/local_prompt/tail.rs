//! Pinned BF16 continuation of the quantized 50-layer H3 encoder.
use crate::{
    checkpoint::Checkpoint,
    model::*,
    weights::{interleave16, rows_of, widen_f32, Recipe, Weights},
    Result,
};
use hrx::artifacts::safetensors::DType;
use std::{collections::BTreeMap, path::PathBuf};
pub const REVISION: &str = "0cfaf48183f594c314753d30a4c4974bc75f3ccb";
pub const VOCAB: usize = 151936;
pub const LAYERS: usize = 14;

const SHAS: [&str; 4] = [
    "3820ffe8d8d6477f6fe8d614ef3c87abb264ee39accebf43a1507b970d80946f",
    "05ad2d08ce71963121c9b03f1d9ec5d7641052f4b23c6c12b80d71065eb8e98e",
    "b64f2289871261fdd1abbd3b78bcd66011b341de3dc8eeb2ed1a473ee7c8d95c",
    "e45b6c9998c77ee5a6577f9f47bc76416c1d4d387169e50c4c9d3134ea51b13b",
];
pub fn paths(dir: Option<&std::path::Path>, offline: bool) -> Result<Vec<PathBuf>> {
    use hrx::artifacts::hf::{HubFile, Repository, Resolver};
    let resolver = Resolver::new(Repository::new("Qwen", "Qwen3-VL-32B-Instruct").at(REVISION))
        .offline(offline);
    (11..=14)
        .map(|i| {
            let name = format!("model-{i:05}-of-00014.safetensors");
            if let Some(dir) = dir {
                let path = dir.join(name);
                if hrx::bundle::file_digest(&path)? != SHAS[i - 11] {
                    return crate::error::invalid(format!(
                        "{} does not match the pinned Qwen checkpoint",
                        path.display()
                    ));
                }
                Ok(path)
            } else {
                Ok(resolver.resolve(&HubFile::new(name).sha256(SHAS[i - 11]))?)
            }
        })
        .collect()
}
/// # Safety
/// The shards must remain unchanged while mapped.
pub unsafe fn open(paths: &[PathBuf]) -> Result<Weights> {
    Ok(unsafe { Weights::open_shards(paths, plan) }?)
}
fn plan(ck: &Checkpoint, out: &mut BTreeMap<String, Recipe>) -> crate::weights::Result<()> {
    let bf = |name: String, shape: &[i64]| -> crate::weights::Result<String> {
        ck.at_checked(&name, DType::BF16, shape)?;
        Ok(name)
    };
    let hid = TE_HID as i64;
    let inner = (TE_HEADS * HEAD_DIM) as i64;
    let kv = (TE_KV * HEAD_DIM) as i64;
    let pitch = gemm_pitch(TE_HID, 16) * 2;
    for i in 0..LAYERS {
        let src = format!("model.language_model.layers.{}.", i + 50);
        let dst = format!("blocks.{i}.");
        let q = bf(format!("{src}self_attn.q_proj.weight"), &[inner, hid])?;
        let k = bf(format!("{src}self_attn.k_proj.weight"), &[kv, hid])?;
        let v = bf(format!("{src}self_attn.v_proj.weight"), &[kv, hid])?;
        out.insert(format!("{dst}qkv.q"), rows_of(ck, &[&q, &k, &v], pitch)?);
        let o = bf(format!("{src}self_attn.o_proj.weight"), &[hid, inner])?;
        out.insert(
            format!("{dst}out.q"),
            rows_of(ck, &[&o], gemm_pitch(inner as usize, 16) * 2)?,
        );
        let g = bf(format!("{src}mlp.gate_proj.weight"), &[TE_FFN as i64, hid])?;
        let u = bf(format!("{src}mlp.up_proj.weight"), &[TE_FFN as i64, hid])?;
        out.insert(
            format!("{dst}gu.q"),
            interleave16(ck, (&g, 0), (&u, 0), TE_FFN, pitch)?,
        );
        let d = bf(format!("{src}mlp.down_proj.weight"), &[hid, TE_FFN as i64])?;
        out.insert(
            format!("{dst}down.q"),
            rows_of(ck, &[&d], gemm_pitch(TE_FFN, 16) * 2)?,
        );
        for (dst_name, src_name, size) in [
            ("norm1", "input_layernorm", hid),
            ("norm2", "post_attention_layernorm", hid),
            ("qnorm", "self_attn.q_norm", HEAD_DIM as i64),
            ("knorm", "self_attn.k_norm", HEAD_DIM as i64),
        ] {
            let name = bf(format!("{src}{src_name}.weight"), &[size])?;
            out.insert(format!("{dst}{dst_name}"), widen_f32(ck, &[&name])?);
        }
    }
    let norm = bf("model.language_model.norm.weight".into(), &[hid])?;
    out.insert("final_norm".into(), widen_f32(ck, &[&norm])?);
    let head = bf("lm_head.weight".into(), &[VOCAB as i64, hid])?;
    out.insert("lm_head".into(), rows_of(ck, &[&head], 0)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires pinned Qwen tail shards; H3_LOCAL_FETCH=1 permits provisioning"]
    fn pinned_tail_shards_have_the_required_shapes_and_only_load_the_continuation() {
        let paths = paths(None, std::env::var("H3_LOCAL_FETCH").as_deref() != Ok("1")).unwrap();
        // Safety: this test does not modify cached checkpoints.
        let weights = unsafe { open(&paths) }.unwrap();
        assert!(weights.has("blocks.13.down.q"));
        assert!(!weights.has("blocks.14.down.q"));
        assert!(!weights.has("vis.patch.w"));
        assert!(weights.device_bytes() > 14usize << 30);
        assert!(weights.device_bytes() < 16usize << 30);
        assert_eq!(
            weights
                .file()
                .bytes(weights.file().at("lm_head.weight").unwrap())
                .len(),
            VOCAB * TE_HID * 2
        );
        eprintln!("local tail resident bytes: {}", weights.device_bytes());
    }
}
