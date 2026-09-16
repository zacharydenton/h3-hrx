//! ComfyUI's Qwen3-VL-32B text encoder (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`): the 50
//! layers' int8 ConvRot rows with their scales, the bf16 vision tower as stored — zero-padded only where
//! a kernel's K or N demands it — and the norms widened to f32.
//!
//! The embedding table is not a recipe: the text path reads one row per token straight from the mapping.
use crate::checkpoint::Checkpoint;
use crate::model::*;
use crate::weights::{
    interleave16, regroup, rows_of, rows_padded, scales_interleave16, scales_rows, widen_f32,
    widen_padded, Recipe, Result,
};
use hrx::artifacts::safetensors::DType as Dtype;
use std::collections::BTreeMap;

// The vision tower's own shapes.
const VHID: usize = 1152;
const VHEADS: usize = 16;
/// Its head dim is 72, but the rope kernel writes heads at 128.
const VHD: usize = 72;
const VHDP: usize = 128;
/// The MLP's N, and that N rounded up to the multiple of 64 the kernel needs.
const VMLP_SRC: usize = 4304;
const VMLP: usize = 4352;
const VOUT: usize = 5120;
const VMERGE: usize = 4 * VHID;
const VIS_BLOCKS: usize = 27;

type Table = BTreeMap<String, Recipe>;

pub fn plan(ck: &Checkpoint, out: &mut Table) -> Result<()> {
    // Validated even though it is never a recipe, so a wrong checkpoint fails at open like the rest.
    ck.at_checked(
        "model.embed_tokens.weight",
        Dtype::BF16,
        &[-1, TEXT_DIM as i64],
    )?;

    let i8w = |n: &str, rows: usize, cols: usize| -> Result<String> {
        let name = format!("{n}.weight");
        ck.at_checked(&name, Dtype::I8, &[rows as i64, cols as i64])?;
        Ok(name)
    };
    let scale = |n: &str, rows: usize| -> Result<String> {
        let name = format!("{n}.weight_scale");
        ck.at_checked(&name, Dtype::F32, &[rows as i64, 1])?;
        Ok(name)
    };
    let bf = |n: &str, shape: &[i64]| -> Result<String> {
        ck.at_checked(n, Dtype::BF16, shape)?;
        Ok(n.to_string())
    };

    let inner = TE_HEADS * HEAD_DIM;
    let kv = TE_KV * HEAD_DIM;
    let pitch = gemm_pitch(TE_HID, 8);

    for i in 0..TE_LAYERS {
        let p = format!("blocks.{i}.");
        let l = format!("model.layers.{i}.");
        let (q, k) = (
            format!("{l}self_attn.q_proj"),
            format!("{l}self_attn.k_proj"),
        );
        let (v, o) = (
            format!("{l}self_attn.v_proj"),
            format!("{l}self_attn.o_proj"),
        );

        out.insert(
            format!("{p}qkv.q"),
            rows_of(
                ck,
                &[
                    &i8w(&q, inner, TE_HID)?,
                    &i8w(&k, kv, TE_HID)?,
                    &i8w(&v, kv, TE_HID)?,
                ],
                pitch,
            )?,
        );
        out.insert(
            format!("{p}qkv.s"),
            scales_rows(ck, &[&scale(&q, inner)?, &scale(&k, kv)?, &scale(&v, kv)?])?,
        );
        out.insert(
            format!("{p}out.q"),
            rows_of(ck, &[&i8w(&o, TE_HID, inner)?], gemm_pitch(inner, 8))?,
        );
        out.insert(
            format!("{p}out.s"),
            scales_rows(ck, &[&scale(&o, TE_HID)?])?,
        );

        // gate and up are separate tensors here: interleaved in 16-row runs
        let (g, u) = (format!("{l}mlp.gate_proj"), format!("{l}mlp.up_proj"));
        let d = format!("{l}mlp.down_proj");
        let ge = i8w(&g, TE_FFN, TE_HID)?;
        let ue = i8w(&u, TE_FFN, TE_HID)?;
        out.insert(
            format!("{p}gu.q"),
            interleave16(ck, (&ge, 0), (&ue, 0), TE_FFN, pitch)?,
        );
        out.insert(
            format!("{p}gu.s"),
            scales_interleave16(
                ck,
                (&scale(&g, TE_FFN)?, 0),
                (&scale(&u, TE_FFN)?, 0),
                TE_FFN,
            )?,
        );
        out.insert(
            format!("{p}down.q"),
            rows_of(ck, &[&i8w(&d, TE_HID, TE_FFN)?], gemm_pitch(TE_FFN, 8))?,
        );
        out.insert(
            format!("{p}down.s"),
            scales_rows(ck, &[&scale(&d, TE_HID)?])?,
        );

        out.insert(
            format!("{p}norm1"),
            widen_f32(
                ck,
                &[&bf(
                    &format!("{l}input_layernorm.weight"),
                    &[TE_HID as i64],
                )?],
            )?,
        );
        out.insert(
            format!("{p}norm2"),
            widen_f32(
                ck,
                &[&bf(
                    &format!("{l}post_attention_layernorm.weight"),
                    &[TE_HID as i64],
                )?],
            )?,
        );
        out.insert(
            format!("{p}qnorm"),
            widen_f32(
                ck,
                &[&bf(
                    &format!("{l}self_attn.q_norm.weight"),
                    &[HEAD_DIM as i64],
                )?],
            )?,
        );
        out.insert(
            format!("{p}knorm"),
            widen_f32(
                ck,
                &[&bf(
                    &format!("{l}self_attn.k_norm.weight"),
                    &[HEAD_DIM as i64],
                )?],
            )?,
        );
    }

    // The vision tower. [1152][3*2*16*16 = 1536] is a contiguous reshape, so the rows travel as they are.
    out.insert(
        "vis.patch.w".into(),
        rows_of(
            ck,
            &[&bf(
                "visual.patch_embed.proj.weight",
                &[VHID as i64, 3, 2, 16, 16],
            )?],
            0,
        )?,
    );
    out.insert(
        "vis.patch.b".into(),
        widen_f32(ck, &[&bf("visual.patch_embed.proj.bias", &[VHID as i64])?])?,
    );
    out.insert(
        "vis.pos".into(),
        widen_f32(ck, &[&bf("visual.pos_embed.weight", &[2304, VHID as i64])?])?,
    );

    for i in 0..VIS_BLOCKS {
        let b = format!("vis.b{i}.");
        let src = format!("visual.blocks.{i}.");
        for n in ["norm1", "norm2"] {
            out.insert(
                format!("{b}{n}.w"),
                widen_f32(ck, &[&bf(&format!("{src}{n}.weight"), &[VHID as i64])?])?,
            );
            out.insert(
                format!("{b}{n}.b"),
                widen_f32(ck, &[&bf(&format!("{src}{n}.bias"), &[VHID as i64])?])?,
            );
        }
        out.insert(
            format!("{b}qkv.w"),
            rows_of(
                ck,
                &[&bf(
                    &format!("{src}attn.qkv.weight"),
                    &[(3 * VHEADS * VHD) as i64, VHID as i64],
                )?],
                0,
            )?,
        );
        out.insert(
            format!("{b}qkv.b"),
            widen_f32(
                ck,
                &[&bf(
                    &format!("{src}attn.qkv.bias"),
                    &[(3 * VHEADS * VHD) as i64],
                )?],
            )?,
        );
        // the rope kernel writes heads at 128, so each head's 72 channels are zero-extended
        out.insert(
            format!("{b}proj.w"),
            regroup(
                ck,
                &bf(
                    &format!("{src}attn.proj.weight"),
                    &[VHID as i64, (VHEADS * VHD) as i64],
                )?,
                VHEADS,
                VHD,
                VHDP,
                0,
            )?,
        );
        out.insert(
            format!("{b}proj.b"),
            widen_f32(ck, &[&bf(&format!("{src}attn.proj.bias"), &[VHID as i64])?])?,
        );
        // N to a multiple of 64, and the bias padded beside it
        out.insert(
            format!("{b}fc1.w"),
            rows_padded(
                ck,
                &bf(
                    &format!("{src}mlp.linear_fc1.weight"),
                    &[VMLP_SRC as i64, VHID as i64],
                )?,
                VMLP - VMLP_SRC,
                0,
            )?,
        );
        out.insert(
            format!("{b}fc1.b"),
            widen_padded(
                ck,
                &bf(&format!("{src}mlp.linear_fc1.bias"), &[VMLP_SRC as i64])?,
                VMLP,
            )?,
        );
        // and K to match on the way back down
        out.insert(
            format!("{b}fc2.w"),
            regroup(
                ck,
                &bf(
                    &format!("{src}mlp.linear_fc2.weight"),
                    &[VHID as i64, VMLP_SRC as i64],
                )?,
                1,
                VMLP_SRC,
                VMLP,
                0,
            )?,
        );
        out.insert(
            format!("{b}fc2.b"),
            widen_f32(
                ck,
                &[&bf(&format!("{src}mlp.linear_fc2.bias"), &[VHID as i64])?],
            )?,
        );
    }

    // The three DeepStack mergers and the patch merger. The mergers view [n][1152] as [n/4][4608], and
    // only the DeepStack norms are that wide.
    for j in 0..4 {
        let d = if j < 3 {
            format!("vis.ds{j}.")
        } else {
            "vis.merger.".to_string()
        };
        let src = if j < 3 {
            format!("visual.deepstack_merger_list.{j}.")
        } else {
            "visual.merger.".to_string()
        };
        let nw = if j < 3 { VMERGE } else { VHID };
        out.insert(
            format!("{d}norm.w"),
            widen_f32(ck, &[&bf(&format!("{src}norm.weight"), &[nw as i64])?])?,
        );
        out.insert(
            format!("{d}norm.b"),
            widen_f32(ck, &[&bf(&format!("{src}norm.bias"), &[nw as i64])?])?,
        );
        out.insert(
            format!("{d}fc1.w"),
            rows_of(
                ck,
                &[&bf(
                    &format!("{src}linear_fc1.weight"),
                    &[VMERGE as i64, VMERGE as i64],
                )?],
                0,
            )?,
        );
        out.insert(
            format!("{d}fc1.b"),
            widen_f32(
                ck,
                &[&bf(&format!("{src}linear_fc1.bias"), &[VMERGE as i64])?],
            )?,
        );
        out.insert(
            format!("{d}fc2.w"),
            rows_of(
                ck,
                &[&bf(
                    &format!("{src}linear_fc2.weight"),
                    &[VOUT as i64, VMERGE as i64],
                )?],
                0,
            )?,
        );
        out.insert(
            format!("{d}fc2.b"),
            widen_f32(
                ck,
                &[&bf(&format!("{src}linear_fc2.bias"), &[VOUT as i64])?],
            )?,
        );
    }
    Ok(())
}
