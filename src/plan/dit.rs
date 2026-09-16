//! ComfyUI's H3 checkpoint (`minimax_h3_{fl2va,ref2va}_pruned_int8_convrot.safetensors`) laid out as the
//! DiT stack, the token refiner, the embedders, the final layer and the conditioning tables.
//!
//! The 50 blocks' int8 ConvRot rows travel with their scales; everything else stays in the type the
//! checkpoint stores it in — a bf16 refiner and condition projection, f32 patch projections and heads,
//! f16 AdaLN projections widened for the CPU that reads them. Laid out for the kernels, never converted.
use crate::checkpoint::Checkpoint;
use crate::model::*;
use crate::weights::{
    interleave16, rows_of, scales_interleave16, scales_rows, widen_f32, Recipe, Result,
};
use hrx::artifacts::safetensors::DType as Dtype;
use std::collections::BTreeMap;

type Table = BTreeMap<String, Recipe>;

pub fn plan(ck: &Checkpoint, out: &mut Table) -> Result<()> {
    // Every source tensor is checked here, so a checkpoint that does not match fails at open with the
    // offending tensor named rather than part-way through a generation.
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
    let vec_of = |n: &str, dtype: Dtype, len: usize| -> Result<String> {
        ck.at_checked(n, dtype, &[len as i64])?;
        Ok(n.to_string())
    };
    let mat = |n: &str, dtype: Dtype, rows: usize, cols: usize| -> Result<String> {
        ck.at_checked(n, dtype, &[rows as i64, cols as i64])?;
        Ok(n.to_string())
    };

    for i in 0..BLOCKS {
        let p = format!("blocks.{i}.");
        let (q, o) = (format!("{p}attn.qkv_proj"), format!("{p}attn.out_proj"));
        let (f1, f2) = (format!("{p}mlp.fc1"), format!("{p}mlp.fc2"));

        out.insert(
            format!("{p}qkv.q"),
            rows_of(ck, &[&i8w(&q, QKV, HID)?], gemm_pitch(HID, 8))?,
        );
        out.insert(format!("{p}qkv.s"), scales_rows(ck, &[&scale(&q, QKV)?])?);
        out.insert(
            format!("{p}out.q"),
            rows_of(ck, &[&i8w(&o, HID, INNER)?], gemm_pitch(INNER, 8))?,
        );
        out.insert(format!("{p}out.s"), scales_rows(ck, &[&scale(&o, HID)?])?);

        // gate rows first, then up: interleaved in 16-row runs for the fused SwiGLU
        let gu = i8w(&f1, 2 * FFN, HID)?;
        let gus = scale(&f1, 2 * FFN)?;
        out.insert(
            format!("{p}gu.q"),
            interleave16(ck, (&gu, 0), (&gu, FFN), FFN, gemm_pitch(HID, 8))?,
        );
        out.insert(
            format!("{p}gu.s"),
            scales_interleave16(ck, (&gus, 0), (&gus, FFN), FFN)?,
        );

        out.insert(
            format!("{p}down.q"),
            rows_of(ck, &[&i8w(&f2, HID, FFN)?], gemm_pitch(FFN, 8))?,
        );
        out.insert(format!("{p}down.s"), scales_rows(ck, &[&scale(&f2, HID)?])?);

        out.insert(
            format!("{p}norm1"),
            widen_f32(
                ck,
                &[&vec_of(&format!("{p}norm1.weight"), Dtype::BF16, HID)?],
            )?,
        );
        out.insert(
            format!("{p}norm2"),
            widen_f32(
                ck,
                &[&vec_of(&format!("{p}norm2.weight"), Dtype::BF16, HID)?],
            )?,
        );
        out.insert(
            format!("{p}qnorm"),
            widen_f32(
                ck,
                &[&vec_of(
                    &format!("{p}attn.q_norm.weight"),
                    Dtype::BF16,
                    HEAD_DIM,
                )?],
            )?,
        );
        out.insert(
            format!("{p}knorm"),
            widen_f32(
                ck,
                &[&vec_of(
                    &format!("{p}attn.k_norm.weight"),
                    Dtype::BF16,
                    HEAD_DIM,
                )?],
            )?,
        );

        out.insert(
            format!("h3.blocks.{i}.adaln.w"),
            widen_f32(
                ck,
                &[&mat(
                    &format!("{p}adaln_proj.linear.weight"),
                    Dtype::F16,
                    MODALITIES * 6 * HID,
                    8,
                )?],
            )?,
        );
        out.insert(
            format!("h3.blocks.{i}.adaln.b"),
            widen_f32(
                ck,
                &[&vec_of(
                    &format!("{p}adaln_proj.linear.bias"),
                    Dtype::F16,
                    MODALITIES * 6 * HID,
                )?],
            )?,
        );
    }

    out.insert(
        "h3.adaln_t_table".into(),
        widen_f32(ck, &[&mat("adaln_t_table", Dtype::F32, 1025, 8)?])?,
    );
    out.insert(
        "h3.rope_inv_freq".into(),
        widen_f32(ck, &[&vec_of("rope.inv_freq", Dtype::F32, 16)?])?,
    );
    out.insert(
        "h3.final.adaln.w".into(),
        widen_f32(
            ck,
            &[&mat(
                "final_layer.adaln_proj.linear.weight",
                Dtype::F16,
                2 * HID,
                8,
            )?],
        )?,
    );
    out.insert(
        "h3.final.adaln.b".into(),
        widen_f32(
            ck,
            &[&vec_of(
                "final_layer.adaln_proj.linear.bias",
                Dtype::F16,
                2 * HID,
            )?],
        )?,
    );
    out.insert(
        "h3.final.norm".into(),
        widen_f32(ck, &[&vec_of("final_layer.norm.weight", Dtype::BF16, HID)?])?,
    );

    // the two heads stacked to N = 128 (video 96 | audio 32) for one f32 matmul
    out.insert(
        "h3.final.out.w".into(),
        rows_of(
            ck,
            &[
                &mat("final_layer.video_out.weight", Dtype::F32, VIDEO_PATCH, HID)?,
                &mat("final_layer.audio_out.weight", Dtype::F32, AUDIO_CH, HID)?,
            ],
            0,
        )?,
    );
    out.insert(
        "h3.final.out.b".into(),
        widen_f32(
            ck,
            &[
                &vec_of("final_layer.video_out.bias", Dtype::F32, VIDEO_PATCH)?,
                &vec_of("final_layer.audio_out.bias", Dtype::F32, AUDIO_CH)?,
            ],
        )?,
    );

    out.insert(
        "h3.cond.w".into(),
        rows_of(
            ck,
            &[&mat("condition_proj.weight", Dtype::BF16, HID, TEXT_DIM)?],
            0,
        )?,
    );
    out.insert(
        "h3.cond.b".into(),
        widen_f32(ck, &[&vec_of("condition_proj.bias", Dtype::BF16, HID)?])?,
    );
    out.insert(
        "h3.video_in.w".into(),
        rows_of(
            ck,
            &[&mat(
                "video_patch_proj.weight",
                Dtype::F32,
                HID,
                VIDEO_PATCH,
            )?],
            0,
        )?,
    );
    out.insert(
        "h3.video_in.b".into(),
        widen_f32(ck, &[&vec_of("video_patch_proj.bias", Dtype::F32, HID)?])?,
    );
    out.insert(
        "h3.audio_in.w".into(),
        rows_of(
            ck,
            &[&mat("audio_patch_proj.weight", Dtype::F32, HID, AUDIO_CH)?],
            0,
        )?,
    );
    out.insert(
        "h3.audio_in.b".into(),
        widen_f32(ck, &[&vec_of("audio_patch_proj.bias", Dtype::F32, HID)?])?,
    );

    // The token refiner: bf16 rows as stored (unrotated), on the bf16 GEMMs. Its pitch is in bytes
    // because these operands are 16-bit, so the element pitch doubles.
    for j in 0..REFINER_BLOCKS {
        let p = format!("h3.refiner.{j}.");
        let src = format!("token_refiner.blocks.{j}.");
        out.insert(
            format!("{p}qkv.q"),
            rows_of(
                ck,
                &[&mat(
                    &format!("{src}attn.qkv_proj.weight"),
                    Dtype::BF16,
                    QKV,
                    HID,
                )?],
                gemm_pitch(HID, 16) * 2,
            )?,
        );
        out.insert(
            format!("{p}out.q"),
            rows_of(
                ck,
                &[&mat(
                    &format!("{src}attn.out_proj.weight"),
                    Dtype::BF16,
                    HID,
                    INNER,
                )?],
                gemm_pitch(INNER, 16) * 2,
            )?,
        );
        let gu = mat(&format!("{src}mlp.fc1.weight"), Dtype::BF16, 2 * FFN, HID)?;
        out.insert(
            format!("{p}gu.q"),
            interleave16(ck, (&gu, 0), (&gu, FFN), FFN, gemm_pitch(HID, 16) * 2)?,
        );
        out.insert(
            format!("{p}down.q"),
            rows_of(
                ck,
                &[&mat(
                    &format!("{src}mlp.fc2.weight"),
                    Dtype::BF16,
                    HID,
                    FFN,
                )?],
                gemm_pitch(FFN, 16) * 2,
            )?,
        );
        out.insert(
            format!("{p}norm1"),
            widen_f32(
                ck,
                &[&vec_of(&format!("{src}norm1.weight"), Dtype::BF16, HID)?],
            )?,
        );
        out.insert(
            format!("{p}norm2"),
            widen_f32(
                ck,
                &[&vec_of(&format!("{src}norm2.weight"), Dtype::BF16, HID)?],
            )?,
        );
        out.insert(
            format!("{p}qnorm"),
            widen_f32(
                ck,
                &[&vec_of(
                    &format!("{src}attn.q_norm.weight"),
                    Dtype::BF16,
                    HEAD_DIM,
                )?],
            )?,
        );
        out.insert(
            format!("{p}knorm"),
            widen_f32(
                ck,
                &[&vec_of(
                    &format!("{src}attn.k_norm.weight"),
                    Dtype::BF16,
                    HEAD_DIM,
                )?],
            )?,
        );
    }
    out.insert(
        "h3.refiner.final_norm".into(),
        widen_f32(
            ck,
            &[&vec_of(
                "token_refiner.final_norm.weight",
                Dtype::BF16,
                HID,
            )?],
        )?,
    );
    Ok(())
}
