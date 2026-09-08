//! ComfyUI's video VAE (`minimax_h3_video_vae_fp16.safetensors`), f16 throughout and run as f16.
//!
//! The decoder's 36 blocks need two reorderings, both done here rather than converted: its fused
//! `to_qkv` rows are head-interleaved and the rope kernel wants `[Q | K | V]`, and its gate|up half is
//! `w1`'s second half first, because the SwiGLU epilogue takes the linear half before the gate. The
//! encoder's causal 3-D convs become the implicit-GEMM operands the conv kernels take.
use crate::checkpoint::Checkpoint;
use crate::model::*;
use crate::weights::{
    conv3d_taps, interleaved_runs, regroup, rows_of, rows_permuted, up, widen_f32, widen_padded,
    widen_runs, Recipe, Result, Run,
};
use safetensors::tensor::Dtype;
use std::collections::BTreeMap;

type Table = BTreeMap<String, Recipe>;

/// The encoder's channel widths per level; its length is the number of levels.
const MID: [usize; 6] = [128, 256, 256, 512, 512, 1024];
/// The encoder's output channels before the quantiser, and the quantiser's own width.
const ENC_OUT: usize = 48;

pub fn plan(ck: &Checkpoint, out: &mut Table) -> Result<()> {
    let f16 = |n: &str, shape: &[i64]| -> Result<String> {
        ck.at_checked(n, Dtype::F16, shape)?;
        Ok(n.to_string())
    };

    let inner = VAE_HEADS * VAE_D;
    // operand row pitches, in bytes: these operands are 16-bit
    let hid_pitch = gemm_pitch(VAE_HID, 16) * 2;
    let inner_pitch = gemm_pitch(inner, 16) * 2;
    let ffn_pitch = gemm_pitch(VAE_FFN, 16) * 2;

    for i in 0..VAE_BLOCKS {
        let p = format!("blocks.{i}.");
        let src = format!("decoder.transformer_blocks.{i}.");

        // to_qkv holds [q | k | v] per head of 64; the rope kernel wants [Q | K | V]
        let mut qkv_rows: Vec<Run> = Vec::new();
        for part in 0..3 {
            for head in 0..VAE_HEADS {
                qkv_rows.push(((head * 3 + part) * VAE_D, VAE_D));
            }
        }
        out.insert(
            format!("{p}qkv.q"),
            rows_permuted(
                ck,
                &f16(
                    &format!("{src}attn.to_qkv.weight"),
                    &[(3 * inner) as i64, VAE_HID as i64],
                )?,
                &qkv_rows,
                hid_pitch,
            )?,
        );
        out.insert(
            format!("{p}qkv.b"),
            widen_runs(
                ck,
                &f16(&format!("{src}attn.to_qkv.bias"), &[(3 * inner) as i64])?,
                &qkv_rows,
            )?,
        );
        out.insert(
            format!("{p}out.q"),
            rows_of(
                ck,
                &[&f16(
                    &format!("{src}attn.to_out.weight"),
                    &[VAE_HID as i64, inner as i64],
                )?],
                inner_pitch,
            )?,
        );
        out.insert(
            format!("{p}out.b"),
            widen_f32(
                ck,
                &[&f16(&format!("{src}attn.to_out.bias"), &[VAE_HID as i64])?],
            )?,
        );

        // w1 is gate | linear (comfy/ldm/minimax/vae.py: `gate, x = w1(x).chunk(2)`); the epilogue takes
        // the linear half first, so the runs start at VAE_FFN.
        let gu = interleaved_runs(VAE_FFN, 0, VAE_FFN, 16);
        out.insert(
            format!("{p}gu.q"),
            rows_permuted(
                ck,
                &f16(
                    &format!("{src}ff.w1.weight"),
                    &[(2 * VAE_FFN) as i64, VAE_HID as i64],
                )?,
                &gu,
                hid_pitch,
            )?,
        );
        out.insert(
            format!("{p}gu.b"),
            widen_runs(
                ck,
                &f16(&format!("{src}ff.w1.bias"), &[(2 * VAE_FFN) as i64])?,
                &gu,
            )?,
        );
        out.insert(
            format!("{p}down.q"),
            rows_of(
                ck,
                &[&f16(
                    &format!("{src}ff.w2.weight"),
                    &[VAE_HID as i64, VAE_FFN as i64],
                )?],
                ffn_pitch,
            )?,
        );
        out.insert(
            format!("{p}down.b"),
            widen_f32(ck, &[&f16(&format!("{src}ff.w2.bias"), &[VAE_HID as i64])?])?,
        );
        for n in ["norm1", "norm2"] {
            out.insert(
                format!("{p}{n}"),
                widen_f32(ck, &[&f16(&format!("{src}{n}.weight"), &[VAE_HID as i64])?])?,
            );
        }
        for n in ["scale1", "scale2"] {
            out.insert(
                format!("{p}{n}"),
                widen_f32(ck, &[&f16(&format!("{src}{n}"), &[VAE_HID as i64])?])?,
            );
        }
    }

    // K to the f16 GEMM's multiple of 64
    out.insert(
        "vae.proj_in.w".into(),
        regroup(
            ck,
            &f16(
                "decoder.x_embedder.weight",
                &[VAE_HID as i64, LATENT_CH as i64],
            )?,
            1,
            LATENT_CH,
            VAE_KIN,
            0,
        )?,
    );
    out.insert(
        "vae.proj_in.b".into(),
        widen_f32(ck, &[&f16("decoder.x_embedder.bias", &[VAE_HID as i64])?])?,
    );
    out.insert(
        "vae.register_tokens".into(),
        widen_f32(
            ck,
            &[&f16(
                "decoder.register_tokens",
                &[1, VAE_REG as i64, VAE_HID as i64],
            )?],
        )?,
    );
    out.insert(
        "vae.norm_out.w".into(),
        widen_f32(ck, &[&f16("decoder.norm_out.weight", &[VAE_HID as i64])?])?,
    );
    out.insert(
        "vae.norm_out.b".into(),
        widen_f32(ck, &[&f16("decoder.norm_out.bias", &[VAE_HID as i64])?])?,
    );
    out.insert(
        "vae.proj_out.w".into(),
        rows_of(
            ck,
            &[&f16(
                "decoder.proj_out.weight",
                &[VAE_OUT as i64, VAE_HID as i64],
            )?],
            0,
        )?,
    );
    out.insert(
        "vae.proj_out.b".into(),
        widen_f32(ck, &[&f16("decoder.proj_out.bias", &[VAE_OUT as i64])?])?,
    );
    out.insert(
        "vae.post_quant_conv.w".into(),
        widen_f32(
            ck,
            &[&f16(
                "post_quant_conv.weight",
                &[LATENT_CH as i64, LATENT_CH as i64, 1, 1, 1],
            )?],
        )?,
    );
    out.insert(
        "vae.post_quant_conv.b".into(),
        widen_f32(ck, &[&f16("post_quant_conv.bias", &[LATENT_CH as i64])?])?,
    );
    out.insert(
        "vae.latents_mean".into(),
        widen_f32(ck, &[&f16("latents_mean", &[LATENT_CH as i64])?])?,
    );
    out.insert(
        "vae.latents_std".into(),
        widen_f32(ck, &[&f16("latents_std", &[LATENT_CH as i64])?])?,
    );

    encoder(ck, out)
}

/// Every conv weight as the implicit GEMM's operand rows, in both the clip and image forms, plus the
/// bias padded to the channels the conv actually writes.
fn conv(
    ck: &Checkpoint,
    out: &mut Table,
    name: &str,
    src: &str,
    cout: usize,
    cin: usize,
) -> Result<()> {
    let weight = format!("{src}.weight");
    ck.at_checked(&weight, Dtype::F16, &[cout as i64, cin as i64, 3, 3, 3])?;
    out.insert(
        format!("{name}.w3"),
        conv3d_taps(ck, &weight, cout, cin, 27)?,
    );
    out.insert(
        format!("{name}.w2"),
        conv3d_taps(ck, &weight, cout, cin, 9)?,
    );
    let bias = format!("{src}.bias");
    ck.at_checked(&bias, Dtype::F16, &[cout as i64])?;
    out.insert(format!("{name}.b"), widen_padded(ck, &bias, up(cout, 64))?);
    Ok(())
}

/// A 1x1x1 conv as `[Cout_pad][K]`.
fn matmul_w(
    ck: &Checkpoint,
    out: &mut Table,
    name: &str,
    src: &str,
    cout: usize,
    cin: usize,
) -> Result<()> {
    let weight = format!("{src}.weight");
    ck.at_checked(&weight, Dtype::F16, &[cout as i64, cin as i64, 1, 1, 1])?;
    out.insert(
        format!("{name}.wm"),
        regroup(ck, &weight, 1, cin, up(up(cin, 8), 32), up(cout, 64))?,
    );
    let bias = format!("{src}.bias");
    ck.at_checked(&bias, Dtype::F16, &[cout as i64])?;
    out.insert(format!("{name}.b"), widen_padded(ck, &bias, up(cout, 64))?);
    Ok(())
}

fn encoder(ck: &Checkpoint, out: &mut Table) -> Result<()> {
    let f16 = |n: &str, len: usize| -> Result<String> {
        ck.at_checked(n, Dtype::F16, &[len as i64])?;
        Ok(n.to_string())
    };

    conv(ck, out, "venc.conv_in", "encoder.conv_in", 128, 3)?;
    let mut c = 128usize;
    for (l, &mid) in MID.iter().enumerate() {
        for r in 0..2 {
            let b = format!("venc.l{l}.r{r}.");
            let src = format!("encoder.down.{l}.block.{r}.");
            // norm1 sees the incoming channels, norm2 the level's
            for (n, ch) in [("norm1", c), ("norm2", mid)] {
                out.insert(
                    format!("{b}{n}.g"),
                    widen_f32(ck, &[&f16(&format!("{src}{n}.weight"), ch)?])?,
                );
                out.insert(
                    format!("{b}{n}.b"),
                    widen_f32(ck, &[&f16(&format!("{src}{n}.bias"), ch)?])?,
                );
            }
            conv(
                ck,
                out,
                &format!("{b}conv1"),
                &format!("{src}conv1"),
                mid,
                c,
            )?;
            conv(
                ck,
                out,
                &format!("{b}conv2"),
                &format!("{src}conv2"),
                mid,
                mid,
            )?;
            if c != mid {
                matmul_w(
                    ck,
                    out,
                    &format!("{b}nin"),
                    &format!("{src}nin_shortcut"),
                    mid,
                    c,
                )?;
            }
            c = mid;
        }
        // the last level does not downsample
        if ck.has(&format!("encoder.down.{l}.downsample.conv.weight")) {
            conv(
                ck,
                out,
                &format!("venc.l{l}.down"),
                &format!("encoder.down.{l}.downsample.conv"),
                c,
                c,
            )?;
        }
    }
    out.insert(
        "venc.norm_out.g".into(),
        widen_f32(ck, &[&f16("encoder.norm_out.weight", c)?])?,
    );
    out.insert(
        "venc.norm_out.b".into(),
        widen_f32(ck, &[&f16("encoder.norm_out.bias", c)?])?,
    );
    conv(ck, out, "venc.conv_out", "encoder.conv_out", ENC_OUT, c)?;
    matmul_w(ck, out, "venc.quant", "quant_conv", ENC_OUT, ENC_OUT)?;
    out.insert(
        "venc.latents_mean".into(),
        widen_f32(ck, &[&f16("latents_mean", LATENT_CH)?])?,
    );
    out.insert(
        "venc.latents_std".into(),
        widen_f32(ck, &[&f16("latents_std", LATENT_CH)?])?,
    );
    Ok(())
}
