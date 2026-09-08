//! ComfyUI's audio VAE (`minimax_h3_audio_vae_fp32.safetensors`), f32 throughout with its weight norm
//! already folded: the BigVGAN vocoder and the encoder's DAC stack.
//!
//! Verbatim except for two things: the conv reshapes the kernels' layouts want, and `exp()` of the
//! decoder's SnakeBeta parameters, which the module stores as logs and evaluates per call
//! (`comfy/ldm/minimax/audio_vae.py`). The encoder's Snake takes its alpha as stored.
use crate::checkpoint::Checkpoint;
use crate::model::*;
use crate::weights::{rows_of, widen_f32, Recipe, Result};
use safetensors::tensor::Dtype;
use std::collections::BTreeMap;

type Table = BTreeMap<String, Recipe>;

/// The vocoder's upsample kernel widths, its residual kernel widths, and the encoder's rates.
const UPK: [usize; 7] = [9, 9, 4, 4, 4, 4, 4];
const RESK: [usize; 3] = [3, 7, 11];
const ARATES: [usize; 5] = [2, 4, 4, 5, 5];

pub fn plan(ck: &Checkpoint, out: &mut Table) -> Result<()> {
    let f32v = |n: &str, len: usize| -> Result<String> {
        ck.at_checked(n, Dtype::F32, &[len as i64])?;
        Ok(n.to_string())
    };

    out.insert(
        "audio.latents_mean".into(),
        widen_f32(ck, &[&f32v("latents_mean", AUDIO_CH)?])?,
    );
    out.insert(
        "audio.latents_std".into(),
        widen_f32(ck, &[&f32v("latents_std", AUDIO_CH)?])?,
    );

    flat(
        ck,
        out,
        "audio.dec_in_proj",
        "dec_in_proj",
        &[2048, AUDIO_CH as i64, 1],
        None,
    )?;
    flat(
        ck,
        out,
        "audio.conv_pre",
        "decoder.conv_pre",
        &[1024, 2048, 7],
        None,
    )?;

    // The decoder's rates {5,5,2,2,2,2,2} halve the channels, which the loop tracks.
    let mut c = 1024usize;
    for (i, upk) in UPK.iter().enumerate() {
        let cout = c / 2;
        // a transposed conv: [Cin][Cout][k], so the bias is Cout wide, not the first dimension
        flat(
            ck,
            out,
            &format!("audio.ups.{i}"),
            &format!("decoder.ups.{i}.0"),
            &[c as i64, cout as i64, *upk as i64],
            Some(cout),
        )?;
        for (j, resk) in RESK.iter().enumerate() {
            let r = i * 3 + j;
            let src = format!("decoder.resblocks.{r}");
            for d in 0..3 {
                flat(
                    ck,
                    out,
                    &format!("audio.res.{r}.c1.{d}"),
                    &format!("{src}.convs1.{d}"),
                    &[cout as i64, cout as i64, *resk as i64],
                    None,
                )?;
                flat(
                    ck,
                    out,
                    &format!("audio.res.{r}.c2.{d}"),
                    &format!("{src}.convs2.{d}"),
                    &[cout as i64, cout as i64, *resk as i64],
                    None,
                )?;
            }
            for act in 0..6 {
                out.insert(
                    format!("audio.res.{r}.act.{act}.alpha"),
                    exp_f32(ck, &format!("{src}.activations.{act}.act.alpha"), cout)?,
                );
                out.insert(
                    format!("audio.res.{r}.act.{act}.beta"),
                    exp_f32(ck, &format!("{src}.activations.{act}.act.beta"), cout)?,
                );
            }
        }
        c = cout;
    }
    out.insert(
        "audio.post.alpha".into(),
        exp_f32(ck, "decoder.activation_post.act.alpha", c)?,
    );
    out.insert(
        "audio.post.beta".into(),
        exp_f32(ck, "decoder.activation_post.act.beta", c)?,
    );

    // The same 12-tap Kaiser-sinc appears in every activation; sharing one copy is only valid if they
    // really are identical, so that is checked rather than assumed.
    let filter = "decoder.activation_post.upsample.filter";
    let reference = ck.at_checked(filter, Dtype::F32, &[1, 1, 12])?;
    let want = ck.bytes(reference);
    for (name, entry) in ck.entries() {
        if name.ends_with(".filter") && (entry.bytes != reference.bytes || ck.bytes(entry) != want)
        {
            return Err(crate::weights::Error::Layout(format!(
                "the resampling filters differ: {name}"
            )));
        }
    }
    out.insert("audio.fir".into(), rows_of(ck, &[filter], 0)?);
    flat(
        ck,
        out,
        "audio.conv_post",
        "decoder.conv_post",
        &[1, c as i64, 7],
        Some(0),
    )?;

    encoder(ck, out)
}

/// `[Cout][Cin][k]` (or a transposed conv's `[Cin][Cout][k]`) as the kernels' flat `[Cout][Cin * k]`
/// rows. `bias_width` of `None` takes the first dimension; `Some(0)` means the conv has no bias.
fn flat(
    ck: &Checkpoint,
    out: &mut Table,
    name: &str,
    src: &str,
    shape: &[i64],
    bias_width: Option<usize>,
) -> Result<()> {
    let weight = format!("{src}.weight");
    ck.at_checked(&weight, Dtype::F32, shape)?;
    // one row of everything: the kernels index it flat
    let entry = ck.at(&weight)?;
    out.insert(
        format!("{name}.w"),
        Recipe::Rows {
            rows: 1,
            row_bytes: entry.bytes,
            pitch_bytes: entry.bytes,
            segments: vec![crate::weights::Segment {
                tensor: weight,
                row0: 0,
                rows: 1,
            }],
        },
    );
    let bw = bias_width.unwrap_or(shape[0] as usize);
    if bw != 0 {
        let bias = format!("{src}.bias");
        ck.at_checked(&bias, Dtype::F32, &[bw as i64])?;
        out.insert(format!("{name}.b"), widen_f32(ck, &[&bias])?);
    }
    Ok(())
}

/// The SnakeBeta parameters, which the checkpoint stores as logs.
fn exp_f32(ck: &Checkpoint, src: &str, n: usize) -> Result<Recipe> {
    ck.at_checked(src, Dtype::F32, &[n as i64])?;
    let src = src.to_string();
    Ok(Recipe::Built {
        bytes: n * 4,
        build: Box::new(move |ck| {
            let bytes = ck.bytes(ck.at(&src)?);
            let mut out = Vec::with_capacity(n * 4);
            for chunk in bytes.chunks_exact(4) {
                let v = f32::from_le_bytes(chunk.try_into().unwrap());
                out.extend_from_slice(&v.exp().to_le_bytes());
            }
            Ok(out)
        }),
    })
}

fn encoder(ck: &Checkpoint, out: &mut Table) -> Result<()> {
    let alpha = |n: &str, dim: usize| -> Result<String> {
        ck.at_checked(n, Dtype::F32, &[1, dim as i64, 1])?;
        Ok(n.to_string())
    };

    flat(
        ck,
        out,
        "aenc.conv_in",
        "encoder.block.0",
        &[64, 1, 7],
        None,
    )?;
    let mut dim = 64usize;
    for i in 1..=5 {
        let p = format!("encoder.block.{i}");
        for r in 0..3 {
            let q = format!("{p}.block.{r}.block");
            out.insert(
                format!("aenc.b{i}.r{r}.act0"),
                widen_f32(ck, &[&alpha(&format!("{q}.0.alpha"), dim)?])?,
            );
            flat(
                ck,
                out,
                &format!("aenc.b{i}.r{r}.c1"),
                &format!("{q}.1"),
                &[dim as i64, dim as i64, 7],
                None,
            )?;
            out.insert(
                format!("aenc.b{i}.r{r}.act1"),
                widen_f32(ck, &[&alpha(&format!("{q}.2.alpha"), dim)?])?,
            );
            flat(
                ck,
                out,
                &format!("aenc.b{i}.r{r}.c2"),
                &format!("{q}.3"),
                &[dim as i64, dim as i64, 1],
                None,
            )?;
        }
        out.insert(
            format!("aenc.b{i}.act"),
            widen_f32(ck, &[&alpha(&format!("{p}.block.3.alpha"), dim)?])?,
        );
        flat(
            ck,
            out,
            &format!("aenc.b{i}.down"),
            &format!("{p}.block.4"),
            &[(2 * dim) as i64, dim as i64, (2 * ARATES[i - 1]) as i64],
            None,
        )?;
        dim *= 2;
    }
    out.insert(
        "aenc.act_out".into(),
        widen_f32(ck, &[&alpha("encoder.block.6.alpha", dim)?])?,
    );
    flat(
        ck,
        out,
        "aenc.conv_out",
        "encoder.block.7",
        &[dim as i64, dim as i64, 3],
        None,
    )?;

    for n in ["norm1", "norm2", "norm3"] {
        let width = if n == "norm2" { AUDIO_CH } else { 2048 };
        let w = format!("pre_block.{n}.weight");
        let b = format!("pre_block.{n}.bias");
        ck.at_checked(&w, Dtype::F32, &[width as i64])?;
        ck.at_checked(&b, Dtype::F32, &[width as i64])?;
        out.insert(format!("aenc.pre.{n}.w"), widen_f32(ck, &[&w])?);
        out.insert(format!("aenc.pre.{n}.b"), widen_f32(ck, &[&b])?);
    }

    ck.at_checked("pre_block.attn.qkv.weight", Dtype::F32, &[6144, 2048])?;
    out.insert(
        "aenc.pre.qkv.w".into(),
        rows_of(ck, &["pre_block.attn.qkv.weight"], 0)?,
    );
    // q, a zero k, and v, concatenated: the attention takes one bias
    for n in ["q_bias", "zero_k_bias", "v_bias"] {
        ck.at_checked(&format!("pre_block.attn.{n}"), Dtype::F32, &[2048])?;
    }
    out.insert(
        "aenc.pre.qkv.b".into(),
        widen_f32(
            ck,
            &[
                "pre_block.attn.q_bias",
                "pre_block.attn.zero_k_bias",
                "pre_block.attn.v_bias",
            ],
        )?,
    );
    flat(
        ck,
        out,
        "aenc.pre.attn_proj",
        "pre_block.attn.proj",
        &[AUDIO_CH as i64, AUDIO_CH as i64],
        None,
    )?;
    flat(
        ck,
        out,
        "aenc.pre.proj",
        "pre_block.proj",
        &[AUDIO_CH as i64, 2048],
        None,
    )?;
    flat(
        ck,
        out,
        "aenc.pre.mlp.norm",
        "pre_block.mlp.norm",
        &[AUDIO_CH as i64],
        None,
    )?;
    flat(
        ck,
        out,
        "aenc.pre.mlp.w0",
        "pre_block.mlp.w0",
        &[64, AUDIO_CH as i64],
        None,
    )?;
    flat(
        ck,
        out,
        "aenc.pre.mlp.w1",
        "pre_block.mlp.w1",
        &[64, AUDIO_CH as i64],
        None,
    )?;
    flat(
        ck,
        out,
        "aenc.pre.mlp.w2",
        "pre_block.mlp.w2",
        &[AUDIO_CH as i64, 64],
        None,
    )?;
    flat(
        ck,
        out,
        "aenc.mean_proj",
        "mean_proj",
        &[AUDIO_CH as i64, AUDIO_CH as i64, 1],
        None,
    )?;

    let (mean, std) = ("latents_mean", "latents_std");
    ck.at_checked(mean, Dtype::F32, &[AUDIO_CH as i64])?;
    ck.at_checked(std, Dtype::F32, &[AUDIO_CH as i64])?;
    out.insert("aenc.latents_mean".into(), widen_f32(ck, &[mean])?);
    out.insert("aenc.latents_std".into(), widen_f32(ck, &[std])?);
    Ok(())
}
