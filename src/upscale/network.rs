use super::UpscaleSettings;
use crate::{
    checkpoint::Checkpoint,
    compile::Compiler,
    dispatch::{self, Conv3d, Matmul, Profile},
    error::invalid,
    weights::{Recipe, Weights},
    Result, Shape,
};
use half::f16;
use std::{collections::BTreeMap, path::Path};

pub const CHECKPOINT: &str =
    "minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors";

/// The released non-attention 3D network. Checkpoint owners follow Session's mmap contract.
pub struct Upscaler {
    weights: Weights,
    channels: usize,
    blocks: [Vec<Block>; 2],
    temporal: usize,
}
#[derive(Clone)]
struct Block {
    name: String,
    temporal: bool,
}

fn values(ck: &Checkpoint, name: &str) -> crate::weights::Result<Vec<f32>> {
    let bytes = crate::weights::widen_f32(ck, &[name])?.assemble(ck)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect())
}
fn architecture(ck: &Checkpoint) -> Result<(usize, [Vec<Block>; 2], usize)> {
    let shape = &ck.at("conv_in.weight")?.shape;
    if shape.len() != 5
        || shape[1..] != [24, 3, 3, 3]
        || !(64..=1024).contains(&shape[0])
        || shape[0] % 64 != 0
    {
        return invalid(
            "upscaler requires conv_in [channels,24,3,3,3], channels a multiple of 64 up to 1024",
        );
    }
    let c = shape[0];
    let mut blocks = [Vec::new(), Vec::new()];
    let mut temporal = 0;
    let mut expected = BTreeMap::<String, Vec<usize>>::new();
    let mut tensor = |name: String, shape: Vec<usize>| {
        expected.insert(name, shape);
    };
    for (name, dims) in [
        ("conv_in", vec![c, 24, 3, 3, 3]),
        ("conv_out", vec![24, c, 3, 3, 3]),
        ("embed.0", vec![64, 1]),
        ("embed.2", vec![64, 64]),
    ] {
        tensor(format!("{name}.bias"), vec![dims[0]]);
        tensor(format!("{name}.weight"), dims);
    }
    for suffix in ["weight", "bias"] {
        tensor(format!("norm_out.{suffix}"), vec![c]);
    }
    for (side, prefix) in ["in_blocks", "out_blocks"].iter().enumerate() {
        let mut i = 0;
        while ck
            .entries()
            .keys()
            .any(|k| k.starts_with(&format!("{prefix}.{i}.")))
        {
            let name = format!("{prefix}.{i}");
            let is_temporal = ck.has(&format!("{name}.dwconv.weight"));
            if is_temporal {
                let dw = &ck.at(&format!("{name}.dwconv.weight"))?.shape;
                if dw.len() != 5
                    || dw[0] != c
                    || dw[1] != 1
                    || dw[3..] != [1, 1]
                    || dw[2] % 2 != 1
                    || dw[2] > 15
                {
                    return invalid("unsupported upscaler temporal convolution");
                }
                if temporal != 0 && temporal != dw[2] {
                    return invalid("upscaler temporal kernels differ");
                }
                temporal = dw[2];
                tensor(format!("{name}.dwconv.weight"), dw.clone());
                tensor(format!("{name}.dwconv.bias"), vec![c]);
                tensor(format!("{name}.pwconv.weight"), vec![c, c, 1, 1, 1]);
                tensor(format!("{name}.pwconv.bias"), vec![c]);
                for suffix in ["weight", "bias"] {
                    tensor(format!("{name}.norm.{suffix}"), vec![c]);
                }
            } else {
                for norm in ["in_layers.0", "out_norm"] {
                    for suffix in ["weight", "bias"] {
                        tensor(format!("{name}.{norm}.{suffix}"), vec![c]);
                    }
                }
                for conv in ["in_layers.2", "out_layers.2"] {
                    tensor(format!("{name}.{conv}.weight"), vec![c, c, 3, 3, 3]);
                    tensor(format!("{name}.{conv}.bias"), vec![c]);
                }
                tensor(format!("{name}.emb_layers.1.weight"), vec![2 * c, 64]);
                tensor(format!("{name}.emb_layers.1.bias"), vec![2 * c]);
            }
            blocks[side].push(Block {
                name,
                temporal: is_temporal,
            });
            i += 1;
        }
        if i == 0 {
            return invalid("upscaler has no residual blocks");
        }
    }
    if expected.len() != ck.entries().len() {
        return invalid("upscaler checkpoint contains missing or unsupported tensors (attention and 2D models are not supported)");
    }
    for (name, shape) in expected {
        let e = ck.at(&name)?;
        if e.shape != shape || e.dtype != hrx::artifacts::safetensors::DType::F16 {
            return invalid(format!(
                "upscaler tensor {name}: expected F16 {shape:?}, got {:?} {:?}",
                e.dtype, e.shape
            ));
        }
    }
    Ok((c, blocks, temporal))
}

impl Upscaler {
    /// # Safety
    /// The checkpoint must remain immutable until this model is dropped.
    pub unsafe fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut arch = None;
        let weights = unsafe {
            Weights::open(path, |ck, recipes| {
                let a =
                    architecture(ck).map_err(|e| crate::weights::Error::Layout(e.to_string()))?;
                for (name, e) in ck.entries() {
                    if e.shape.len() == 5 && !name.contains("dwconv") {
                        let (co, ci, kt, kh, kw) =
                            (e.shape[0], e.shape[1], e.shape[2], e.shape[3], e.shape[4]);
                        if (kt, kh, kw) == (3, 3, 3) {
                            recipes.insert(
                                name.clone(),
                                crate::weights::conv3d_taps(ck, name, co, ci, 27)?,
                            );
                            continue;
                        }
                        let cp = ci.div_ceil(8) * 8;
                        let op = co.div_ceil(64) * 64;
                        let k = (cp * kt * kh * kw).div_ceil(32) * 32;
                        let n = name.clone();
                        recipes.insert(
                            name.clone(),
                            Recipe::Built {
                                bytes: op * k * 2,
                                build: Box::new(move |ck| {
                                    let e = ck.at(&n)?;
                                    let src = ck.bytes(e);
                                    let mut out = vec![0u8; op * k * 2];
                                    for o in 0..co {
                                        for i in 0..ci {
                                            for tap in 0..kt * kh * kw {
                                                let from = ((o * ci + i) * kt * kh * kw + tap) * 2;
                                                let to = (o * k + tap * cp + i) * 2;
                                                out[to..to + 2]
                                                    .copy_from_slice(&src[from..from + 2]);
                                            }
                                        }
                                    }
                                    Ok(out)
                                }),
                            },
                        );
                    } else if name.ends_with("bias")
                        || e.shape.len() == 1
                        || name.starts_with("embed")
                        || name.contains("emb_layers")
                    {
                        let n = name.clone();
                        let count = e.shape.iter().product::<usize>();
                        let padded = if name == "conv_out.bias" { 64 } else { count };
                        recipes.insert(
                            name.clone(),
                            Recipe::Built {
                                bytes: padded * 4,
                                build: Box::new(move |ck| {
                                    let mut v = values(ck, &n)?;
                                    v.resize(padded, 0.0);
                                    Ok(crate::vvae::as_bytes(&v).to_vec())
                                }),
                            },
                        );
                    } else {
                        recipes.insert(
                            name.clone(),
                            crate::weights::Recipe::Rows {
                                rows: 1,
                                row_bytes: e.bytes,
                                pitch_bytes: e.bytes,
                                segments: vec![crate::weights::Segment {
                                    tensor: name.clone(),
                                    row0: 0,
                                    rows: 1,
                                }],
                            },
                        );
                    }
                }
                arch = Some(a);
                Ok(())
            })
        }?;
        let (channels, blocks, temporal) = arch.expect("validated architecture");
        Ok(Self {
            weights,
            channels,
            blocks,
            temporal,
        })
    }
    pub fn device_bytes(&self) -> usize {
        self.weights.device_bytes()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        input: &[f32],
        shape: &Shape,
        settings: &UpscaleSettings,
        mut progress: Option<&mut dyn FnMut(usize, usize, f64) -> crate::Control>,
    ) -> Result<(Shape, Vec<f32>)> {
        let (out, scale) = settings.resolve(shape)?;
        let (t, h, w) = (
            shape.latent_t as usize,
            shape.lat_h as usize,
            shape.lat_w as usize,
        );
        let (oh, ow) = (out.lat_h as usize, out.lat_w as usize);
        if input.len() != 24 * t * h * w
            || input.iter().any(|v| !v.is_finite() || v.abs() > 65504.0)
        {
            return invalid("upscaler requires finite FP16-representable [24,T,H,W] latents");
        }
        if (h, w) == (oh, ow) {
            return Ok((out, input.to_vec()));
        }
        let chunked = settings.temporal_chunking && t > 32;
        let overlap = if chunked { self.temporal } else { 0 };
        let segment_t = if chunked {
            (32 + 4 * overlap).min(t + 2 * overlap)
        } else {
            t
        };
        if segment_t > 64
            || segment_t * oh * ow > 1_048_576
            || segment_t * oh * ow * self.channels * 2 > u32::MAX as usize
        {
            return invalid("upscale volume exceeds kernel limits; enable temporal chunking or reduce the target");
        }
        let host_bytes =
            24 * t * oh * ow * 4 + segment_t * h * w * 24 * 4 + segment_t * oh * ow * 64 * 4;
        let _host_reservation = stream
            .memory_budget()
            .map(|budget| budget.reserve(host_bytes))
            .transpose()?;
        let mut output = vec![0f32; 24 * t * oh * ow];
        let mut total = vec![0f32; t];
        let chunks = if chunked { t.div_ceil(32) } else { 1 };
        let started = std::time::Instant::now();
        for chunk in 0..chunks {
            let start = if chunked { chunk * 32 } else { 0 };
            let end = if chunked { (start + 32).min(t) } else { t };
            let out_start = start.saturating_sub(overlap);
            let out_end = (end + overlap).min(t);
            let lo = out_start.saturating_sub(overlap);
            let hi = (out_end + overlap).min(t + 2 * overlap);
            let nt = hi - lo;
            let mut segment = vec![f16::ZERO; nt * h * w * 24];
            for z in 0..nt {
                let source = (lo + z).saturating_sub(overlap).min(t - 1);
                for p in 0..h * w {
                    for ch in 0..24 {
                        segment[(z * h * w + p) * 24 + ch] =
                            f16::from_f32(input[(ch * t + source) * h * w + p]);
                    }
                }
            }
            let values = self.segment(stream, c, prof, &segment, nt, h, w, oh, ow, scale)?;
            for z in out_start..out_end {
                let weight = if z < start {
                    (z - out_start + 1) as f32 / (start - out_start + 1) as f32
                } else if z >= end {
                    (out_end - z) as f32 / (out_end - end + 1) as f32
                } else {
                    1.0
                };
                total[z] += weight;
                let source = z + overlap - lo;
                for p in 0..oh * ow {
                    for ch in 0..24 {
                        output[(ch * t + z) * oh * ow + p] +=
                            values[(source * oh * ow + p) * 64 + ch].to_f32() * weight;
                    }
                }
            }
            if let Some(cb) = progress.as_deref_mut() {
                if cb(chunk + 1, chunks, started.elapsed().as_secs_f64()) == crate::Control::Cancel
                {
                    return Err(crate::Error::Cancelled);
                }
            }
        }
        for ch in 0..24 {
            for z in 0..t {
                for p in 0..oh * ow {
                    output[(ch * t + z) * oh * ow + p] /= total[z];
                }
            }
        }
        crate::error::finite_output("upscale", "video latents", &output)?;
        Ok((out, output))
    }
    fn linear(&self, name: &str, x: &[f32], out: usize) -> Result<Vec<f32>> {
        let w = self
            .weights
            .host_f32(&format!("{name}.weight"), out * x.len())?;
        let b = self.weights.host_f32(&format!("{name}.bias"), out)?;
        Ok((0..out)
            .map(|i| {
                f16::from_f32(
                    w[i * x.len()..(i + 1) * x.len()]
                        .iter()
                        .zip(x)
                        .fold(b[i], |s, (a, b)| a.mul_add(*b, s)),
                )
                .to_f32()
            })
            .collect())
    }
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        s: &mut hrx::Stream,
        c: &Compiler,
        p: &mut Profile,
        name: &str,
        x: &hrx::Buffer,
        out: &hrx::Buffer,
        t: usize,
        h: usize,
        w: usize,
        ci: usize,
        co: usize,
    ) -> Result<()> {
        let k = (27 * ci).div_ceil(32) * 32;
        let co = co.div_ceil(64) * 64;
        let weight = self.weights.at(s, &format!("{name}.weight"), co * k * 2)?;
        let bias = self.weights.at(s, &format!("{name}.bias"), co * 4)?;
        Conv3d::build_padding(c, s, false, t, h, w, 1, 1, 3, ci, ci, k, co, true)?.run(
            s,
            Some(p),
            "upscale conv3d",
            x.binding(),
            weight.binding(),
            bias.binding(),
            out.binding(),
            None,
        )?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        s: &mut hrx::Stream,
        c: &Compiler,
        p: &mut Profile,
        name: &str,
        x: &hrx::Buffer,
        out: &hrx::Buffer,
        stats: &hrx::Buffer,
        rows: usize,
        modulation: Option<&[f32]>,
    ) -> Result<()> {
        let gamma = self
            .weights
            .at(s, &format!("{name}.weight"), self.channels * 4)?;
        let beta = self
            .weights
            .at(s, &format!("{name}.bias"), self.channels * 4)?;
        let identity = vec![0.0; self.channels * 2];
        let modulation = modulation.unwrap_or(&identity);
        let mods = s.allocate(modulation.len() * 4)?;
        s.upload(mods.binding(), crate::vvae::as_bytes(modulation))?;
        let cfg = |stem: &str| {
            vec![
                (format!("h3.{stem}.channels"), self.channels.to_string()),
                (format!("h3.{stem}.groups"), "32".into()),
                (format!("h3.{stem}.plane"), rows.to_string()),
                (
                    format!("h3.{stem}.rows_bound"),
                    (rows.div_ceil(64) * 64).to_string(),
                ),
            ]
        };
        let stat = c.get(
            s,
            "upscale_gn_stats",
            "h3_upscale_gn_stats",
            &cfg("upscale_gn_stats"),
        )?;
        let mut apply_cfg = cfg("upscale_gn_silu");
        apply_cfg.push(("h3.upscale_gn_silu.eps".into(), crate::compile::num(1e-5)));
        let apply = c.get(s, "upscale_gn_silu", "h3_upscale_gn_silu", &apply_cfg)?;
        let bytes = rows * self.channels * 2;
        dispatch::checked(
            s,
            &stat,
            Some(p),
            "upscale groupnorm stats",
            [1, 32, 1],
            [32, 1, 1],
            &[1],
            &[x.binding(), stats.binding()],
            &[bytes, 256],
        )?;
        dispatch::checked(
            s,
            &apply,
            Some(p),
            "upscale groupnorm silu",
            [(rows * self.channels).div_ceil(256) as u32, 1, 1],
            [256, 1, 1],
            &[1],
            &[
                x.binding(),
                stats.binding(),
                gamma.binding(),
                beta.binding(),
                out.binding(),
                mods.binding(),
            ],
            &[
                bytes,
                256,
                self.channels * 4,
                self.channels * 4,
                bytes,
                self.channels * 8,
            ],
        )?;
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn segment(
        &self,
        s: &mut hrx::Stream,
        c: &Compiler,
        p: &mut Profile,
        input: &[f16],
        t: usize,
        h: usize,
        w: usize,
        oh: usize,
        ow: usize,
        scale: f32,
    ) -> Result<Vec<f16>> {
        let rows = t * oh * ow;
        let bytes = rows * self.channels * 2;
        let mut x = s.allocate(bytes)?;
        let mut y = s.allocate(bytes)?;
        let mut tmp = s.allocate(bytes)?;
        let stats = s.allocate(256)?;
        let src = s.allocate(input.len() * 2)?;
        s.upload(
            src.binding(),
            &input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        self.conv(s, c, p, "conv_in", &src, &x, t, h, w, 24, self.channels)?;
        drop(src);
        let silu = |v: f32| f16::from_f32(v / (1.0 + (-v).exp())).to_f32();
        let emb = self
            .linear("embed.0", &[f16::from_f32(scale - 1.0).to_f32()], 64)?
            .into_iter()
            .map(silu)
            .collect::<Vec<_>>();
        let emb = self
            .linear("embed.2", &emb, 64)?
            .into_iter()
            .map(silu)
            .collect::<Vec<_>>();
        let mut hh = h;
        let mut ww = w;
        for side in 0..2 {
            if side == 1 {
                let (indices, weights) = resize_map(h, w, oh, ow);
                let ix = s.allocate(indices.len() * 4)?;
                let wt = s.allocate(weights.len() * 4)?;
                s.upload(
                    ix.binding(),
                    &indices
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                s.upload(wt.binding(), crate::vvae::as_bytes(&weights))?;
                pointwise(
                    s,
                    c,
                    p,
                    "upscale_resize",
                    &[
                        ("channels", self.channels),
                        ("frames", t),
                        ("in_plane", h * w),
                        ("out_plane", oh * ow),
                    ],
                    rows * self.channels,
                    &[x.binding(), ix.binding(), wt.binding(), y.binding()],
                    &[
                        t * h * w * self.channels * 2,
                        indices.len() * 4,
                        weights.len() * 4,
                        bytes,
                    ],
                )?;
                std::mem::swap(&mut x, &mut y);
                hh = oh;
                ww = ow;
            }
            let n = t * hh * ww;
            let plane = n * self.channels * 2;
            for block in &self.blocks[side] {
                let b = &block.name;
                if block.temporal {
                    self.norm(s, c, p, &format!("{b}.norm"), &x, &y, &stats, n, None)?;
                    let wt = self.weights.at(
                        s,
                        &format!("{b}.dwconv.weight"),
                        self.channels * self.temporal * 2,
                    )?;
                    let bias =
                        self.weights
                            .at(s, &format!("{b}.dwconv.bias"), self.channels * 4)?;
                    pointwise(
                        s,
                        c,
                        p,
                        "upscale_temporal",
                        &[
                            ("channels", self.channels),
                            ("frames", t),
                            ("plane", hh * ww),
                            ("taps", self.temporal),
                        ],
                        n * self.channels,
                        &[y.binding(), wt.binding(), bias.binding(), tmp.binding()],
                        &[
                            plane,
                            self.channels * self.temporal * 2,
                            self.channels * 4,
                            plane,
                        ],
                    )?;
                    let wt = self.weights.at(
                        s,
                        &format!("{b}.pwconv.weight"),
                        self.channels * self.channels * 2,
                    )?;
                    let bias =
                        self.weights
                            .at(s, &format!("{b}.pwconv.bias"), self.channels * 4)?;
                    Matmul::build(c, s, self.channels, self.channels)?.run(
                        s,
                        Some(p),
                        "upscale pointwise",
                        n,
                        tmp.binding(),
                        wt.binding(),
                        bias.binding(),
                        y.binding(),
                    )?;
                } else {
                    self.norm(
                        s,
                        c,
                        p,
                        &format!("{b}.in_layers.0"),
                        &x,
                        &y,
                        &stats,
                        n,
                        None,
                    )?;
                    self.conv(
                        s,
                        c,
                        p,
                        &format!("{b}.in_layers.2"),
                        &y,
                        &tmp,
                        t,
                        hh,
                        ww,
                        self.channels,
                        self.channels,
                    )?;
                    let m = self.linear(&format!("{b}.emb_layers.1"), &emb, 2 * self.channels)?;
                    self.norm(
                        s,
                        c,
                        p,
                        &format!("{b}.out_norm"),
                        &tmp,
                        &y,
                        &stats,
                        n,
                        Some(&m),
                    )?;
                    self.conv(
                        s,
                        c,
                        p,
                        &format!("{b}.out_layers.2"),
                        &y,
                        &tmp,
                        t,
                        hh,
                        ww,
                        self.channels,
                        self.channels,
                    )?;
                    std::mem::swap(&mut y, &mut tmp);
                }
                pointwise(
                    s,
                    c,
                    p,
                    "upscale_add",
                    &[],
                    n * self.channels,
                    &[y.binding(), x.binding()],
                    &[plane, plane],
                )?;
                std::mem::swap(&mut x, &mut y);
            }
        }
        self.norm(s, c, p, "norm_out", &x, &y, &stats, rows, None)?;
        self.conv(s, c, p, "conv_out", &y, &tmp, t, oh, ow, self.channels, 24)?;
        let mut result = vec![f16::ZERO; rows * 64];
        let mut bytes = vec![0u8; result.len() * 2];
        s.read_blocking(tmp.slice(0, bytes.len()), &mut bytes)?;
        for (v, b) in result.iter_mut().zip(bytes.chunks_exact(2)) {
            *v = f16::from_le_bytes(b.try_into().unwrap());
        }
        Ok(result)
    }
}
#[allow(clippy::too_many_arguments)]
fn pointwise(
    s: &mut hrx::Stream,
    c: &Compiler,
    p: &mut Profile,
    name: &str,
    cfg: &[(&str, usize)],
    count: usize,
    views: &[hrx::View<'_>],
    sizes: &[usize],
) -> Result<()> {
    let cfg = cfg
        .iter()
        .map(|(k, v)| (format!("h3.{name}.{k}"), v.to_string()))
        .collect();
    let kernel = c.get(s, name, &format!("h3_{name}"), &cfg)?;
    dispatch::checked(
        s,
        &kernel,
        Some(p),
        name,
        [count.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        &[count as u32],
        views,
        sizes,
    )?;
    Ok(())
}
fn resize_map(h: usize, w: usize, oh: usize, ow: usize) -> (Vec<i32>, Vec<f32>) {
    let mut indices = Vec::with_capacity(oh * ow * 4);
    let mut weights = Vec::with_capacity(oh * ow * 4);
    for y in 0..oh {
        for x in 0..ow {
            let fy = ((y as f64 + 0.5) * h as f64 / oh as f64 - 0.5).max(0.0);
            let fx = ((x as f64 + 0.5) * w as f64 / ow as f64 - 0.5).max(0.0);
            let (y0, x0) = (fy.floor() as usize, fx.floor() as usize);
            let (dy, dx) = ((fy - y0 as f64) as f32, (fx - x0 as f64) as f32);
            indices.extend(
                [
                    y0 * w + x0,
                    y0 * w + (x0 + 1).min(w - 1),
                    (y0 + 1).min(h - 1) * w + x0,
                    (y0 + 1).min(h - 1) * w + (x0 + 1).min(w - 1),
                ]
                .map(|v| v as i32),
            );
            weights.extend([
                (1. - dy) * (1. - dx),
                (1. - dy) * dx,
                dy * (1. - dx),
                dy * dx,
            ]);
        }
    }
    (indices, weights)
}
