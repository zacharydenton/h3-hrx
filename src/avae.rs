//! The audio VAE: a DAC-style convolutional encoder with a small attention head, and BigVGAN as the
//! decoder. Every kernel here is plain f32 SIMT — no WMMA, no quantisation.
//!
//! The two directions are not mirror images. The encoder ends in an AttnProjection whose attention is
//! pooled down to 32 channels; the decoder is seven transposed-convolution upsamples, each followed by
//! three anti-aliased residual blocks whose outputs are *averaged*, not summed. Both run one channel of
//! stereo at a time, since the model is mono and the two channels are independent.
use crate::compile::{Cfg, Compiler};
use crate::dispatch::{axpy, checked, MatmulF32, Profile};
use crate::error::{invalid, other, Result};
use crate::model::THREADS;
use crate::weights::Weights;
use hrx::View;

/// Samples per latent frame, at 32 kHz.
pub const HOP: usize = 800;
pub use crate::model::AUDIO_CH;

/// The encoder's downsampling rates, and the decoder's upsampling ones with their kernel sizes.
const ENC_RATES: [usize; 5] = [2, 4, 4, 5, 5];
const DEC_RATES: [usize; 7] = [5, 5, 2, 2, 2, 2, 2];
const DEC_UPK: [usize; 7] = [9, 9, 4, 4, 4, 4, 4];
/// The three residual kernel sizes and their dilations, the AMP block's fixed shape.
const RES_K: [usize; 3] = [3, 7, 11];
const RES_DIL: [usize; 3] = [1, 3, 5];

/// The encoder's length bounds are powers of two; the decoder's are multiples of 256. Both are
/// compiled into their kernels, so the two conventions are kept apart.
fn pow2_bound(n: usize) -> usize {
    let mut b = 256;
    while b < n {
        b *= 2;
    }
    b
}

fn round256(n: usize) -> usize {
    n.div_ceil(256) * 256
}

/// A strided, dilated 1-D convolution over `[cin][in_len]` f32.
#[allow(clippy::too_many_arguments)]
fn conv_s(
    c: &Compiler,
    stream: &mut hrx::Stream,
    prof: &mut Profile,
    stage: &str,
    (cin, cout, ksize, dil, pad, stride): (usize, usize, usize, usize, usize, usize),
    in_len: usize,
    out_len: usize,
    x: View<'_>,
    w: View<'_>,
    b: View<'_>,
    out: View<'_>,
) -> Result<()> {
    let ns = "h3.conv1d_s_f32.";
    let cfg: Cfg = vec![
        (format!("{ns}cin"), cin.to_string()),
        (format!("{ns}cout"), cout.to_string()),
        (format!("{ns}ksize"), ksize.to_string()),
        (format!("{ns}dilation"), dil.to_string()),
        (format!("{ns}pad"), pad.to_string()),
        (format!("{ns}stride"), stride.to_string()),
        (format!("{ns}in_bound"), pow2_bound(in_len).to_string()),
        (format!("{ns}out_bound"), pow2_bound(out_len).to_string()),
    ];
    let k = c.get(stream, "conv1d_s_f32", "h3_conv1d_s_f32", &cfg)?;
    checked(
        stream,
        &k,
        Some(prof),
        stage,
        [out_len.div_ceil(256) as u32, cout as u32, 1],
        [THREADS, 1, 1],
        &[out_len as u32, in_len as u32],
        &[x, w, b, out],
        &[
            cin * in_len * 4,
            cout * cin * ksize * 4,
            cout * 4,
            cout * out_len * 4,
        ],
    )?;
    Ok(())
}

/// The plain Snake activation, `x + sin^2(alpha x) / alpha`.
#[allow(clippy::too_many_arguments)]
fn snake_plain(
    c: &Compiler,
    stream: &mut hrx::Stream,
    prof: &mut Profile,
    channels: usize,
    len: usize,
    x: View<'_>,
    alpha: View<'_>,
    out: View<'_>,
) -> Result<()> {
    let ns = "h3.snake_f32.";
    let cfg: Cfg = vec![
        (format!("{ns}channels"), channels.to_string()),
        (format!("{ns}len_bound"), pow2_bound(len).to_string()),
    ];
    let k = c.get(stream, "snake_f32", "h3_snake_f32", &cfg)?;
    checked(
        stream,
        &k,
        Some(prof),
        "aenc snake",
        [len.div_ceil(256) as u32, channels as u32, 1],
        [THREADS, 1, 1],
        &[len as u32],
        &[x, alpha, out],
        &[channels * len * 4, channels * 4, channels * len * 4],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn layernorm(
    c: &Compiler,
    stream: &mut hrx::Stream,
    prof: &mut Profile,
    rows: usize,
    width: usize,
    x: View<'_>,
    w: View<'_>,
    b: View<'_>,
    out: View<'_>,
) -> Result<()> {
    let ns = "h3.layernorm_f32.";
    let cfg: Cfg = vec![
        (format!("{ns}width"), width.to_string()),
        (format!("{ns}eps"), crate::compile::num(1e-5)),
    ];
    let k = c.get(stream, "layernorm_f32", "h3_layernorm_f32", &cfg)?;
    checked(
        stream,
        &k,
        Some(prof),
        "aenc layernorm",
        [rows as u32, 1, 1],
        [32, 1, 1],
        &[rows as u32],
        &[x, w, b, out],
        &[rows * width * 4, width * 4, width * 4, rows * width * 4],
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn transpose(
    c: &Compiler,
    stream: &mut hrx::Stream,
    prof: &mut Profile,
    rows: usize,
    cols: usize,
    x: View<'_>,
    out: View<'_>,
) -> Result<()> {
    let ns = "h3.transpose_f32.";
    let cfg: Cfg = vec![(format!("{ns}cols"), cols.to_string())];
    let k = c.get(stream, "transpose_f32", "h3_transpose_f32", &cfg)?;
    checked(
        stream,
        &k,
        Some(prof),
        "aenc transpose",
        [(rows * cols).div_ceil(256) as u32, 1, 1],
        [THREADS, 1, 1],
        &[rows as u32],
        &[x, out],
        &[rows * cols * 4, rows * cols * 4],
    )?;
    Ok(())
}

/// A non-strided, dilated 1-D convolution over channel-major FP32 activations. The `accumulate` form
/// adds into its output, which is how a residual block's second convolution closes the skip.
#[allow(clippy::too_many_arguments)]
fn conv4(
    c: &Compiler,
    stream: &mut hrx::Stream,
    prof: &mut Profile,
    stage: &str,
    (cin, cout, ksize, dil, pad): (usize, usize, usize, usize, usize),
    accumulate: bool,
    len: usize,
    x: View<'_>,
    w: View<'_>,
    b: View<'_>,
    out: View<'_>,
) -> Result<()> {
    let blocked = crate::plan::avae::packed_conv(cin, cout);
    let narrow =
        (4..=16).contains(&cin) && cin.is_multiple_of(4) && cout <= 16 && (2..=11).contains(&ksize);
    // Operand lookahead helps bounded windows; longer convolutions retain
    // the two-sample tile for throughput. Small channel groups cross over later.
    let prefetch_window = len <= 512
        || (cin <= 512 && cout <= 512 && len <= 2048)
        || (cin <= 64 && cout <= 64 && len <= 4096);
    let prefetched =
        blocked && cin.is_multiple_of(4) && (2..=11).contains(&ksize) && prefetch_window;
    // Share the packed channel group across lanes when time lanes would be idle.
    // Wider/longer convolutions benefit more from sharing each input in registers.
    let channel_lanes = prefetched
        && cin.is_multiple_of(16)
        && (len <= 32 || (cin <= 256 && cout <= 256 && ksize <= 7 && len <= 128));
    let stem = if narrow {
        "conv1d_narrow_f32"
    } else if channel_lanes && ksize == 3 {
        "conv1d_k3_f32"
    } else if channel_lanes {
        "conv1d_lane_f32"
    } else if prefetched {
        "conv1d_prefetch_f32"
    } else if blocked {
        "conv1d_block_f32"
    } else {
        "conv1d4_f32"
    };
    let ns = format!("h3.{stem}.");
    let cfg: Cfg = vec![
        (format!("{ns}cin"), cin.to_string()),
        (format!("{ns}cout"), cout.to_string()),
        (format!("{ns}ksize"), ksize.to_string()),
        (format!("{ns}dilation"), dil.to_string()),
        (format!("{ns}pad"), pad.to_string()),
        (
            format!("{ns}accumulate"),
            if accumulate { "1" } else { "0" }.to_string(),
        ),
        (format!("{ns}len_bound"), round256(len).to_string()),
    ];
    let k = c.get(stream, stem, &format!("h3_{stem}"), &cfg)?;
    // Packed convolutions share inputs across eight output channels. Narrow
    // prefetch keeps one sample per lane; the fallback retains four samples.
    let (span, outputs) = if narrow {
        (64, cout)
    } else if channel_lanes {
        (8, cout / 8)
    } else if prefetched {
        (64, cout / 8)
    } else if blocked {
        (128, cout / 8)
    } else {
        (256, cout)
    };
    checked(
        stream,
        &k,
        Some(prof),
        stage,
        [len.div_ceil(span) as u32, outputs as u32, 1],
        [64, 1, 1],
        &[len as u32],
        &[x, w, b, out],
        &[
            cin * len * 4,
            cout * cin * ksize * 4,
            cout * 4,
            cout * len * 4,
        ],
    )?;
    Ok(())
}

pub struct AudioVae {
    weights: Weights,
    /// the decoder's conv_post has no bias in the checkpoint, so it is handed zeros
    zeros: hrx::Buffer,
    dec: Option<DecBuffers>,
    enc: Option<EncBuffers>,
    dec_mean: Vec<f32>,
    dec_std: Vec<f32>,
    enc_mean: Vec<f32>,
    enc_std: Vec<f32>,
}

/// The decoder's scratch, sized for the widest `[C][len]` plane the stack reaches.
struct DecBuffers {
    cap: usize,
    h: hrx::Buffer,
    acc: hrx::Buffer,
    hj: hrx::Buffer,
    r: hrx::Buffer,
    r2: hrx::Buffer,
    input: hrx::Buffer,
}

/// Resident encoder scratch; stable bindings reuse native dispatch preparation.
struct EncBuffers {
    cap: usize,
    x0: hrx::Buffer,
    h: hrx::Buffer,
    h2: hrx::Buffer,
    y: hrx::Buffer,
    y2: hrx::Buffer,
    rows: hrx::Buffer,
    n1: hrx::Buffer,
    qkv: hrx::Buffer,
    pattn: hrx::Buffer,
    pool: hrx::Buffer,
    xa: hrx::Buffer,
    xb: hrx::Buffer,
    xc: hrx::Buffer,
    a0: hrx::Buffer,
    a1: hrx::Buffer,
    g: hrx::Buffer,
}

impl AudioVae {
    /// # Safety
    ///
    /// Maps the checkpoint; see [`crate::Session::new`].
    pub unsafe fn open(
        stream: &mut hrx::Stream,
        path: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let weights = unsafe { Weights::open(path, crate::plan::avae::plan) }?;
        let zeros = stream.allocate_zeroed(2048 * 4)?;
        Ok(Self {
            dec_mean: weights.host_f32("audio.latents_mean", AUDIO_CH)?,
            dec_std: weights.host_f32("audio.latents_std", AUDIO_CH)?,
            enc_mean: weights.host_f32("aenc.latents_mean", AUDIO_CH)?,
            enc_std: weights.host_f32("aenc.latents_std", AUDIO_CH)?,
            weights,
            zeros,
            dec: None,
            enc: None,
        })
    }

    /// The anti-aliased SnakeBeta: up two, activate, down two, through a FIR the checkpoint carries.
    #[allow(clippy::too_many_arguments)]
    fn snake_beta(
        &self,
        c: &Compiler,
        stream: &mut hrx::Stream,
        prof: &mut Profile,
        channels: usize,
        len: usize,
        x: View<'_>,
        alpha: View<'_>,
        beta: View<'_>,
        out: View<'_>,
    ) -> Result<()> {
        let ns = "h3.snake_fused_f32.";
        let cfg: Cfg = vec![
            (format!("{ns}channels"), channels.to_string()),
            (format!("{ns}len_bound"), round256(len).to_string()),
        ];
        let kernel = c.get(stream, "snake_fused_f32", "h3_snake_fused_f32", &cfg)?;
        let fir = self.weights.at(stream, "audio.fir", 12 * 4)?;
        // The complete anti-aliased activation keeps its rounded upsampled
        // values in a workgroup tile, including the downsample's replicated halo.
        checked(
            stream,
            &kernel,
            Some(prof),
            "audio snake",
            [len.div_ceil(64) as u32, channels as u32, 1],
            [64, 1, 1],
            &[len as u32],
            &[x, fir.binding(), alpha, beta, out],
            &[
                channels * len * 4,
                12 * 4,
                channels * 4,
                channels * 4,
                channels * len * 4,
            ],
        )?;
        Ok(())
    }

    fn ensure_dec(&mut self, stream: &mut hrx::Stream, t: usize) -> Result<()> {
        let l_out = t * HOP;
        let cap = (2048 * t).max(8 * l_out) + 4096;
        if self.dec.as_ref().is_some_and(|d| d.cap >= cap) {
            return Ok(());
        }
        self.dec = None;
        self.dec = Some(DecBuffers {
            cap,
            h: stream.allocate(cap * 4)?,
            acc: stream.allocate(cap * 4)?,
            hj: stream.allocate(cap * 4)?,
            r: stream.allocate(cap * 4)?,
            r2: stream.allocate(cap * 4)?,
            input: stream.allocate(AUDIO_CH * t * 4 + 4096)?,
        });
        Ok(())
    }

    /// Model-space latents `[2][32][audio_t]` to stereo samples `[2][audio_t * 800]` at 32 kHz.
    pub fn decode(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        latents: &[f32],
        audio_t: usize,
        samples: &mut [f32],
    ) -> Result<()> {
        let t = audio_t;
        let l_out = t * HOP;
        self.ensure_dec(stream, t)?;

        if samples.len() < 2 * l_out {
            return invalid(format!(
                "{audio_t} audio latents decode to {} samples, {} given",
                2 * l_out,
                samples.len()
            ));
        }
        // Both channels are staged and committed together. The second one can fail after the first
        // has been decoded, and a caller that got an error back would otherwise find half its buffer
        // rewritten and half of it as it was.
        let mut staged = vec![0.0f32; 2 * l_out];
        for ch in 0..2 {
            // undo the latent normalisation into the [32][T] the first projection reads
            let mut input = vec![0.0f32; AUDIO_CH * t];
            for k in 0..AUDIO_CH {
                for i in 0..t {
                    input[k * t + i] =
                        latents[(ch * AUDIO_CH + k) * t + i] * self.dec_std[k] + self.dec_mean[k];
                }
            }
            let d = self.dec.as_ref().expect("sized above");
            stream.upload_at(&d.input, 0, crate::vvae::as_bytes(&input))?;

            let held_1 = self
                .weights
                .at(stream, "audio.dec_in_proj.w", 2048 * AUDIO_CH * 4)?;
            let held_2 = self.weights.at(stream, "audio.dec_in_proj.b", 2048 * 4)?;
            conv4(
                c,
                stream,
                prof,
                "audio dec_in_proj",
                (AUDIO_CH, 2048, 1, 1, 0),
                false,
                t,
                d.input.binding(),
                held_1.binding(),
                held_2.binding(),
                d.r.binding(),
            )?;
            let held_1 = self
                .weights
                .at(stream, "audio.conv_pre.w", 1024 * 2048 * 7 * 4)?;
            let held_2 = self.weights.at(stream, "audio.conv_pre.b", 1024 * 4)?;
            conv4(
                c,
                stream,
                prof,
                "audio conv_pre",
                (2048, 1024, 7, 1, 3),
                false,
                t,
                d.r.binding(),
                held_1.binding(),
                held_2.binding(),
                d.h.binding(),
            )?;

            let (mut len, mut chan) = (t, 1024usize);
            for i in 0..7 {
                let (cout, k, rate) = (chan / 2, DEC_UPK[i], DEC_RATES[i]);
                let pad = (k - rate) / 2;
                let olen = (len - 1) * rate + k - 2 * pad;
                {
                    let blocked = crate::plan::avae::packed_upsample(cout);
                    // Short sequences expose too few waves to hide each channel's
                    // loads. Look ahead four channels without changing FMA order.
                    let prefetch =
                        blocked && chan.is_multiple_of(4) && k <= 2 * rate && olen <= 512;
                    let stem = if prefetch {
                        "convt1d_prefetch_f32"
                    } else if blocked {
                        "convt1d_block_f32"
                    } else {
                        "convt1d_f32"
                    };
                    let ns = format!("h3.{stem}.");
                    let cfg: Cfg = vec![
                        (format!("{ns}cin"), chan.to_string()),
                        (format!("{ns}cout"), cout.to_string()),
                        (format!("{ns}ksize"), k.to_string()),
                        (format!("{ns}stride"), rate.to_string()),
                        (format!("{ns}pad"), pad.to_string()),
                        (format!("{ns}len_bound"), round256(olen).to_string()),
                    ];
                    let kt = c.get(stream, stem, &format!("h3_{stem}"), &cfg)?;
                    let (threads, outputs) = if blocked {
                        (64, cout / 16)
                    } else {
                        (THREADS, cout)
                    };
                    let d = self.dec.as_ref().expect("sized above");
                    let held_1 = self.weights.at(
                        stream,
                        &format!("audio.ups.{i}.w"),
                        chan * cout * k * 4,
                    )?;
                    let held_2 = self
                        .weights
                        .at(stream, &format!("audio.ups.{i}.b"), cout * 4)?;
                    checked(
                        stream,
                        &kt,
                        Some(prof),
                        "audio upsample",
                        [olen.div_ceil(threads as usize) as u32, outputs as u32, 1],
                        [threads, 1, 1],
                        &[len as u32, olen as u32],
                        &[
                            d.h.binding(),
                            held_1.binding(),
                            held_2.binding(),
                            d.r.binding(),
                        ],
                        &[
                            chan * len * 4,
                            chan * cout * k * 4,
                            cout * 4,
                            cout * olen * 4,
                        ],
                    )?;
                }
                len = olen;
                chan = cout;
                let plane = chan * len;
                let d = self.dec.as_ref().expect("sized above");
                stream.copy(d.h.slice(0, plane * 4), d.r.slice(0, plane * 4))?;
                stream.fill(d.acc.slice(0, plane * 4), 0)?;

                // three AMP blocks, averaged rather than summed
                for (j, &kk) in RES_K.iter().enumerate() {
                    let r = i * 3 + j;
                    let d = self.dec.as_ref().expect("sized above");
                    stream.copy(d.hj.slice(0, plane * 4), d.h.slice(0, plane * 4))?;
                    for (dl, &dil) in RES_DIL.iter().enumerate() {
                        let act1 = format!("audio.res.{r}.act.{}.", 2 * dl);
                        let act2 = format!("audio.res.{r}.act.{}.", 2 * dl + 1);
                        let d = self.dec.as_ref().expect("sized above");
                        let (hj, rb, r2b) = (d.hj.binding(), d.r.binding(), d.r2.binding());
                        // held for the call: a view borrows the allocation it names
                        let (a1, b1) = (
                            self.weights.at(stream, &format!("{act1}alpha"), chan * 4)?,
                            self.weights.at(stream, &format!("{act1}beta"), chan * 4)?,
                        );
                        self.snake_beta(
                            c,
                            stream,
                            prof,
                            chan,
                            len,
                            hj,
                            a1.binding(),
                            b1.binding(),
                            rb,
                        )?;
                        let held_0 = self.weights.at(
                            stream,
                            &format!("audio.res.{r}.c1.{dl}.w"),
                            chan * chan * kk * 4,
                        )?;
                        let held_1 = self.weights.at(
                            stream,
                            &format!("audio.res.{r}.c1.{dl}.b"),
                            chan * 4,
                        )?;
                        conv4(
                            c,
                            stream,
                            prof,
                            "audio res conv1",
                            (chan, chan, kk, dil, (kk * dil - dil) / 2),
                            false,
                            len,
                            rb,
                            held_0.binding(),
                            held_1.binding(),
                            r2b,
                        )?;
                        let (a2, b2) = (
                            self.weights.at(stream, &format!("{act2}alpha"), chan * 4)?,
                            self.weights.at(stream, &format!("{act2}beta"), chan * 4)?,
                        );
                        self.snake_beta(
                            c,
                            stream,
                            prof,
                            chan,
                            len,
                            r2b,
                            a2.binding(),
                            b2.binding(),
                            rb,
                        )?;
                        // the accumulating form closes the skip in place
                        let held_0 = self.weights.at(
                            stream,
                            &format!("audio.res.{r}.c2.{dl}.w"),
                            chan * chan * kk * 4,
                        )?;
                        let held_1 = self.weights.at(
                            stream,
                            &format!("audio.res.{r}.c2.{dl}.b"),
                            chan * 4,
                        )?;
                        conv4(
                            c,
                            stream,
                            prof,
                            "audio res conv2",
                            (chan, chan, kk, 1, (kk - 1) / 2),
                            true,
                            len,
                            rb,
                            held_0.binding(),
                            held_1.binding(),
                            hj,
                        )?;
                    }
                    let d = self.dec.as_ref().expect("sized above");
                    axpy(
                        c,
                        stream,
                        Some(prof),
                        "audio axpy",
                        1.0,
                        1.0,
                        plane,
                        d.hj.binding(),
                        d.acc.binding(),
                    )?;
                }
                let d = self.dec.as_ref().expect("sized above");
                axpy(
                    c,
                    stream,
                    Some(prof),
                    "audio axpy",
                    1.0 / 3.0,
                    0.0,
                    plane,
                    d.acc.binding(),
                    d.h.binding(),
                )?;
            }

            let d = self.dec.as_ref().expect("sized above");
            let (hb, rb, r2b) = (d.h.binding(), d.r.binding(), d.r2.binding());
            let (pa, pb) = (
                self.weights.at(stream, "audio.post.alpha", chan * 4)?,
                self.weights.at(stream, "audio.post.beta", chan * 4)?,
            );
            self.snake_beta(
                c,
                stream,
                prof,
                chan,
                len,
                hb,
                pa.binding(),
                pb.binding(),
                rb,
            )?;
            let held_1 = self.weights.at(stream, "audio.conv_post.w", chan * 7 * 4)?;
            conv4(
                c,
                stream,
                prof,
                "audio conv_post",
                (chan, 1, 7, 1, 3),
                false,
                len,
                rb,
                held_1.binding(),
                self.zeros.binding(),
                r2b,
            )?;
            if len != l_out {
                return other(format!("audio length {len} != {l_out}"));
            }
            stream.read_blocking(
                r2b,
                crate::vvae::as_bytes_mut(&mut staged[ch * l_out..(ch + 1) * l_out]),
            )?;
        }
        for (out, v) in samples.iter_mut().zip(&staged) {
            *out = v.clamp(-1.0, 1.0);
        }
        Ok(())
    }

    fn ensure_enc(&mut self, stream: &mut hrx::Stream, lp: usize) -> Result<()> {
        if self.enc.as_ref().is_some_and(|e| e.cap >= lp) {
            return Ok(());
        }
        // An earlier failed invocation may still have queued work. Fence before
        // evicting its backing and the stream's cached native commands.
        stream.synchronize()?;
        self.enc = None;
        let t = lp / HOP;
        self.enc = Some(EncBuffers {
            cap: lp,
            x0: stream.allocate((lp) * 4)?,
            h: stream.allocate((64 * lp) * 4)?,
            h2: stream.allocate((64 * lp) * 4)?,
            y: stream.allocate((64 * lp) * 4)?,
            y2: stream.allocate((64 * lp) * 4)?,
            rows: stream.allocate((t * 2048) * 4)?,
            n1: stream.allocate((t * 2048) * 4)?,
            qkv: stream.allocate((t * 6144) * 4)?,
            pattn: stream.allocate((8 * t * t) * 4)?,
            pool: stream.allocate((t * AUDIO_CH) * 4)?,
            xa: stream.allocate((t * AUDIO_CH) * 4)?,
            xb: stream.allocate((t * AUDIO_CH) * 4)?,
            xc: stream.allocate((t * AUDIO_CH) * 4)?,
            a0: stream.allocate((t * 64) * 4)?,
            a1: stream.allocate((t * 64) * 4)?,
            g: stream.allocate((t * 64) * 4)?,
        });
        Ok(())
    }

    /// Stereo samples `[2][n]` at 32 kHz to model-space latents `[2][32][audio_t]`.
    ///
    /// The sample count is padded up to a whole number of 800-sample frames; the tail is silence, not
    /// a repeat, because the encoder is causal and a repeated tail would be heard.
    pub fn encode(
        &mut self,
        stream: &mut hrx::Stream,
        c: &Compiler,
        prof: &mut Profile,
        samples: &[f32],
        n: usize,
    ) -> Result<(Vec<f32>, usize)> {
        let lp = n.div_ceil(HOP) * HOP;
        let t = lp / HOP;
        self.ensure_enc(stream, lp)?;
        let workspace = self.enc.as_ref().expect("sized above");
        let x0 = &workspace.x0;
        let mut h = &workspace.h;
        let mut h2 = &workspace.h2;
        let y = &workspace.y;
        let y2 = &workspace.y2;
        let rows = &workspace.rows;
        let n1 = &workspace.n1;
        let qkv = &workspace.qkv;
        let pattn = &workspace.pattn;
        let pool = &workspace.pool;
        let xa = &workspace.xa;
        let xb = &workspace.xb;
        let xc = &workspace.xc;
        let a0 = &workspace.a0;
        let a1 = &workspace.a1;
        let g = &workspace.g;

        let mut out = vec![0.0f32; 2 * AUDIO_CH * t];
        let mut host = vec![0.0f32; lp];
        let w = |stream: &mut hrx::Stream, nm: &str, count: usize| {
            self.weights.at(stream, nm, count * 4)
        };

        for ch in 0..2 {
            host.fill(0.0);
            host[..n].copy_from_slice(&samples[ch * n..ch * n + n]);
            stream.upload(x0.binding(), crate::vvae::as_bytes(&host))?;

            let (mut len, mut dim) = (lp, 64usize);
            let hoisted_1 = w(stream, "aenc.conv_in.w", 64 * 7)?;
            let hoisted_2 = w(stream, "aenc.conv_in.b", 64)?;
            conv_s(
                c,
                stream,
                prof,
                "aenc conv_in",
                (1, 64, 7, 1, 3, 1),
                lp,
                lp,
                x0.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                h.binding(),
            )?;
            for i in 1..=5 {
                for r in 0..3 {
                    let dil = [1usize, 3, 9][r];
                    let p = format!("aenc.b{i}.r{r}.");
                    let hoisted_1 = w(stream, &format!("{p}act0"), dim)?;
                    snake_plain(
                        c,
                        stream,
                        prof,
                        dim,
                        len,
                        h.binding(),
                        hoisted_1.binding(),
                        y.binding(),
                    )?;
                    let hoisted_1 = w(stream, &format!("{p}c1.w"), dim * dim * 7)?;
                    let hoisted_2 = w(stream, &format!("{p}c1.b"), dim)?;
                    conv_s(
                        c,
                        stream,
                        prof,
                        "aenc res conv7",
                        (dim, dim, 7, dil, 3 * dil, 1),
                        len,
                        len,
                        y.binding(),
                        hoisted_1.binding(),
                        hoisted_2.binding(),
                        y2.binding(),
                    )?;
                    let hoisted_1 = w(stream, &format!("{p}act1"), dim)?;
                    snake_plain(
                        c,
                        stream,
                        prof,
                        dim,
                        len,
                        y2.binding(),
                        hoisted_1.binding(),
                        y.binding(),
                    )?;
                    let hoisted_1 = w(stream, &format!("{p}c2.w"), dim * dim)?;
                    let hoisted_2 = w(stream, &format!("{p}c2.b"), dim)?;
                    conv_s(
                        c,
                        stream,
                        prof,
                        "aenc res conv1",
                        (dim, dim, 1, 1, 0, 1),
                        len,
                        len,
                        y.binding(),
                        hoisted_1.binding(),
                        hoisted_2.binding(),
                        y2.binding(),
                    )?;
                    axpy(
                        c,
                        stream,
                        Some(prof),
                        "audio axpy",
                        1.0,
                        1.0,
                        dim * len,
                        y2.binding(),
                        h.binding(),
                    )?;
                }
                let s = ENC_RATES[i - 1];
                let out_len = len / s;
                let p = format!("aenc.b{i}.");
                let hoisted_1 = w(stream, &format!("{p}act"), dim)?;
                snake_plain(
                    c,
                    stream,
                    prof,
                    dim,
                    len,
                    h.binding(),
                    hoisted_1.binding(),
                    y.binding(),
                )?;
                let hoisted_1 = w(stream, &format!("{p}down.w"), 2 * dim * dim * 2 * s)?;
                let hoisted_2 = w(stream, &format!("{p}down.b"), 2 * dim)?;
                conv_s(
                    c,
                    stream,
                    prof,
                    "aenc down",
                    (dim, 2 * dim, 2 * s, 1, s.div_ceil(2), s),
                    len,
                    out_len,
                    y.binding(),
                    hoisted_1.binding(),
                    hoisted_2.binding(),
                    h2.binding(),
                )?;
                std::mem::swap(&mut h, &mut h2);
                dim *= 2;
                len = out_len;
            }
            if dim != 2048 || len != t {
                return other("audio encoder shape mismatch");
            }
            let hoisted_1 = w(stream, "aenc.act_out", 2048)?;
            snake_plain(
                c,
                stream,
                prof,
                2048,
                len,
                h.binding(),
                hoisted_1.binding(),
                y.binding(),
            )?;
            let hoisted_1 = w(stream, "aenc.conv_out.w", 2048 * 2048 * 3)?;
            let hoisted_2 = w(stream, "aenc.conv_out.b", 2048)?;
            conv_s(
                c,
                stream,
                prof,
                "aenc conv_out",
                (2048, 2048, 3, 1, 1, 1),
                len,
                len,
                y.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                h.binding(),
            )?;
            transpose(c, stream, prof, 2048, t, h.binding(), rows.binding())?;

            // AttnProjection: x = proj(norm3(x)) + attn(norm1(x)); x += mlp(norm2(x))
            let hoisted_1 = w(stream, "aenc.pre.norm3.w", 2048)?;
            let hoisted_2 = w(stream, "aenc.pre.norm3.b", 2048)?;
            layernorm(
                c,
                stream,
                prof,
                t,
                2048,
                rows.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                n1.binding(),
            )?;
            let hoisted_1 = w(stream, "aenc.pre.proj.w", AUDIO_CH * 2048)?;
            let hoisted_2 = w(stream, "aenc.pre.proj.b", AUDIO_CH)?;
            MatmulF32::build(c, stream, 2048, AUDIO_CH)?.run(
                stream,
                Some(prof),
                "aenc matmul",
                t,
                n1.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                xa.binding(),
            )?;
            let hoisted_1 = w(stream, "aenc.pre.norm1.w", 2048)?;
            let hoisted_2 = w(stream, "aenc.pre.norm1.b", 2048)?;
            layernorm(
                c,
                stream,
                prof,
                t,
                2048,
                rows.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                n1.binding(),
            )?;
            let hoisted_1 = w(stream, "aenc.pre.qkv.w", 6144 * 2048)?;
            let hoisted_2 = w(stream, "aenc.pre.qkv.b", 6144)?;
            MatmulF32::build(c, stream, 2048, 6144)?.run(
                stream,
                Some(prof),
                "aenc matmul",
                t,
                n1.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                qkv.binding(),
            )?;
            {
                let ns = "h3.attn_scores_f32.";
                let cfg: Cfg = vec![
                    (format!("{ns}heads"), "8".into()),
                    (format!("{ns}hd"), "256".into()),
                    (format!("{ns}scale"), crate::compile::num(1.0 / 16.0)),
                ];
                let k = c.get(stream, "attn_scores_f32", "h3_attn_scores_f32", &cfg)?;
                // eight heads of 256, packed q|k|v, against a per-head t-by-t score plane
                checked(
                    stream,
                    &k,
                    Some(prof),
                    "aenc attention",
                    [t.div_ceil(256) as u32, 8, 1],
                    [THREADS, 1, 1],
                    &[t as u32],
                    &[qkv.binding(), pattn.binding()],
                    &[t * 3 * 8 * 256 * 4, 8 * t * t * 4],
                )?;
            }
            {
                let ns = "h3.attn_pv_pool_f32.";
                let cfg: Cfg = vec![
                    (format!("{ns}heads"), "8".into()),
                    (format!("{ns}hd"), "256".into()),
                    (format!("{ns}pool"), "8".into()),
                ];
                let k = c.get(stream, "attn_pv_pool_f32", "h3_attn_pv_pool_f32", &cfg)?;
                // the pool of eight narrows each head's 256 to 32, which is AUDIO_CH
                checked(
                    stream,
                    &k,
                    Some(prof),
                    "aenc attention",
                    [t as u32, 1, 1],
                    [32, 1, 1],
                    &[t as u32],
                    &[qkv.binding(), pattn.binding(), pool.binding()],
                    &[t * 3 * 8 * 256 * 4, 8 * t * t * 4, t * (256 / 8) * 4],
                )?;
            }
            let hoisted_1 = w(stream, "aenc.pre.attn_proj.w", AUDIO_CH * AUDIO_CH)?;
            let hoisted_2 = w(stream, "aenc.pre.attn_proj.b", AUDIO_CH)?;
            MatmulF32::build(c, stream, AUDIO_CH, AUDIO_CH)?.run(
                stream,
                Some(prof),
                "aenc matmul",
                t,
                pool.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                xb.binding(),
            )?;
            axpy(
                c,
                stream,
                Some(prof),
                "audio axpy",
                1.0,
                1.0,
                t * AUDIO_CH,
                xb.binding(),
                xa.binding(),
            )?;

            let hoisted_1 = w(stream, "aenc.pre.norm2.w", AUDIO_CH)?;
            let hoisted_2 = w(stream, "aenc.pre.norm2.b", AUDIO_CH)?;
            layernorm(
                c,
                stream,
                prof,
                t,
                AUDIO_CH,
                xa.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                xb.binding(),
            )?;
            let hoisted_1 = w(stream, "aenc.pre.mlp.norm.w", AUDIO_CH)?;
            let hoisted_2 = w(stream, "aenc.pre.mlp.norm.b", AUDIO_CH)?;
            layernorm(
                c,
                stream,
                prof,
                t,
                AUDIO_CH,
                xb.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                xc.binding(),
            )?;
            let mlp = MatmulF32::build(c, stream, AUDIO_CH, 64)?;
            let hoisted_1 = w(stream, "aenc.pre.mlp.w0.w", 64 * AUDIO_CH)?;
            let hoisted_2 = w(stream, "aenc.pre.mlp.w0.b", 64)?;
            mlp.run(
                stream,
                Some(prof),
                "aenc matmul",
                t,
                xc.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                a0.binding(),
            )?;
            let hoisted_1 = w(stream, "aenc.pre.mlp.w1.w", 64 * AUDIO_CH)?;
            let hoisted_2 = w(stream, "aenc.pre.mlp.w1.b", 64)?;
            mlp.run(
                stream,
                Some(prof),
                "aenc matmul",
                t,
                xc.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                a1.binding(),
            )?;
            {
                let k = c.get(stream, "geglu_tanh_f32", "h3_geglu_tanh_f32", &Cfg::new())?;
                checked(
                    stream,
                    &k,
                    Some(prof),
                    "aenc geglu",
                    [(t * 64).div_ceil(256) as u32, 1, 1],
                    [THREADS, 1, 1],
                    &[(t * 64) as u32],
                    &[a0.binding(), a1.binding(), g.binding()],
                    &[t * 64 * 4, t * 64 * 4, t * 64 * 4],
                )?;
            }
            let hoisted_1 = w(stream, "aenc.pre.mlp.w2.w", AUDIO_CH * 64)?;
            let hoisted_2 = w(stream, "aenc.pre.mlp.w2.b", AUDIO_CH)?;
            MatmulF32::build(c, stream, 64, AUDIO_CH)?.run(
                stream,
                Some(prof),
                "aenc matmul",
                t,
                g.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                xb.binding(),
            )?;
            axpy(
                c,
                stream,
                Some(prof),
                "audio axpy",
                1.0,
                1.0,
                t * AUDIO_CH,
                xb.binding(),
                xa.binding(),
            )?;
            let hoisted_1 = w(stream, "aenc.mean_proj.w", AUDIO_CH * AUDIO_CH)?;
            let hoisted_2 = w(stream, "aenc.mean_proj.b", AUDIO_CH)?;
            MatmulF32::build(c, stream, AUDIO_CH, AUDIO_CH)?.run(
                stream,
                Some(prof),
                "aenc matmul",
                t,
                xa.binding(),
                hoisted_1.binding(),
                hoisted_2.binding(),
                xc.binding(),
            )?;

            let mut zrow = vec![0.0f32; t * AUDIO_CH];
            stream.read_blocking(xc.binding(), crate::vvae::as_bytes_mut(&mut zrow))?;
            for i in 0..t {
                for k in 0..AUDIO_CH {
                    out[(ch * AUDIO_CH + k) * t + i] =
                        (zrow[i * AUDIO_CH + k] - self.enc_mean[k]) / self.enc_std[k];
                }
            }
        }
        stream.synchronize()?;
        Ok((out, t))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_encoder_bounds_round_up_to_powers_of_two_from_256() {
        assert_eq!(pow2_bound(1), 256);
        assert_eq!(pow2_bound(256), 256);
        assert_eq!(pow2_bound(257), 512);
        assert_eq!(pow2_bound(800), 1024);
        assert_eq!(pow2_bound(1 << 20), 1 << 20);
        // and the decoder's do not: they are multiples of 256
        assert_eq!(round256(257), 512);
        assert_eq!(round256(800), 1024);
        assert_eq!(round256(1000), 1024);
        assert_eq!(round256(1025), 1280);
    }

    #[test]
    fn the_rates_multiply_out_to_the_hop() {
        assert_eq!(ENC_RATES.iter().product::<usize>(), HOP);
        assert_eq!(DEC_RATES.iter().product::<usize>(), HOP);
    }

    #[test]
    fn the_decoder_halves_its_width_at_every_upsample() {
        // 1024 down to 8 over seven levels, which is what makes 8 * l_out the plane bound
        let mut chan = 1024usize;
        for _ in 0..7 {
            chan /= 2;
        }
        assert_eq!(chan, 8);
    }

    #[test]
    fn the_residual_paddings_keep_the_length() {
        // a dilated convolution keeps its length when pad = (k*d - d) / 2, and the second, undilated
        // one when pad = (k - 1) / 2
        for (i, k) in RES_K.iter().enumerate() {
            let d = RES_DIL[i];
            assert_eq!(
                2 * ((k * d - d) / 2),
                (k - 1) * d,
                "kernel {k} dilation {d}"
            );
            assert_eq!(2 * ((k - 1) / 2), k - 1, "kernel {k}");
        }
    }
}
