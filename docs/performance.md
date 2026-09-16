# Performance and numerical validation

The current comparison measures h3-hrx and ComfyUI on AMD Strix Halo (`gfx1151`),
with 128 GB unified memory. Timings depend on sequence length, prompt, cache state,
and competing CPU/GPU work on the APU.

## September 14: complete native 768p comparison

At 1344×768, 124 frames, and 20 evaluations, h3 completed video and audio output
in **35 min 59 s**. Updated native ComfyUI took **4 h 3 min 53 s** with default
PyTorch BF16 attention, or **43 min 33 s** with its built-in Comfy Kitchen INT8
attention. The measured end-to-end advantages are **6.78×** and **1.21×**,
respectively. Comfy Kitchen needs no custom node or source patch.

Steady medians per evaluation were **103.9 s** for h3, **721.3 s** for default
ComfyUI, and **121.7 s** for Comfy Kitchen. Both engines already use INT8 linear
kernels for these quantized checkpoints. The large default-backend gap is mainly
an attention issue, rather than evidence that only h3 computes INT8 products.
A representative attention microbenchmark also finds a major PyTorch slowdown
with the model's strided QKV layout; the complete Kitchen run demonstrates how
much an available attention replacement changes the practical comparison.

h3 decoded video/audio in **75.2 s**, versus **147.5 s** for the Kitchen run.
This is a useful remaining advantage beyond sampling. Startup and conditioning
also contribute to full process time, and vary with filesystem/cache state.

Sampled full-pipeline GPU residency peaked at **56.63 GiB for h3**, versus
**26.64 GiB for either ComfyUI configuration**. Process PSS peaks were
**2.70 GiB**, **26.45 GiB** (default), and **26.06 GiB** (Kitchen). These are
overlapping memory views on the APU and must not be added. The maximum observed
drop in system available RAM was **58.36 GiB**, **51.24 GiB**, and **50.00 GiB**,
respectively; this whole-machine view includes other applications and caching.
h3's small PSS does not establish a smaller total footprint.

The headless ComfyUI runner releases conditioning models before sampling and
the DiT before decoding. The h3 run above retained models throughout the session.
The CLI now defaults to stage-scoped ownership. A complete repeat of alien-fjord
reduced peak GPU residency to **27.20 GiB (52% less)** and process PSS to
**1.80 GiB**, with byte-identical video and audio latents. It took **38m02s**,
including 116.3 seconds preparation, 2,085.9 seconds sampling (103.7-second steady
median), and 78.8 seconds decoding. The historical timing above had much warmer
startup. These changes establish a memory reduction, not an additional speedup.
The [implementation report](benchmarks/20260915-optimization/implementation.md)
includes stage peaks, raw telemetry, and latent digests.

There is one complete trajectory per configuration, with cached checkpoints and
sequential fresh processes. Steady medians exclude evaluation one. Equal seeds
across engines use different random streams, so output images are not identical.
See the [full report, videos, raw data, and memory measurements](benchmarks/20260914/README.md).

The h3 F16 attention control took **1 h 33 min 43 s**, with a **272.6 s** steady
median: 2.60× the total time of default h3. It retains the same overall creature
and composition in the sampled frames, with detail differences but no obvious
visual improvement for this prompt. F16 uses an older attention kernel family;
its cost is not a pure comparison of data types. Updating its operand layout and
tiling is another concrete improvement target. Full numerical checks and both
videos are in the current report.

### Earlier 768p measurements

The [September 13 report](benchmarks/20260913-768p/README.md) recorded a 103.2 s
h3 steady median versus 772.4 s for ComfyUI's default attention, with the ComfyUI
run intentionally stopped during evaluation five. Its roughly 4 h 20 min full
ComfyUI time was extrapolated from observed sampling and earlier overhead.
The complete native runs above supersede that estimate for the current comparison;
the original logs and calculation remain available as historical measurements.

## September 2026: 480p

For 864×480, 124 frames and 20 evaluations, h3-hrx's steady median was **28.1 s per
evaluation**, versus **85.5 s** for ComfyUI: **3.04× faster denoising**. This uses the
same prompt and Comfy-Org quantized checkpoints, with one trajectory per engine
and the first evaluation excluded. h3 computes int8 products with default int8 QK
attention; ComfyUI uses quantized linear kernels with BF16 activations and
default BF16 PyTorch attention.

Sampled peak GPU-resident buffers were **51.74 GiB for h3-hrx** and **25.45 GiB for
ComfyUI**. Process PSS peaks were **1.32 GiB** and **23.12 GiB**, respectively.
These views overlap on unified memory and must not be added together; PSS alone
does not represent the complete memory footprint. Five-second samples may miss
brief peaks.

h3 produced its MP4/WAV in 731.7 seconds with cached checkpoints. ComfyUI finished
sampling and video/audio decoding, then failed at MP4 encoding because the
container's FFmpeg lacked `libx264`. No end-to-end speedup is reported.

The [benchmark report](benchmarks/20260913/README.md) includes commands, versions,
checkpoint hashes, raw logs, telemetry, and a memory timeline. These are practical
within-run observations, not independent repetitions or a statistical-significance
claim. The output images and random noise streams are not identical.

## Historical development measurements

These older timings used `H3_PROFILE=1`, int8 checkpoint rows, and default int8 QK
attention. ComfyUI was measured with `bench_comfyui_h3.py`, a retired development
script, in the Strix Halo ComfyUI container. They are retained as historical context;
they are not the current reproducible comparison above.

| Clip | h3-hrx per evaluation | ComfyUI per evaluation | Observed speedup |
| --- | ---: | ---: | ---: |
| 1344×768, 124 frames | 101 s | 771 s | 7.6× |
| 864×480, 124 frames | 26.7 s | 103 s | 3.9× |
| 864×480, 22 frames | 4.2 s | 3.8 s | 0.9× |

A full generation also pays for weight loading, text encoding, kernel compilation
on a new shape, decoding, and output encoding. The default is 30 evaluations.
Short sequences do not show the long-clip speedup.

For the 864×480, 124-frame decoder workload, the current representative
resident video-plus-audio timing is **33.02 s**, down from **44.1 s**. It is
one uncontended comparison; a second confirmation was interrupted by other
GPU jobs. The 30-second target has not been reached. See the
[decoder report](archive/vae-30s.md) for timing boundaries, output checks,
and pending experiments. At 768p the demo run recorded 82.8 s for decode;
that measurement also has only one sample.

## Numerical agreement

Recorded video-row residual cosine against ComfyUI's own run of the same
denoising evaluation:

| Block | Default int8 QK attention | f16 attention |
| ---: | ---: | ---: |
| 0–10 | 1.0000 | 1.0000 |
| 20 | 0.9999 | 0.9999 |
| 30 | 0.9991 | 0.9992 |
| 40 | 0.9942 | 0.9947 |
| 49 | 0.9992 | 0.9993 |

In the recorded 20-evaluation trajectory, relative error was about 0.01 after
five evaluations and final latent cosine about 0.90. This is numerical
agreement, not identical clips. The implementations differ in attention
precision, quantization details, and floating-point operation order. Small
rounding differences grow through the sampling trajectory.

Encoder comparisons recorded audio cosine 1.0000000, vision 0.99996, and video
0.9994 against ComfyUI; the text encoder reached 1.0000 after 50 layers against
transformers bf16. The optimized f16 video decoder measured 63.95 dB PSNR
against the f32 diffusers decoder on an 864×480×22 case. These measurements
cover specific inputs, not all possible prompts or shapes.

See [test coverage](testing.md) for the numerical gates and their
required reference data.

## Implementation and profiling

The DiT uses an f32 residual stream: recorded activations exceed f16 range
by block 23. Core GEMMs retain the checkpoints' int8, bf16, f16, or f32 types.
Int8 QK attention uses rotated per-token Q/K quantization, f16 PV, and f32
online softmax. Padded operand pitches reduce cache aliasing.

The video decoder uses 128×256 f16 GEMM tiles, padded SwiGLU output, fused
QKV/normalization/RoPE, and head-major attention with 32-key staging. Audio
convolutions compute four samples per thread while retaining f32 accumulation
order. Detailed kernel results and unsuccessful variants are in the
[research archive](archive/README.md).

| Option | Purpose |
| --- | --- |
| `H3_PROFILE=1` | Print per-stage timings after each evaluation |
| `H3_TRACE=1` | Print and synchronize every launch; changes timing |
| `--attn f16` | Compare against f16 QK attention |
| `--attn i4` | Experimental int4 attention; recorded ghosting on conditioned clips |
| `H3_VAE_FAST=0` | Restore the original video decoder kernels for comparison |

Measure with an idle GPU and no concurrent CPU compilation. Rotating-weight
microbenchmarks better approximate decoder traffic than repeatedly using one
cached matrix; neither replaces a complete resident decoder measurement.
