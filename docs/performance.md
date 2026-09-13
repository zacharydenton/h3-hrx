# Performance and numerical validation

The current comparison measures h3-hrx and ComfyUI on AMD Strix Halo (`gfx1151`),
with 128 GB unified memory. Timings depend on sequence length, prompt, cache state,
and competing CPU/GPU work on the APU.

## September 2026 comparison

For 864×480, 124 frames and 20 evaluations, h3-hrx's steady median was **28.1 s per
evaluation**, versus **85.5 s** for ComfyUI: **3.04× faster denoising**. This uses the
same prompt and Comfy-Org quantized checkpoints, with one trajectory per engine
and the first evaluation excluded. h3 computes int8 products with default int8 QK
attention; ComfyUI dequantizes the weights for bf16 compute.

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
agreement, not identical clips: ComfyUI dequantizes weights to bf16, while
this implementation uses int8 products with int32 accumulation and stored
row scales. Small rounding differences grow through the sampling trajectory.

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
