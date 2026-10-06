# Performance and numerical validation

Measurements below used AMD Strix Halo (`gfx1151`) with 128 GB unified memory.
Timings depend on shape, prompt length, cache state and competing CPU/GPU work.
The September timings precede the October FP32 feed-forward changes.

## September 14: complete native 768p comparison

Same alien-fjord prompt, Comfy-Org checkpoints, 1344×768, 124 frames and
20 evaluations. End-to-end time includes loading, conditioning, sampling,
video/audio decoding and completed output files.

| Configuration | End to end | Steady median/evaluation | GPU residency peak | Process PSS peak |
| --- | ---: | ---: | ---: | ---: |
| h3-hrx, retained models | 35 min 59 s | 103.9 s | 56.63 GiB | 2.70 GiB |
| ComfyUI, PyTorch BF16 attention | 4 h 3 min 53 s | 721.3 s | 26.64 GiB | 26.45 GiB |
| ComfyUI, Comfy Kitchen INT8 attention | 43 min 33 s | 121.7 s | 26.64 GiB | 26.06 GiB |
| h3-hrx, stage-scoped repeat | 38 min 02 s | 103.7 s | 27.20 GiB | 1.80 GiB |

Both engines use INT8 linear kernels. Comfy Kitchen is a built-in attention
backend. The measured end-to-end speedups for the first h3 run are 6.78× against
PyTorch attention and 1.21× against Kitchen. Video/audio decoding took 75.2 s
for h3 and 147.5 s for Kitchen.

Stage-scoped ownership reduced h3's GPU residency by 52%, with byte-identical
video/audio latents. Its colder startup took 116.3 s rather than 11 s.
GPU residency and PSS overlap on unified memory and must not be added.

Each configuration has one complete trajectory, with cached checkpoints and
sequential execution. Steady medians exclude the first evaluation. Equal seeds
use different random streams across engines. The FP16 attention control took
1 h 33 min 43 s, but uses a different kernel family, so this does not isolate
the cost of precision alone.

[Full comparison, videos and raw data](benchmarks/20260914/README.md) ·
[Stage-scoped memory and latent checks](benchmarks/20260915-optimization/implementation.md)

## September 22: audio command reuse

With HRX 0.8.4 and retained audio scratch, five alternating fresh-process pairs
reduced a 3,200-sample stereo encode/decode benchmark from 430.811 ms to
298.476 ms (30.7%). Each process collected 11 warm samples; outputs were
byte-identical. The budgeted session retains scratch until eviction or teardown;
a stage-scoped session unloading the VAE does not retain it.

Use `examples/bench_runtime.rs` for warm session latency and
`examples/profile_audio.rs` for synchronized stage diagnostics.

## Earlier measurements

| Workload | Result | Record |
| --- | --- | --- |
| 864×480, 124 frames, 20 evaluations | 28.1 s/evaluation for h3; 85.5 s for ComfyUI | [480p report](benchmarks/20260913/README.md) |
| Early 768p comparison | ComfyUI stopped during evaluation five; total time extrapolated | [Estimate and logs](benchmarks/20260913-768p/README.md) |
| Resident 864×480×124 video/audio decode | 44.1 s reduced to 33.02 s in one uncontended comparison | [Decoder tuning](archive/vae-30s.md) |

The 480p ComfyUI run completed sampling and decoding but failed during MP4
encoding; it supplies no complete end-to-end comparison. Historical kernel
measurements and retired variants are in the [research archive](archive/README.md).

## Numerical agreement

Recorded video-row residual cosine against ComfyUI for one denoising evaluation:

| Block | INT8 QK attention | FP16 attention |
| ---: | ---: | ---: |
| 0–10 | 1.0000 | 1.0000 |
| 20 | 0.9999 | 0.9999 |
| 30 | 0.9991 | 0.9992 |
| 40 | 0.9942 | 0.9947 |
| 49 | 0.9992 | 0.9993 |

The recorded 20-evaluation trajectory ended near 0.90 latent cosine. Encoder
comparisons recorded audio cosine 1.0000000, vision 0.99996, video 0.9994 and
text 1.0000 after 50 layers. These cover specific inputs; quantization and
floating-point ordering differences accumulate through sampling.

The October FP32 fixes preserve feed-forward products above the former FP16 cap,
use FP32 Hadamard scratch, and preserve non-finite results through clamps.
A 384×256 replay reached an activation magnitude of 287,332 and completed with
finite latents under a 32 GiB budget (26.47 GiB sampled allocation reservations).
Video latents changed by 1.72% relative L2. Fixed baseline latents still decoded
byte-identically, and the frame-17/34 pulses remained. The instrumented run is
not a controlled timing comparison. See the
[FP32 validation record](benchmarks/20261006-fp32-feedforward.json) and
[test coverage](testing.md).

## Implementation and profiling

DiT residuals and feed-forward intermediates use FP32. Matrix operands retain
INT8/BF16/FP16 formats as appropriate. INT8 QK attention uses rotated per-token
quantization, FP16 PV and FP32 online softmax. Audio convolutions retain FP32
accumulation order.

| Option | Purpose |
| --- | --- |
| `H3_PROFILE=1` | Synchronized kernel timings; changes execution costs |
| `H3_STAGE_TRACE=1` | JSON events for model loading, packed rows, sampling and release |
| `--attn f16` | Compare against FP16 QK attention |
| `--attn i4` | Experimental INT4 attention; conditioned clips have shown ghosting |
| `H3_VAE_FAST=0` | Compare with the original decoder kernels |

Measure on an idle CPU and GPU. Rotating-weight microbenchmarks approximate
model traffic better than repeatedly using one cached matrix, but still need
confirmation with complete output and peak-memory checks.
