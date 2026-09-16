# Implementation progress

The [audit](README.md) describes the original `f212ae8` runtime. Work below
implements the accepted roadmap; experimental features are not performance claims.

## Measured stage-scoped result

The complete alien-fjord run at 1344×768, 124 frames, seed 1618, and twenty
base evaluations produced **byte-identical video and audio latents** to the
saved September 14 baseline. Peak GPU residency fell from **56.63 to 27.20 GiB
(52%)**. Peak process PSS was 1.80 GiB. Conditioning peaked at 26.10 GiB GPU,
sampling at 27.20 GiB, and video decoding at 5.26 GiB.

Complete process time was 2,282.06 seconds (38m02s): 116.3 seconds preparation,
2,085.9 seconds sampling, and 78.8 seconds decoding. The historical run took
2,158.91 seconds with only 11 seconds preparation and 2,071.4 seconds sampling.
This validates a memory reduction, not a speedup. The new steady median was
103.7 seconds per evaluation versus 103.9 previously.

System available memory dropped 37.56 GiB, with 33.98 GiB remaining at its
minimum. External applications contribute to these counters; GPU residency,
process PSS, and system pressure overlap on UMA and must not be added. This
does not establish compatibility with a 32 GiB machine.

[Raw timing](stage-scoped/timing.json), [stage events](stage-scoped/stages.json),
[compressed telemetry](stage-scoped/telemetry.jsonl.gz),
[source/binary identity](stage-scoped/metadata.json), and
[latent digests](stage-scoped/parity.json) preserve the measurement.

## Completed checks

- Stage-scoped ownership and prefix-sized `text_copy` implemented. Existing
  `Session::new` retains models; CLI defaults to stage-scoped sessions.
- GPU comparison passed: retained and stage-scoped sampling produced identical
  latent bytes. Cancellation released model owners; the same session then
  completed a subsequent request correctly.
- The test's raw timing, stage events, and UMA telemetry are currently in
  `build/optimization-20260915/residency-parity.*`. This is a small-shape parity
  test, not a measured 768p memory requirement.
- Both pinned Turbo adapters downloaded into the normal HF Hub cache and all
  624 tensors validated, covering 208 projections across 50 sampling and two
  refiner blocks. Four-step alpha/rank is 1; eight-step alpha/rank is 0.0625.
- Cache policy has separate conditioning/audio/video gates, two full warmup and
  final evaluations, no consecutive skips, and an observation-only mode. CPU
  tests pass. GPU observation and forced-full runs preserve both uncached latent
  streams exactly; an eligible actual skip returns finite, different output.
  Thresholds are not calibrated or promoted.

- New FP32 base GEMMs preserve values above FP16 range. INT8 and floating GEMM
  regressions passed, as did adapter-add-before-activation/residual tests.
- Real adapter A/B projections passed independent CPU matrix-product comparisons
  for both presets, all four projection types, and the DiT/refiner stacks.
- Real adapted DiT/refiner blocks passed three changing-input eager/graph replays
  byte-for-byte for both presets. The CPU reference's initial 1% full-block gate
  exposed 1.40–1.43% propagated error in the synthetic DiT case. Intermediate
  checks located only 0.011% QKV and 0.066–0.080% attention error; supplying the
  same attention output reduced the remaining block error to 0.090–0.113%.
  Refiner full-block error was 0.197–0.200%. No runtime arithmetic was changed
  to resolve this: the reference now checks QKV below 0.1%, attention below
  0.2%, remaining projections below 0.3% with identical attention input, and
  the independent complete block below 2% with cosine above 0.999. This captures
  INT8 rounding propagation while retaining separate projection/adapter gates.
- All four uncached denoising goldens passed unchanged. The old cache golden
  intentionally changed because early/consecutive skips are now forbidden;
  new observation/forced-full trajectory comparisons replace that expectation.
- All four uncached denoising goldens also passed with `H3_REUSE_SCRATCH=1`.

## First native Turbo result

The four-evaluation alien-fjord render completed in **605.70 seconds (10m06s)**
at 1344×768, 124 frames, seed 1618: 80.85 seconds preparation, 448.8 seconds
sampling, and 74.84 seconds decoding. Peak GPU residency was **38.21 GiB** and
process PSS was **1.84 GiB**. The adapter adds weights and scratch; it reduces
the evaluation count rather than the cost of each evaluation.

This is 3.77× faster than the stage-scoped twenty-evaluation run above, using a
different trained schedule and adapter. It does not establish equivalent quality
or the same advantage against an equally configured Comfy run.

[Watch the clip](../../media/benchmarks/20260915/alien-turbo4.mp4) ·
[16-frame contact sheet](../../media/benchmarks/20260915/alien-turbo4-frames.jpg) ·
[measurement and validation](alien-turbo4/summary.json).

The sampled frames preserve creature identity, lighting, and composition, with
visible membrane/filament motion. Both streams passed complete FFmpeg decoding;
both latent streams are finite. Audio has RMS 0.0429, peak 0.2166, and no near-full-
scale samples. Auditory quality and synchronization have not been reviewed.
The matched Comfy Kitchen run and the remaining four/eight-evaluation cases are
still in progress.

## Active implementation

- Native low-rank BF16 branches preserve the INT8 base checkpoint and add before
  activation/gating. Base projections preserve FP32 output. Low-rank intermediate
  rounding requires numerical and perceptual qualification.
- Experimental CLI presets are hidden from help until qualification. No default
  quality, sampler, or base evaluation-count change is intended.
- `H3_STAGE_TRACE=1` emits structured layout, checkpoint identity, schedule, cache,
  and ownership events without enabling the kernel profiler.
- `scripts/optimization_benchmark.py` wraps ordinary generation with source and
  binary identity, process timing, stage events, and separate UMA memory views.

## Remaining release gates

1. New adapter kernels and shared-GEMM regressions passed; CPU workspace tests
   and clippy passed. Repeat after final changes and verify the package.
2. Qualify real adapted projections/refiner/block output against an independent
   reference, then evaluate both 768p Turbo presets on alien-fjord and Glass.
3. Stage-scoped 768p base run and stage memory measurements passed, as above.
4. Compare the accepted Turbo presets with native Comfy Kitchen using the same
   adapter and schedule, including complete-process time and separate memory views.
5. Calibrate cache observations before exposing a conservative CLI cache mode.
6. Run the bounded attention/GEMM experiments, operand-fusion work, and conditional
   sparse-attention/decoder follow-ups in the accepted plan. Record rejected
   experiments as well as wins.
7. Update user-facing documentation only with validated behavior and measurements.

The separate PyTorch/HIP profiler associated with the earlier host freeze is not
part of this work. Existing external GPU workloads are left undisturbed.

## Bounded kernel experiments

`h3-dev compile-probes` now compiles fused QK normalization/RoPE/INT8 preparation
and direct fused-QKV V transpose with the installed compiler. The runtime switch
`H3_FUSED_OPERANDS=1` avoids separate Q/K/V allocations for compatible unsmoothed
INT8 attention. The FP16 boundary after RoPE is preserved. Its GPU comparison
passed with identical Q/K codes, scales, and transposed V, including padded rows.
`H3_REUSE_SCRATCH=1`
aliases gate/up output with the dead QKV allocation when it fits; the attention
branch join precedes reuse. A 1344×768, 22-frame, one-evaluation comparison with
both switches produced identical video/audio latents. Sampling was 10.1 seconds
split and 9.5 seconds fused; these short single samples do not establish a speed
claim. Full-length parity and sustained timing gates remain pending.

The same scratch switch now also stores the adapter's BF16 input in its later
FP32 base-projection workspace. The A projection consumes that input before the
base projection writes the allocation, with explicit dependencies throughout.
This removes another approximately 1 GiB at the alien shape; adapted graph and
projection checks for this additional alias are queued separately.

The subgroup-shuffle diagnostic compiles after expressing the partner lane bounds
explicitly. All 64 repeated GPU comparisons passed against the LDS preparation
kernel. Timing and whole-trajectory checks precede changing the production
workaround. The archived phase diagnostic has been ported to the
current assembly target contract but still requires the unsupported
`s_getreg_b32_shader_cycles` descriptor. The installed shared compiler rejects it;
no production compiler or kernel has been changed to hide that limitation.

The installed compiler's ELF metadata reports 28 VGPRs and 4,096 bytes of LDS
for fused QK preparation, 8 VGPRs and 2,304 bytes of LDS for the direct V
transpose, and 15 VGPRs with no LDS for the shuffle probe. All report zero
private-segment bytes. These are resource checks, not measured speedups.

The two archived centered K128 sources and their isolated compiler artifacts are
not present in this checkout, including ignored build files. Their historical
compile-only reports do not establish a working candidate. The bounded GEMM
harness is available as `h3-dev tune-gemm`, covering all four projections at the
actual alien/Glass row counts, four real layer weights, and groups 2/3/4/8.

The removed legacy cache golden used 256×256, 22 frames, six sigma grid points,
ResMultistep, seed 7, threshold 0.15. Its video/audio digests were
`b91dc015f89185f972c0e4152a2558ca6e350ce1c79487680351de45c6a7c0c1` and
`333e54b3354e7643637e0f0275864334f1d72c7bcb52bb68d4e0ec36db9db39b`.
Those record the retired skip policy and are not updated to bless new output.
The replacement compares observation and forced-full execution directly against
an uncached trajectory, and separately checks an eligible actual skip.

## Conditional follow-ups

The measured four-evaluation pipeline spends 74.84 of 605.70 seconds decoding
(12.4%). Even halving decode would save 6.2% of this whole render. The existing
decoder already takes about half the time of the historical Comfy decoder.
Spatial batching would require independent attention domains and temporal-state
ownership; changing tile boundaries also changes the numerical reference. It
remains a separate, lower-priority experiment rather than a default tile change.
Turbo already uses the resident Euler update, so moving the base multistep
sampler to the device would not accelerate this preset.

FastH3 is not compatible with the dense Turbo path as a drop-in adapter. The
[official preview](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
requires its VSA-H3 backend and supports distilled T2VA only. The
[pruned-INT8 conversion](https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo)
adds compressed AdaLN adapter tensors and fifty new gate projections, alongside
a separate gate checkpoint. A native integration needs new routing/attention
semantics and an independently checked AdaLN conversion. The strict current
loader rejects these extra tensors; no sparse preset is advertised for Glass
Leviathan or enabled by silently loading only the familiar projection keys.
