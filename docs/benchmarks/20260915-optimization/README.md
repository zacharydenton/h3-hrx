# 768p optimization audit — 2026-09-15

The largest opportunities are fewer transformer evaluations and shorter model
lifetimes. Dense attention and matrix multiplication still deserve targeted
work, but the existing kernels have already survived extensive tuning. Another
large speed multiplier is more plausibly available through distilled adapters
or controlled computation reuse than through changing a tile constant.

This audit covers h3-hrx at `f212ae860d0dad5c9ffd47dc5015b1f233ebb034`, the
[completed 768p measurements](../20260914/README.md), current source, archived
kernel experiments, and the primary upstream sources linked below. No new GPU
workload was run for this audit. Subsequent runtime changes and validation are
tracked in the [implementation report](implementation.md); the audit below describes
the original revision. [analyze.py](analyze.py) reproduces the static allocation calculations
and timing scenarios in [accounting.json](accounting.json) using only Python's
standard library. Scenarios are arithmetic projections, not measured gains.

**What the current measurements tell us**

The alien-fjord baseline is 1344×768, 124 frames, synchronized audio, and twenty
transformer evaluations. h3 takes 2,158.9 seconds (35m59s), including 2,071.4
seconds sampling and 75.2 seconds decoding. Default ComfyUI takes 4h03m53s;
Comfy Kitchen takes 43m33s. Thus the measured advantage is **6.78× over default
ComfyUI and 1.21× over Comfy Kitchen**. These are one complete trajectory per
configuration, not repeated statistical estimates. Any adapter comparison
should give both engines the same adapter and inference schedule.

The separate two-evaluation h3 profile attributes its instrumented denoising
time as follows:

| Operation | Share |
| --- | ---: |
| Attention | 58.5% |
| Gate/up projection and SwiGLU | 13.7% |
| QKV projection | 10.3% |
| Down projection and residual | 7.1% |
| Output projection and residual | 3.5% |
| Attention operand preparation | 2.8% |
| Other instrumented operations | 4.1% |

Profiling synchronizes dispatches and disables graph replay. Applying these
shares to ordinary sampling time gives useful prioritization, not exact
unprofiled component timings:

| Hypothetical change | Projected whole render | Time saved |
| --- | ---: | ---: |
| Attention latency reduced 20% | 31m57s | 11.2% |
| All four GEMM latencies reduced 20% | 33m36s | 6.6% |
| Both reductions | 29m33s | 17.9% |
| Attention twice as fast | 25m53s | 28.1% |
| Decoder twice as fast | 35m21s | 1.7% |

Even an exceptional attention-only improvement cannot make this whole pipeline
twice as fast. The rest of the transformer remains substantial.

**1. Native distilled-adapter support has the largest practical speed upside**

LightX2V publishes FL2VA/T2VA adapters trained at **1344×768**, with both four-
and eight-evaluation versions. The 768p versions use video/audio shifts 6/3;
the lower-resolution adapters use different settings. The task support makes
these more relevant to the Glass Leviathan image-to-video workflow than a
text-only distillation. See the author's
[model specifications](https://github.com/ModelTC/Minimax-H3-Turbo).

Comfy-Org distributes
`minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors` alongside the
quantized base weights. This is an optional adapter track that can retain the
existing Comfy-Org INT8 base checkpoint. Both files should resolve through the
standard Hugging Face Hub cache. The adapter's filename and source repository
do not require adopting ComfyUI's local directory layout.
[Comfy-Org model inventory](https://huggingface.co/Comfy-Org/MiniMax-H3).

With unchanged average transformer cost and the baseline's 87.5 seconds of
non-sampling overhead, eight calls project to **15m16s**, and four to **8m22s**.
These omit adapter execution and loading costs and do not establish equivalent
quality. They explain why this integration deserves priority.

The integration is real work: h3's current weight recipes rearrange existing
checkpoint bytes without implementing a general adapter path. Validate tensor
mapping, fused QKV branches, rank/alpha scaling, compressed AdaLN, and the
effective weight basis used by the ConvRot checkpoint. A low-rank side branch
can retain the INT8 base; merging and requantizing creates a different numerical
path and needs separate validation. Never add BF16 adapter tensors directly to
INT8 storage. Adapter contributions must enter before the corresponding
activation, normalization, or residual operation; current fused epilogues
make that placement consequential.

The upstream workflows default to Euler and require matching the adapter,
step count, and modality shifts together. h3's `Schedule::new` currently takes
grid points including the final zero, so four evaluations require five grid
points. The first integration should explicitly check the complete sigma arrays,
including the terminal zero, rather than copying a CLI step integer.
[Upstream inference instructions](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/COMFYUI_SETUP_AND_INFERENCE.md),
[h3 schedule](../../../src/layout.rs).

FastH3 is another promising track: its recommended preview uses four forwards
and 90% sparse attention, but requires VSA-H3 semantics and currently supports
T2VA, not distilled FL2VA/Ref2VA. It is a larger attention/backend integration.
It should follow the dense 768p adapter work, not block it.
[Official FastH3 model card](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree).

An experimental third-party FastH3 conversion explicitly targets our pruned
INT8 base, but needs an additional gate-weight file and injected sparse
attention layers; a generic LoRA loader alone is insufficient. Its existence
helps identify conversion requirements, not establish working h3 support.
[Conversion author's model card](https://huggingface.co/barelymining/ComfyUI-MiniMax-H3-FastVideo).

**2. Release completed stages to substantially reduce memory pressure**

`Session` retains the text encoder, DiT, and VAEs for reuse. That is useful for
an interactive service, but a single-render CLI pays for simultaneous residency
after earlier stages have finished. The measured h3 GPU-residency peak is
56.63 GiB and occurs during decoding; Comfy's is 26.64 GiB. Whole-machine
available-memory drops are 58.36 versus 51.24 GiB for default Comfy, respectively.
On this UMA machine, GPU residency, process PSS, and system pressure are
overlapping views; adding them produces a misleading total.

The text encoder's fifty layers hold **22.92 GiB of main INT8 linear matrices**,
including the actual padded pitches. That excludes scales, norms, the vision
tower, and scratch. Those matrices remain allocated after prompt preparation.
The DiT's main INT8 matrices similarly account for 17.98 GiB. These are static
allocation counts, not measurements of a revised pipeline peak.

Split `Dit::denoise` into conditioning preparation and sampling so the encoder
can be released after `vision_for`/`text_in`, before `ensure_blocks` loads the
fifty DiT blocks. Release the DiT before final VAE decoding. For image-to-video,
the input VAE also participates earlier; stage ownership must accommodate this.
Provide a lifecycle policy so reusable sessions can retain weights when desired.
[Session ownership](../../../src/session.rs),
[conditioning and sampling](../../../src/dit.rs).

`Weights` caches uploaded buffers in `Arc`s, and stacks/graphs may retain their
own references. Clearing only the cache map is insufficient: release the owners
and graph executions and synchronize in-flight work at the transition. Do not
introduce per-layer eviction during sampling. Measure the new conditioning,
sampling, and decode peaks separately; subtracting 22.92 GiB from the old global
peak does not prove the new machine requirement. In particular, 32 GiB
compatibility remains unverified. [Weight ownership](../../../src/weights.rs).

There is also a straightforward oversized buffer: `Blocks.text_copy` allocates
for the entire sequence but copies/restores only the text/reference prefix.
For alien-fjord it allocates 0.765 GiB for 0.0068 GiB of used prefix; for Glass
Leviathan, 0.806 GiB for 0.0468 GiB. Right-sizing saves approximately **0.76 GiB**
in either case without changing arithmetic. Track prefix capacity separately
and invalidate affected graph bindings if storage changes.

**3. The existing step cache could skip substantial work, but needs hardening**

The optional cache runs block zero every evaluation and can reuse a recorded
residual in place of blocks 1–49. It is disabled by default and is not exposed
in the normal CLI. If five of twenty suffixes were safely skipped, an idealized
equal-block-cost model projects about **27m31s**; ten skips project **19m04s**.
That approximation ignores cache overhead and small per-evaluation work outside
the blocks. Achievable skip rates and perceptual quality are not established.
[Decision code](../../../src/cache.rs),
[actual caller](../../../src/dit.rs).

Two issues should precede any user-facing speed preset:

- The documentation says the first two evaluations are always full. The caller
  records a residual after evaluation zero, allowing evaluation one to skip.
  The existing test omits that `recorded()` call. The CPU reproduction
  confirmed the mismatch; [cache_contract.rs](cache_contract.rs) now
  checks the repaired contract. This establishes
  a broken warmup claim, not that reusing the first residual is inherently
  invalid. Choose and enforce the intended warmup policy before calibration.
- The error metric combines all rows. Generated video contributes 37,296 rows
  and audio just 414. A single aggregate can hide important audio changes.
  Track video, audio, and conditioning changes separately and require each
  relevant gate to pass; calibrate thresholds rather than assuming equal scales.

Use mandatory full evaluations at the beginning/end and a maximum consecutive
skip count. First observe metrics on a full trajectory without skipping, then
evaluate one conservative candidate against saved identical h3 input noise.
Check identity, motion, temporal coherence, prompt adherence, dialogue and
audio synchronization. A still landscape is an insufficient calibration case.

Caching also costs memory: three full FP32 residual buffers add **2.29 GiB**
for alien-fjord and **2.41 GiB** for Glass Leviathan, plus reduction scratch.
Stage release should come first. The existing behavior resembles a first-block
cache; published TeaCache gains on other model families do not validate H3's
threshold or audio behavior.

**4. Target attention's data movement and synchronization, not another broad sweep**

The current production attention already has online softmax, fused score/value
processing, head-major quantized operands, and graph execution. It does not
materialize the quadratic score matrix. Generic advice to add FlashAttention,
INT8, or graphs misses optimizations already present.

The archived tuning measured roughly 36–37 trillion matrix-operation-equivalents
per second for production attention against a roughly 51 trillion-operation
matrix-only calibration. That calibration is an arithmetic reference, not a
proven throughput ceiling. Production uses **192 VGPRs, 27,648 bytes of LDS,
and no spills**. Prefetch variants grew to 224/256 registers and lost; more
waves, larger K tiles, exp approximations, alternate staging, and many layout
variants were already tested. [Detailed experiment history](../../archive/attention-int8-45.md).

The most useful unfinished experiment is the archived five-phase shader-cycle
instrumentation: staging/barriers, QK and shared-memory access, softmax/scales,
rescaling/PV, and final synchronization. Its small-shape check passed; a clean
large-shape phase measurement was not completed. Port that bounded harness to
the current runtime, verify unchanged output/register use, and profile the exact
production shape before selecting a new kernel design. Per-wave cycle counts
include waits and are not additive whole-GPU wall time.

Two centered K128 variants became compilable without spills after a Loom
allocator fix but were not GPU-validated. They warrant a small controlled
comparison, not an assumption that K128 is better. Changes to quantization
granularity or value quantization are a separate precision experiment.

On AMD's RDNA3 WMMA table, INT8 and FP16 have the same listed arithmetic rate;
INT4 is higher. The local INT8/FP16 calibration is consistent with this. An
INT8 value product may reduce storage or register pressure, but one cannot
assume it doubles arithmetic throughput as on some other architectures.
[AMD WMMA documentation](https://gpuopen.com/learn/wmma_on_rdna3/).
The existing INT4 attention path has produced ghosting in conditioned clips;
it is unsuitable as a default speed recommendation without new quality evidence.

**5. Fuse operand preparation and reuse temporary storage**

The DiT separately allocates fused QKV output and split FP16 Q/K/V. The split
set occupies **1.53 GiB** at the alien shape and **1.61 GiB** at the Glass shape.
A post-GEMM kernel that applies normalization/RoPE and directly prepares INT8
head-major Q/K plus the value layout could eliminate that intermediate set.
Maintain the current FP16 rounding boundaries if claiming numerical parity;
otherwise treat the fusion as a precision change and test it accordingly.
[Scratch allocations and dispatch](../../../src/stack.rs).

Also prove temporary lifetimes before aliasing attention and MLP workspaces.
Graph branches already overlap independent Q/K/value preparation, so buffer
aliasing must respect actual dependencies, not just source order. Operand
preparation is only 2.8% of instrumented denoising: its direct time-saving
ceiling is small, although avoiding intermediates may reduce memory pressure.

One preparation kernel retains shared-memory butterfly stages because a past
Loom subgroup-shuffle lowering produced stale register reads. A focused compiler
regression check could determine whether that workaround is still necessary.
It is a useful Loom opportunity, but H3's profile bounds its pipeline payoff.

**6. Tune the large GEMMs by their real shapes**

Gate/up and QKV together account for 24% of denoising, substantially more than
down projection alone. At the alien shape, M is 38,048, and the (K,N) dimensions
are (5376,28672), (5376,21504), (14336,5376), and (7168,5376).
Use these shapes and rotating real weight buffers in an alternating baseline/
candidate harness. The current small GEMM comparison shapes do not represent
this workload.

The long down-projection group-size tuning and the decoder's 1,797-token group
specializations are already selected in `model.rs`. Avoid rediscovering those.
Screen a bounded set of traversal/group choices on gate/up and QKV, inspect
spills and sustained clocks, and keep only reproducible improvements.
[Current selection](../../../src/model.rs),
[GEMM tuning history](../../archive/gemm-int8-tuning.md).

**Lower priorities and measurement corrections**

The decoder is already much faster than Comfy Kitchen's measured decode, and
halving its current 75 seconds saves less than 2% of the whole render. Batching
independent spatial tiles could improve weight reuse once stage release frees
memory. Larger tiles are not automatically faster: they reduce overlap but
increase attention work and change boundary context. Decoder optimization
becomes more important after reducing sampling to four evaluations.

The default multistep sampler transfers generated rows to the CPU every
evaluation. Porting it to the device can remove transfers, but one synchronization
per roughly 104-second evaluation is not the dominant bottleneck. Euler already
has a GPU update path. Preserve the sampler's documented floating-point operation
order when evaluating parity. Likewise, blindly changing the FP32 residual stream
to FP16 is inappropriate: prior experiments encountered values outside FP16's
finite range.

Startup varies substantially with loading/cache state. Instrument file reads,
upload, compilation, and conditioning separately before changing mmap advice
or adding prefetch. `MADV_DONTNEED` on file mappings does not by itself prove
physical global-page-cache eviction. Keep the standard HF cache and measure
fresh-process and reused-session latency separately.

The attention microbenchmark used N=37,754. The actual alien run has
37,296 video + 414 audio + 338 text = **38,048 rows**; Glass has another 1,008
keyframe rows and 1,329 text rows, totaling **40,047**. The earlier prose omitted
one audio stream in relating the microbenchmark to the prompt. Its description
has been corrected; all raw observations remain unchanged. Future harnesses
should derive and record every layout component rather than manually copying N.

**Recommended order of work**

1. Split stage ownership, release finished models for single renders, and shrink
   `text_copy`. Verify output parity and stage-specific memory peaks.
2. Add native adapter support and the 768p Turbo schedule. Start with the
   image-conditioned alien case and a matching Comfy run using the same adapter.
3. Repair the cache warmup contract and collect modality-specific metrics on a
   full trajectory; enable a conservative candidate only after quality review.
4. Recover the bounded attention phase diagnostic, then select one informed
   kernel experiment. Validate sustained performance with ordinary GPU timing.
5. Fuse preparation/reuse scratch and tune the remaining large GEMMs. Revisit
   decoding once sampling has become much shorter.

For numerical changes, reuse bit-identical exported noise within each engine;
equal integer seeds across engines do not generate identical initial noise.
For performance claims, use complete unprofiled runs and report prompt, reference,
audio/video rows, actual evaluation count, attention backend, adapter and revision,
sigma schedule, stage timings, and all three separate memory views. Use the saved
baseline and a small paired evaluation first, expanding only if results conflict.
The PyTorch/HIP profiler associated with the prior host freeze is not needed for
this plan; its causal role remains unproven.

CPU reproduction from the repository root:

```sh
python docs/benchmarks/20260915-optimization/analyze.py
rustc --edition=2024 docs/benchmarks/20260915-optimization/cache_contract.rs -o /tmp/h3-cache-contract
/tmp/h3-cache-contract
```
