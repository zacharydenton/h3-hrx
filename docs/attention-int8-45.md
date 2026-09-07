# Toward 45 TFLOP/s-equivalent in Loom

**45 has not been reached.** The new 64-key kernel is now selected by the C
host's long-sequence INT8 attention path. Final production-source measurements
on the Radeon 8060S (`gfx1151`) are:

| Tokens | Heads × dimension | Median time | TFLOP/s-equivalent |
| ---: | ---: | ---: | ---: |
| 16,000 | 56 × 128 | 195.521 ms | **37.541** |
| 37,723 | 56 × 128 | 1,104.313 ms | **36.947** |

Separate alternating comparisons measured 37.668 versus 35.421 at 16,000
and 37.104 versus 35.297 at 37,723: approximately **6.3% and 5.1% faster**
than the previous 32-key head-major kernel. Production trials used three rounds,
one warmup per round, and 12 measured launches at 16,000 or four at 37,723.
All timed runs waited for an idle GPU and low aggregate CPU use.

The metric remains `4 * N * N * heads * 128 / kernel_seconds / 1e12`, counting
both dense matrix products. QK uses INT8 with INT32 accumulation; online softmax
and output accumulation use FP32; P and V use FP16. It is kernel-only timing,
excluding quantization, preparation, allocations, and transfers. No HIP-compiled
GPU code participates in this production kernel. Full model rendering was not
benchmarked in this optimization run.

Clocks and power limits were unchanged; the driver remains in `auto` mode.
The 16,000-token trials recorded median busy clocks around 2.42 GHz. The
advertised 2.9 GHz maximum is not a demonstrated sustained clock for this kernel;
scaling measured throughput by a clock ratio would not establish a 45 result.
All rounds, hashes, compiler identity, and environment samples are retained in
[attention-int8-45-benchmarks.json](attention-int8-45-benchmarks.json).

## What changed

- Eight wave32 query tiles share 64 keys in one LDS buffer, with a barrier before
  reuse. This amortizes loop and output-rescaling work across more keys.
- Two 32-register vectors carry the 64 FP32 output accumulators. This arrangement
  reduces allocator copies and register pressure relative to independent matrix
  fragments.
- K loads look ahead by one 16-channel fragment; V loads look ahead by two
  fragments. Short public `low.invoke` helpers preserve WMMA ordering.
- Both large specializations use **192 VGPRs, 27,648 bytes of LDS, and no spills**.

Q/K/scales retain the existing head-major layout. Buffer sizes, preparation,
launch geometry, and output layout are unchanged. The 32-key kernel remains
available as a benchmark reference. The C host selects
`attention_i8qkhm_mha8_k64_lds_f16_wmma` for its eight-wave INT8 path.

## Validation and reproduction

Twenty full-output cases pass against FP32 attention on identical quantized
operands: token counts 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
255, 256, 257, and 1001, plus score-scale stress cases 0, 32, and 256 at 257
tokens. Outputs are deterministic. Maximum relative L2 error in these cases
is below 0.00030. Large runs check sampled rows against FP32 and verify that
every output is finite; their relative L2 error is approximately 0.000295 and
0.000314. CPU host regressions and both host-runtime builds pass.

```bash
python3 tools/gen_attention_i8_head_major_64.py
python3 tests/test_attention_i8_head_major.py
OPENBLAS_NUM_THREADS=4 python3 tools/bench_attention_i8.py 16000 \
  attention_i8qkhm_mha8_k64_lds_f16_wmma \
  --head-major --rounds 3 --repeat 12 --wait-idle \
  --output build/attention45/reproduce
```

Use a Python environment with NumPy. The generator uses `loom-format` from
`HRX_BUILD` (default `~/code/hrx-system/build-cuda`) and reproduces the kernel
byte-for-byte from the retained 32-key template.

## Remaining gap

Experiments covered larger workgroups, 128-key tiles, two query tiles per wave,
several operand-prefetch schedules, vector scale loads, alternative probability
packing, and skipping mathematically redundant rescaling. None reached 45.
Some failed register allocation or accuracy checks and were excluded from
performance claims. Local prototypes and generation scripts are archived under
`build/attention45/prototypes/`; they are not selected by the host.

ROCm counters on the 64-key variant before LDS lookahead reported zero LDS bank
conflicts and roughly 22% memory-unit activity. Those counters do not identify
one definitive bottleneck. The cache-counter collection stalled and was
terminated, so there is no cache-hit conclusion. A temporary allocator-capacity
probe enabled the 128-key experiments; the compiler change was restored.


## Follow-up screening

Additional two-round, four-launch trials at 16,000 tokens did not establish a
faster production replacement. Each row below comes from its own alternating
comparison; compare against that row's control rather than across rows.

| Experiment | Best candidate TFLOP/s-equivalent | Production control |
| --- | ---: | ---: |
| Four- or six-wave workgroups | 35.467 | 38.049 |
| First-tile peeling | 37.227 | 37.634 |
| Twofold loop unrolling | 34.572 | 37.634 |
| Query-scale factoring with FMA | 38.071 | 37.826 |
| Earlier key-scale loads | 36.978 | 37.911 |
| Reloading query fragments | 36.801 | 37.357 |

The scale-FMA variant's small screening gain is insufficient evidence for
promotion: it still needs the full stress suite and long-sequence validation.
The query-reload experiment reached 160 VGPRs without spills, but the added
loads outweighed the concurrency benefit.

A four-wave wave64 kernel passed a 257-token, three-head full-output check
(relative L2 0.000292), then measured 29.166 TFLOP/s-equivalent at 16,000 tokens.
It used Loom throughout, with a temporary descriptor for `v_permlane64_b32`.
That compiler change was restored; its patch and prototype remain in the local
archive. A two-query wave64 variant spilled and was not timed. The benchmark
harness now records `--wave-size` and uses it for launch geometry.

The production kernel and its compiled binary remain unchanged. These short
screens do not supersede the production measurements above. Raw follow-up
reports are included in the benchmark JSON; power management remains `auto`.


## Sustained arithmetic and additional scheduling probes

A Loom matrix-only calibration measured **51.727 TOP/s for INT8**, **51.219
TFLOP/s for FP16**, and **51.205 TFLOP/s-equivalent for a 50/50 mixture**.
These are not attention results. The microkernels reuse constant nonzero operands
in registers, execute 64 WMMA instructions per loop iteration, and check every
output exactly. They exclude softmax, quantization, and attention's operand
traffic. Sources, binaries, and raw records are under `build/attention45/roof/`;
the reproducible driver is archived as `prototypes/attention45-roof.py`.
The restored compiler reproduces all three binaries byte-for-byte.

This measured similarity is consistent with AMD's published RDNA 3 table, which
lists the same per-clock WMMA rate for INT8 and FP16/BF16
([AMD WMMA guide](https://gpuopen.com/learn/wmma_on_rdna3/)). The calibration
establishes an arithmetic reference, not an upper bound or a prediction for
full attention.

A longer production attention run, with two rounds of 100 measured launches,
returned **37.235 TFLOP/s-equivalent** (197.066 and 197.189 ms per launch).
The reported busy clocks ramped and then settled near 2.55 GHz. Earlier short
trials' roughly 2.42 GHz median therefore should not be treated as a fixed
sustained clock or used to extrapolate a high-performance-mode result. Power
management remained `auto` throughout.

Additional short alternating screens were:

| Experiment | Best candidate TFLOP/s-equivalent | Production control |
| --- | ---: | ---: |
| Uniform vector key-scale loads | 37.134 | 37.570 |
| CU placement descriptor probe | 37.446 | 37.388 |
| Explicit scalar-memory key-scale loads | 36.034 | 37.454 |
| Contiguous 64-token V tiles | 37.584 | 37.533 |
| Packed probabilities for 128-key tiles | 34.614 | 37.912 |
| Tied rescaling with owned accumulator copies | 30.618 | 37.218 |
| Tied rescaling without those copies | 36.370 | 37.406 |

The 257-token, three-head full-output checks passed for CU placement, scalar
memory, tiled V, packed probabilities, and tied rescaling; tied rescaling without
copies passed a 17-token full-output check. Larger runs used the existing sampled
FP32 checks and whole-output finiteness check. None was promoted.

The compiler's temporary allocation-capacity and tied-multiply descriptor probes
were restored. The current production binary is byte-identical after restoration.
All above reports are included in the benchmark JSON. Initial tied-multiply trials
that inadvertently retained the original rescale operation were excluded.

The queued 9/10/11-wave workgroup screen also completed: 34.413 / 35.680 /
34.449 TFLOP/s-equivalent, versus 37.472 for production (16,000 tokens,
two rounds of four launches). Sampled outputs match production exactly and
all outputs are finite. These larger workgroups were not promoted.

An explicit memory-clause probe also failed to improve throughput: production
37.459, batched key-scale loads 37.337, and those loads with a 31-instruction
`s_clause` 37.355 TFLOP/s-equivalent (three rounds, eight launches, N=16,000).
Both candidates passed the 257-token full-output gate and produced the same
large-run samples as production. All three kernels used 192 VGPRs without
spills. The final scale load sits outside the clause to keep its address-register
overwrite outside the grouped instructions; disassembly was inspected before
launch. This was motivated by LLVM's explicit
[hard-clause pass](https://llvm.org/docs/doxygen/SIInsertHardClauses_8cpp_source.html),
but did not show a benefit here. The temporary Loom descriptor patch was
removed, and the probe sources, patch, and disassembly were archived under
`build/attention45/prototypes/`.

Further N=16,000 screens retained all 56 heads and D=128:

| Experiment | Candidate TFLOP/s-equivalent | Production control |
| --- | ---: | ---: |
| Batch K global loads before LDS stores | 37.441 | 37.619 |
| Batch V global loads before LDS stores | 37.519 | 37.619 |
| Batch both K and V loads | 37.363 | 37.619 |
| Source `vector.mma` for QK | 36.020 | 36.718 |
| Source `vector.mma` for PV | 32.179 | 36.718 |
| Source `vector.mma` for both | 31.845 | 36.718 |
| Zero-accumulator instruction for the first QK products | 37.236 | 36.718 |
| Compact LDS rows, six waves, separate 64-bit accesses | 34.724 | 37.409 |
| Compact LDS rows, eight waves, separate 64-bit accesses | 37.062 | 37.409 |
| Interleave two PV output-channel chains, one-load lookahead | 37.312 | 37.363 |
| Same interleave, two-load lookahead | 36.889 | 37.363 |
| Compact LDS, six waves, paired 64-bit instructions | 30.847 | 37.282 |
| Compact LDS, eight waves, paired 64-bit instructions | 34.604 | 37.282 |

These candidates all passed a 257-token, three-head full-output check, and
the large runs passed the existing sampled FP32 reference and whole-output
finiteness checks. The source-MMA/zero-accumulator screen used two rounds of
four launches; the others used three rounds of eight. The zero-accumulator
screen's small relative gain was not promoted: it does not establish a
sustained improvement over the previously validated production result.

Batched staging retained 192 VGPRs and 27,648 bytes of LDS without spills.
Compact rows reduced LDS to 26,112 bytes, retaining 192 VGPRs; both ordinary
64-bit accesses and public `ds_read2_b64`/`ds_write2_b64` helpers were tested.
The smaller allocation was intended to permit more resident workgroups,
but actual residency was not measured. Brief CPU compilation probes overlapped
the ordinary compact-layout screen, so its small differences should not be
interpreted as a precise isolated comparison. No result from that screen is
used as a production performance claim.

PV channel interleaving preserves the order of additions within each output
element while placing independent output accumulators between updates. It
required 192 or 200 VGPRs at the two tested lookaheads and did not improve the
full attention kernel. Four- and eight-channel forms compiled at 216 and 248
VGPRs but were not timed. Sources and generators were archived under
`build/attention45/prototypes/`; raw reports are in the benchmark JSON.

## Independent-operand matrix calibration

A single Loom binary measured **51.101** TFLOP/s-equivalent with constant ones
and **50.708** with independent operand matrices per wave. Each result is the
median of three alternating rounds, with one warmup and two timed launches per
round. Every output matched the exact reference. No competing compute client
was observed by the benchmark monitor; power management remained `auto`.

This is **matrix-only throughput, not attention**. The kernel retains its
operands in registers and alternates 32 INT8 and 32 FP16 WMMA instructions per
loop, across eight accumulator chains. It uses 168 VGPRs, no LDS and no spills.
INT8 operands vary across signed codes; FP16 A0 contains codes 0/1, A1 is its
complement, and B contains positive codes 1 through 7, all divided by eight.
The FP16 data therefore exercise limited mantissas and no negative inputs.
The small measured difference does not establish a roof for arbitrary data
or account for attention's memory traffic, softmax and synchronization.

An earlier signed-FP16 cancellation construction failed its proposed exactness
check, despite exact INT8 outputs. At 32,768 loop iterations its maximum FP16
absolute error was approximately 0.0625; a one-iteration diagnostic already
differed by up to 1.83e-6. Smaller FP16 code ranges also failed. These timings
are excluded from validated varied-data throughput claims. The cause remains
unresolved; these tests do not establish a hardware or compiler defect.
Production precision and accuracy thresholds were unchanged.

The complete positive-input report and rejected diagnostic are retained under
`independent_operand_matrix_calibration` in the benchmark JSON. Reproduction
scripts are archived in `build/attention45/prototypes/`, and generated inputs,
sources, disassembly and outputs remain under `build/attention45/independent-roof*`.

## Global K prefetch at 64 keys

Carrying the next 64-key tile in registers did not improve attention. At
N=16,000, 56 heads and D=128, prefetch before QK measured **36.584** and prefetch
before PV measured **36.391**, versus **37.247** TFLOP/s-equivalent for the paired
production control (three alternating rounds, eight launches each). Both
candidates used 224 VGPRs versus 192 for production, with 27,648 bytes of LDS
and no spills. Their sampled outputs matched production exactly, and both
passed the 257-token, three-head full-output check. The final prefetch clamps
to the last tile to keep speculative reads within the allocation.

A variant carrying all four K packets as one 16-register vector compiled at
256 VGPRs without spills; it was not timed. No prefetch variant was promoted.
Sources and generators were archived under `build/attention45/prototypes/`;
the gate and large-run reports are retained in `k64_global_prefetch_follow_up`
in the benchmark JSON.

## Centered FP16 logit experiment

An experimental softmax path centered each lane's eight-score chunks against
their FP32 maxima, stored the differences temporarily in FP16, then promoted
them before FP32 exponentiation and summation. The chunk-to-global maximum
offset remained FP32. A second variant completed and compressed two QK score
blocks before computing the other two. Neither variant changed production.

Both retained 192 VGPRs without spills. At N=16,000, 56 heads and D=128, the
ordinary and reordered forms measured **35.617** and **35.969**, versus **37.303**
TFLOP/s-equivalent for production (three alternating rounds of eight launches).
Both large samples had relative L2 error 0.000293 and every output was finite.
The ordinary form passed full-output N=257, three-head cases at score scales
0, 1, 32 and 256; the reordered form passed the scale-1 gate and produced the
same large-run samples. The existing 0.002 error limit was unchanged. The extra
rounding did not buy a register reduction or a throughput improvement, so these
variants were archived. Reports are in `centered_fp16_logit_follow_up`.

## Polynomial exponentials

Replacing the weight exponentials with a cubic approximation also lost. For
nonpositive x, the experiment clamps x to -126, takes n=trunc(x) and r=x-n,
and evaluates `2^n * (1 + r*(a + r*(b + r*c)))`. The coefficients are
approximately 0.69151765, 0.23102762 and 0.03950999. A CPU sample of 100,001
points on [-1,0] measured maximum relative error 0.000147; this is a sampled
numerical check, not a formal error bound. Old-output rescaling retains native
exp2. Values below -126 receive a tiny nonzero weight instead of underflowing
to zero, so this is an approximate arithmetic experiment.

Replacing all 32 weight exponentials measured **34.576** TFLOP/s-equivalent;
replacing half measured **35.694**, versus **37.247** for production. These
are N=16,000, 56-head, D=128 results from three alternating rounds of eight
launches. Both candidates used 200 VGPRs without spills. Both passed full-output
N=257, three-head checks at score scales 0, 1, 32 and 256. Large relative L2
errors were 0.000298 and 0.000299, with finite outputs and the unchanged 0.002
limit. Extra arithmetic and registers did not yield a throughput benefit.
Neither variant was promoted; raw reports are in `polynomial_exp2_follow_up`.


## Zero-accumulator confirmation

The four initial QK WMMA instructions were switched to the zero-accumulator
encoding, then compared with production in three alternating rounds of 32
measured launches per round. The early screening gain did not persist:

| Tokens | Production median | Zero-accumulator median |
| ---: | ---: | ---: |
| 16,000 | 196.952 ms | 197.384 ms |
| 37,723 | 1,120.040 ms | 1,125.719 ms |

Both kernels passed the 257-token, three-head full-output gate, and the large
checks produced identical sampled output hashes. The candidate was not promoted.
Power management remained `auto`; these longer confirmation runs put current
production throughput around 37.27 and 36.43 TFLOP/s-equivalent, respectively.
Raw reports are retained under `zero_accumulator_confirmation` in the benchmark
JSON and `build/attention45/zero-confirm/`.


## Current latency breakdown: measurement limits

The latest sustained production attention medians are **196.952 ms at 16,000
tokens** and **1,120.040 ms at 37,723 tokens**, with 56 heads and dimension 128.
The separately validated down-projection GEMM at 37,723 tokens takes
**140.542 ms**, including its residual epilogue. These are individual kernel
times, not a complete generation profile. No fresh timings for all other GEMMs,
preparation, encoder/decoder, transfers, or host work were collected here.
Fifty calls at the larger shape imply about 56.0 seconds of attention and
7.0 seconds of down projection per model evaluation, before other work.

An experimental Loom kernel now records five regions using same-wave shader
cycle deltas: K/V staging and its barrier; QK matrix work and LDS reads; key
scales, softmax and probability packing; output rescaling, PV matrix work and
LDS reads; and the final barrier. It retains 192 VGPRs, 27,648 bytes of LDS, and
zero spills. The 257-token, three-head full output is bit-identical to the
validated control. The descriptor lives only in an isolated compiler checkout.

Repeated competing GPU jobs prevented a clean large-shape comparison. The
waiting benchmark was stopped; instrumentation overhead and useful large-shape
phase percentages remain unmeasured. One uncontested short control launch batch
returned 203.872 ms at 16,000 tokens; this incomplete experiment does not
supersede the longer alternating production medians. Cycle regions mark issue
boundaries and include waits; they do not directly measure isolated arithmetic
costs or additive portions of global GPU wall time.

Reaching 45 TFLOP/s-equivalent at 37,723 tokens requires 906.688 ms, a further
19.05% latency reduction from the sustained baseline. That remains an unachieved
target. Profiling artifacts and scripts are under `build/attention45/phase-profile/`
and `build/attention45/prototypes/attention45-{phase-profile,run-phase}.py`.

The isolated compiler also fixes mandatory tied-concat alias reservations
outgrowing a 16-entry list. An authored regression fails on the original
compiler and passes with the fix; all 17 allocation fixture cases and six
allocation unit cases pass. The production attention binary is identical under
both compilers. Two centered K128 variants now compile without spills but have
not been GPU validated and are not production candidates yet. Separate allocator
and experimental clock patches are retained in `build/attention45/allocator-fix/`.
