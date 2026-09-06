# INT8 attention above 30 TFLOP/s in Loom

On the Radeon 8060S (`gfx1151`), the Loom-compiled head-major kernel exceeds
30 TFLOP/s-equivalent on both tested sequence lengths. Both matrix products,
softmax, and output normalization execute in one Loom kernel. No HIP-compiled
GPU code is used by this path.

| Kernel | Tokens | Heads × dimension | Median kernel time | TFLOP/s-equivalent |
| --- | ---: | ---: | ---: | ---: |
| Head-major Loom | 16,000 | 56 × 128 | 204.441 ms | **35.903** |
| Head-major Loom | 37,723 | 56 × 128 | 1,178.682 ms | **34.616** |
| Previous Loom K-prefetch default | 37,723 | 56 × 128 | 1,785.599 ms | 22.850 |

The production-size kernel is **1.515× faster** than the previous default in
these measurements. The metric is `4 * tokens² * heads * dimension / seconds`:
INT8 QK with INT32 accumulation, FP32 online softmax and output accumulation,
and FP16 P/V multiplication. Timing excludes operand preparation and transfers.
This is a kernel result; full model rendering was not benchmarked in this run.

Three rounds used four measured launches per round at 16,000 tokens and three
at 37,723, after a warmup launch. The previous default used two rounds of three
launches. Every round waited for five idle GPU readings and less than 20%
aggregate CPU activity. No clocks or power limits were changed. The complete
rounds, environment readings, source/binary hashes, and compiler identity are in
[attention-int8-benchmarks.json](attention-int8-benchmarks.json).
The compiler came from the shared `hrx-system` working tree; its uncommitted
paths are recorded there as well.

The largest improvement came from storing Q, K, and their scales by head:
`[heads][capacity][128]` INT8 codes and `[heads][capacity]` FP32 scales.
The same 32-key algorithm measured approximately 28–29 TFLOP/s with the original
layout and 35–36 after this change. V already used a transposed layout.
The preparation kernel writes the new layout directly, so the host needs no
additional buffers or transpose launch for Q/K.

Each workgroup has eight wave32 query tiles and shares 32 keys. Computing
`K Qᵀ` keeps each query's softmax reduction within a lane pair. Packing
probabilities as half pairs avoids an LDS round trip. Small public `low.invoke`
helpers specify probability packing and the order of QK/PV matrix operations;
addressing, loops, memory operations, and softmax remain ordinary Loom source.
The final specializations use 216/224 VGPRs respectively, 29,696 bytes of LDS,
and no spills.
Accumulator copies in the helpers preserve value semantics when short loops
specialize to shared zero constants. Inactive query waves do not write outputs.

The C host selects `attention_i8qkhm_mha8_lds_f16_wmma` and
`prepare_qk_i8hm` for INT8 attention at 4,096 or more tokens. Select INT8 as usual
with `H3_ATTN_QK=i8`. Smaller sequences retain the existing four-wave path.
The INT4 and FP16 choices retain their existing kernels.

Validation passed:

- All outputs against a dense FP32 oracle on identical quantized operands at
  1, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, and 1,001 tokens.
- Uniform and sharp softmax cases at 257 tokens (score multipliers 0, 32, 256).
  All 17 cases were deterministic; relative L2 error was at most 0.030%.
- Sampled large-shape error was 0.030% at 16,000 and 0.032% at 37,723 tokens;
  every output was finite. This measures kernel arithmetic error against the
  quantized-input oracle, separately from INT8 quantization error itself.
- Preparation produced bit-identical codes/scales to the existing quantizer,
  including nonzero input offsets, 3/9/56 heads, and zero padded capacity.
- The CPU host regression suite and both HIP-runtime and HRX host builds passed.

Reproduce with the project's NumPy environment:

```bash
python3 tools/gen_attention_i8_head_major.py
python3 tests/test_prepare_qk_i8_head_major.py
python3 tests/test_attention_i8_head_major.py
OPENBLAS_NUM_THREADS=4 python3 tools/bench_attention_i8.py 16000 \
  attention_i8qkhm_mha8_lds_f16_wmma --head-major \
  --rounds 3 --repeat 4 --wait-idle --output build/attention30/loom-final
OPENBLAS_NUM_THREADS=4 python3 tools/bench_attention_i8.py 37723 \
  attention_i8qkhm_mha8_lds_f16_wmma --head-major \
  --rounds 3 --repeat 3 --wait-idle --output build/attention30/loom-final
```

The generator uses the checked-in 32-key transposed template under
`experiments/`. Intermediate scheduling variants and their measurements remain
locally under `build/attention30/prototypes` and `build/attention30`.
