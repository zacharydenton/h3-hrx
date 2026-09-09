# Native test coverage

`scripts/test.sh --cpu` runs formatting, Clippy, workspace tests and C-header drift checks.
`scripts/test.sh --gpu` additionally runs `h3/tests/kernels.rs` and the resident
Euler and conditioning unit tests on gfx1151.
The native tests compile through `hrx::loom::Compiler`, upload owned buffers,
dispatch through `hrx::Stream`, and read results back through HRX staging. They
are ignored by default; explicitly running them requires working hardware and
the provisioned native bundle. No Python or Torch dependency is involved.

The GPU suite compares independent scalar CPU references against:

- GroupNorm and SiLU at zero, small and normal variance.
- Four experimental 32-key attention layouts with the softmax maximum in the upper key tile.
- The attention stems the host actually selects — `mha`, `mha8`, `mha64`, `mha648`, `mha64t32` and
  the text encoder's causal `gqa8c` — against scaled dot-product attention in f64.
- `prepare_qk_i8`: the Hadamard rotation, int8 quantisation, packing and scales that the int8
  attention consumes, and `attention_i8qk_mha` against the attention those operands define.
- Scalar and four-output audio convolutions at eleven boundary lengths, with
  padding, dilation, residual accumulation and untouched output guards.
- Float32 matrix multiplication past the former 32,768-row limit.
- BF16 vision matmul with bias, residual scaling, tanh GELU and erf GELU.
- RMSNorm, LayerNorm and plain FP16/BF16 preparation at three widths, including
  large inputs, per-row classes and padded output strides.
- All three rotary Q/K normalization layouts, including grouped heads and exact V copying.
- INT4/INT8 GEMM families, both tile sizes, bias, residual classes, SwiGLU ordering
  and padded input strides on kernels that support them.
- The f16 and BF16 GEMM families across all three modes and both epilogues.

Workspace unit tests cover model shapes, checkpoint layouts, CPU sampling,
tokenization, compiler/source behavior, dispatch bounds, and C ABI contracts.
The C smoke test in the shared HRX repository builds against generated H3 and
Krea headers and loads both model libraries in one process. Rustler remains an
application adapter under `examples/rustler`.

Loom sources are maintained directly. Python generators, reference model
implementations, wrappers and one-off studies are retired. Historical reports
remain historical; they are not automatically revalidated by this suite.
Whole-model parity lives in `scripts/parity.py`, outside this suite and outside
`scripts/test.sh`: it needs the checkpoints, a device and dumps produced inside the
ComfyUI container by `scripts/comfy_dump.py`, and it takes about eleven minutes.
Run `python3 scripts/parity.py gate --require` before a release. This suite
establishes the numerical cases above; that script establishes full-model parity.

Still uncovered here, in rough order of how much they matter: the int4 and
head-major attention variants (`attention_i4qk*`, `attention_i8qkhm_mha8_k64`,
`attention_mha64hm32`) with their `prepare_qk_i4` and `prepare_qk_i8hm` operands;
the wide and fast decoder GEMMs; the fused decoder QKV GEMM; `norm_mod_f32`; and
the smaller shape kernels (`layernorm_f32`, `layernorm_f16_f32`, `transpose_f16`,
`transpose_f32`, `gn_stats_f16`, `prepare_plain16_i8`, `conv1d_s_f32`,
`rope2d_qkv_f16`, `matmul_bias_f16_wmma_af16_cf16`). The resident Euler sampler is checked bit-for-bit
against the CPU as a unit test, and its five reference denoise cases were compared
byte-for-byte against the host sampler it replaced; neither check is in a committed
harness, so re-run the comparison by hand when that path changes.
