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
- Four attention layouts with the softmax maximum in the upper key tile.
- Scalar and four-output audio convolutions at eleven boundary lengths, with
  padding, dilation, residual accumulation and untouched output guards.
- Float32 matrix multiplication past the former 32,768-row limit.
- BF16 vision matmul with bias, residual scaling, tanh GELU and erf GELU.
- RMSNorm, LayerNorm and plain FP16/BF16 preparation at three widths, including
  large inputs, per-row classes and padded output strides.
- All three rotary Q/K normalization layouts, including grouped heads and exact V copying.
- INT4/INT8 GEMM families, both tile sizes, bias, residual classes, SwiGLU ordering
  and padded input strides on kernels that support them.

Workspace unit tests cover model shapes, checkpoint layouts, CPU sampling,
tokenization, compiler/source behavior, dispatch bounds, and C ABI contracts.
The C smoke test in the shared HRX repository builds against generated H3 and
Krea headers and loads both model libraries in one process. Rustler remains an
application adapter under `examples/rustler`.

Loom sources are maintained directly. Python generators, reference model
implementations, wrappers and one-off studies are retired. Historical reports
remain historical; they are not automatically revalidated by this suite.
Whole-model ComfyUI parity, encoder reference dumps and long video quality tests
need versioned golden outputs before equivalent independent Rust tests can be
claimed. This suite establishes the numerical cases above, not full-model parity.
