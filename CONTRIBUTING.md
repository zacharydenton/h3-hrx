# Contributing

Use [setup](docs/setup.md) to build the host and the
[pinned Loom toolchain](patches/loom/README.md) for kernel work. Include a
description of the changed behavior and the checks you ran with a contribution.
For performance changes, record the shape, precision, hardware, toolchain,
timing boundary, and numerical comparison alongside the result.

## Repository layout

| Directory | Purpose |
| --- | --- |
| `host/` | C++ pipeline, checkpoint loader, tokenizer, HIP/HRX runtime bindings |
| `cli/` | the `h3` command in Rust, and the worked example of the C API |
| `kernels/` | Runtime Loom sources, mostly emitted by `tools/gen_*.py` |
| `tools/` | Generators, Python client, benchmarks, and ComfyUI comparison tools |
| `tests/` | CPU host regressions, kernel numerical checks, and pipeline comparisons |
| `reference/` | PyTorch/diffusers numerical oracles; unused during normal inference |
| `experiments/` | Kernel alternatives, including templates and GPU regression inputs |
| `examples/` | C, Rust, Go, and Python clients |
| `assets/` | Tokenizer data embedded into the library |
| `docs/archive/` | Historical measurements and tuning logs |

Edit a generated kernel's generator and regenerate the source together.
The CPU suite checks the canonical generated sources it covers. Do not delete
`experiments/` wholesale: production head-major attention generation reads
`attention_i8qkt32_mha8_lds_f16_wmma.loom`, and the kernel regressions exercise
several experimental 32-key attention variants.

## Tests

```sh
bash scripts/test.sh --cpu
bash scripts/test.sh --quick
bash scripts/test.sh
```

- `--cpu`: Python syntax, host regressions (including ASan/UBSan), generated
  source checks, and kernel compilation. No GPU, weights, or containers.
  Loom-dependent checks are explicitly skipped when its tools are absent.
- `--quick`: adds the host build, GPU kernel checks, tokenizer comparison,
  and a toy ComfyUI comparison when its container runtime is available.
- Full suite: adds model and pipeline comparisons; some checks require saved
  reference data and report a skip if it is absent.

Set `H3_PYTHON` to a Python 3 interpreter with NumPy. Set
`H3_REFERENCE_PYTHON` to an environment with ROCm PyTorch, diffusers, and
transformers for GPU/reference checks; it defaults to `H3_PYTHON`.
`bash scripts/test_host.sh` runs the CPU host checks alone.

The ComfyUI harness uses podman and
`docker.io/kyuz0/amd-strix-halo-comfyui:latest`. `H3_REQUIRE_COMFY=1` makes
the toy comparison mandatory. Reference tools run inside that image with the
checkout and model directory mounted; their module docstrings describe their
inputs. In particular:

- [`comfy_clip.py`](tools/comfy_clip.py) produces block and trajectory dumps
  for [`test_comfy_parity.py`](tests/test_comfy_parity.py).
- [`ref_truth_comfy.py`](tools/ref_truth_comfy.py) produces encoder and
  reference-conditioning fixtures under `build/ref_truth`.
- `scripts/download.sh [DIR] [--all]` fetches original MiniMax files used by
  reference tools; `--all` also downloads the large bf16 transformer and
  text encoder. Production uses the separate ComfyUI-format checkpoints.

For a release numerical check, require the dumps instead of accepting a skip:

```sh
python3 tests/test_comfy_parity.py --require
```

Use the reference interpreter for this command. Passing the CPU suite alone
does not establish GPU correctness or model parity.

## Benchmarks

[`docs/performance.md`](docs/performance.md) summarizes recorded results.
`tools/bench_attention_i8.py`, `tools/bench_gemm_i8_tuning.py`,
`tools/bench_vae_gemm.py`, and `tools/bench_vae_decode.py` contain the focused
drivers; consult their `--help` for inputs. Keep logs, binaries, checkpoints,
and generated clips under ignored `build/`, and commit compact measurements
only when they support a documented change.
