#!/usr/bin/env bash
# The one test command, in three tiers:
#   bash scripts/test.sh --cpu     no GPU, weights or containers: Python parses, CPU host regressions, generated kernels vs their generators
#   bash scripts/test.sh --quick   plus the host build, the kernel tests on the GPU, the tokenizer, the ComfyUI toy comparison
#   bash scripts/test.sh           plus the reference tier: decoder, encoder, block, text encoder, pipeline and ComfyUI parity checks
# Python: H3_PYTHON (default python3, NumPy) runs the CPU tier; the GPU kernel tests (torch float64 references) and the
# reference tier (torch, diffusers, transformers) take H3_REFERENCE_PYTHON (default: H3_PYTHON). The ComfyUI toy comparison
# needs podman and its image; it is skipped when podman is absent unless H3_REQUIRE_COMFY=1.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
tier=full; case "${1:-}" in --cpu) tier=cpu;; --quick) tier=quick;; "") ;; *) echo "usage: scripts/test.sh [--cpu | --quick]" >&2; exit 64;; esac
PY="${H3_PYTHON:-python3}"; REF_PY="${H3_REFERENCE_PYTHON:-$PY}"
status=0
step() { local name="$1"; shift; printf '\n=== %s ===\n' "$name"; if "$@"; then printf '  ok\n'; else printf '  FAILED: %s\n' "$name"; status=1; fi; }
skip() { printf '\n=== %s ===\n  skipped: %s\n' "$1" "$2"; }
gpu() { env -u LD_LIBRARY_PATH "$REF_PY" "$@"; }    # the kernel tests drive the GPU through the Python harness (tools/kernel_test.py) against torch references
ref() { env -u LD_LIBRARY_PATH "$REF_PY" "$@"; }
step "tracked Python parses" bash -c 'git ls-files "*.py" | xargs "$0" -m py_compile' "$PY"
if [ -x "${LOOM_FORMAT:-}" ] && ls kernels/*.loom >/dev/null 2>&1; then step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom'; else skip "loom sources are canonically formatted" "no loom-format (scripts/env.sh)"; fi
if [ -x "${LOOM_FORMAT:-}" ]; then
step "generated kernels match their generators" bash -c '
  tmpdir=$(mktemp -d); trap "rm -rf $tmpdir" EXIT
  cp -r kernels "$tmpdir/kernels" && cp -r tools "$tmpdir/tools" && cp -r experiments "$tmpdir/experiments" && cp -r scripts "$tmpdir/scripts" && cd "$tmpdir" &&
  "$0" tools/gen_prepare.py >/dev/null && "$0" tools/gen_attention_lds.py >/dev/null && ATTN_WAVES=8 "$0" tools/gen_attention_lds.py >/dev/null && ATTN_GQA=8 ATTN_WAVES=8 ATTN_CAUSAL=1 "$0" tools/gen_attention_lds.py >/dev/null && "$0" tools/gen_gemm.py >/dev/null && "$0" tools/gen_rope.py >/dev/null && sh scripts/gen_attention_i4.sh >/dev/null &&
  "$0" tools/gen_attention_i8_head_major.py >/dev/null && "$0" tools/gen_attention_i8_head_major_64.py >/dev/null && "$0" tools/gen_gemm_f16.py >/dev/null && "$0" tools/gen_matmul_bf16.py >/dev/null &&
  for f in attention_i4qk_mha8_lds_f16_wmma attention_i4qk_mha_lds_f16_wmma attention_i4qkl_mha8_lds_f16_wmma attention_i4qks_mha8_lds_f16_wmma attention_i4qks_mha_lds_f16_wmma attention_i4qksl_mha8_lds_f16_wmma prepare_norm_i4 prepare_plain_i4 prepare_norm_i8 prepare_plain_i8 prepare_plain16_i8 prepare_lnorm_i8 prepare_norm_f16 prepare_lnorm_f16 prepare_plain_f16 prepare_norm_bf16 prepare_lnorm_bf16 prepare_plain_bf16 attention_mha_lds_f16_wmma attention_mha8_lds_f16_wmma attention_gqa8c_lds_f16_wmma gemm_i4_256 gemm_i4_resid_256 gemm_i4_swiglu_256 gemm_i8_256b gemm_i8_resid_256b gemm_i8_swiglu_256b_gs gemm_i8_256 gemm_i8_resid_256 gemm_i8_swiglu_256 gemm_f16_256 gemm_f16_256b gemm_f16_resid_256 gemm_f16_resid_256b gemm_f16_swiglu_256 gemm_f16_swiglu_256b_gs gemm_bf16_256 gemm_bf16_256b gemm_bf16_resid_256 gemm_bf16_resid_256b gemm_bf16_swiglu_256 gemm_bf16_swiglu_256b_gs matmul_bias_bf16_wmma matmul_resid_bf16_wmma matmul_gelu_bf16_wmma matmul_gelu_erf_bf16_wmma rope_qknorm_f16 rope64_qknorm_f16 rope128_qknorm_f16 attention_i8qkhm_mha8_lds_f16_wmma attention_i8qkhm_mha8_k64_lds_f16_wmma prepare_qk_i8hm; do "$LOOM_FORMAT" --in-place "kernels/$f.loom" >/dev/null && cmp -s "kernels/$f.loom" "$OLDPWD/kernels/$f.loom" || { echo "  $f differs"; exit 1; }; done' "$PY"
else skip "generated kernels match their generators" "no loom-format (scripts/env.sh)"; fi
step "CPU host regressions" env H3_PYTHON="$PY" bash scripts/test_host.sh
if [ "$tier" = cpu ]; then printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'; exit $status; fi

# ComfyUI imports only inside its image (its runtime packages are not in the shared venv); CPU, toy size
if command -v podman >/dev/null 2>&1 || [ "${H3_REQUIRE_COMFY:-}" = 1 ]; then
step "reference vs ComfyUI's MiniMaxH3Model (toy)" podman run --rm -v "$HOME:$HOME" -w "$PWD" -e PYTHONPATH=/opt/ComfyUI --entrypoint /opt/venv/bin/python docker.io/kyuz0/amd-strix-halo-comfyui:latest tests/test_ref_vs_comfy.py
else skip "reference vs ComfyUI's MiniMaxH3Model (toy)" "no podman (H3_REQUIRE_COMFY=1 makes this a failure)"; fi
step "build host"        ./scripts/build_host.sh
step "small kernel regressions" gpu tests/test_kernel_regressions.py
step "prepare kernels"   gpu tests/test_prepare.py
step "prepare Q/K, int8 head-major (the production attention operands)" gpu tests/test_prepare_qk_i8_head_major.py
step "qk norm + rope"    gpu tests/test_rope_qknorm.py
step "qk norm + rope (text encoder: 128 channels, 8 kv heads)" env -u LD_LIBRARY_PATH ROPE_D=128 ROPE_R=128 ROPE_KV=8 "$REF_PY" tests/test_rope_qknorm.py
step "attention"         gpu tests/test_attention.py
step "attention (text encoder: causal, 8 query heads per kv head)" env -u LD_LIBRARY_PATH ATTN_GQA=8 "$REF_PY" tests/test_attention.py
step "attention, int8 QK^T head-major (the production long-sequence kernel)" gpu tests/test_attention_i8_head_major.py
step "gemm family (M=512)" gpu tests/test_gemm.py 512
step "gemm int8 family (text encoder, M=300)" gpu tests/test_gemm.py 300 i8
step "gemm f16 family (the video VAE decoder, M=512)" gpu tests/test_gemm_f16.py 512
step "gemm bf16 family (the token refiner, M=512)" env -u LD_LIBRARY_PATH GEMM_ELEM=bf16 "$REF_PY" tests/test_gemm_f16.py 512
step "prepare kernels, f16 and bf16 unrotated" gpu tests/test_prepare_float.py
step "vision matmuls, bf16; matmul_f32 past 32768 rows" gpu tests/test_matmul_bf16.py
step "C tokenizer vs transformers" ref tests/test_tokenizer.py
if [ "$tier" = full ]; then
  # the reference tier: model weights, exports and the reference dumps (README, Weights and Tests)
  step "DiT blocks vs the reference (the checkpoint's int8 rows)" ref tests/test_stack_parity.py --stack dit
  step "text encoder layers vs transformers" ref tests/test_stack_parity.py --stack te
  step "C pipeline (libh3pipe) vs the Python stages" ref tests/test_pipe.py --decode --audio
  if [ -d build/ref_truth ]; then
    step "audio encoder vs ComfyUI" ref tests/test_audio_encoder.py
    step "vision tower vs ComfyUI" ref tests/test_vision.py
    step "video VAE encoder vs ComfyUI" ref tests/test_vae_encoder.py
    if [ -f "${H3_MODELS:-$HOME/comfy-models}/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors" ]; then
    step "ref2va step vs ComfyUI (the ref2va checkpoint)" ref tests/test_ref2va.py
    else skip "ref2va step vs ComfyUI" "no ref2va checkpoint (README, Weights)"; fi
  else skip "encoder and ref2va checks vs ComfyUI" "no build/ref_truth (tools/ref_truth_comfy.py)"; fi
  step "C host vs ComfyUI's own run (int8 rows + f16 attention)" ref tests/test_comfy_parity.py
fi
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
