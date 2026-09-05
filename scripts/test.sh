#!/usr/bin/env bash
# The one test command. Fills in as the pieces land: format check, generated kernels
# against their generators, host build, reference vs diffusers, kernel tests, blocks vs
# the fixture.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/env.sh
status=0
step() { local name="$1"; shift; printf '\n=== %s ===\n' "$name"; if "$@"; then printf '  ok\n'; else printf '  FAILED: %s\n' "$name"; status=1; fi; }
if ls kernels/*.loom >/dev/null 2>&1; then step "loom sources are canonically formatted" bash -c '"$LOOM_FORMAT" --check kernels/*.loom'; fi
# ComfyUI imports only inside its image (its runtime packages are not in the shared venv); CPU, toy size
step "reference vs ComfyUI's MiniMaxH3Model (toy)" bash -c 'podman run --rm -v "$HOME:$HOME" -w "$PWD" -e PYTHONPATH=/opt/ComfyUI --entrypoint /opt/venv/bin/python docker.io/kyuz0/amd-strix-halo-comfyui:latest tests/test_ref_vs_comfy.py 2>&1 | grep -E "PASS|FAIL|Error" | tee /dev/stderr | grep -q FAIL && exit 1 || exit 0'
step "generated kernels match their generators" bash -c '
  tmpdir=$(mktemp -d); trap "rm -rf $tmpdir" EXIT
  cp -r kernels "$tmpdir/kernels" && cp -r tools "$tmpdir/tools" && cd "$tmpdir" &&
  python3 tools/gen_prepare.py >/dev/null && python3 tools/gen_attention_lds.py >/dev/null && ATTN_WAVES=8 python3 tools/gen_attention_lds.py >/dev/null && ATTN_GQA=8 ATTN_WAVES=8 ATTN_CAUSAL=1 python3 tools/gen_attention_lds.py >/dev/null && python3 tools/gen_gemm.py >/dev/null && python3 tools/gen_rope.py >/dev/null &&
  for f in prepare_norm_i4 prepare_plain_i4 prepare_norm_i8 prepare_plain_i8 prepare_plain16_i8 attention_mha_lds_f16_wmma attention_mha8_lds_f16_wmma attention_gqa8c_lds_f16_wmma gemm_i4_256 gemm_i4_resid_256 gemm_i4_swiglu_256 gemm_i8_256b gemm_i8_resid_256b gemm_i8_swiglu_256b_gs gemm_i8_256 gemm_i8_resid_256 gemm_i8_swiglu_256 rope_qknorm_f16 rope64_qknorm_f16 rope128_qknorm_f16; do "$LOOM_FORMAT" --in-place "kernels/$f.loom" >/dev/null && cmp -s "kernels/$f.loom" "$OLDPWD/kernels/$f.loom" || { echo "  $f differs"; exit 1; }; done'
step "build host"        ./scripts/build_host.sh
step "prepare kernels"   bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_prepare.py'
step "qk norm + rope"    bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_rope_qknorm.py'
step "qk norm + rope (text encoder: 128 channels, 8 kv heads)" bash -c 'env -u LD_LIBRARY_PATH ROPE_D=128 ROPE_R=128 ROPE_KV=8 python3 tests/test_rope_qknorm.py'
step "attention"         bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_attention.py'
step "attention (text encoder: causal, 8 query heads per kv head)" bash -c 'env -u LD_LIBRARY_PATH ATTN_GQA=8 python3 tests/test_attention.py'
step "gemm family (M=512)" bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_gemm.py 512'
step "gemm int8 family (text encoder, M=300)" bash -c 'env -u LD_LIBRARY_PATH python3 tests/test_gemm.py 300 i8'
if [ "${1:-}" != "--quick" ]; then
  step "native blocks vs reference (fixture)" bash -c 'source ~/code/krea2-loom/.venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_blocks.py --curve 1,50'
  step "text encoder layers vs transformers" bash -c 'source ~/code/krea2-loom/.venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_te_blocks.py --curve 1,50'
  step "video VAE decoder blocks vs diffusers (W8A8)" bash -c 'source ~/code/krea2-loom/.venv/bin/activate && env -u LD_LIBRARY_PATH python3 tests/test_vae_blocks.py --curve 36 --bits 8'
fi
printf '\n'; [ "$status" = 0 ] && printf 'all checks passed\n' || printf 'SOME CHECKS FAILED\n'
exit $status
