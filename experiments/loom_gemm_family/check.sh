#!/usr/bin/env bash
# Compile and inspect the motif. Never opens a GPU or launches a kernel.
set -euo pipefail
cd "$(dirname "$0")/../.."
compiler=${LOOM_COMPILE:-loom-compile}
objdump=${LLVM_OBJDUMP:-llvm-objdump}
out=build/loom-family-probe/check
mkdir -p "$out"

for registers in 2 4; do
  bits=$((registers * 2))
  for epilogue in 0 1; do
    stem="$out/i${bits}-epilogue${epilogue}"
    if ! "$compiler" experiments/loom_gemm_family/packed.loom \
        --product=kernel --format=amdgpu-hsaco --target=amdgpu:gfx1151 \
        --config="probe.registers=$registers" --config="probe.epilogue=$epilogue" \
        --output="$stem.hsaco" \
        --dump-ir-after-all --dump-ir-output="$stem-ir/" > "$stem.log" 2>&1; then
      cat "$stem.log" >&2
      exit 1
    fi
    "$objdump" -d --mcpu=gfx1151 "$stem.hsaco" > "$stem.asm"
    awk -v opcode="v_wmma_i32_16x16x16_iu$bits" \
        -v expected_mma="$((16 / registers))" -v expected_add="$((8 * epilogue))" '
      $1 == opcode { mma++ }
      $1 ~ /^v_wmma/ && $1 != opcode { bad++ }
      $1 ~ /^v_add_nc_u32/ { add++ }
      $1 ~ /^s_(cbranch|branch|call|swappc|setpc)/ { bad++ }
      END {
        if (mma != expected_mma || add != expected_add || bad) {
          printf "Unexpected code: WMMA=%d, adds=%d, forbidden=%d\n", mma, add, bad
          exit 1
        }
      }' "$stem.asm"
    printf 'PASS i%s epilogue=%s: %s WMMAs, %s epilogue adds, no branches/calls\n' \
      "$bits" "$epilogue" "$((16 / registers))" "$((8 * epilogue))"
  done
done

# Exact encoding configs and dependent vector arguments compile end to end.
for registers in 2 4; do
  bits=$((registers * 2))
  stem="$out/config-i$bits"
  "$compiler" experiments/loom_gemm_family/encoding_config.loom \
    --product=kernel --format=amdgpu-hsaco --target=amdgpu:gfx1151 \
    --config="probe.registers=$registers" \
    --config="probe.schema=#encoding.operand<element_format=i$bits, payload_elements=16, payload_registers=$registers>" \
    --output="$stem.hsaco"
  "$objdump" -d --mcpu=gfx1151 "$stem.hsaco" > "$stem.asm"
  awk -v opcode="v_wmma_i32_16x16x16_iu$bits" '
    $1 == opcode { mma++ }
    $1 ~ /^v_wmma/ && $1 != opcode { bad++ }
    END { exit mma != 1 || bad }' "$stem.asm"
  "$compiler" experiments/loom_gemm_family/dynamic_argument.loom \
    --product=kernel --format=amdgpu-hsaco --target=amdgpu:gfx1151 --config="probe.registers=$registers" \
    --output="$out/argument-$registers.hsaco"
  printf 'PASS i%s encoding config and vector<%sxi32> template argument\n' "$bits" "$registers"
done

# The public config contracts must reject unsupported family members.
for invalid in probe.registers=3 probe.epilogue=2; do
  case "$invalid" in
    probe.registers=*) config=(--config="$invalid" --config=probe.epilogue=0) ;;
    *) config=(--config=probe.registers=2 --config="$invalid") ;;
  esac
  if "$compiler" experiments/loom_gemm_family/packed.loom \
      --product=kernel --format=amdgpu-hsaco --target=amdgpu:gfx1151 "${config[@]}" \
      --output="$out/invalid.hsaco" > "$out/$invalid.log" 2>&1; then
    printf 'Unexpected success: %s\n' "$invalid" >&2
    exit 1
  fi
  if ! awk '/violates constraint/ { found = 1 } END { exit !found }' "$out/$invalid.log"; then
    cat "$out/$invalid.log" >&2
    exit 1
  fi
  printf 'PASS rejected %s (see %s/%s.log)\n' "$invalid" "$out" "$invalid"
done
