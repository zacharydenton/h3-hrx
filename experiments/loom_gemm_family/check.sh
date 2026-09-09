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
        --backend=amdgpu-hal --target=gfx1151 \
        --config="probe.registers=$registers" --config="probe.epilogue=$epilogue" \
        --output="$stem.hal" --emit-target-artifact="$stem.hsaco" \
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

# The public config contracts must reject unsupported family members.
for invalid in probe.registers=3 probe.epilogue=2; do
  case "$invalid" in
    probe.registers=*) config=(--config="$invalid" --config=probe.epilogue=0) ;;
    *) config=(--config=probe.registers=2 --config="$invalid") ;;
  esac
  if "$compiler" experiments/loom_gemm_family/packed.loom \
      --backend=amdgpu-hal --target=gfx1151 "${config[@]}" \
      --output="$out/invalid.hal" > "$out/$invalid.log" 2>&1; then
    printf 'Unexpected success: %s\n' "$invalid" >&2
    exit 1
  fi
  if ! awk '/violates constraint/ { found = 1 } END { exit !found }' "$out/$invalid.log"; then
    cat "$out/$invalid.log" >&2
    exit 1
  fi
  printf 'PASS rejected %s (see %s/%s.log)\n' "$invalid" "$out" "$invalid"
done
