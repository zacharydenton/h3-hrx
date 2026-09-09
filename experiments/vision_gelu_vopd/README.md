# Vision GELU: illegal dual-FMA register pairing

The rejected vision family exposes a bug in Loom's native AMDGPU VOPD planner.
The template preserves the GELU arithmetic. Its different register allocation
causes the planner to combine two FMAs whose addends use the same SRC2 bank.

## Minimal CPU reproducer

```sh
loom-check experiments/vision_gelu_vopd/bank_conflict.loom-test
```

With the tested compiler, the conflicting case fails and the legal control
passes. This is an expected-failing compiler regression fixture, outside the
model's test suite. It needs no GPU, model weights, or generated sources.

The failing case pins the addends to `v0` and `v2`. Loom emits:

```asm
v_dual_fmamk_f32 v4, v1, 0x3d372713, v0 :: v_dual_fmamk_f32 v5, v3, 0x3d372713, v2
```

Put that instruction in a `.s` file and run:

```sh
llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1151 -filetype=obj pair.s -o pair.o
```

LLVM 22.1.8 rejects it with `src2 operands must use different VGPR banks`.
The control changes the addends to opposite-parity registers and retains the
legal dual instruction.

## Cause and fix location

In the sibling Loom checkout, `loom_amdgpu_vopd_register_constraint_flags` in
`src/loom/target/arch/amdgpu/planning/vopd_plan.c:994` checks every encoded
`vsrc1` using mask **3**. For `FMAMK`, that field contains the addend routed
through **SRC2**, which uses mask **1**. Registers 0 and 2 pass the current
modulo-four check but conflict in the two-bank SRC2 cache. LLVM independently
records the operand bank masks as `{1, 3, 3, 1}` for destination and SRC0/1/2.
See [LLVM's operand-bank definitions](https://github.com/llvm/llvm-project/blob/llvmorg-22.1.8/llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.h#L634-L637).

The fix belongs in the planner's operand-to-source-cache mapping, with checks
for each actual cache used by both components. Cover mixed instruction forms
as well as two FMAMKs; changing every `vsrc1` check to mask 1 would incorrectly
restrict ordinary two-source operations. Keep legal pairing enabled. Changing
the GELU formula, its tolerance, or source ordering would only hide this trigger.

## GPU evidence

On gfx1151, the rejected family was compared with the production BF16 kernels
at M=65, K=128, N=64, using the deterministic inputs from
`vision_bf16_matmuls_match_rounded_operands_and_epilogues`:

| Candidate | Differing f32 outputs / 4,160 | Maximum absolute difference |
| --- | ---: | ---: |
| Shared bias | 0 | 0 |
| Shared tanh GELU | 1,040 | 0.4168412 |
| Shared erf GELU | 0 | 0 |
| Tanh GELU with the offending pair split | 0 | 0 |

Every incorrect output was component 2 of a four-value vector. The actual
offending pair used addends `v30` and `v28`, both even. Replacing that pair and
its preceding four-byte delay with two ordinary FMAs preserved code size and
branch targets and restored bitwise agreement. Replacing all 121 `s_delay_alu`
instructions with long `s_nop` delays did **not** correct the mismatch. These
were diagnostic artifact edits; production kernels and compiler are unchanged.

Tested 2026-09-09 with native bundle
`750f265ce4fd6a194fbac12a795c96cb19cc9ed3696fd5123c5edd5589a4cd05`, compiler SHA-256
`a2902bba66bec779d95d15f6bac573072c1940dccd34215663c9a59842941dfa`.
The sibling build's compiler has the same digest; source inspection used Loom
checkout `8b2d1e882`. Raw GPU artifacts remain in `build/vision-gelu/`.
