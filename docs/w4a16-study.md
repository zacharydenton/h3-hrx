# W4A16 accuracy study (2026-09-06)

Removing activation quantization from the existing GPTQ export helps substantially on
its calibration fixture, but the same export still has a large error on a real prompt.
This does not establish a general four-bit accuracy limit for H3.

`tools/w4a16_study.py` evaluates all 50 transformer blocks and the final velocity head.
The GPTQ A4 and A16 cases load identical signed four-bit codes and per-row scales from
`build/weights_gptq`, including reversing the export's gate/up interleaving. A16 uses
BF16 linear operands, with no integer activation quantization. Every mode uses FP16
attention operands, with no INT4 Q/K approximation. The residual updates use BF16 as
in the Torch reference, whereas the native C implementation keeps its residual in FP32.

The baseline uses the source checkpoint's dequantized INT8 weights with BF16 activations.
This is an accuracy emulation using dequantized matrices; it is not a native W4A16
implementation or a throughput benchmark. One block's weights are held at a time, and
attention processes 128 queries at a time against all keys. Signed nibble packing and
gate/up row order were checked with round-trip tests.

Video-velocity relative L2 error against that baseline:

| Weight / activation treatment | Calibration fixture | Real T2VA prompt, first evaluation |
| --- | ---: | ---: |
| W8A8 | 2.64% | 1.40% |
| GPTQ W4A4, per-row weights | 12.86% | 27.70% |
| GPTQ W4A16, same per-row weights | 5.56% | 23.00% |
| W4A16, round-to-nearest weights, group 128 | 13.02% | 20.00% |
| W4A16, round-to-nearest weights, group 32 | 11.51% | 14.24% |

The fixture has 2097 tokens, random text conditioning, and video sigma 0.9. The real
prompt is the saved fox T2VA case at 864x480, 22 frames, 2922 tokens, sigma 1. Its input
is reconstructed from saved noise and refined text. Older `h_in.npy` dumps can contain
block 0's output because the hook read its argument after an in-place update, so they
are deliberately not used. ComfyUI audio noise is transposed from [32, 2, A] into the
reference's channel-major token rows.

For the real case, the baseline video velocity differs from ComfyUI's saved first
prediction by 2.01%; GPTQ W4A16 differs by 22.94%. This independent check supports the
paired comparison. It does not measure a full sampling trajectory or rendered quality.

Audio-velocity error also improves with the same GPTQ weights: 19.02% to 9.82% on the
fixture, and 17.57% to 12.47% on the real prompt. The GPTQ A16 cases peak at 2.54 and
2.79 GB of Torch device allocations respectively; this excludes CPU tensors and file
cache. Runtime timings in the JSON files include weight loading and are not native
kernel benchmarks.

The strong improvement on the fixture contradicts an explanation that attributes
all of that configuration's loss to weights. The larger remaining error on the real
case also shows that activation precision alone is insufficient. More representative
weight calibration is worth testing; these two inputs cannot isolate whether prompt,
timestep, layout, or another distribution difference accounts for the remaining gap.
Group-32 W4A16 generalizes better to this real input than the per-row GPTQ export, even
though GPTQ does better on its own calibration fixture. The best tested real-prompt
four-bit configuration still has substantially more velocity error than W8A8. No full
reference-conditioned render or native W4A16 throughput test was run.

Reproduce with the project's Torch environment:

```bash
OMP_NUM_THREADS=4 python tools/w4a16_study.py --out build/w4a16/fixture.json
OMP_NUM_THREADS=4 python tools/w4a16_study.py --comfy-truth build/comfy_t2va_blocks --modes none,w8a8,gptq_a4,gptq_a16 --out build/w4a16/prompt.json
OMP_NUM_THREADS=4 python tools/w4a16_study.py --comfy-truth build/comfy_t2va_blocks --modes none,w4g128a16,w4g32a16 --out build/w4a16/prompt_groups.json
```

The group-128 and group-32 cases use round-to-nearest weight quantization of the source
checkpoint, not the per-row GPTQ codes. They are separate quantizer comparisons.
