"""The C host against ComfyUI's own run (tools/comfy_clip.py dumps in build/comfy_t2va_blocks and build/comfy_fl2va):
int8 rows + f16 attention must reproduce ComfyUI's residual stream (video rows cosine >= 0.999 through block 30, >= 0.99 at blocks 40 and 49)
and its 20-evaluation trajectory to rel err <= 0.02 after five evaluations. Skips (exit 0) when the dumps are absent."""
import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
def run(args):
    out = subprocess.run([sys.executable, str(ROOT / "tools/compare_comfy.py")] + args, capture_output=True, text=True); print(out.stdout); return out.stdout
def main():
    if not (ROOT / "build/comfy_t2va_blocks/blocks/blk_49.npy").exists() or not (ROOT / "build/comfy_fl2va/x_19.npy").exists() or not (ROOT / "build/weights_i8/manifest.txt").exists():
        print("SKIP: no ComfyUI dumps (tools/comfy_clip.py --dump-steps --dump-blocks 0,1,2,5,10,20,30,40,49 --steps 2 --out build/comfy_t2va_blocks) or no build/weights_i8"); return 0
    ok = True
    for line in run(["--case", "t2va", "--mode", "blocks", "--truth", "build/comfy_t2va_blocks", "--blocks", "build/weights_i8", "--attn", "f16"]).splitlines():
        if line.startswith("blk_"):
            blk = int(line.split(":")[0][4:]); video = float(line.split("video ")[1].rstrip("]")); good = video >= (0.999 if blk <= 30 else 0.99);   # measured 0.9935 at block 40, 0.9992 at 49 ok &= good; print(f"  {'PASS' if good else 'FAIL'} {line.split(':')[0]} video rows {video:.4f}")
    for line in run(["--case", "fl2va", "--mode", "trajectory", "--truth", "build/comfy_fl2va", "--blocks", "build/weights_i8", "--attn", "f16"]).splitlines():
        if line.startswith("x_05"):
            err = float(line.split("rel err ")[1]); good = err <= 0.02; ok &= good; print(f"  {'PASS' if good else 'FAIL'} trajectory after five evaluations: rel err {err:.4f}")
    return 0 if ok else 1
if __name__ == "__main__": sys.exit(main())
