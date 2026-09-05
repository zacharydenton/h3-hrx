"""The C pipeline (libh3pipe.so) against the Python reference, stage by stage:
  text_in: the refined text rows for a cached prompt vs reference/h3_ref.py's text_in on the same embeddings.
    python3 tests/test_pipe.py [--prompt-file build/prompts/<hash>.pt]"""
import argparse, glob, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from h3pipe_loom import H3Pipe


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prompt-file", default=None); a = ap.parse_args()
    pf = Path(a.prompt_file) if a.prompt_file else min((Path(p) for p in glob.glob(str(ROOT / "build/prompts/*.pt"))), key=lambda p: p.stat().st_size)
    prompt = torch.load(pf); ids = prompt["ids"].numpy().astype(np.int32); print(f"prompt {prompt['prompt'][:60]!r}: {ids.size} tokens")
    ok = True
    ckpt = R.Checkpoint(device="cuda", dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none")
    with torch.no_grad(): want = ref.text_in(prompt["embeds"].cuda()).float().cpu().numpy()
    del ref, ckpt; torch.cuda.empty_cache()
    t0 = time.time(); pipe = H3Pipe(); print(f"session in {time.time() - t0:.1f} s")
    t0 = time.time(); got = pipe.text_in(ids); print(f"text_in in {time.time() - t0:.2f} s")
    c = float(np.dot(got.ravel(), want.ravel()) / (np.linalg.norm(got) * np.linalg.norm(want) + 1e-30)); err = float(np.linalg.norm(got - want) / (np.linalg.norm(want) + 1e-30))
    print(f"  {'PASS' if c > 0.999 else 'FAIL'} text_in: cosine {c:.5f}, rel err {err:.4f}  (the C path re-encodes the prompt in Loom; the reference uses the cached Loom embeddings)")
    ok &= c > 0.999
    pipe.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
