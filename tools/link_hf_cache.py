"""Register an existing ComfyUI models directory in the shared Hugging Face cache.

    python3 tools/link_hf_cache.py [--apply] [--models DIR]

`hf download --local-dir` populates a directory but not the cache, so checkpoints downloaded that way
are invisible to anything that resolves through it. This links them in: every cache reader, this
project's included, resolves a file through `snapshots/<commit>/<path>`, so a symlink there is enough
and no bytes are copied. That matters when the two live on different filesystems, where a hard link is
impossible and a copy would be another 115 GB.

Each file's size is checked against the repository before it is linked, so a differently named or
truncated local file is not registered as something it is not. A dry run prints what it would do.
"""
import json, os, sys, urllib.request
from pathlib import Path

REPO = "Comfy-Org/MiniMax-H3"
_argv = sys.argv[1:]
MODELS = Path(_argv[_argv.index("--models") + 1]) if "--models" in _argv else Path(
    os.environ.get("H3_MODELS") or Path.home() / "comfy-models"
)
CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
FOLDER = CACHE / ("models--" + REPO.replace("/", "--"))
apply = "--apply" in sys.argv

with urllib.request.urlopen(f"https://huggingface.co/api/models/{REPO}?blobs=true", timeout=60) as r:
    meta = json.load(r)
sizes = {s["rfilename"]: s.get("size") for s in meta.get("siblings", [])}
ref = FOLDER / "refs/main"
if not ref.is_file():
    sys.exit(
        f"no cache entry for {REPO}: fetch one small file first, e.g.\n"
        f"  hf download {REPO} README.md"
    )
commit = ref.read_text().strip()
snap = FOLDER / "snapshots" / commit

linked = skipped = 0
for rel, want in sorted(sizes.items()):
    local = MODELS / rel
    if not local.is_file():
        continue
    have = local.stat().st_size
    target = snap / rel
    if target.exists() or target.is_symlink():
        print(f"  already in the cache   {rel}")
        skipped += 1
        continue
    if want is not None and have != want:
        print(f"  SIZE MISMATCH, skipped {rel}: local {have}, repo {want}")
        skipped += 1
        continue
    print(f"  link {have / 1e9:6.1f} GB       {rel}")
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(local.resolve())
    linked += 1
print(f"{'linked' if apply else 'would link'} {linked}, skipped {skipped}")
if not apply:
    print("(dry run; pass --apply)")
