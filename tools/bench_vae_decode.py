"""Compare decoder variants on identical saved latents; requires explicit GPU use.

The first call includes lazy decoder loading/compilation, subsequent calls use
resident weights. Session teardown is reported separately. This decoder-only
session does not reproduce teardown of a CLI session holding DiT/text weights.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h3pipe_loom import H3Pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", required=True)
    parser.add_argument("--latents", type=Path, required=True, help="video .npy or .npz with video and optional audio arrays")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--variants", nargs="+", choices=("base", "fast", "wide"), default=["base", "fast"])
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--frames", type=int, help="requires --repeat-latents when different from the saved input")
    parser.add_argument("--repeat-latents", action="store_true", help="explicitly synthesize a longer input by repeating saved latents")
    parser.add_argument("--output", type=Path)
    opt = parser.parse_args()
    if opt.repeat < 1: parser.error("repeat must be positive")
    if os.environ.get("H3_PROFILE") or os.environ.get("H3_TRACE"):
        parser.error("unset H3_PROFILE and H3_TRACE for unprofiled timings")
    loaded = np.load(opt.latents, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        with loaded:
            z = loaded["video"].copy()
            audio = loaded["audio"].copy() if "audio" in loaded.files else None
    else:
        z, audio = loaded, None
    if z.ndim != 4 or z.shape[0] != 24 or z.shape[1] < 2 or (z.shape[1] - 2) % 5:
        parser.error("video must have shape [24, 2+5*n, h, w]")
    frames = opt.frames or (z.shape[1] - 2) // 5 * 17 + 5
    if frames < 5 or (frames - 5) % 17: parser.error("frames must be 5+17*n")
    target_t = (frames - 5) // 17 * 5 + 2
    synthetic = target_t != z.shape[1]
    if synthetic:
        if not opt.repeat_latents: parser.error("different frame count requires --repeat-latents")
        z = np.tile(z, (1, math.ceil(target_t / z.shape[1]), 1, 1))[:, :target_t].copy()
    results = dict(latents=str(opt.latents), synthetic_repeated_latents=synthetic,
                   frames=frames, height=z.shape[2] * 16, width=z.shape[3] * 16,
                   includes_audio=audio is not None, measurements=[])
    def record(item):
        results["measurements"].append(item)
        print(json.dumps(item), flush=True)
        if opt.output:
            opt.output.parent.mkdir(parents=True, exist_ok=True)
            opt.output.write_text(json.dumps(results, indent=2) + "\n")
    reference = None
    saved_env = {key: os.environ.get(key) for key in ("H3_VAE_FAST", "H3_VAE_WIDE")}
    try:
        for variant in opt.variants:
            os.environ["H3_VAE_FAST"] = "1" if variant == "fast" else "0"
            os.environ["H3_VAE_WIDE"] = "1" if variant == "wide" else "0"
            pipe = H3Pipe(dit="/unused/dit", te="/unused/te", library=opt.library)
            try:
                params = pipe.params(height=results["height"], width=results["width"], frames=frames)
                av = audio
                if av is not None:
                    count = pipe.shape(params).audio_t
                    if av.shape != (2, 32, count):
                        if not opt.repeat_latents or av.ndim != 3 or av.shape[:2] != (2, 32) or av.shape[2] < 1:
                            parser.error("audio shape does not match the requested frame count")
                        av = np.tile(av, (1, 1, math.ceil(count / av.shape[2])))[:, :, :count].copy()
                        results["synthetic_repeated_latents"] = True
                for run in range(opt.repeat):
                    start = time.perf_counter()
                    output = pipe.decode_video(params, z)
                    video_end = time.perf_counter()
                    if av is not None: pipe.decode_audio(av)
                    end = time.perf_counter()
                    record(dict(variant=variant, run=run, resident=run > 0,
                                video_seconds=video_end - start, audio_seconds=end - video_end,
                                decode_seconds=end - start))
            finally:
                start = time.perf_counter(); pipe.close()
                record(dict(variant=variant, teardown_seconds=time.perf_counter() - start))
            if reference is None:
                reference = output
            else:
                squared, maximum = 0.0, 0
                for first in range(output.shape[0]):
                    delta = output[first].astype(np.float64) - reference[first]
                    squared += float(np.sum(delta * delta)); maximum = max(maximum, int(np.max(np.abs(delta))))
                mse = squared / output.size
                record(dict(variant=variant, reference_variant=opt.variants[0], rgb_max_difference=maximum,
                            rgb_psnr_db=10 * math.log10(255 ** 2 / max(mse, 1e-12))))
    finally:
        for key, value in saved_env.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    return 0


if __name__ == "__main__":
    sys.exit(main())
