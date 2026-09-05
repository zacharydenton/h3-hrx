"""The Loom text-encoder layers against transformers' bf16 Qwen3-VL layers (the same int8 ConvRot
weights, dequantised) on a real prompt: cosine of the hidden state per depth, and of the final
conditioning after layer 50.
    python3 tests/test_te_blocks.py [--curve 1,4,12,50] [--prompt "..."] [--profile]"""
import argparse, gc, hashlib, sys, time
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from encode_prompt import load_encoder, rotate_inputs, TOK
from h3te_loom import H3TeBlocks, rope_tables

PROMPT = ("A red fox trotting through a snowy forest at dawn, low golden light through the trees, "
          "steam from its breath, cinematic tracking shot, shallow depth of field.")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--curve", default="1,4,12,50"); ap.add_argument("--prompt", default=PROMPT); ap.add_argument("--profile", action="store_true"); ap.add_argument("--weights", default=None)
    a = ap.parse_args(); dev = "cuda"
    from transformers import AutoTokenizer
    ids = AutoTokenizer.from_pretrained(str(TOK))(a.prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    n = ids.shape[1]; print(f"prompt tokens {n}")
    depths = [int(v) for v in a.curve.split(",")]
    cache = ROOT / "build/te_ref" / (hashlib.sha1(a.prompt.encode()).hexdigest()[:12] + ".pt")
    if cache.exists() and all(str(d) in torch.load(cache) for d in depths):
        hs = torch.load(cache)
    else:                                       # the bf16 reference (~30 GB) runs alone, then is freed before the Loom session
        model = load_encoder(dev); rotate_inputs(model, dev)
        with torch.no_grad():
            full = model.model.language_model(input_ids=ids.to(dev), output_hidden_states=True).hidden_states
        hs = {str(d): full[d][0].float().cpu() for d in [0] + depths}
        del full, model; gc.collect(); torch.cuda.empty_cache()
        cache.parent.mkdir(parents=True, exist_ok=True); torch.save(hs, cache)
    x0 = hs["0"]
    cos, sin = rope_tables(n)
    ok = True
    for depth in depths:
        ref = hs[str(depth)]
        loom = H3TeBlocks(n, layers=depth, weights=a.weights)
        if a.profile: loom.profile(True)
        t0 = time.time(); got = loom.forward(x0, cos, sin); dt = time.time() - t0; loom.close()
        c = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0).item()
        cu = torch.nn.functional.cosine_similarity((got - x0).flatten(), (ref - x0).flatten(), dim=0).item()
        err = ((got - ref).norm() / ref.norm()).item()
        print(f"  {depth:2d} layers {dt * 1e3:8.0f} ms: hidden cosine vs bf16 {c:.5f}, update cosine {cu:.5f}, rel err {err:.4f}")
        ok &= c > 0.99
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
