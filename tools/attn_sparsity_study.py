"""How many 16-key tiles can an online-softmax kernel skip on H3's real attention? For each query tile (16
rows) and head, a key tile is skippable when its max score is below the row max minus tau. Two skip rules:
'final' (the true row max: an upper bound) and 'running' (the max seen so far in key order: what the kernel
does). Reports the skip fraction per layer and the attention output error of the running rule.
    python3 tools/attn_sparsity_study.py [--layers 0,10,25,40,49] [--sigma 0.9] [--frames 37] [--taus 6,8,10,12]"""
import argparse, math, sys
from pathlib import Path
import torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "reference")); sys.path.insert(0, str(ROOT / "tools"))
import h3_ref as R
from encode_prompt import prompt_path
PROMPT = "A red fox trotting through a snowy forest at dawn, cinematic"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--layers", default="0,10,25,40,49"); ap.add_argument("--sigma", type=float, default=0.9); ap.add_argument("--frames", type=int, default=37); ap.add_argument("--taus", default="6,8,10,12"); ap.add_argument("--heads", type=int, default=8)
    a = ap.parse_args(); dev = "cuda"; layers = [int(v) for v in a.layers.split(",")]; taus = [float(v) for v in a.taus.split(",")]
    fx = torch.load(ROOT / "build/fox_480p_5s_latents.pt"); prompt = torch.load(prompt_path(PROMPT, ROOT / "build/prompts"))
    latents = fx["video"][0, :, :a.frames].float(); T, Hh, W = latents.shape[1:]
    audio = torch.nan_to_num(fx["audio"].float())[:, :, :round(T / 5 * 17 / 24 * 40)] if a.frames < 37 else torch.nan_to_num(fx["audio"].float())
    layout = R.Layout(prompt["embeds"].shape[0], T, Hh, W, audio.shape[-1])
    ckpt = R.Checkpoint(device=dev, dtype=torch.bfloat16); ref = R.H3Ref(ckpt, quant="none", cache_linears=False)
    with torch.no_grad():
        text_x = ref.text_in(prompt["embeds"].to(dev)); cos, sin = R.rope_tables(layout.position_ids, ref.inv_freq, dev); rows = layout.adaln_rows.to(dev)
        sigma = a.sigma; t_v, t_a = 1.0 - sigma, 1.0 - R.time_shift_sigma(sigma, 12.0, 3.0); temb = ref.t_emb(torch.tensor([t_v, t_a]))
        gen = torch.Generator(dev).manual_seed(0); vid = latents.to(dev); xt = (1 - sigma) * vid + sigma * torch.randn(vid.shape, generator=gen, device=dev)
        video_rows = xt[None].reshape(1, 24, T, Hh // 2, 2, W // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(-1, 96)
        arows = audio.to(dev).permute(0, 2, 1).reshape(-1, 32); sig_a = R.time_shift_sigma(sigma, 12.0, 3.0); arows = (1 - sig_a) * arows + sig_a * torch.randn(arows.shape, generator=gen, device=dev)
        x = torch.cat([text_x, ref.audio_in(arows), ref.video_in(video_rows)], dim=0); S = x.shape[0]
        print(f"{S} rows ({T}x{Hh}x{W} latents), sigma {sigma}, {len(taus)} thresholds, {a.heads} heads sampled per layer")
        scale = 1.0 / math.sqrt(R.HEAD_DIM); nq = (S + 15) // 16; nk = (S + 15) // 16
        for i in range(max(layers) + 1):
            mods = ref.block_mods(i, temb)
            if i in layers:
                p = f"blocks.{i}"; m = mods[rows]; h = R.rms_norm(x, ref.t(f"{p}.norm1.weight"), ref.eps) * (1.0 + m[:, 1].to(x.dtype)) + m[:, 0].to(x.dtype)
                q, k, v = ref.qkv_heads(f"{p}.attn", ref.lin(f"{p}.attn.qkv_proj")(h), cos, sin)
                q, k, v = q.float(), k.float(), v.float()
                heads = torch.linspace(0, R.HEADS - 1, a.heads).long().tolist()
                skip_final = {t: 0.0 for t in taus}; skip_run = {t: 0.0 for t in taus}; err_run = {t: 0.0 for t in taus}; base_norm = 0.0
                for hd in heads:
                    s = (q[:, hd] @ k[:, hd].T) * scale                                       # [S][S] f32
                    exact = torch.softmax(s, -1) @ v[:, hd]; base_norm += exact.norm() ** 2
                    sp = F.pad(s, (0, nk * 16 - S, 0, nq * 16 - S), value=-1e9)
                    tile_max = sp.reshape(nq, 16, nk, 16).amax(dim=(1, 3))                   # [nq][nk]: the tile's max over its 16 rows x 16 keys
                    row_tile_max = sp.reshape(nq, 16, nk, 16).amax(dim=3)                     # [nq][16][nk]: per row per key tile
                    final_max = sp.amax(dim=1).reshape(nq, 16)                                 # [nq][16]
                    running = torch.cummax(row_tile_max, dim=2).values                         # [nq][16][nk]: the running max per row after each key tile
                    running_prev = torch.cat([torch.full_like(running[:, :, :1], -1e9), running[:, :, :-1]], dim=2)
                    for t in taus:
                        # final rule: the whole 16-row tile is skippable when every row's max in the key tile is below its final max - tau
                        skip_f = (row_tile_max < (final_max[:, :, None] - t)).all(dim=1)       # [nq][nk]
                        skip_final[t] += skip_f.float().mean().item() / len(heads)
                        # running rule: the kernel compares the tile's rows against the running max before this tile (tile max update happens after)
                        run_before = torch.maximum(running_prev, row_tile_max)                 # the running max after this tile's own scores
                        skip_r = (row_tile_max < (running_prev - t)).all(dim=1)                # skippable only if every row is far below what has been seen before
                        skip_run[t] += skip_r.float().mean().item() / len(heads)
                        mask = skip_r[:, None, :, None].expand(nq, 16, nk, 16).reshape(nq * 16, nk * 16)[:S, :S]
                        approx = torch.softmax(s.masked_fill(mask, -1e9), -1) @ v[:, hd]
                        err_run[t] += ((approx - exact).norm() ** 2).item()
                    del s, sp, exact
                print(f"  layer {i:2d}: " + "  ".join(f"tau {t:g}: skip final {100 * skip_final[t]:4.1f}% running {100 * skip_run[t]:4.1f}% err {math.sqrt(err_run[t] / base_norm):.4f}" for t in taus))
            x = ref.block(i, x, mods, rows, cos, sin)


if __name__ == "__main__":
    main()
