"""Launch local ComfyUI with H3 checkpoints discovered in the standard HF cache."""
import argparse
import json
import os
from pathlib import Path
import sys

from huggingface_hub import try_to_load_from_cache

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--comfy', type=Path, default=Path.home() / 'code/ComfyUI')
args, server_args = parser.parse_known_args()
args.comfy = args.comfy.expanduser().resolve()
files = {
    'diffusion_models': 'diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors',
    'text_encoders': 'text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors',
    'vae': 'vae/minimax_h3_video_vae_fp16.safetensors',
}
paths = {}
for category, filename in files.items():
    cached = try_to_load_from_cache('Comfy-Org/MiniMax-H3', filename)
    if not isinstance(cached, str):
        raise SystemExit(f'H3 checkpoint is not cached: {filename}')
    paths[category] = str(Path(cached).parent)
config = Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')) / 'h3-hrx/comfy-hf-paths.json'
config.parent.mkdir(parents=True, exist_ok=True)
config.write_text(json.dumps({'h3_huggingface_cache': paths}, indent=2) + '\n')
os.environ.setdefault('TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL', '1')
os.chdir(args.comfy)
os.execv(sys.executable, [sys.executable, str(args.comfy / 'main.py'),
                         '--extra-model-paths-config', str(config),
                         '--disable-api-nodes', *server_args])
