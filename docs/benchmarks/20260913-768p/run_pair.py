"""Run the matched 768p workloads sequentially with memory telemetry.

Provision build/benchmark-768p/bin/ffmpeg before invoking this script.
"""
import json
import os
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[3]
os.chdir(repo)
out = Path('build/benchmark-768p')
bin_dir = (out / 'bin').resolve()
cache = Path.home() / '.cache/huggingface/hub'
image = 'docker.io/kyuz0/amd-strix-halo-comfyui@sha256:384aa1fecef6a841832e0d5552949977330308d8c25e212a94f5e8dfcc061cae'
container = 'h3-benchmark-768p-20260913'
case = 'floating_glacier_768p'
prompt = 'docs/prompts/floating_glacier.txt'
wrapper = [
    'python3', 'docs/benchmarks/20260913/run_timed.py', '--out-dir', str(out),
    '--sample-interval', '1', '--min-available-gib', '80', '--abort-available-gib', '12',
]
h3 = wrapper + [
    '--name', case, '--stdin', prompt, '--',
    'env', f'PATH={bin_dir}:{os.environ["PATH"]}', 'HF_HUB_OFFLINE=1', 'HRX_OFFLINE=1',
    'target/release/h3', '--width', '1344', '--height', '768', '--frames', '124',
    '--steps', '21', '--seed', '1618', '--out', str(out / f'{case}.mp4'),
]
comfy = wrapper + [
    '--name', f'comfy_{case}', '--container', container, '--',
    'podman', 'run', '--rm', '--name', container,
    '--device', '/dev/kfd', '--device', '/dev/dri', '--group-add', 'keep-groups',
    '--security-opt', 'label=disable', '-v', f'{repo}:{repo}',
    '-v', f'{cache}:{cache}:ro', '-w', str(repo),
    '-e', f'HF_HUB_CACHE={cache}', '-e', 'HF_HUB_OFFLINE=1',
    '-e', f'PATH={bin_dir}:/opt/venv/bin:/usr/local/bin:/usr/bin',
    '--entrypoint', '/opt/venv/bin/python', image, 'scripts/comfy_dump.py',
    '--comfy', '/opt/ComfyUI',
    '--prompt-file', prompt, '--width', '1344', '--height', '768', '--length', '124',
    '--steps', '20', '--seed', '1618', '--out', str(out / f'comfy_{case}'),
    '--video-out', str(out / f'comfy_{case}/video.mp4'),
]
out.mkdir(parents=True, exist_ok=True)
if any((out / f'{name}.log').exists() for name in [case, f'comfy_{case}']):
    raise SystemExit('Existing run logs found; preserve or move them before rerunning.')
(out / 'commands.json').write_text(json.dumps({'h3': h3, 'comfyui': comfy}, indent=2) + '\n')
for command in [h3, comfy]:
    subprocess.run(command, check=True)
