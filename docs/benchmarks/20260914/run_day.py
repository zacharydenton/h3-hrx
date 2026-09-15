"""Run native 768p comparisons and showcase renders sequentially on an idle GPU."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=repo / 'build/benchmark-20260914')
parser.add_argument('--comfy', type=Path, default=Path.home() / 'code/ComfyUI')
parser.add_argument('--job', action='append', help='run only these named jobs, in the listed order')
args = parser.parse_args()
os.chdir(repo)
args.out = args.out.resolve()
args.out.mkdir(parents=True, exist_ok=True)
raw = args.out / 'raw'
python = args.comfy / '.venv-rocm/bin/python'
environment = dict(os.environ, HF_HUB_OFFLINE='1', HRX_OFFLINE='1',
                   TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL='1',
                   HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
for key in ['H3_PROFILE', 'H3_TRACE', 'H3_CACHE_TRACE']:
    environment.pop(key, None)
jobs = {}


def h3(name, prompt, seed, *, attention='i8', steps=21, profile=False):
    command = ['target/release/h3', '--width', '1344', '--height', '768', '--frames', '124',
               '--steps', str(steps), '--seed', str(seed), '--attn', attention,
               '--out', str(raw / f'{name}.mp4'), '--latents', str(raw / name)]
    if profile:
        command = ['env', 'H3_PROFILE=1', *command]
    jobs[name] = {'prompt': f'docs/prompts/{prompt}.txt', 'command': command}


def comfy(name, backend='default', *, steps=20, profile=False):
    command = [str(python), 'scripts/comfy_dump.py', '--comfy', str(args.comfy),
               '--width', '1344', '--height', '768', '--length', '124',
               '--steps', str(steps), '--seed', '1618', '--attention-backend', backend,
               '--prompt-file', 'docs/prompts/alien_fjord.txt', '--save-latents',
               '--out', str(raw / name), '--video-out', str(raw / name / 'video.mp4')]
    if profile:
        command.append('--profile-sampling')
    jobs[name] = {'command': command}


h3('alien_fjord', 'alien_fjord', 1618)
comfy('comfy_alien_fjord')
comfy('comfy_kitchen_alien_fjord', 'comfy-kitchen-int8')
h3('tidal_sky', 'tidal_sky', 31415)
h3('midnight_tram', 'midnight_tram', 27182)
h3('alien_fjord_f16', 'alien_fjord', 1618, attention='f16')
h3('h3_profile_alien_fjord', 'alien_fjord', 1618, steps=3, profile=True)
comfy('comfy_kitchen_profile_alien_fjord', 'comfy-kitchen-int8', steps=2, profile=True)
# Profiling is explicit: the native ComfyUI profiler coincided with a host freeze.
# See incident.json; ordinary reproduction runs only the complete comparisons.
selected = args.job or [name for name in jobs if '_profile_' not in name]
for name in selected:
    if name not in jobs:
        raise SystemExit(f'Unknown job: {name}')
    if (raw / f'{name}.log').exists():
        raise SystemExit(f'Existing run found: {name}; select unfinished jobs explicitly.')
for name in selected:
    job = jobs[name]
    wrapper = ['python3', 'docs/benchmarks/20260913/run_timed.py', '--name', name,
               '--out-dir', str(raw), '--sample-interval', '1',
               '--min-available-gib', '80', '--abort-available-gib', '12']
    if 'prompt' in job:
        wrapper += ['--stdin', job['prompt']]
    command = wrapper + ['--', *job['command']]
    status = {'job': name, 'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'command': command, 'status': 'running'}
    (args.out / 'queue-status.json').write_text(json.dumps(status, indent=2) + '\n')
    print(json.dumps(status), flush=True)
    result = subprocess.run(command, env=environment)
    status.update(status='completed' if result.returncode == 0 else 'failed', exit_code=result.returncode,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (args.out / 'queue-status.json').write_text(json.dumps(status, indent=2) + '\n')
    with (args.out / 'queue-history.jsonl').open('a') as history:
        history.write(json.dumps(status) + '\n')
    if result.returncode:
        raise SystemExit(result.returncode)
