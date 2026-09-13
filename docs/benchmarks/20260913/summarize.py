"""Summarize a completed h3 / ComfyUI pair; first evaluations are excluded from steady timing."""
import argparse
import json
import re
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('run_dir', type=Path)
ap.add_argument('--case', default='floating_glacier')
ap.add_argument('--inference-only', action='store_true',
                help='compare completed sampling/decoding despite a later MP4 encoder failure')
a = ap.parse_args()
h3_run = json.loads((a.run_dir / f'{a.case}.timing.json').read_text())
comfy_run = json.loads((a.run_dir / f'comfy_{a.case}.timing.json').read_text())
comfy = json.loads((a.run_dir / f'comfy_{a.case}/metrics.json').read_text())
if h3_run['exit_code']:
    raise SystemExit('h3 must finish successfully before comparing runs.')
if comfy_run['exit_code']:
    comfy_log = (a.run_dir / f'comfy_{a.case}.log').read_text()
    if not (a.inference_only and comfy.get('decode_and_save_seconds', 0) > 0
            and "Unknown encoder 'libx264'" in comfy_log
            and f"frames ({comfy['frames']}, {comfy['height']}, {comfy['width']}, 3)" in comfy_log):
        raise SystemExit('ComfyUI failed; only the documented post-decode encoder failure can be summarized with --inference-only.')
log = (a.run_dir / f'{a.case}.log').read_text()
steps = re.findall(r'step\s+(\d+)/(\d+)\s+([0-9.]+) s', log)
expected = comfy['evaluations']
if [int(step[0]) for step in steps] != list(range(1, expected + 1)):
    raise SystemExit('Incomplete or duplicate h3 evaluations.')
if any(int(step[1]) != expected for step in steps) or len(comfy['step_seconds']) != expected:
    raise SystemExit('Evaluation counts do not match.')
cumulative = [0.0] + [float(step[2]) for step in steps]
h3_seconds = [round(b - a, 1) for a, b in zip(cumulative, cumulative[1:])]

def summary(seconds):
    steady = seconds[1:]
    return {
        'evaluation_seconds': seconds,
        'steady_evaluations': len(steady),
        'steady_median_seconds': statistics.median(steady),
        'steady_mean_seconds': statistics.mean(steady),
        'steady_min_seconds': min(steady),
        'steady_max_seconds': max(steady),
    }

result = {'h3': summary(h3_seconds), 'comfyui': summary(comfy['step_seconds'])}
result['steady_median_speedup'] = result['comfyui']['steady_median_seconds'] / result['h3']['steady_median_seconds']
result['memory'] = {engine: {key: value for key, value in run.items() if key.startswith('sampled_peak_') or key in ['memory_sampling_interval_seconds', 'baseline_system_memory_bytes', 'minimum_system_available_bytes', 'maximum_system_swap_used_bytes', 'memory_note']} for engine, run in [('h3', h3_run), ('comfyui', comfy_run)]}
result['h3_wall_seconds'] = h3_run['wall_seconds']
result['comfyui_observed_process_seconds'] = comfy_run['wall_seconds']
result['exit_codes'] = {'h3': h3_run['exit_code'], 'comfyui': comfy_run['exit_code']}
if not a.inference_only:
    result['process_to_output_speedup'] = comfy_run['wall_seconds'] / h3_run['wall_seconds']
result['wall_time_note'] = ('ComfyUI completed sampling and video/audio decoding but MP4 encoding failed because its FFmpeg lacks libx264. No end-to-end speedup is reported. Memory covers the observed processes, including decoding and output attempts.' if a.inference_only else 'Both exported MP4/WAV. Checkpoints were cached; runtime/kernel caches were not reset.')
result['method'] = 'One completed sampling trajectory per engine; discard the first evaluation. The remaining evaluations are within-run observations, not independent repeated runs. h3 callback timestamps are rounded to 0.1 seconds.'
print(json.dumps(result, indent=2))
