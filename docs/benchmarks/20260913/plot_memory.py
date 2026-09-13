"""Plot separate memory views from the completed benchmark telemetry."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path)
parser.add_argument('--case', default='floating_glacier')
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
metrics = json.loads((args.run_dir / f'comfy_{args.case}/metrics.json').read_text())
intervals = set()

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
for prefix, label, color in [('', 'h3-hrx · Loom / HRX', '#0d9488'),
                             ('comfy_', 'ComfyUI', '#d97706')]:
    path = args.run_dir / f'{prefix}{args.case}.telemetry.jsonl'
    timing = json.loads((args.run_dir / f'{prefix}{args.case}.timing.json').read_text())
    intervals.add(timing['memory_sampling_interval_seconds'])
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    minutes = [sample['elapsed_s'] / 60 for sample in samples]
    gpu = [sample['drm_memory']['resident_bytes'] / 1024**3 for sample in samples]
    pss = [sample['process_memory_bytes']['Pss'] / 1024**3 for sample in samples]
    axes[0].plot(minutes, gpu, label=label, color=color, linewidth=2)
    axes[1].plot(minutes, pss, label=label, color=color, linewidth=2)

axes[0].set_title('GPU buffer residency · engine DRM clients only', loc='left')
axes[1].set_title('Process memory · proportional set size (PSS)', loc='left')
for axis in axes:
    axis.set_ylabel('GiB')
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.2)
    axis.spines[['top', 'right']].set_visible(False)
axes[0].legend(frameon=False)
axes[1].set_xlabel('Minutes since process launch')
interval_label = '/'.join(f'{value:g}' for value in sorted(intervals))
title = f"Strix Halo · {metrics['width']}×{metrics['height']} · {metrics['frames']} frames"
if len(metrics['step_seconds']) < metrics['evaluations']:
    title += f"\n{metrics['evaluations']} evaluations planned · ComfyUI stopped after {len(metrics['step_seconds'])} completed"
else:
    title += f" · {metrics['evaluations']} evaluations"
fig.suptitle(title + '\n' +
             f'{interval_label}-second samples; these memory views must not be added together', fontsize=13)
args.out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(args.out, dpi=180)
