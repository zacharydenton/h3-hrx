"""Plot full generation time, stage breakdown, and separate UMA memory views."""
import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path)
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--out-dir', type=Path, required=True)
args = parser.parse_args()
results = json.loads(args.results.read_text())['runs']
engines = [
    ('h3_i8', 'h3 · INT8 QK', '#0d9488'),
    ('comfy_pytorch', 'ComfyUI · PyTorch BF16', '#c77920'),
    ('comfy_kitchen', 'ComfyUI · Kitchen INT8', '#9054bb'),
    ('h3_f16', 'h3 · F16 QK', '#3a73b9'),
]
args.out_dir.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
for key, label, color in engines:
    case = results[key]['case']
    telemetry = args.run_dir / f'{case}.telemetry.jsonl'
    if telemetry.exists():
        lines = telemetry.read_text().splitlines()
    else:
        with gzip.open(str(telemetry) + '.gz', 'rt') as source:
            lines = source.read().splitlines()
    samples = [json.loads(line) for line in lines if line]
    minutes = [sample['elapsed_s'] / 60 for sample in samples]
    views = [
        [sample['drm_memory']['resident_bytes'] / 2**30 for sample in samples],
        [sample['process_memory_bytes']['Pss'] / 2**30 for sample in samples],
        [sample['system_memory_bytes']['MemAvailable'] / 2**30 for sample in samples],
    ]
    for axis, values in zip(axes, views):
        axis.plot(minutes, values, label=label, color=color, linewidth=1.6)
for axis, label in zip(axes, ['GPU-resident buffers (GiB)', 'Process PSS (GiB)', 'System available RAM (GiB)']):
    axis.set_ylabel(label)
    axis.set_ylim(bottom=0)
    axis.grid(alpha=.2)
    axis.spines[['top', 'right']].set_visible(False)
axes[0].legend(frameon=False, ncol=2)
axes[-1].set_xlabel('Minutes since each process launch (runs were sequential)')
fig.suptitle('Strix Halo · 1344×768 · 124 frames · 20 evaluations\n'
             'Full pipelines; one-second samples. Memory views overlap and must not be added.', fontsize=13)
fig.savefig(args.out_dir / 'memory.png', dpi=180)
plt.close(fig)

fig, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
sampling = np.array([results[key]['stages_seconds']['sampling'] / 60 for key, _, _ in engines])
decoding = np.array([(results[key]['stages_seconds'].get('decode', 0)
                      + results[key]['stages_seconds'].get('decode_and_save', 0)) / 60 for key, _, _ in engines])
total = np.array([results[key]['wall_seconds'] / 60 for key, _, _ in engines])
other = total - sampling - decoding
y = np.arange(len(engines))
axis.barh(y, other, label='Startup, conditioning, output and other', color='#a9b8bd')
axis.barh(y, sampling, left=other, label='Sampling (includes first evaluation)', color='#299d8b')
axis.barh(y, decoding, left=other + sampling, label='Video/audio decode', color='#ce914b')
for i, minutes in enumerate(total):
    axis.text(minutes + max(total) * .012, i, f'{minutes:.1f} min', va='center', fontsize=10)
axis.set_yticks(y, [label for _, label, _ in engines])
axis.invert_yaxis()
axis.set_xlim(0, max(total) * 1.17)
axis.set_xlabel('Minutes from process launch through completed output and process exit')
axis.spines[['top', 'right']].set_visible(False)
axis.grid(axis='x', alpha=.2)
axis.set_axisbelow(True)
axis.legend(frameon=False, loc='lower right', fontsize=9)
axis.set_title('One complete 768p alien-fjord clip per configuration\n'
               'Same prompt, quantized checkpoints, and 20-evaluation schedule')
fig.savefig(args.out_dir / 'generation-time.png', dpi=180)
