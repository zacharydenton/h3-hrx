"""Summarize completed native 768p renders; never extrapolate incomplete jobs."""
import argparse
import json
from pathlib import Path
import re
import statistics
import subprocess

CASES = {
    'h3_i8': ('alien_fjord', 'h3'),
    'comfy_pytorch': ('comfy_alien_fjord', 'comfy'),
    'comfy_kitchen': ('comfy_kitchen_alien_fjord', 'comfy'),
    'h3_f16': ('alien_fjord_f16', 'h3'),
    'tidal_sky': ('tidal_sky', 'h3'),
    'midnight_tram': ('midnight_tram', 'h3'),
}


def summarize(directory, case, engine):
    timing = json.loads((directory / f'{case}.timing.json').read_text())
    if timing['exit_code'] or timing.get('aborted_for_memory_pressure'):
        raise ValueError(f'{case} did not finish successfully')
    log = (directory / f'{case}.log').read_text()
    steps = re.findall(r'step\s+(\d+)/(\d+)\s+([0-9.]+) s', log)
    if [int(step[0]) for step in steps] != list(range(1, 21)) or any(step[1] != '20' for step in steps):
        raise ValueError(f'{case} must have 20 consecutive completed evaluations')
    command = timing['command']
    if engine == 'h3':
        cumulative = [0.0] + [float(step[2]) for step in steps]
        intervals = [round(b - a, 1) for a, b in zip(cumulative, cumulative[1:])]
        denoised = float(re.search(r'denoised in ([0-9.]+) s', log)[1])
        stages = {
            'session': float(re.search(r'session in ([0-9.]+) s', log)[1]),
            'preparation': round(denoised - cumulative[-1], 1),
            'sampling': cumulative[-1],
            'decode': float(re.search(r'decoded in ([0-9.]+) s', log)[1]),
        }
        output = Path(command[command.index('--out') + 1])
        tokens = int(re.search(r'; (\d+) prompt tokens', log)[1])
    else:
        metrics = json.loads((directory / case / 'metrics.json').read_text())
        if metrics['profile_sampling'] or metrics['evaluations'] != 20:
            raise ValueError(f'{case} is an intrusive diagnostic or a different schedule')
        intervals = metrics['step_seconds']
        if len(intervals) != 20:
            raise ValueError(f'{case} metrics are incomplete')
        stages = {name: metrics[key] for name, key in {
            'initialization': 'initialization_seconds',
            'conditioning': 'conditioning_seconds',
            'dit_load_and_conditioning_unload': 'dit_load_and_conditioning_unload_seconds',
            'sampling': 'sampling_seconds',
            'dit_unload': 'dit_unload_seconds',
            'decode_and_save': 'decode_and_save_seconds',
            'mux': 'mux_seconds',
        }.items()}
        output = Path(metrics['video_out'])
        tokens = None
    stages['other_process_time'] = timing['wall_seconds'] - sum(stages.values())
    # Reports remain reproducible after copying raw data away from the original host.
    local_output = directory / case / 'video.mp4' if engine == 'comfy' else directory / f'{case}.mp4'
    if local_output.exists():
        output = local_output
    probe_record = directory / f'{case}.ffprobe.json'
    if probe_record.exists():
        streams = json.loads(probe_record.read_text())['streams']
    else:
        streams = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(output),
        ]))['streams']
    video = next(stream for stream in streams if stream['codec_type'] == 'video')
    audio = next(stream for stream in streams if stream['codec_type'] == 'audio')
    if (video['width'], video['height'], int(video['nb_frames'])) != (1344, 768, 124):
        raise ValueError(f'{case} has unexpected video dimensions or frame count')
    memory = {key: value for key, value in timing.items() if key.startswith('sampled_peak_')}
    memory['minimum_system_available_bytes'] = timing['minimum_system_available_bytes']
    memory['peak_drop_in_system_available_bytes'] = timing['baseline_system_memory_bytes']['MemAvailable'] - timing['minimum_system_available_bytes']
    steady = intervals[1:]
    return {
        'case': case, 'engine': engine, 'wall_seconds': timing['wall_seconds'],
        'evaluations': len(intervals), 'evaluation_seconds': intervals,
        'steady_median_seconds': statistics.median(steady),
        'steady_range_seconds': [min(steady), max(steady)],
        'stages_seconds': stages, 'prompt_tokens': tokens, 'memory': memory,
        'output': {'width': video['width'], 'height': video['height'], 'frames': int(video['nb_frames']),
                   'video_codec': video['codec_name'], 'audio_codec': audio['codec_name'],
                   'duration_seconds': float(video['duration'])},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--partial', action='store_true', help='show only completed jobs while the queue runs')
    args = parser.parse_args()
    results = {}
    for label, (case, engine) in CASES.items():
        if args.partial and not (args.run_dir / f'{case}.timing.json').exists():
            continue
        results[label] = summarize(args.run_dir, case, engine)
    comparisons = {}
    if 'h3_i8' in results:
        for label in ['comfy_pytorch', 'comfy_kitchen', 'h3_f16']:
            if label in results:
                comparisons[label] = {
                    'wall_time_ratio_to_h3_i8': results[label]['wall_seconds'] / results['h3_i8']['wall_seconds'],
                    'steady_time_ratio_to_h3_i8': results[label]['steady_median_seconds'] / results['h3_i8']['steady_median_seconds'],
                }
    print(json.dumps({'runs': results, 'comparisons': comparisons,
                      'memory_note': 'Process PSS and GPU residency overlap on UMA; do not add. System available-memory changes include other applications and cache behavior.',
                      'method': 'Full 124-frame 1344x768 outputs; 20 evaluations; cached checkpoints. Separate fresh processes, sequential runs. One trajectory per configuration; steady statistics omit the first evaluation. Intrusive profiles excluded.'}, indent=2))


if __name__ == '__main__':
    main()
