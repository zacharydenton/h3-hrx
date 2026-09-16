"""Propose at most two conservative cache trials from a complete observation run.

Input: optimization_benchmark's *.stages.json. Predictions replay the policy on
an uncached history; a cached trajectory can diverge and must be rendered/reviewed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def quantile(values, p):
    values = sorted(values)
    at = (len(values) - 1) * p
    low, high = math.floor(at), math.ceil(at)
    return values[low] + (values[high] - values[low]) * (at - low)


def proposals(report):
    events = report['events']
    schedules = [e for e in events if e['stage'] == 'schedule']
    if len(schedules) != 1 or schedules[0].get('cache_mode') != 'observe':
        raise ValueError('expected one complete observation-only trajectory')
    decisions = [e for e in events if e['stage'] == 'cache_decision']
    count = schedules[0]['evaluations']
    if count < 6 or [e['evaluation'] for e in decisions] != list(range(1, count + 1)):
        raise ValueError('missing evaluations or trajectory too short to calibrate')
    if any(e['skipped'] for e in decisions):
        raise ValueError('calibration requires all evaluations to run in full')
    metrics = [e['relative_conditioning_audio_video'] for e in decisions]
    if any(len(v) != 3 or any(x is None or not math.isfinite(x) or x < 0 for x in v) for v in metrics):
        raise ValueError('nonfinite or malformed modality metrics')
    trials = []
    for percentile in (0.25, 0.5):
        thresholds = [max(quantile([v[k] for v in metrics[2:-2]], percentile), 1e-12) for k in range(3)]
        accumulated = [0.0, 0.0, 0.0]
        last_skipped = False
        skips = []
        for step, values in enumerate(metrics):
            accumulated = [a + v for a, v in zip(accumulated, values)]
            skip = 2 <= step < count - 2 and not last_skipped and all(a < t for a, t in zip(accumulated, thresholds))
            if skip:
                skips.append(step + 1)
            else:
                accumulated = [0.0, 0.0, 0.0]
            last_skipped = skip
        trials.append({'quantile': percentile, 'conditioning_audio_video_thresholds': thresholds,
                       'predicted_skipped_evaluations': skips,
                       'cli_arguments': ['--cache-thresholds', *[format(t, '.9g') for t in thresholds]]})
    return trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stages', type=Path)
    args = parser.parse_args()
    source = args.stages.read_bytes()
    try:
        trials = proposals(json.loads(source))
    except (ValueError, KeyError, TypeError) as e:
        parser.error(str(e))
    print(json.dumps({'observation_sha256': hashlib.sha256(source).hexdigest(), 'trials': trials,
                      'release_gate': 'Require at least 10% measured whole-render improvement and acceptable identity, motion, temporal coherence, dialogue and audio synchronization against identical initial noise.'}, indent=2))


if __name__ == '__main__':
    main()
