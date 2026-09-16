"""Record an unprofiled h3 run with source identity, stage events and UMA telemetry.

Example: python scripts/optimization_benchmark.py --name alien-stage-scoped
  --out build/optimization --prompt docs/prompts/alien_fjord.txt --
  target/release/h3 --width 1344 --height 768 --steps 21 --seed 1618 --out clip.mp4
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def parse_events(log):
    # A callback may leave a carriage-return progress line open before the event.
    # Test harnesses likewise prefix their first stderr line with the test name.
    return [json.loads(line.partition('H3_STAGE ')[2])
            for line in log.splitlines() if 'H3_STAGE {' in line]


def stage_report(log, timing, telemetry):
    events = parse_events(log.read_text())
    start = datetime.datetime.fromisoformat(timing['started_utc']).timestamp()
    samples = [json.loads(line) for line in telemetry.read_text().splitlines()]
    intervals = []
    for i, event in enumerate(events):
        end = events[i + 1]['unix_seconds'] if i + 1 < len(events) else start + timing['wall_seconds']
        selected = [s for s in samples if event['unix_seconds'] <= start + s['elapsed_s'] < end]
        intervals.append({
            'stage': event['stage'], 'seconds_to_next_event': max(0, end - event['unix_seconds']),
            'samples': len(selected),
            'peak_drm_resident_bytes': max((s['drm_memory']['resident_bytes'] for s in selected), default=None),
            'peak_process_pss_bytes': max((s['process_memory_bytes']['Pss'] for s in selected), default=None),
            'minimum_system_available_bytes': min((s['system_memory_bytes']['MemAvailable'] for s in selected), default=None),
        })
    return {'events': events, 'intervals': intervals,
            'note': 'Host submission events do not synchronize kernels. Residency release events follow a fence; loading and kernel execution can overlap elsewhere. UMA memory views must not be summed.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prompt', type=Path, required=True)
    parser.add_argument('--min-available-gib', type=float, default=70)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('pass an h3 command after --')
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a filename component')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / f'{args.name}.metadata.json'
    if any((out / f'{args.name}.{suffix}').exists() for suffix in ('metadata.json', 'log', 'timing.json')):
        parser.error('this run already exists; use a new name')
    executable = Path(shutil.which(command[0]) or command[0]).resolve()
    metadata = {
        'command': command, 'executable': str(executable), 'executable_sha256': sha256(executable),
        'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'tracked_patch_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p)
                          for tree in ('src', 'kernels', 'scripts') for p in sorted((ROOT / tree).rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
        'prompt_sha256': sha256(args.prompt),
        'environment': {k: os.environ[k] for k in (
            'HF_HOME', 'HF_HUB_CACHE', 'HF_HUB_OFFLINE', 'HRX_OFFLINE',
            'HRX_LOOM_LIBRARY', 'H3_GRAPH', 'H3_KEEP_MAPPED', 'H3_CACHE_TRACE',
            'H3_FUSED_OPERANDS', 'H3_REUSE_SCRATCH', 'H3_KSMOOTH', 'H3_VAE_FAST', 'H3_VAE_WIDE') if k in os.environ},
        'method': 'Fresh process; existing filesystem/compiler caches. No kernel profiler. Wait for six idle GPU samples before starting.',
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + '\n')
    env = dict(os.environ, H3_STAGE_TRACE='1')
    env.pop('H3_PROFILE', None)
    wrapper = [sys.executable, str(ROOT / 'docs/benchmarks/20260913/run_timed.py'),
               '--name', args.name, '--out-dir', str(out), '--stdin', str(args.prompt.resolve()),
               '--sample-interval', '1', '--min-available-gib', str(args.min_available_gib),
               '--abort-available-gib', '12', '--', *command]
    result = subprocess.run(wrapper, env=env)
    timing_path = out / f'{args.name}.timing.json'
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        report = stage_report(out / f'{args.name}.log', timing, out / f'{args.name}.telemetry.jsonl')
        (out / f'{args.name}.stages.json').write_text(json.dumps(report, indent=2) + '\n')
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
