"""Run one GPU workload, recording its command, wall time, and device telemetry."""
import argparse
import datetime
import fcntl
import json
import os
import pathlib
import resource
import signal
import subprocess
import time

from memory import descendants, drm_memory, process_memory, system_memory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--name', required=True)
parser.add_argument('--out-dir', type=pathlib.Path, required=True)
parser.add_argument('--stdin', type=pathlib.Path)
parser.add_argument('--wait-pid', type=int)
parser.add_argument('--container', help='Podman container name, for tracking its host PID')
parser.add_argument('--min-available-gib', type=float, default=70)
parser.add_argument('--abort-available-gib', type=float, default=8)
parser.add_argument('--sample-interval', type=float, default=5,
                    help='seconds between memory samples and pressure checks')
parser.add_argument('--gpu-device', type=pathlib.Path,
                    default=pathlib.Path('/sys/class/drm/card1/device'))
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.command and args.command[0] == '--':
    args.command.pop(0)
if not args.command:
    parser.error('a command is required after --')
if args.sample_interval <= 0:
    parser.error('--sample-interval must be positive')

root = args.out_dir
root.mkdir(parents=True, exist_ok=True)
# Refuse a second run in this result directory instead of overlapping measurements.
lock = (root / 'gpu-run.lock').open('w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
while args.wait_pid and pathlib.Path(f'/proc/{args.wait_pid}').exists():
    time.sleep(5)

busy_sensor = args.gpu_device / 'gpu_busy_percent'
temperature_sensor = next(args.gpu_device.glob('hwmon/hwmon*/temp1_input'), None)
idle_samples = 0
while idle_samples < 6:
    busy = int(busy_sensor.read_text())
    available = system_memory()['MemAvailable']
    ready = busy < 10 and available >= args.min_available_gib * 1024**3
    idle_samples = idle_samples + 1 if ready else 0
    time.sleep(5)

baseline_system = system_memory()
samples = []
container_pid = None
memory_abort = False


def stop_workload(process):
    if args.container:
        subprocess.run(['podman', 'kill', args.container], capture_output=True)
    # A launcher may have children; terminate our isolated process group together.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


started = datetime.datetime.now(datetime.timezone.utc).isoformat()
start = time.perf_counter()
input_file = args.stdin.open() if args.stdin else subprocess.DEVNULL
with (root / f'{args.name}.log').open('w') as log, \
        (root / f'{args.name}.telemetry.jsonl').open('w') as telemetry:
    process = subprocess.Popen(args.command, stdin=input_file, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while process.poll() is None:
            if args.container and container_pid is None:
                inspected = subprocess.run(
                    ['podman', 'inspect', '--format', '{{.State.Pid}}', args.container],
                    capture_output=True, text=True,
                )
                if inspected.returncode == 0 and inspected.stdout.strip().isdigit():
                    value = int(inspected.stdout.strip())
                    container_pid = value if value > 0 else None
            roots = [process.pid] + ([container_pid] if container_pid else [])
            pids = descendants(roots)
            entry = {
                'elapsed_s': time.perf_counter() - start,
                'engine_pids': pids,
                'process_memory_bytes': process_memory(pids),
                'drm_memory': drm_memory(pids),
                'system_memory_bytes': system_memory(),
            }
            for sensor_name in ['mem_info_gtt_used', 'mem_info_vram_used']:
                try:
                    entry[sensor_name] = int((args.gpu_device / sensor_name).read_text())
                except OSError:
                    pass
            samples.append(entry)
            for key, sensor in [('gpu_busy_percent', busy_sensor),
                                ('temperature_mC', temperature_sensor)]:
                if sensor is not None:
                    try:
                        entry[key] = int(sensor.read_text())
                    except OSError:
                        pass
            entry['kfd_pids'] = subprocess.run(
                ['fuser', '/dev/kfd'], capture_output=True, text=True
            ).stdout.split()
            telemetry.write(json.dumps(entry) + '\n')
            telemetry.flush()
            if entry['system_memory_bytes']['MemAvailable'] < args.abort_available_gib * 1024**3:
                memory_abort = True
                print('Stopping our workload: available system memory fell below the guard.', flush=True)
                stop_workload(process)
                break
            try:
                process.wait(timeout=args.sample_interval)
            except subprocess.TimeoutExpired:
                pass
        wall_seconds = time.perf_counter() - start
    finally:
        if process.poll() is None:
            stop_workload(process)
        if args.stdin:
            input_file.close()

result = {
    'started_utc': started,
    'command': args.command,
    'stdin': str(args.stdin) if args.stdin else None,
    'wall_seconds': wall_seconds,
    'exit_code': process.returncode if not memory_abort else 1,
    'aborted_for_memory_pressure': memory_abort,
    'minimum_starting_available_gib': args.min_available_gib,
    'abort_available_gib': args.abort_available_gib,
    'max_rss_kib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    'memory_sampling_interval_seconds': args.sample_interval,
    'baseline_system_memory_bytes': baseline_system,
    'sampled_peak_process_rss_bytes': max((s['process_memory_bytes']['Rss'] for s in samples), default=0),
    'sampled_peak_process_pss_bytes': max((s['process_memory_bytes']['Pss'] for s in samples), default=0),
    'sampled_peak_drm_resident_bytes': max((s['drm_memory']['resident_bytes'] for s in samples), default=0),
    'sampled_peak_drm_allocated_bytes': max((s['drm_memory']['allocated_bytes'] for s in samples), default=0),
    'minimum_system_available_bytes': min((s['system_memory_bytes']['MemAvailable'] for s in samples), default=None),
    'maximum_system_swap_used_bytes': max((s['system_memory_bytes']['SwapTotal'] - s['system_memory_bytes']['SwapFree'] for s in samples), default=0),
    'memory_note': f'Sampled process PSS and DRM memory are separate views and must not be summed. {args.sample_interval:g}-second sampling may miss brief peaks. System counters include other applications.',
}
(root / f'{args.name}.timing.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(result['exit_code'])
