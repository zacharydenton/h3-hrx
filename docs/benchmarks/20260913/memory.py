"""Linux process, DRM, and system memory counters for the benchmark sampler.

Process PSS and DRM allocations are separate views; adding them can double-count
shared or mapped pages. System counters include other applications and caches.
"""
from pathlib import Path


def counters(path):
    result = {}
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return result
    for line in lines:
        key, _, value = line.partition(':')
        parts = value.split()
        if parts and parts[0].isdigit():
            amount = int(parts[0])
            if len(parts) > 1:
                amount *= {'kB': 1024, 'KiB': 1024, 'MiB': 1024**2}.get(parts[1], 1)
            result[key] = amount
    return result


def system_memory():
    values = counters('/proc/meminfo')
    return {key: values.get(key) for key in
            ['MemTotal', 'MemAvailable', 'MemFree', 'Cached', 'SwapTotal', 'SwapFree', 'Dirty']}


def descendants(roots):
    """Include subprocesses; container init PIDs must be supplied explicitly."""
    parents = {}
    for path in Path('/proc').glob('[0-9]*/status'):
        parent = counters(path).get('PPid')
        if parent is not None:
            parents[int(path.parent.name)] = parent
    owned = set(roots)
    while True:
        children = {pid for pid, parent in parents.items() if parent in owned}
        expanded = owned | children
        if expanded == owned:
            return sorted(owned)
        owned = expanded


def process_memory(pids):
    totals = {key: 0 for key in ['Rss', 'Pss', 'Pss_Anon', 'Pss_File', 'Pss_Shmem', 'SwapPss']}
    sampled = []
    for pid in pids:
        values = counters(f'/proc/{pid}/smaps_rollup')
        if 'Pss' not in values:
            continue
        sampled.append(pid)
        for key in totals:
            totals[key] += values.get(key, 0)
    return {'sampled_pids': sampled, **totals}


def drm_memory(pids):
    """Count each AMD DRM client once, even if it has multiple file descriptors."""
    clients = {}
    for pid in pids:
        try:
            descriptors = list(Path(f'/proc/{pid}/fdinfo').iterdir())
        except OSError:
            continue
        for descriptor in descriptors:
            try:
                text = descriptor.read_text()
            except OSError:
                continue
            fields = dict(line.split(':', 1) for line in text.splitlines() if ':' in line)
            if fields.get('drm-driver', '').strip() != 'amdgpu':
                continue
            key = (fields.get('drm-pdev', '').strip(), fields.get('drm-client-id', '').strip())
            values = counters(descriptor)
            clients[key] = {
                'resident_gtt': values.get('drm-resident-gtt', values.get('drm-memory-gtt', 0)),
                'resident_vram': values.get('drm-resident-vram', values.get('drm-memory-vram', 0)),
                'allocated_gtt': values.get('drm-total-gtt', values.get('drm-memory-gtt', 0)),
                'allocated_vram': values.get('drm-total-vram', values.get('drm-memory-vram', 0)),
            }
    return {
        'client_count': len(clients),
        'resident_bytes': sum(c['resident_gtt'] + c['resident_vram'] for c in clients.values()),
        'allocated_bytes': sum(c['allocated_gtt'] + c['allocated_vram'] for c in clients.values()),
    }
