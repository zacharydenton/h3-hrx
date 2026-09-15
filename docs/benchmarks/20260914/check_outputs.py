"""Check completed outputs and quantify within-engine attention differences."""
import argparse
import hashlib
import json
from pathlib import Path
import wave

import numpy as np

RUNS = {
    'h3_i8': ('alien_fjord', 'h3'),
    'h3_f16': ('alien_fjord_f16', 'h3'),
    'comfy_pytorch': ('comfy_alien_fjord', 'comfy'),
    'comfy_kitchen': ('comfy_kitchen_alien_fjord', 'comfy'),
    'tidal_sky': ('tidal_sky', 'h3'),
    'midnight_tram': ('midnight_tram', 'h3'),
}


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def latent(directory, case, engine, kind):
    path = directory / f'{case}.{kind}.f32' if engine == 'h3' else directory / case / f'{kind}_latent.npy'
    values = np.memmap(path, dtype='<f4', mode='r') if engine == 'h3' else np.load(path, mmap_mode='r')
    if not np.isfinite(values).all():
        raise ValueError(f'Non-finite values in {path}')
    return values, path


def difference(reference, candidate):
    if reference.shape != candidate.shape:
        raise ValueError('Within-engine latent shapes differ')
    a, b = reference.reshape(-1).astype(np.float64), candidate.reshape(-1).astype(np.float64)
    aa, bb = float(a @ a), float(b @ b)
    delta = b - a
    return {
        'cosine': float(a @ b / np.sqrt(aa * bb)) if aa and bb else None,
        'relative_l2': float(np.sqrt(delta @ delta / aa)) if aa else None,
        'rmse': float(np.sqrt(np.mean(delta * delta))),
        'max_abs_difference': float(np.max(np.abs(delta))),
    }


def sound(path):
    with wave.open(str(path)) as source:
        if source.getsampwidth() != 2:
            raise ValueError(f'Expected PCM16 WAV: {path}')
        channels, rate, frames = source.getnchannels(), source.getframerate(), source.getnframes()
        values = np.frombuffer(source.readframes(frames), dtype='<i2').astype(np.float64) / 32768
    return {'channels': channels, 'sample_rate': rate, 'duration_seconds': frames / rate,
            'rms': float(np.sqrt(np.mean(values * values))), 'peak': float(np.max(np.abs(values))),
            'near_full_scale_fraction': float(np.mean(np.abs(values) >= 32766 / 32768)),
            'sha256': digest(path)}


def shared_h3_command(command):
    args = iter(command)
    result = []
    for item in args:
        if item in {'--attn', '--out', '--latents'}:
            next(args)
        else:
            result.append(item)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--partial', action='store_true', help='check only completed runs')
    args = parser.parse_args()
    arrays, results, commands = {}, {}, {}
    for label, (case, engine) in RUNS.items():
        record = args.run_dir / f'{case}.timing.json'
        if args.partial and not record.exists():
            continue
        timing = json.loads(record.read_text())
        if timing['exit_code'] or timing.get('aborted_for_memory_pressure'):
            raise ValueError(f'{case} did not finish successfully')
        commands[label] = timing['command']
        arrays[label], results[label] = {}, {}
        for kind in ['video', 'audio']:
            values, path = latent(args.run_dir, case, engine, kind)
            arrays[label][kind] = values
            results[label][kind + '_latent'] = {'finite': True, 'elements': values.size,
                                               'shape': list(values.shape), 'sha256': digest(path)}
        if engine == 'comfy':
            results[label]['initial_noise'] = {}
            for kind in ['video', 'audio']:
                path = args.run_dir / case / f'noise_{kind}.npy'
                values = np.load(path, mmap_mode='r')
                if not np.isfinite(values).all():
                    raise ValueError(f'Non-finite initial noise in {path}')
                results[label]['initial_noise'][kind] = {
                    'shape': list(values.shape), 'sha256': digest(path), 'finite': True}
        wav = args.run_dir / f'{case}.wav' if engine == 'h3' else args.run_dir / case / 'audio.wav'
        results[label]['audio'] = sound(wav)
    pairs = {}
    for reference, candidate in [('h3_f16', 'h3_i8'), ('comfy_pytorch', 'comfy_kitchen')]:
        if reference not in arrays or candidate not in arrays:
            continue
        if reference == 'h3_f16':
            if shared_h3_command(commands[reference]) != shared_h3_command(commands[candidate]):
                raise ValueError('h3 commands differ beyond attention and output paths')
        else:
            for kind in ['video', 'audio']:
                a = np.load(args.run_dir / RUNS[reference][0] / f'noise_{kind}.npy', mmap_mode='r')
                b = np.load(args.run_dir / RUNS[candidate][0] / f'noise_{kind}.npy', mmap_mode='r')
                if not np.array_equal(a, b):
                    raise ValueError(f'ComfyUI initial {kind} noise differs')
        pairs[candidate + '_vs_' + reference] = {
            'reference': reference, 'candidate': candidate,
            'initial_noise': ('Identical saved video/audio arrays' if reference == 'comfy_pytorch'
                              else 'Same deterministic h3 RNG, seed, shape, and schedule; initial arrays were not exported'),
            **{kind: difference(arrays[reference][kind], arrays[candidate][kind]) for kind in ['video', 'audio']},
        }
    print(json.dumps({'runs': results, 'within_engine_differences': pairs,
                      'interpretation': 'Latent differences measure numerical divergence, not perceptual quality. No cross-engine similarity score is computed because initial random streams differ.'},
                     indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
