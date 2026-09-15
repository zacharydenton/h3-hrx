"""Export completed benchmark clips, previews, and portable measurement records."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from summarize import CASES, summarize

repo = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('run_dir', type=Path)
parser.add_argument('--case', action='append', choices=list(CASES), required=True)
parser.add_argument('--records', type=Path, default=Path(__file__).parent / 'raw')
args = parser.parse_args()
args.records.mkdir(parents=True, exist_ok=True)
names = {'h3_i8': 'h3-i8', 'comfy_pytorch': 'comfy-pytorch',
         'comfy_kitchen': 'comfy-kitchen', 'h3_f16': 'h3-f16'}
for label in args.case:
    case, engine = CASES[label]
    result = summarize(args.run_dir, case, engine)
    source = args.run_dir / case / 'video.mp4' if engine == 'comfy' else args.run_dir / f'{case}.mp4'
    media_dir = repo / 'docs/media' / ('benchmarks/20260914' if label in names else 'showcase')
    media_dir.mkdir(parents=True, exist_ok=True)
    name = names.get(label, label.replace('_', '-'))
    output = media_dir / f'{name}.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source),
                    '-map', '0:v:0', '-map', '0:a:0', '-c', 'copy',
                    '-movflags', '+faststart', str(output)], check=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-i', str(output),
                    '-vf', 'select=eq(n\\,0)+eq(n\\,24)+eq(n\\,48)+eq(n\\,72)+eq(n\\,96)+eq(n\\,123),scale=448:256,tile=3x2',
                    '-frames:v', '1', '-threads', '2', str(media_dir / f'{name}-frames.jpg')], check=True)
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-threads', '2', '-i', str(output),
                    '-frames:v', '1', '-threads', '2', str(media_dir / f'{name}.jpg')], check=True)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(output)]))
    probe['artifact'] = {'path': str(output.relative_to(repo)),
                         'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    (args.records / f'{case}.ffprobe.json').write_text(json.dumps(probe, indent=2) + '\n')
    for suffix in ['log', 'timing.json']:
        shutil.copyfile(args.run_dir / f'{case}.{suffix}', args.records / f'{case}.{suffix}')
    with (args.run_dir / f'{case}.telemetry.jsonl').open('rb') as source_log:
        with (args.records / f'{case}.telemetry.jsonl.gz').open('wb') as destination:
            with gzip.GzipFile(filename='', mode='wb', fileobj=destination, mtime=0) as compressed:
                shutil.copyfileobj(source_log, compressed)
    if engine == 'comfy':
        (args.records / case).mkdir(exist_ok=True)
        shutil.copyfile(args.run_dir / case / 'metrics.json', args.records / case / 'metrics.json')
    print(json.dumps({'case': label, 'video': str(output), 'wall_seconds': result['wall_seconds']}), flush=True)
