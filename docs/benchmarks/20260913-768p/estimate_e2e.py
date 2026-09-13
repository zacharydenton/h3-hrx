"""Extrapolate ComfyUI's full 768p time from observed sampling and prior overhead."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / '20260913' / 'raw'
current = json.loads((HERE / 'results.json').read_text())
shape = json.loads((HERE / 'raw/comfy_floating_glacier_768p/metrics.json').read_text())
prior = json.loads((PRIOR / 'comfy_floating_glacier/metrics.json').read_text())
prior_run = json.loads((PRIOR / 'comfy_floating_glacier.timing.json').read_text())
assert shape['frames'] == prior['frames'] == 124
assert shape['comfy_commit'] == prior['comfy_commit']

area_ratio = shape['width'] * shape['height'] / (prior['width'] * prior['height'])
sampling = current['planned_evaluations'] * current['comfyui']['steady_median_seconds']
decode = prior['decode_and_save_seconds'] * area_ratio
overhead = prior_run['wall_seconds'] - prior['sampling_seconds'] - prior['decode_and_save_seconds']
encoding_allowance = 5.0
total = sampling + decode + overhead + encoding_allowance
print(json.dumps({
    'method': '20 times the observed 768p steady median, plus prior process overhead, area-scaled decode/save, and an assumed five seconds for encoding.',
    'prior_source_commit': '416a3c4c6a5fe0ee20fcad60c18e35c027692aaf',
    'historical_768p_source_commit': '52827db9f05ca0b6012dfeb05f535abc0d4f9c81',
    'historical_768p_evaluation_seconds': [772.7, 772.4, 768.8],
    'prior_decode_and_save_seconds': prior['decode_and_save_seconds'],
    'spatial_area_ratio': area_ratio,
    'projected_sampling_seconds': sampling,
    'projected_decode_and_save_seconds': decode,
    'prior_other_process_seconds': overhead,
    'assumed_encoding_seconds': encoding_allowance,
    'comfyui_estimated_wall_seconds': total,
    'h3_measured_wall_seconds': current['h3_wall_seconds'],
    'estimated_end_to_end_speedup': total / current['h3_wall_seconds'],
}, indent=2))
