"""Regressions for benchmark stage extraction and telemetry boundaries."""
import datetime
import json
from pathlib import Path
import tempfile
import unittest
from optimization_benchmark import parse_events, stage_report
from cache_calibrate import proposals


class BenchmarkTests(unittest.TestCase):
    def test_progress_and_test_prefixes_do_not_hide_release_events(self):
        events = parse_events('test fixture ... H3_STAGE {"stage":"conditioning_start"}\n'
                              '\r  step 1/2  100 s\r  step 2/2  200 sH3_STAGE {"stage":"denoise_finished"}\n'
                              'ordinary output\nH3_STAGE {"stage":"video_decode_start"}\n')
        self.assertEqual([e['stage'] for e in events], ['conditioning_start', 'denoise_finished', 'video_decode_start'])

    def test_stage_intervals_do_not_double_count_boundary_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'run.log').write_text('H3_STAGE {"stage":"sampling","unix_seconds":1001}\n'
                                           'H3_STAGE {"stage":"decode","unix_seconds":1003}\n')
            samples = [{'elapsed_s': second, 'drm_memory': {'resident_bytes': gpu},
                        'process_memory_bytes': {'Pss': gpu // 2},
                        'system_memory_bytes': {'MemAvailable': 100 - gpu}}
                       for second, gpu in [(1, 20), (2, 30), (3, 10), (4, 15)]]
            (root / 'telemetry').write_text('\n'.join(json.dumps(s) for s in samples))
            timing = {'started_utc': datetime.datetime.fromtimestamp(1000, datetime.timezone.utc).isoformat(), 'wall_seconds': 5}
            report = stage_report(root / 'run.log', timing, root / 'telemetry')
            self.assertEqual([s['peak_drm_resident_bytes'] for s in report['intervals']], [30, 15])
            self.assertEqual([s['samples'] for s in report['intervals']], [2, 2])

    def test_calibration_refuses_partial_or_cached_trajectories(self):
        report = {'events': [{'stage': 'schedule', 'cache_mode': 'observe', 'evaluations': 20}]}
        with self.assertRaises(ValueError):
            proposals(report)
        report['events'] += [{'stage': 'cache_decision', 'evaluation': i + 1, 'skipped': False,
                              'relative_conditioning_audio_video': [0.01 * i, 0.02 * i, 0.03 * i]}
                             for i in range(20)]
        trials = proposals(report)
        self.assertEqual(len(trials), 2)
        for trial in trials:
            skips = trial['predicted_skipped_evaluations']
            self.assertTrue(all(3 <= step <= 18 for step in skips))
            self.assertTrue(all(b - a > 1 for a, b in zip(skips, skips[1:])))
        report['events'][3]['skipped'] = True
        with self.assertRaises(ValueError):
            proposals(report)


if __name__ == '__main__':
    unittest.main()
