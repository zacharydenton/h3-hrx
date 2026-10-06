"""Cache calibration requires a complete, uncached observation trace."""
import unittest
from cache_calibrate import parse_events, proposals


class CalibrationTests(unittest.TestCase):
    def test_progress_and_test_prefixes_do_not_hide_release_events(self):
        events = parse_events('test fixture ... H3_STAGE {"stage":"conditioning_start"}\n'
                              '\r  step 1/2  100 s\r  step 2/2  200 sH3_STAGE {"stage":"denoise_finished"}\n'
                              'ordinary output\nH3_STAGE {"stage":"video_decode_start"}\n')
        self.assertEqual([e['stage'] for e in events], ['conditioning_start', 'denoise_finished', 'video_decode_start'])

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
