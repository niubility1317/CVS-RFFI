import gzip
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class UReceiverConflictTests(unittest.TestCase):
    def test_frozen_aggregate_resolves_receivers_and_conflict(self):
        self.assertIsNotNone(importlib.util.find_spec('experiments.pamf_repl_step_guard.analyze_u_receiver'))
        from experiments.pamf_repl_step_guard.analyze_u_receiver import analyze_segment
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            observations = [
                dict(step=4441, physical_ids=['a', 'b'], pseudo=[0, 0], accepted=[True, True]),
                dict(step=4442, physical_ids=['a', 'c'], pseudo=[0, 1], accepted=[False, True]),
            ]
            guards = [dict(step=i, groups={g: {'anchor_cosines_before': {'L_full': cosine}}
                        for g in ('t2', 'cls_head', 'other')})
                      for i, cosine in ((1, -.3), (2, .2))]
            for name, rows in [('obs.gz', observations), ('guard.gz', guards)]:
                with gzip.open(root/name, 'wt', encoding='utf-8') as stream:
                    for row in rows:
                        stream.write(json.dumps(row)+'\n')
            done = dict(status='SOURCE_FROZEN', epoch=50, updates=11100,
                        target_access=False, U_truth_read=False)
            result = analyze_segment(root/'obs.gz', root/'guard.gz', truth={'a': 0, 'b': 1, 'c': 1},
                receivers={'a': 1, 'b': 1, 'c': 2}, step_offset=4440, expected_steps=2,
                source_completion=done)
            self.assertEqual(result['overall'], {'seen': 4, 'accepted': 3, 'accepted_correct': 2})
            self.assertEqual(result['receivers']['1']['unique_seen'], 2)
            self.assertEqual(result['receivers']['1']['unique_accepted'], 2)
            self.assertEqual(result['receivers']['1']['accepted_correct'], 1)
            self.assertEqual(result['receivers']['2']['accepted_correct'], 1)
            self.assertEqual(result['groups']['t2']['conflict']['accepted_correct'], 1)
            self.assertEqual(result['groups']['t2']['nonconflict']['accepted_correct'], 1)
            self.assertEqual(result['receivers']['1']['groups']['t2']['conflict']['accepted'], 2)
            self.assertEqual(result['epochs']['21']['accepted'], 3)

    def test_unfrozen_or_unaligned_data_are_rejected(self):
        if importlib.util.find_spec('experiments.pamf_repl_step_guard.analyze_u_receiver') is None:
            self.skipTest('Analysis module not implemented yet')
        from experiments.pamf_repl_step_guard.analyze_u_receiver import analyze_segment
        with self.assertRaises(ValueError):
            analyze_segment('not_opened.gz', 'not_opened.gz', truth={}, receivers={},
                step_offset=4440, expected_steps=1,
                source_completion=dict(status='RUNNING', epoch=49, updates=10878,
                                       target_access=False, U_truth_read=False))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with gzip.open(root/'obs.gz', 'wt', encoding='utf-8') as f:
                f.write(json.dumps(dict(step=4442, physical_ids=['a'], pseudo=[0], accepted=[True]))+'\n')
            with gzip.open(root/'guard.gz', 'wt', encoding='utf-8') as f:
                f.write(json.dumps(dict(step=1, groups={g: {'anchor_cosines_before': {'L_full': 0.}}
                    for g in ('t2', 'cls_head', 'other')}))+'\n')
            with self.assertRaises(ValueError):
                analyze_segment(root/'obs.gz', root/'guard.gz', truth={'a': 0}, receivers={'a': 1},
                    step_offset=4440, expected_steps=1,
                    source_completion=dict(status='SOURCE_FROZEN', epoch=50, updates=11100,
                                           target_access=False, U_truth_read=False))

    def test_full_trajectory_requires_independent_score_and_matches_counts(self):
        import numpy as np
        from experiments.pamf_repl_step_guard.analyze_u_receiver import analyze_frozen_run
        from tests.test_pamf_step_guard_full_analysis import fixture

        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)/'run'
            source = run/'F'/'test'/'source'
            source.mkdir(parents=True)
            fixture(source)
            cache = run/'cache'
            cache.mkdir()
            np.savez(cache/'U_s_metadata.npz', ids=np.array(['x']), rx_i=np.array([7]))
            done = json.loads((source/'complete.json').read_text())
            done['config']['source_cache'] = str(cache)
            (source/'complete.json').write_text(json.dumps(done))
            with patch('experiments.pamf_repl_step_guard.u_scoring.load_truth',
                       return_value=(['x'], {'x': 0})) as truth:
                with self.assertRaises(FileNotFoundError):
                    analyze_frozen_run(source)
                truth.assert_not_called()
                score = run/'readouts'/'test'/'U_diagnostic'
                score.mkdir(parents=True)
                (score/'complete.json').write_text(json.dumps(dict(status='SCORED_U_DIAGNOSTIC',
                    truth_last=True, source_epoch=50)))
                (score/'scores.json').write_text(json.dumps(dict(training_or_selection_use=False,
                    overall=dict(seen=6660, accepted=6660, accepted_correct=6660))))
                result = analyze_frozen_run(source)
                self.assertEqual(result['overall']['accepted_correct'], 6660)
                self.assertEqual(result['segments']['30']['receivers']['7']['unique_seen'], 1)
                self.assertEqual(result['segments']['50']['groups']['t2']['conflict']['steps'], 4440)
                truth.assert_called_once()


if __name__ == '__main__':
    unittest.main()
