import json
from pathlib import Path
import tempfile
import unittest

from experiments.pamf_phase3_open_world import star_selective_uda_launch as launch


class SelectiveUDALaunchTests(unittest.TestCase):
    def test_predictor_has_no_truth_and_scorer_has_all_three_truth_roles(self):
        r = dict(python='python', run_root='/run', source_novel_truth='/private/source',
                 clean_truth='/private/clean', star_truth='/private/star')
        predict = launch._command(r, 'predict')
        score = launch._command(r, 'score')
        self.assertTrue(predict[4].endswith('star_selective_uda_run'))
        self.assertFalse(any('/private/' in value or 'truth' in value for value in predict))
        for role, path in [('source-novel', '/private/source'), ('clean', '/private/clean'), ('star', '/private/star')]:
            with self.subTest(role=role):
                i = score.index('--'+role+'-truth')
                self.assertEqual(score[i+1], path)

    def test_partial_or_wrong_predictions_cannot_advance_to_truth(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root/'predict').mkdir()
            r = dict(run_root=d, code_commit='a'*40)
            good = dict(run_id=launch.RUN, code_commit=r['code_commit'], final_phase3_complete=False,
                        status='SELECTIVE_UDA_PREDICTIONS_COMPLETE', source_models_frozen_before_U=True,
                        adapters_frozen_before_source_and_Q=True, all_three_DURA_conditions_complete=True,
                        all_thresholds_reused_exactly=True, private_truth_reads=0, target_U_reads=384,
                        U_features_reused=True)
            marker = root/'predict/complete.json'
            marker.write_text(json.dumps(good), encoding='utf-8')
            self.assertEqual(launch._terminal(r, 'predict'), good)
            for key, value in [('all_three_DURA_conditions_complete', False), ('adapters_frozen_before_source_and_Q', False),
                               ('status', 'NUISANCE_MARGINAL_PREDICTIONS_COMPLETE'), ('private_truth_reads', 1)]:
                with self.subTest(key=key):
                    marker.write_text(json.dumps(dict(good, **{key:value})), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        launch._terminal(r, 'predict')


if __name__ == '__main__':
    unittest.main()
