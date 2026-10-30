import json
from pathlib import Path
import tempfile
import unittest

from experiments.pamf_phase3_open_world import star_class_competition_launch as launch


class ClassCompetitionLaunchTests(unittest.TestCase):
    def test_predictor_has_no_truth_and_scorer_has_all_three_truth_roles(self):
        r = dict(python='python', run_root='/run', source_novel_truth='/private/source',
                 clean_truth='/private/clean', star_truth='/private/star')
        predict = launch._command(r, 'predict')
        score = launch._command(r, 'score')
        self.assertTrue(predict[4].endswith('star_class_competition_run'))
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
                        status='CLASS_COMPETITION_PREDICTIONS_COMPLETE', models_frozen_before_V=True,
                        all_thresholds_frozen_before_novel_and_Q=True, all_three_conditions_complete=True,
                        all_conditions_through_DURA=True, private_truth_reads=0, target_U_reads=0)
            marker = root/'predict/complete.json'
            marker.write_text(json.dumps(good), encoding='utf-8')
            self.assertEqual(launch._terminal(r, 'predict'), good)
            for key, value in [('all_three_conditions_complete', False), ('all_conditions_through_DURA', False),
                               ('status', 'NUISANCE_MARGINAL_PREDICTIONS_COMPLETE'), ('private_truth_reads', 1)]:
                with self.subTest(key=key):
                    marker.write_text(json.dumps(dict(good, **{key:value})), encoding='utf-8')
                    with self.assertRaises(ValueError):
                        launch._terminal(r, 'predict')


if __name__ == '__main__':
    unittest.main()
