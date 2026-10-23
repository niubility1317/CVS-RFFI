import copy
import tempfile
from pathlib import Path
import unittest

from experiments.pamf_ssl_research.dura_supplement import prepare, validate, SEEDS
from experiments.cric_dura_comparison.evaluation import preflight
from experiments.cric_dura_comparison.io import write


class FrozenDuraSupplementTests(unittest.TestCase):
    def configuration(self):
        project = '/project'
        checkpoint = project + '/runs/20261021-phase1-restoration-R65-four-route-s2026092701-m4-r69/stage2/native_map_prior/final.pt'
        template = dict(predict=dict(readouts={'raw': None, 'DURA-LU': dict(
            kind='original_F0_LU', roles='LU', checkpoint=checkpoint)},
            tests={'main': {'views_root': '/received'}}, frozen_sources={}),
            score=dict(output_root='/baseline-scores', readouts=['raw', 'DURA-LU']))
        context = {str(seed): {arm: copy.deepcopy(template) for arm in ('F20', 'F21')}
                   for seed in SEEDS}
        return prepare(context, project, '/release', 'a' * 40)

    def test_reject_truth_in_prediction_configuration(self):
        runtime, blobs = self.configuration()
        blobs['configs/PSM-s2026092701-predict.json']['tests']['main']['truth'] = '/truth'
        with self.assertRaisesRegex(ValueError, 'rejects truth'):
            validate(runtime, blobs)

    def test_reject_another_source_model_or_restorer_or_raw_archive(self):
        for key in ('source_output', 'raw_predictions_from'):
            runtime, blobs = self.configuration()
            blobs['configs/PSM-s2026092701-predict.json'][key] = '/other-row'
            with self.assertRaisesRegex(ValueError, 'binding differs'):
                validate(runtime, blobs)
        runtime, blobs = self.configuration()
        blobs['configs/PSM-s2026092701-predict.json']['readouts']['DURA-LU']['checkpoint'] = '/other-G'
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            validate(runtime, blobs)

    def test_reject_score_pointing_at_another_prediction_row(self):
        runtime, blobs = self.configuration()
        blobs['configs/PSM-s2026092701-score.json']['prediction_output'] = '/other-predictions'
        with self.assertRaisesRegex(ValueError, 'binding differs'):
            validate(runtime, blobs)

    def test_scorer_refuses_incomplete_predictions_before_opening_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            write(Path(directory) / 'complete.json', dict(status='INCOMPLETE'))
            with self.assertRaisesRegex(ValueError, 'Truth closed'):
                preflight(dict(prediction_output=directory, row_id='row', run_id='run',
                               tests={'main': {'truth': '/must-never-open'}}))


if __name__ == '__main__':
    unittest.main()
