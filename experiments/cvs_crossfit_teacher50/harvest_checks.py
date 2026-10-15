"""Synthetic reporting checks; these do not provide trained-model evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from .harvest import validate, store


def fixture():
    files = {
        'complete.json': dict(status='ANALYZED', source_only=True, query_read=False, actual_model_count=2,
            seed_count=1, student_models_trained=0, automatic_bank_expansion=False, target_predictions=0),
        'comparison/results.json': dict(status='ANALYZED', source_only=True, query_read=False, U_truth_read=False,
            student_models_trained=0, head_fit_updates=0, comparison_forward_packets=1440, teacher_evaluation_forward_packets=10084,
            records=[dict(view='clean', region='RX1', count=144, repairs=3, new_errors=1, net=2)]),
        'resources.json': dict(teacher_training_updates=22200,
            rows=[dict(arm=a, full_step_log_read=11100) for a in ('SAMPLE', 'EXCLUDE_RX1')]),
    }
    for arm in ('SAMPLE', 'EXCLUDE_RX1'):
        prefix = arm+'-s2026092701'
        files[prefix+'/source/completion.json'] = dict(epoch=50, optimizer_steps=11100, logged_steps=11100,
            target_access=False, head_fit_updates=0)
        files[prefix+'/source/training_mechanism_complete.json'] = dict(status='VERIFIED', unique_train_samples=4464,
            forward_packets=dict(train=1420800), U_forward_packets=0, EMA_updates=0, U_truth_read=False, head_fit_updates=0,
            train_receivers=[1, 3, 4, 6, 8] if arm=='SAMPLE' else [3, 4, 6, 8])
        files[prefix+'/source_evaluation/complete.json'] = dict(status='SOURCE_PREDICTIONS_COMPLETE', forward_packets=5042,
            model_state_unchanged=True, query_read=False, U_truth_read=False)
    return dict(status='COMPLETE_SOURCE_EVIDENCE', files=files)


class HarvestChecks(unittest.TestCase):
    def test_incomplete_does_not_claim_complete(self):
        self.assertFalse(validate(dict(status='WAITING_FOR_MATRIX', no_partial_curve_claim=True)))

    def test_complete_synthetic_schema_and_actual_budget_rejection(self):
        self.assertTrue(validate(fixture()))
        for name, field, value in (
            ('comparison/results.json', 'U_truth_read', True),
            ('SAMPLE-s2026092701/source/completion.json', 'optimizer_steps', 11099),
            ('EXCLUDE_RX1-s2026092701/source/training_mechanism_complete.json', 'train_receivers', [1, 3, 4, 6]),
        ):
            payload = fixture();payload['files'][name][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(payload)

    def test_repeat_read_reuses_and_conflict_preserves_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp);original = {'x.json': dict(value=1), 'a/y.md': 'actual source report\n'}
            store(out, original);store(out, original)
            with self.assertRaises(FileExistsError):
                store(out, {'new.json': {}, 'x.json': dict(value=2)})
            self.assertFalse((out/'new.json').exists())
            self.assertEqual(json.loads((out/'x.json').read_text()), dict(value=1))


if __name__ == '__main__':
    unittest.main(verbosity=2)
