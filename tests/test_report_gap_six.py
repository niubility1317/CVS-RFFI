import tempfile
import unittest
from pathlib import Path

from experiments.cric_dura_comparison.evaluation import reject_truth
from experiments.pamf_repl_source_adapt import report_gap_six as gap


class TestReportBackfill(unittest.TestCase):
    def test_declared_matrix_and_fixed_model_seeds(self):
        rows = gap.matrix()
        self.assertEqual(len(rows), 18)
        self.assertEqual(len({r['row_id'] for r in rows}), 18)
        self.assertEqual({r['setting'] for r in rows}, set(gap.SETTINGS))
        for setting in gap.SETTINGS:
            self.assertEqual({r['model_seed'] for r in rows if r['setting'] == setting}, set(gap.SEEDS))

    def test_prediction_rejects_truth_even_when_nested(self):
        with self.assertRaises(ValueError):
            reject_truth({'tests': {'cross_rx': {'truth': '/forbidden'}}})

    def test_scorer_rejects_incomplete_prediction_before_truth_open(self):
        from experiments.cric_dura_comparison import evaluation
        with tempfile.TemporaryDirectory() as tmp:
            config = {'prediction_output': str(Path(tmp)/'absent'), 'tests': {'cross_rx': {'truth': '/forbidden'}}}
            with self.assertRaises(FileNotFoundError):
                evaluation.preflight(config)

    def test_scope_restores_shared_evaluator_configuration(self):
        from experiments.pamf_single_view_paper import evaluation
        old=(evaluation.TASKS,evaluation.READOUTS,evaluation.load_reader)
        with gap.cross_rx_scope() as scoped:
            self.assertEqual(scoped.TASKS, ('cross_rx',))
            self.assertEqual(scoped.READOUTS, ('raw','DURA-LU'))
        self.assertEqual((evaluation.TASKS,evaluation.READOUTS,evaluation.load_reader),old)


if __name__ == '__main__':
    unittest.main()
