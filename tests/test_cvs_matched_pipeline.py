import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

spec = importlib.util.spec_from_file_location('pipeline', Path(__file__).resolve().parents[1] / 'tools/cvs_matched_pipeline.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class FinalScoringTests(unittest.TestCase):
    def test_phase1_scores_despite_phase2_failure_and_checks_before_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            out = root / 'row' / 'final_eval'
            out.mkdir(parents=True)
            capsule = root / 'capsule'
            capsule.mkdir()
            pipeline.write(root / 'state.json', {'row': {'status': 'TECHNICAL_FAILURE'}})
            pipeline.write(capsule / 'manifest.json', {'classes': ['a', 'b'], 'scenes': ['high']})
            np.savez(capsule / 'index.npz', ids=['a', 'b'], scenes=[0, 0])
            truth = root / 'truth.json'
            pipeline.write(truth, {'a': {'label': 0, 'receiver': 'x'}, 'b': {'label': 1, 'receiver': 'x'}})
            run = {'execution': {'remote_run_root': str(root)}, 'rows': [{'row_id': 'row', 'output_root': str(out.parent)}], 'final_capsule': str(capsule), 'final_truth': str(truth)}
            original_read = pipeline.read
            def guarded_read(path):
                if str(path) == str(truth):
                    raise AssertionError('truth opened before prediction validation')
                return original_read(path)
            pipeline.write(out / 'predictions_complete.json', {'status': 'INCOMPLETE'})
            with patch.object(pipeline, 'read', guarded_read), self.assertRaises(ValueError):
                pipeline.final_score(run)
            pipeline.write(out / 'predictions_complete.json', {'status': 'PREDICTIONS_COMPLETE'})
            np.savez(out / 'predictions.npz', ids=['wrong', 'b'], clean=[0, 1], satellite=[1, 1])
            with patch.object(pipeline, 'read', guarded_read), self.assertRaises(ValueError):
                pipeline.final_score(run)
            np.savez(out / 'predictions.npz', ids=['a', 'b'], clean=[0, 1], satellite=[1, 1])
            pipeline.final_score(run)
            results = original_read(root / 'phase1_final_results.json')['results']
            self.assertEqual([r['accuracy'] for r in results if r['receiver'] == 'ALL'], [1.0, 0.5])
            with self.assertRaises(FileExistsError):
                pipeline.final_score(run)


if __name__ == '__main__':
    unittest.main()
