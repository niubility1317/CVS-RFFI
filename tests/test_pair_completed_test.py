import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('completed_test', Path(__file__).resolve().parents[1] / 'tools/pair_completed_test.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_truth_join_uses_ids_not_order():
    actual = mod.aligned_truth(np.array(['a', 'b']), np.array(['b', 'a']), np.array([1, 0]))
    assert actual.tolist() == [0, 1]


@pytest.mark.parametrize('pred,truth', [(['a', 'a'], ['a', 'b']), (['a', 'b'], ['a', 'a']), (['a'], ['b'])])
def test_invalid_coverage_rejected(pred, truth):
    with pytest.raises(ValueError):
        mod.aligned_truth(np.array(pred), np.array(truth), np.zeros(len(truth), dtype=int))


def test_scene_metrics_exclude_other_samples():
    m = mod.group_metrics(np.array([0, 1, 2, 2]), np.array([0, 0, 2, 0]), np.array([1, 1, 1, 0], dtype=bool), 3)
    assert m['total'] == 3 and m['correct'] == 2
    assert m['confusion_matrix'] == [[1, 0, 0], [1, 0, 0], [0, 0, 1]]


def test_score_requires_all_predictions_before_truth(tmp_path):
    with pytest.raises(RuntimeError, match='all predictions'):
        mod.score({'rows': [{'row_id': 'missing'}]}, tmp_path)


def test_output_not_overwritten(tmp_path):
    p = tmp_path / 'result.json'
    mod.write_json(p, {'first': 1})
    with pytest.raises(FileExistsError):
        mod.write_json(p, {'second': 2})
