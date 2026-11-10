import pytest
from pathlib import Path

from experiments.pamf_branch_training.weak_evaluation import score
from experiments.pamf_branch_training.prepare import score_dependencies


def test_weak_scorer_rejects_source_handles_before_reading_truth():
    with pytest.raises(ValueError, match='Independent scorer'):
        score(dict(source_output='/forbidden', truth='/not-opened'))


def test_new_model_truth_waits_for_practical_and_weak_predictions():
    root = Path('row')
    assert score_dependencies(root, True) == [str(root / 'predictions/complete.json'),
                                              str(root / 'weak_predictions/complete.json')]
    assert score_dependencies(root, False) == [str(root / 'weak_predictions/complete.json')]
