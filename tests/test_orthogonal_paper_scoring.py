import numpy as np
import pytest
from experiments.orthogonal_paper_repro.score import score_stage, score_all


def test_scores_accuracy_and_harmonic_mean_from_frozen_predictions():
    truth = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    result = score_stage(predicted, truth, old_classes=2, seen_classes=3)
    assert result['accuracy'] == pytest.approx(4 / 6)
    assert result['old_accuracy'] == pytest.approx(.75)
    assert result['new_accuracy'] == pytest.approx(.5)
    assert result['harmonic'] == pytest.approx(.6)


def test_rejects_missing_class_or_misaligned_predictions():
    with pytest.raises(ValueError):
        score_stage(np.array([0, 1]), np.array([0, 0]), old_classes=1, seen_classes=2)
    with pytest.raises(ValueError):
        score_stage(np.array([0]), np.array([0, 1]), old_classes=1, seen_classes=2)
