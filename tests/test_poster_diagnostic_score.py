import numpy as np
import pytest
from experiments.poster_paper_repro.diagnostic_score import score


def test_accuracy_and_per_class_accuracy_are_computed_from_fixed_predictions():
    truth = np.repeat(np.arange(5), 2)
    prediction = truth.copy()
    prediction[0] = 1
    result = score(prediction, truth)
    assert result['accuracy'] == pytest.approx(.9)
    assert result['per_class']['0'] == pytest.approx(.5)


def test_missing_classes_are_rejected():
    with pytest.raises(ValueError):
        score(np.array([0]), np.array([0]))
