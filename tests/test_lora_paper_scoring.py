import numpy as np
import pytest
from experiments.lora_paper_repro.score import score


def test_scores_predeclared_devices_only():
    truth = np.repeat(np.arange(30, 40), 2)
    prediction = truth.copy()
    prediction[0] = 31
    result = score(prediction, truth)
    assert result['accuracy'] == pytest.approx(.95)
    assert result['per_device_accuracy']['31'] == pytest.approx(.5)


def test_rejects_incomplete_or_invalid_labels():
    with pytest.raises(ValueError):
        score(np.array([30]), np.array([30]))
