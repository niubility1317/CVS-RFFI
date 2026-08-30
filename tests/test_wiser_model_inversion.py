from __future__ import annotations

import inspect

import torch
from torch import nn

from cvsrffi.wiser_model_inversion import invert_source_head_iq


class _SignedMeanClassifier(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        score = value.mean(dim=(1, 2)) * 8.0
        return torch.stack((score, -score), dim=1)


def test_model_inversion_has_no_source_data_input_surface() -> None:
    names = set(inspect.signature(invert_source_head_iq).parameters)

    assert "source_iq" not in names
    assert "source_loader" not in names
    assert "source_features" not in names


def test_model_inversion_is_deterministic_frozen_and_nonformal() -> None:
    model = _SignedMeanClassifier()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    first = invert_source_head_iq(
        model,
        class_ids=[0, 1],
        samples_per_class=2,
        input_channels=2,
        input_length=16,
        steps=80,
        learning_rate=0.08,
        seed=23,
        target_rms=0.35,
    )
    second = invert_source_head_iq(
        model,
        class_ids=[0, 1],
        samples_per_class=2,
        input_channels=2,
        input_length=16,
        steps=80,
        learning_rate=0.08,
        seed=23,
        target_rms=0.35,
    )

    assert torch.equal(first.pseudo_iq, second.pseudo_iq)
    assert first.pseudo_iq.shape == (4, 2, 16)
    assert first.class_ids.tolist() == [0, 0, 1, 1]
    assert first.audit["status"] == "DIAGNOSTIC_MODEL_INVERSION_NON_FORMAL"
    assert first.audit["formal_phase2_eligible"] is False
    assert first.audit["source_sample_access"] is False
    assert first.audit["source_feature_access"] is False
    assert first.pseudo_iq.abs().max().item() <= 1.0
    predicted = model(first.pseudo_iq).argmax(dim=1)
    assert torch.equal(predicted, first.class_ids)
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
