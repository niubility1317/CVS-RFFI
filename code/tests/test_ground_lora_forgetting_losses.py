from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for path in (str(ROOT), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)

from train_apply_phase1_iq_preadapter_20260703 import (
    _validate_source_only_ground_lora_mode,
    nested_k_worst_prototype_risk,
    prototype_gram_deconfusion_loss,
    relation_gram_preservation_loss,
)


def test_effective_ground_lora_must_stop_before_target_export() -> None:
    with pytest.raises(ValueError, match="requires --source_only_ground_lora"):
        _validate_source_only_ground_lora_mode(
            argparse.Namespace(
                model_adapter_mode="lora_effective_feature",
                source_only_ground_lora=False,
            )
        )
    _validate_source_only_ground_lora_mode(
        argparse.Namespace(
            model_adapter_mode="lora_effective_feature",
            source_only_ground_lora=True,
        )
    )


def test_source_only_switch_cannot_hide_a_nonformal_adapter_export() -> None:
    with pytest.raises(ValueError, match="reserved for lora_effective_feature"):
        _validate_source_only_ground_lora_mode(
            argparse.Namespace(
                model_adapter_mode="lora_full_feature",
                source_only_ground_lora=True,
            )
        )


def test_relation_loss_is_zero_for_identical_geometry_and_positive_after_drift() -> None:
    reference = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32
    )
    same = relation_gram_preservation_loss(reference.clone(), reference)
    drifted = reference.clone()
    drifted[1] = torch.tensor([1.0, 1.0])
    changed = relation_gram_preservation_loss(drifted, reference)
    assert float(same) == 0.0
    assert float(changed) > 0.0


def test_prototype_gram_loss_targets_only_crowded_classes() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    separated = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    crowded = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.95, 0.05], [0.95, 0.05]]
    )
    assert float(prototype_gram_deconfusion_loss(separated, labels)) == 0.0
    assert float(prototype_gram_deconfusion_loss(crowded, labels)) > 0.0


def test_nested_k_risk_is_query_free_differentiable_and_tracks_k() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [1.0, -0.1],
            [0.0, 1.0],
            [0.1, 0.9],
            [-0.1, 1.0],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 0, 1, 1, 1])
    risk, by_k = nested_k_worst_prototype_risk(
        features, labels, k_values=(1, 2, 5), risk_tau=0.2
    )
    assert set(by_k) == {1, 2}
    assert torch.isfinite(risk)
    risk.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
