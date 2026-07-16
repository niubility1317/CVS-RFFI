from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from paper_reproduction.cvs_aligned.k1_support_trust import (
    DEFAULT_ALPHA_GRID,
    leave_one_group_margins,
    scale_lora_trainable_state,
    select_largest_safe_support_scale,
)


def _base_features() -> np.ndarray:
    base = np.asarray(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.98, 0.03, 0.0], [0.02, 0.99, 0.0], [0.0, 0.02, 0.99]],
            [[0.99, 0.0, 0.02], [0.0, 0.98, 0.03], [0.02, 0.0, 0.98]],
        ],
        dtype=np.float32,
    )
    return base


def _feature_path(endpoint: np.ndarray) -> dict[float, np.ndarray]:
    base = _base_features()
    return {
        alpha: ((1.0 - alpha) * base + alpha * endpoint).astype(np.float32)
        for alpha in DEFAULT_ALPHA_GRID
    }


def test_signature_has_no_query_or_role_inputs() -> None:
    names = set(inspect.signature(select_largest_safe_support_scale).parameters)
    assert not any("query" in name for name in names)
    assert not any("role" in name for name in names)
    assert not any("quota" in name for name in names)


def test_leave_one_group_margin_is_class_permutation_equivariant() -> None:
    base = _base_features()
    permutation = np.asarray([2, 0, 1])
    original = leave_one_group_margins(base)
    permuted = leave_one_group_margins(base[:, permutation])
    np.testing.assert_allclose(permuted, original[:, permutation], atol=1.0e-6)


def test_selects_largest_support_safe_nonzero_scale() -> None:
    base = _base_features()
    endpoint = base.copy()
    endpoint[0, 0] = np.asarray([0.82, 0.40, 0.0], dtype=np.float32)
    decision = select_largest_safe_support_scale(
        base,
        _feature_path(endpoint),
        worst_group_tolerance=0.08,
        mean_margin_tolerance=0.02,
        mean_cosine_drift_cap=0.03,
    )
    safe = [row["alpha"] for row in decision.rows if row["safe"] and row["alpha"] > 0]
    assert safe
    assert decision.selected_alpha == max(safe)
    assert decision.status == "support_safe_nonzero_delta"
    assert decision.query_rows_used == 0
    assert decision.role_labels_used is False
    assert decision.class_quota_used is False


def test_falls_back_to_p4_identity_when_every_nonzero_scale_is_unsafe() -> None:
    base = _base_features()
    path = {0.0: base.copy()}
    for alpha in DEFAULT_ALPHA_GRID[1:]:
        broken = base.copy()
        broken[0, 0] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        path[alpha] = broken
    decision = select_largest_safe_support_scale(
        base,
        path,
        worst_group_tolerance=0.0,
        mean_margin_tolerance=0.0,
        mean_cosine_drift_cap=0.0,
    )
    assert decision.selected_alpha == 0.0
    assert decision.status == "fallback_p4_identity_alpha_zero"
    assert all(not row["safe"] for row in decision.rows if row["alpha"] > 0.0)


def test_alpha_zero_must_exactly_reproduce_identity_features() -> None:
    base = _base_features()
    path = _feature_path(base)
    path[0.0] = path[0.0] + 1.0e-3
    with pytest.raises(ValueError, match="alpha=0"):
        select_largest_safe_support_scale(base, path)


def test_scale_lora_state_scales_only_b_factor() -> None:
    state = {
        "id_gate.lora_a.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float16),
        "id_gate.lora_b.weight": torch.tensor([[4.0], [6.0]], dtype=torch.float16),
        "joint_proj.lora_a.weight": torch.tensor([[3.0, 5.0]], dtype=torch.float16),
        "joint_proj.lora_b.weight": torch.tensor([[8.0], [10.0]], dtype=torch.float16),
    }
    scaled = scale_lora_trainable_state(state, 0.25)
    torch.testing.assert_close(scaled["id_gate.lora_a.weight"], state["id_gate.lora_a.weight"])
    torch.testing.assert_close(scaled["joint_proj.lora_a.weight"], state["joint_proj.lora_a.weight"])
    torch.testing.assert_close(
        scaled["id_gate.lora_b.weight"],
        torch.tensor([[1.0], [1.5]], dtype=torch.float16),
    )
    torch.testing.assert_close(
        scaled["joint_proj.lora_b.weight"],
        torch.tensor([[2.0], [2.5]], dtype=torch.float16),
    )


def test_alpha_zero_removes_composed_lora_residual() -> None:
    state = {
        "layer.lora_a.weight": torch.ones((2, 3), dtype=torch.float32),
        "layer.lora_b.weight": torch.ones((4, 2), dtype=torch.float32),
    }
    scaled = scale_lora_trainable_state(state, 0.0)
    delta = scaled["layer.lora_b.weight"] @ scaled["layer.lora_a.weight"]
    torch.testing.assert_close(delta, torch.zeros_like(delta))
