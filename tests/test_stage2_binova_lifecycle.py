from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import cvsrffi.stage2_binova_lifecycle as lifecycle
from cvsrffi.stage2_binova_da import NOVA_DA_Module, NOVA_DA_State, NOVA_DA_Config
from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVAQuery, BiNOVASupport
from cvsrffi.stage2_binova_lifecycle import (
    evaluate_stage_a_continuation_gate,
    freeze_binova_support_states,
    predict_binova_query_read_only,
    select_binova_mode,
)


@dataclass(frozen=True)
class _FakeD92:
    class_ids: tuple[int, ...]
    coefficient: np.ndarray

    def score(self, identity160: np.ndarray, fft96: np.ndarray) -> np.ndarray:
        joined = np.concatenate([identity160[:, :1], fft96[:, :1]], axis=1)
        return joined @ self.coefficient


def _features(rows: int, prefix: str) -> BiNOVAFeatures:
    rng = np.random.default_rng(rows)
    return BiNOVAFeatures(
        identity160=rng.normal(size=(rows, 160)).astype(np.float32),
        late_time160=rng.normal(size=(rows, 160)).astype(np.float32),
        domain160=rng.normal(size=(rows, 160)).astype(np.float32),
        fft96=rng.normal(size=(rows, 96)).astype(np.float32),
        physical6=rng.normal(size=(rows, 6)).astype(np.float32),
        physical_ids=tuple(f"{prefix}-{index}" for index in range(rows)),
    )


def _context() -> dict[str, str]:
    return {
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "cap-life",
        "split_id": "split-life",
    }


def _stage_a() -> NOVA_DA_State:
    module = NOVA_DA_Module()
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return NOVA_DA_State(
        module=module,
        config=NOVA_DA_Config(steps=1),
        domain_context166=np.zeros(166, dtype=np.float32),
        audit={"query_rows_used": 0, "non_affine_fraction": 0.25},
    )


def test_stage_a_gate_and_fallback_are_support_only() -> None:
    control = {"pseudo_h": 0.70, "pseudo_forgetting": 0.03, "pseudo_old_floor": 0.60}
    passing = {"pseudo_h": 0.706, "pseudo_forgetting": 0.03, "pseudo_old_floor": 0.60}
    gate = evaluate_stage_a_continuation_gate(control, passing, non_affine_fraction=0.25)
    assert gate.passed is True and gate.query_rows_used == 0
    assert select_binova_mode(stage_a_gate_passed=True, stage_b_gate_passed=True) == "S2"
    assert select_binova_mode(stage_a_gate_passed=True, stage_b_gate_passed=False) == "S1"
    assert select_binova_mode(stage_a_gate_passed=False, stage_b_gate_passed=True) == "S0"


def test_query_cannot_open_before_support_states_are_frozen() -> None:
    query = BiNOVAQuery(features=_features(3, "query"), context=_context())
    with pytest.raises((TypeError, ValueError), match="frozen"):
        predict_binova_query_read_only(None, query)


def test_four_states_are_frozen_before_query_and_prediction_is_read_only(monkeypatch) -> None:
    old = BiNOVASupport(
        features=_features(60, "old"), labels=np.repeat(np.arange(6), 10),
        ranks=np.tile(np.arange(10), 6), context=_context(),
    )
    registered_features = _features(70, "registered")
    registered = BiNOVASupport(
        features=registered_features, labels=np.repeat(np.arange(7), 10),
        ranks=np.tile(np.arange(10), 7), context=_context(),
    )

    def fake_fit(identity160, fft96, labels, *, class_ids, **kwargs):
        count = len(class_ids)
        coefficient = np.arange(2 * count, dtype=np.float32).reshape(2, count) / 10.0
        return _FakeD92(tuple(class_ids), coefficient)

    monkeypatch.setattr(lifecycle, "exact_d92_fit", fake_fit)
    frozen = freeze_binova_support_states(
        old, registered, stage_a=_stage_a(), stage_b=None,
        selected_mode="S1", seed=9, device="cpu",
    )
    assert tuple(frozen.states) == ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
    before = {name: state.coefficient.copy() for name, state in frozen.states.items()}
    query = BiNOVAQuery(features=_features(5, "query"), context=_context())
    result = predict_binova_query_read_only(frozen, query)
    assert set(result) == {"DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1"}
    assert result["DA0_REG0"]["new_accuracy"] == "N/A"
    for name, state in frozen.states.items():
        np.testing.assert_array_equal(state.coefficient, before[name])
