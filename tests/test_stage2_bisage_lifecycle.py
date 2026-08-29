from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

import cvsrffi.stage2_bisage_lifecycle as lifecycle
from cvsrffi.stage2_binova_features import BiNOVAFeatures, BiNOVAQuery, BiNOVASupport
from cvsrffi.stage2_bisage_da import SAGEDConfig, SAGEDModule, SAGEDState
from cvsrffi.stage2_bisage_lifecycle import (
    BiSAGELifecycleError,
    FrozenBiSAGEStates,
    next_stage,
    freeze_bisage_support_states,
    predict_bisage_query_read_only,
    select_mode_from_metrics,
)


class _Head:
    def __init__(self, class_count: int) -> None:
        self.class_ids = tuple(range(class_count))

    def score(self, identity: np.ndarray, fft: np.ndarray) -> np.ndarray:
        rows = len(identity)
        logits = np.zeros((rows, len(self.class_ids)), dtype=np.float32)
        logits[:, 0] = 1.0
        return logits


def _query(ids: tuple[str, ...] = ("q0", "q1")) -> BiNOVAQuery:
    rows = len(ids)
    zeros160 = np.zeros((rows, 160), dtype=np.float32)
    features = BiNOVAFeatures(
        identity160=zeros160,
        late_time160=zeros160,
        domain160=zeros160,
        fft96=np.zeros((rows, 96), dtype=np.float32),
        physical6=np.zeros((rows, 6), dtype=np.float32),
        physical_ids=ids,
    )
    return BiNOVAQuery(
        features=features,
        context={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "cap-life",
            "split_id": "split-life",
        },
    )


def _frozen() -> FrozenBiSAGEStates:
    module = SAGEDModule(late_rank=2, identity_rank=2, context_dim=2)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    stage_a = SAGEDState(
        module=module,
        config=SAGEDConfig(
            steps=1, late_rank=2, identity_rank=2, context_dim=2, covariance_rank=2
        ),
        domain_context166=np.zeros(166),
        audit={"selected_mode": "S1_CANDIDATE", "query_rows_used": 0},
    )
    return FrozenBiSAGEStates(
        states=MappingProxyType(
            {
                "DA0_REG0": _Head(6),
                "DA1_REG0": _Head(6),
                "DA0_REG1": _Head(7),
                "DA1_REG1": _Head(7),
            }
        ),
        stage_a=stage_a,
        stage_b=None,
        selected_mode="S1",
        old_class_count=6,
        support_physical_ids=frozenset({"s0"}),
        context_binding=("p2_min_v1", "VALIDATED_ONCE", "cap-life", "split-life"),
    )


def test_stage_b_runs_only_after_stage_a_pass() -> None:
    assert next_stage({"stage_a_gate_passed": True}) == "STAGE_B"
    assert next_stage({"stage_a_gate_passed": False}) == "STOPPED_SCIENTIFIC_GATE"


def test_support_only_mode_selection_uses_s2_s1_s0_order() -> None:
    baseline = {
        "old_accuracy": 0.7,
        "new_accuracy": 0.6,
        "h": 0.646,
        "old_floor": 0.5,
        "new_floor": 0.5,
        "old_to_new": 0.2,
        "positive_definite": True,
    }
    s1 = dict(baseline, h=0.65)
    s2 = dict(baseline, old_accuracy=0.72, h=0.66, old_to_new=0.1)
    assert select_mode_from_metrics(baseline, s1, s2) == "S2"
    unsafe_s2 = dict(s2, old_floor=0.4)
    assert select_mode_from_metrics(baseline, s1, unsafe_s2) == "S1"
    unsafe_s1 = dict(s1, h=0.60)
    assert select_mode_from_metrics(baseline, unsafe_s1, unsafe_s2) == "S0"


def test_query_prediction_never_calls_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("fit opened during query prediction")

    monkeypatch.setattr(lifecycle, "fit_sage_d", forbidden)
    monkeypatch.setattr(lifecycle, "fit_sage_r", forbidden)
    output = predict_bisage_query_read_only(_frozen(), _query())
    assert tuple(output) == ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
    assert output["DA0_REG0"]["new_accuracy"] == "N/A"
    assert output["DA1_REG1"]["new_accuracy"] == "PENDING_TRUTH"


def test_query_rejects_support_physical_id_overlap() -> None:
    with pytest.raises(BiSAGELifecycleError, match="disjoint"):
        predict_bisage_query_read_only(_frozen(), _query(("s0",)))


def _support(class_count: int, *, seed: int) -> BiNOVASupport:
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(class_count), 10)
    rows = len(labels)
    identity = rng.normal(size=(rows, 160)).astype(np.float32)
    identity += labels[:, None].astype(np.float32) * 0.15
    features = BiNOVAFeatures(
        identity160=identity,
        late_time160=rng.normal(size=(rows, 160)).astype(np.float32),
        domain160=rng.normal(size=(rows, 160)).astype(np.float32),
        fft96=rng.normal(size=(rows, 96)).astype(np.float32),
        physical6=rng.normal(size=(rows, 6)).astype(np.float32),
        physical_ids=tuple(f"s{class_count}_{index}" for index in range(rows)),
    )
    return BiNOVASupport(
        features=features,
        labels=labels,
        ranks=np.tile(np.arange(10), class_count),
        context={
            "protocol_schema": "p2_min_v1",
            "phase2_data_status": "VALIDATED_ONCE",
            "capsule_id": "cap-freeze",
            "split_id": "split-freeze",
        },
    )


def test_support_freeze_builds_exact_four_state_heads() -> None:
    old = _support(6, seed=11)
    registered = _support(8, seed=12)
    stage_a = _frozen().stage_a
    frozen = freeze_bisage_support_states(
        old,
        registered,
        stage_a=stage_a,
        stage_b=None,
        selected_mode="S1",
        seed=7,
        device="cpu",
    )
    assert tuple(frozen.states) == ("DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1")
    assert len(frozen.states["DA0_REG0"].class_ids) == 6
    assert len(frozen.states["DA1_REG1"].class_ids) == 8
