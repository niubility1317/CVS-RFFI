from __future__ import annotations

import inspect
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_bundle import (
    build_rxid_metabias4_bundle,
)
from cvsrffi.stage2_d104_angq_qknn import ANGQ_SCHEMA
from cvsrffi.stage2_d104_rxid_angq import (
    ARMS,
    D104RXIDANGQError,
    audit_d104_four_arm_int8,
    build_d104_prediction_artifact,
    fit_d104_four_arm_state,
    predict_d104_four_arm_logits,
)
from cvsrffi.stage2_rxid_metabias4 import (
    ACTIVE,
    INACTIVE_NON_PROMOTABLE,
    K1IdentifiabilityReceipt,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
)


CLASSES = ("opaque-2", "opaque-0", "opaque-1")
HASHES = tuple(f"{index:x}" * 64 for index in range(1, 9))


def _bundle():
    rng = np.random.default_rng(103713)
    u = np.zeros((32, 160), dtype=np.float32)
    u[:, :32] = np.eye(32, dtype=np.float32)
    b = rng.normal(0.0, 0.08, (160, 4)).astype(np.float32)
    g = np.zeros((5, 32), dtype=np.float32)
    for index in range(5):
        g[index, index] = 1.0
        g[index, (index + 7) % 32] = 0.3
    t = np.asarray(
        [
            [0.15, 0.08, -0.05, 0.10],
            [0.12, 0.06, -0.03, 0.08],
            [0.10, 0.05, -0.02, 0.07],
            [0.14, 0.07, -0.04, 0.09],
            [0.11, 0.04, -0.01, 0.06],
        ],
        dtype=np.float32,
    )
    return build_rxid_metabias4_bundle(
        u,
        b,
        g,
        t,
        np.full((5, 4), 4.0, dtype=np.float32),
        np.full(5, 1.8, dtype=np.float32),
        cell_min_physical_count=np.full(5, 2, dtype=np.int16),
        cell_class_count=np.full(5, 3, dtype=np.int16),
        checkpoint_sha256=HASHES[0],
        runtime_sha256=HASHES[1],
        method_lock_sha256=HASHES[2],
        training_receipt_sha256=HASHES[3],
        nested_receipt_sha256=HASHES[4],
        tx_probe_receipt_sha256=HASHES[5],
        aggregation_receipt_sha256=HASHES[6],
        quantization_receipt_sha256=HASHES[7],
        tx_probe_mean_balanced_accuracy=0.20,
        tx_probe_max_balanced_accuracy=0.24,
    )


def _qknn(k_shot: int) -> Phase1ZIDStudentTLock:
    return Phase1ZIDStudentTLock(
        active_k=k_shot,
        student_nu=3.0,
        kernel_effective_dim=160,
        kernel_volume_gamma=1.0,
        shared_h0=0.2,
        scale_prior_strength=2.0,
        scale_min_ratio=0.5,
        scale_max_ratio=2.0,
        temperature=1.0,
        phase1_lodo_receipt_sha256="a" * 64,
        quantization_margin_audit_sha256="b" * 64,
    )


def _support(k_shot: int):
    rng = np.random.default_rng(607 + k_shot)
    pre: list[np.ndarray] = []
    zdom: list[np.ndarray] = []
    labels: list[str] = []
    for class_index, label in enumerate(CLASSES):
        for _ in range(k_shot):
            row = rng.normal(0.1, 0.15, 160).astype(np.float32)
            row[20 + class_index] += np.float32(1.0)
            domain = rng.normal(0.0, 0.03, 160).astype(np.float32)
            domain[class_index] += np.float32(1.0)
            pre.append(row)
            zdom.append(domain)
            labels.append(label)
    return (
        np.asarray(pre, dtype=np.float32),
        np.asarray(zdom, dtype=np.float32),
        tuple(labels),
    )


def _k1_receipt(**overrides):
    values = {
        "view_top1_agreement": 1.0,
        "large_margin_flip_count": 0,
        "independent_direction_cosine_median": 0.9,
        "independent_episode_count": 7,
        "receipt_sha256": "c" * 64,
    }
    values.update(overrides)
    return K1IdentifiabilityReceipt(**values)


def _fit(k_shot: int, *, receipt=None):
    pre, zdom, labels = _support(k_shot)
    state = fit_d104_four_arm_state(
        _bundle(),
        pre,
        zdom,
        labels,
        CLASSES,
        qknn_config=_qknn(k_shot),
        stage="S_C",
        support_receipt_sha256="d" * 64,
        k1_identifiability_receipt=receipt,
    )
    return state, pre, zdom, labels


@pytest.mark.parametrize("k_shot", [1, 5, 10])
def test_four_arm_state_is_matched_query_neutral_and_resource_closed(
    k_shot: int,
) -> None:
    receipt = _k1_receipt() if k_shot == 1 else None
    state, _, _, _ = _fit(k_shot, receipt=receipt)
    assert state.active_k == k_shot
    assert state.classes == CLASSES
    assert tuple(state.method_lock["arms"]) == ARMS
    assert state.method_lock["query_truth_read"] is False
    assert state.method_lock["query_state_updates"] == 0
    assert state.method_lock["target25_authorized"] is False
    assert (
        state.m_head_bank.quantization_audit["schema"]
        == ANGQ_SCHEMA
    )
    assert (
        state.m_joint_bank.quantization_audit["schema"]
        == ANGQ_SCHEMA
    )
    for resource in state.resource_receipts.values():
        assert resource["numeric_bank_array_bytes_delta"] == 0
        assert resource["query_mac_delta"] == 0
        assert resource["adaptation_mac_total"] == 32320 * len(CLASSES) * k_shot
        assert resource["passes_d104_resource_gate"] is True
    assert len(state.state_receipt_sha256) == 64


def test_four_arm_prediction_is_chunk_invariant_and_state_immutable() -> None:
    state, pre, _, _ = _fit(5)
    before = state.state_receipt_sha256
    together = predict_d104_four_arm_logits(state, pre[:8])
    first = predict_d104_four_arm_logits(state, pre[:3])
    second = predict_d104_four_arm_logits(state, pre[3:8])
    repeated = predict_d104_four_arm_logits(state, pre[:8])
    assert tuple(together) == ARMS
    for arm in ARMS:
        np.testing.assert_array_equal(
            together[arm],
            np.concatenate((first[arm], second[arm])),
        )
        np.testing.assert_array_equal(together[arm], repeated[arm])
        assert together[arm].shape == (8, len(CLASSES))
    assert state.state_receipt_sha256 == before


def test_prediction_artifact_has_four_arms_and_no_truth_surface() -> None:
    signature = inspect.signature(build_d104_prediction_artifact)
    assert tuple(signature.parameters) == (
        "state",
        "query_pre_relu",
        "query_physical_ids",
    )
    assert "truth" not in str(signature).lower()
    state, pre, _, _ = _fit(5)
    artifact = build_d104_prediction_artifact(
        state,
        pre[:7],
        [f"opaque-query-{index}" for index in range(7)],
    )
    assert tuple(artifact["arm_predictions"]) == ARMS
    assert artifact["all_four_arms_present"] is True
    assert artifact["all_registered_classes_compete"] is True
    assert artifact["query_truth_present"] is False
    assert artifact["query_rows_used_for_fit"] == 0
    assert artifact["query_state_updates"] == 0
    assert artifact["old_new_role_access"] is False
    assert artifact["class_quota_access"] is False
    assert artifact["target25_authorized"] is False
    assert len(artifact["prediction_receipt_sha256"]) == 64
    with pytest.raises(TypeError):
        build_d104_prediction_artifact(
            state,
            pre[:1],
            ["opaque-query-0"],
            query_truth=[CLASSES[0]],
        )


def test_k1_inactive_preserves_factorial_closure_without_promoting_da() -> None:
    state, pre, _, _ = _fit(1, receipt=None)
    assert state.d103_state.status == INACTIVE_NON_PROMOTABLE
    logits = predict_d104_four_arm_logits(state, pre)
    np.testing.assert_array_equal(logits["M_DA"], logits["M0"])
    np.testing.assert_array_equal(logits["M_JOINT"], logits["M_HEAD"])
    assert state.method_lock["target25_authorized"] is False


def test_k1_active_keeps_all_four_arms_and_d103_activity() -> None:
    state, pre, _, _ = _fit(1, receipt=_k1_receipt())
    assert state.d103_state.status == ACTIVE
    logits = predict_d104_four_arm_logits(state, pre)
    assert tuple(logits) == ARMS
    assert all(value.shape == (len(pre), len(CLASSES)) for value in logits.values())


def test_truth_free_int8_audit_uses_stable_registry_and_two_separate_gates() -> None:
    state, pre, _, labels = _fit(5)
    audit = audit_d104_four_arm_int8(
        state,
        pre,
        labels,
        pre,
    )
    assert set(audit) >= {
        "M_HEAD",
        "M_JOINT",
        "passes_d104_int8_gate",
        "receipt_sha256",
    }
    for arm in ("M_HEAD", "M_JOINT"):
        local = audit[arm]
        assert local["validation_row_count"] == len(pre)
        assert local["stable_winner_runner_tie_break"] == (
            "opaque_registry_index_ascending"
        )
        assert local["query_rows_used_for_fit"] == 0
        assert local["query_truth_read"] is False
        assert (
            local["shared_angq_fp16_bandwidth_direction_audit"][
                "promotion_gate"
            ]
            is False
        )
    assert audit["query_truth_read"] is False
    assert audit["query_state_updates"] == 0
    assert audit["target25_authorized"] is False


def test_four_arm_fit_has_support_only_surface() -> None:
    signature = inspect.signature(fit_d104_four_arm_state)
    assert not any(
        token in name.lower()
        for name in signature.parameters
        for token in ("query", "truth", "old", "new", "receiver", "scene")
    )


def test_duplicate_query_physical_ids_fail_closed() -> None:
    state, pre, _, _ = _fit(1, receipt=_k1_receipt())
    with pytest.raises(D104RXIDANGQError, match="unique"):
        build_d104_prediction_artifact(
            state,
            pre[:2],
            ["same", "same"],
        )


def test_real_feature_smoke_cli_has_no_truth_or_query_argument() -> None:
    script = (
        Path(__file__).resolve().parents[1]
        / "code"
        / "scripts"
        / "run_d104_r1_real_feature_noquery_smoke.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--tap-archive" in result.stdout
    assert "--dual-archive" in result.stdout
    assert "--k-shot" in result.stdout
    assert "--output-json" in result.stdout
    assert "--query" not in result.stdout.lower()
    assert "--truth" not in result.stdout.lower()
