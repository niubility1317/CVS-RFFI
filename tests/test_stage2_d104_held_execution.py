from __future__ import annotations

import copy

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_bundle import build_rxid_metabias4_bundle
from cvsrffi.stage2_d104_held_execution import (
    D104HeldExecutionError,
    predict_d104_matched_row,
    score_d104_prediction_artifact,
)


SHA = "a" * 64


def _bundle():
    rng = np.random.default_rng(10)
    u = np.zeros((32, 160), dtype=np.float32)
    u[:, :32] = np.eye(32, dtype=np.float32)
    return build_rxid_metabias4_bundle(
        u,
        rng.normal(0.0, 0.08, size=(160, 4)).astype(np.float32),
        rng.normal(size=(8, 32)).astype(np.float32),
        np.zeros((8, 4), dtype=np.float32),
        np.full((8, 4), 4.0, dtype=np.float32),
        np.full(8, 1.8, dtype=np.float32),
        cell_min_physical_count=np.full(8, 6, dtype=np.int16),
        cell_class_count=np.full(8, 6, dtype=np.int16),
        checkpoint_sha256=SHA,
        runtime_sha256="b" * 64,
        method_lock_sha256="c" * 64,
        training_receipt_sha256="d" * 64,
        nested_receipt_sha256="e" * 64,
        tx_probe_receipt_sha256="f" * 64,
        aggregation_receipt_sha256="1" * 64,
        quantization_receipt_sha256="2" * 64,
        tx_probe_mean_balanced_accuracy=0.0,
        tx_probe_max_balanced_accuracy=0.0,
    )


def _row(k=5):
    rng = np.random.default_rng(104 + k)
    classes = tuple(f"c{i}" for i in range(6))
    support = np.maximum(
        rng.normal(size=(6 * k, 160)).astype(np.float32), 0.0
    )
    support[:, :6] += np.eye(6, dtype=np.float32).repeat(k, axis=0) * 4
    labels = np.repeat(classes, k)
    zdom = rng.normal(size=(6 * k, 160)).astype(np.float32)
    query = np.maximum(rng.normal(size=(24, 160)).astype(np.float32), 0.0)
    return predict_d104_matched_row(
        held_receiver="rx",
        held_class=None,
        k_shot=k,
        support_pre_relu=support,
        support_zdom=zdom,
        support_labels=labels,
        support_physical_ids=[f"s{i}" for i in range(6 * k)],
        query_pre_relu=query,
        query_physical_ids=[f"q{i}" for i in range(24)],
        registered_classes=classes,
        d103_outer_bundle=_bundle(),
        d103_day_bundles=(_bundle(),) * 4,
    )[0]


def test_d104_matched_row_has_4_arm_units_and_no_truth() -> None:
    artifact = _row(5)
    assert tuple(artifact["arm_predictions"]) == (
        "M0",
        "M_DA",
        "M_HEAD",
        "M_JOINT",
    )
    assert len(artifact["arm_prediction_receipts"]) == 4
    assert artifact["query_truth_present"] is False
    assert artifact["query_rows_used_for_fit"] == 0
    assert artifact["int8_audit"]["passes_d104_int8_gate"] is True


def test_d104_truth_side_scores_all_effects_same_row() -> None:
    artifact = _row(5)
    truth = [f"c{i % 6}" for i in range(24)]
    score = score_d104_prediction_artifact(artifact, truth)
    assert set(score["arm_metrics"]) == {
        "M0",
        "M_DA",
        "M_HEAD",
        "M_JOINT",
    }
    assert set(score["simple_effects"]) == {
        "H0_HEAD_at_base",
        "H1_HEAD_at_DA",
        "D0_DA_at_legacy",
        "D1_DA_at_ANGQ",
    }
    assert score["truth_row_count"] == 24


def test_d104_truth_side_rejects_tamper() -> None:
    artifact = _row(5)
    changed = copy.deepcopy(artifact)
    changed["arm_predictions"]["M_HEAD"][0] = "c5"
    with pytest.raises(D104HeldExecutionError):
        score_d104_prediction_artifact(
            changed, [f"c{i % 6}" for i in range(24)]
        )
