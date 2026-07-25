from __future__ import annotations

import copy

import numpy as np
import pytest

from cvsrffi.rxid_metabias4_held_falsifier import (
    D103HeldFalsifierError,
    aggregate_tx_probe_fold,
    build_complete_fit_plan,
    evaluate_complete_gate,
    probe_partition,
    run_fixed_tx_probe,
)


RECEIVERS = [f"r{i}" for i in range(7)]
CLASSES = [f"c{i}" for i in range(6)]
DAYS = [f"d{i}" for i in range(4)]


def _passing_performance_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    keys = [(receiver, None, k) for receiver in RECEIVERS for k in (1, 5, 10)]
    keys += [(receiver, class_id, 1) for receiver in RECEIVERS for class_id in CLASSES]
    for receiver, class_id, k_shot in keys:
        rows.append(
            {
                "held_receiver": receiver,
                "held_class": class_id,
                "K": k_shot,
                "base_ba": 0.80,
                "adapted_ba": 0.81,
                "base_floor": 0.60,
                "adapted_floor": 0.61,
                "wrong_to_correct": 2,
                "correct_to_wrong": 1,
                "joint_score_d102": 0.70,
                "joint_score_d103": 0.71,
                "prediction_artifact_committed_before_truth": True,
                "d102_prediction_committed_before_truth": True,
                "truth_open_event_sha256": "a" * 64,
                "d102_comparator_status": (
                    "DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE"
                ),
                "d102_bundle_content_root_sha256": "1" * 64,
                "active": True,
                "information_rank": 4,
                "min_singular_value": 0.10,
                "condition_number": 2.0,
                "prior_fraction": 0.50,
                "coefficient_norm": 0.01,
                "view_top1_agreement": 1.0,
                "view_large_margin_flip_count": 0,
                "direction_cosine_median": 0.90,
                "k1_receipt_evidence_scope": "support_only_no_held_query",
            }
        )
    return rows


def _stability_rows() -> list[dict[str, object]]:
    keys = [(receiver, None) for receiver in RECEIVERS]
    keys += [
        (receiver, class_id)
        for receiver in RECEIVERS
        for class_id in CLASSES
    ]
    return [
        {
            "held_receiver": receiver,
            "held_class": class_id,
            "outer_shift_norm": 0.01,
            "day_shift_norms": [0.011, 0.012, 0.013, 0.014],
            "direction_cosines": [0.88, 0.89, 0.91, 0.92],
            "direction_cosine_median": 0.90,
            "actual_160d_shift_used": True,
            "query_rows_used": 0,
        }
        for receiver, class_id in keys
    ]


def _d102_provenance() -> dict[str, object]:
    keys = [(receiver, None) for receiver in RECEIVERS]
    keys += [
        (receiver, class_id)
        for receiver in RECEIVERS
        for class_id in CLASSES
    ]
    return {
        "status": "DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE",
        "fold_count": 49,
        "folds": [
            {
                "held_receiver": receiver,
                "held_class": class_id,
                "bundle_content_root_sha256": "1" * 64,
                "l_s_physical_root_sha256": "2" * 64,
                "query_rows_used_for_fit": 0,
            }
            for receiver, class_id in keys
        ],
        "original_rejected_receipt_sha256": (
            "01a45e11fe519389071cf1eb279d293c958fc4fa48e0ed4c51bea9ff20c536b2"
        ),
        "method_lock_sha256": (
            "9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f"
        ),
        "code_sha256": "5" * 64,
        "built_before_source_validation_open": True,
        "target_access": False,
        "formal_query_access": False,
    }


def _probe_rows() -> list[dict[str, object]]:
    return [
        {
            "held_receiver": receiver,
            "fold_score": 0.20,
            "asset_frozen_before_probe": True,
            "probe_state_returned_to_asset": False,
        }
        for receiver in RECEIVERS
    ]


def _quantization() -> dict[str, object]:
    return {
        "top1_agreement": 0.999,
        "large_margin_flip_count": 0,
        "persistent_fp_sidecar": False,
        "learning_arrays_int8_only": True,
    }


def _resource() -> dict[str, object]:
    return {
        "total_gpu_hours": 29.0,
        "peak_memory_bytes": 1024**3,
        "run_root_bytes": 1024**3,
        "phase2_state_bytes": 40_000,
        "post_backbone_mac_per_query": 200_000,
        "completed_fit_count": 246,
        "completed_meta_steps": 98_400,
    }


def _access() -> dict[str, object]:
    return {
        "protocol_schema": "p2_min_v1",
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_validation_ratio": 0.30,
        "u_s_tx_label_access": False,
        "source_validation_gradient_access": False,
        "source_validation_asset_access": False,
        "target_access": False,
        "formal_query_access": False,
        "query_fit_rows": 0,
        "derived_from_fit_access_receipt_count": 246,
        "all_fit_manifests_identity_bound": True,
    }


def _evaluate(
    rows=None,
    probes=None,
    quant=None,
    resource=None,
    access=None,
    stability=None,
    d102=None,
):
    return evaluate_complete_gate(
        receiver_ids=RECEIVERS,
        class_ids=CLASSES,
        performance_rows=rows or _passing_performance_rows(),
        day_stability_rows=stability or _stability_rows(),
        d102_provenance=d102 or _d102_provenance(),
        tx_probe_rows=probes or _probe_rows(),
        quantization_receipt=quant or _quantization(),
        resource_receipt=resource or _resource(),
        access_receipt=access or _access(),
    )


def test_complete_fit_plan_has_246_non_overwriting_specs() -> None:
    plan = build_complete_fit_plan(RECEIVERS, CLASSES, DAYS)
    assert len(plan) == 246
    assert len({row.fit_id for row in plan}) == 246
    assert sum(row.fit_stage == "final" for row in plan) == 1
    assert sum(row.fit_stage == "outer" for row in plan) == 49
    assert sum(row.fit_stage == "leave_one_day" for row in plan) == 196


def test_probe_partition_is_cellwise_and_physical_disjoint() -> None:
    receiver = []
    day = []
    label = []
    physical = []
    for receiver_id in ("r0", "r1"):
        for day_id in ("d0", "d1"):
            for class_id in ("c0", "c1"):
                for index in range(5):
                    receiver.append(receiver_id)
                    day.append(day_id)
                    label.append(class_id)
                    physical.append(f"{receiver_id}-{day_id}-{class_id}-{index}")
    train, test = probe_partition(
        np.asarray(receiver),
        np.asarray(day),
        np.asarray(label),
        np.asarray(physical),
    )
    assert len(train) == 24
    assert len(test) == 16
    assert not set(np.asarray(physical)[train]) & set(np.asarray(physical)[test])


def test_probe_capacity_aggregation_uses_max_over_pooled_and_days() -> None:
    receipts = []
    for index in range(9):
        receipts.append(
            {
                "capacity_id": f"p{index}",
                "pooled_ba": 0.20,
                "per_day_ba": [0.21, 0.22, 0.23, 0.24 if index == 8 else 0.22],
                "probe_train_test_physical_disjoint": True,
            }
        )
    result = aggregate_tx_probe_fold(receipts)
    assert result["capacity_count"] == 9
    assert result["fold_score"] == 0.24


def test_fixed_tx_probe_runs_all_preregistered_capacities() -> None:
    rng = np.random.default_rng(103713)
    encoded = []
    receiver = []
    day = []
    label = []
    physical = []
    for day_index, day_id in enumerate(DAYS):
        for class_index, class_id in enumerate(CLASSES):
            for sample in range(5):
                row = rng.normal(0.0, 0.05, size=32)
                row[class_index] += 0.2
                encoded.append(row)
                receiver.append("held")
                day.append(day_id)
                label.append(class_id)
                physical.append(f"{day_id}-{class_id}-{sample}")
    result = run_fixed_tx_probe(
        np.asarray(encoded),
        np.asarray(receiver),
        np.asarray(day),
        np.asarray(label),
        np.asarray(physical),
    )
    assert result["capacity_count"] == 9
    assert result["train_physical_count"] == 72
    assert result["test_physical_count"] == 48
    assert 0.0 <= result["fold_score"] <= 1.0
    assert result["probe_state_returned_to_asset"] is False


def test_complete_gate_accepts_only_full_passing_receipts() -> None:
    result = _evaluate()
    assert result["status"] == "PHASE1_HELD_ACCEPT"
    assert result["target25_gate_eligible"] is True
    assert result["target25_authorized"] is False
    assert result["performance_row_count"] == 63
    assert result["rejection_reasons"] == []


@pytest.mark.parametrize(
    ("kind", "reason"),
    [
        ("performance", "BA_NEGATIVE"),
        ("k1", "K1_INACTIVE_OR_UNIDENTIFIED"),
        ("tx", "TX_PROBE_LIMIT_EXCEEDED"),
        ("quant", "QUANTIZATION_GATE_FAILED"),
        ("resource", "RESOURCE_GATE_FAILED"),
        ("access", "ACCESS_GATE_FAILED"),
    ],
)
def test_complete_gate_rejects_each_hard_gate(kind: str, reason: str) -> None:
    rows = _passing_performance_rows()
    probes = _probe_rows()
    quant = _quantization()
    resource = _resource()
    access = _access()
    if kind == "performance":
        rows[0]["adapted_ba"] = 0.79
    elif kind == "k1":
        rows[0]["active"] = False
    elif kind == "tx":
        probes[0]["fold_score"] = 0.26
    elif kind == "quant":
        quant["top1_agreement"] = 0.99
    elif kind == "resource":
        resource["completed_fit_count"] = 245
    elif kind == "access":
        access["u_s_tx_label_access"] = True
    result = _evaluate(rows, probes, quant, resource, access)
    assert result["status"] == "PHASE1_HELD_REJECT"
    assert result["target25_gate_eligible"] is False
    assert result["target25_authorized"] is False
    assert any(value.startswith(reason) for value in result["rejection_reasons"])


def test_incomplete_performance_coverage_fails_closed() -> None:
    with pytest.raises(D103HeldFalsifierError, match="exactly 63"):
        _evaluate(rows=_passing_performance_rows()[:-1])


def test_truth_open_event_must_be_shared_by_all_rows() -> None:
    rows = _passing_performance_rows()
    rows[-1]["truth_open_event_sha256"] = "b" * 64
    with pytest.raises(D103HeldFalsifierError, match="one immutable"):
        _evaluate(rows=rows)


def test_leave_day_and_d102_provenance_are_mandatory_and_matched() -> None:
    with pytest.raises(D103HeldFalsifierError, match="exactly 49"):
        _evaluate(stability=_stability_rows()[:-1])
    provenance = _d102_provenance()
    provenance["built_before_source_validation_open"] = False
    with pytest.raises(D103HeldFalsifierError, match="diagnostic provenance"):
        _evaluate(d102=provenance)
    rows = _passing_performance_rows()
    rows[0]["d102_bundle_content_root_sha256"] = "9" * 64
    with pytest.raises(D103HeldFalsifierError, match="invalid performance"):
        _evaluate(rows=rows)
    provenance = _d102_provenance()
    provenance["method_lock_sha256"] = "4" * 64
    with pytest.raises(D103HeldFalsifierError, match="frozen parent"):
        _evaluate(d102=provenance)
