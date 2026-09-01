from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from cvsrffi.stage2_marc_ot_scoring import (
    compare_marc_ot_score_rows,
    score_marc_ot_predictions,
)


def _write_prediction(root, *, arm="R0", predictions=(0, 1, 1, 1), receipt_updates=None):
    root.mkdir()
    tokens = np.asarray(["q0", "q1", "q2", "q3"])
    probs = np.asarray(
        [[0.9, 0.1] if prediction == 0 else [0.1, 0.9] for prediction in predictions],
        dtype=np.float64,
    )
    logits = np.log(probs)
    members = {"query_tokens": tokens}
    for prefix in ("p1", "p2", "p3"):
        members[f"{prefix}_logits"] = logits
        members[f"{prefix}_predictions"] = np.asarray(predictions, dtype=np.int64)
    np.savez_compressed(root / "predictions.npz", **members)
    receipt = {
        "schema": "cvs.phase2.marc_ot.prediction_receipt.v1",
        "status": "PREDICTIONS_COMPLETE",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "outer_key": "outer",
        "capsule_id": "capsule",
        "split_id": "split",
        "receiver": "3-19",
        "scenario": "leo_clear_weak",
        "arm": arm,
        "query_rows": 4,
        "expected_query_tokens": tokens.tolist(),
        "class_registry": ["old0", "old1"],
        "query_truth_opened": False,
        "query_role_opened": False,
        "support_state_frozen_before_query": True,
        "resources": {
            "training_seconds": 1.25,
            "inference_seconds": 0.5,
            "peak_rss_bytes": 1024,
            "peak_cuda_bytes": 2048,
            "peak_rss_status": "MEASURED",
            "peak_cuda_status": "MEASURED",
            "trainable_parameter_count": 8,
        },
    }
    if receipt_updates:
        receipt.update(receipt_updates)
    (root / "prediction_receipt.json").write_text(
        json.dumps(receipt), encoding="utf-8"
    )
    return root


def _truth(path):
    path.write_text(
        json.dumps(
            {
                "receiver": "3-19",
                "capsule_id": "capsule",
                "split_id": "split",
                "rows": [
                    {"query_token": "q0", "true_class_index": 0},
                    {"query_token": "q1", "true_class_index": 0},
                    {"query_token": "q2", "true_class_index": 1},
                    {"query_token": "q3", "true_class_index": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_truth_last_score_outputs_absolute_metrics_per_class_and_resources(tmp_path) -> None:
    root = _write_prediction(tmp_path / "prediction")
    score = score_marc_ot_predictions(root, _truth(tmp_path / "truth.json"))

    p3 = score["probes"]["P3_OLD_D92"]
    assert p3["accuracy"] == pytest.approx(0.75)
    assert p3["balanced_accuracy"] == pytest.approx(0.75)
    assert p3["floor"] == pytest.approx(0.5)
    assert p3["macro_f1"] == pytest.approx((2.0 / 3.0 + 0.8) / 2.0)
    assert p3["nll"] == pytest.approx(
        (-math.log(0.9) - math.log(0.1) - math.log(0.9) - math.log(0.9)) / 4.0
    )
    assert p3["per_class_accuracy"] == {"old0": 0.5, "old1": 1.0}
    assert score["resources"]["training_seconds"] == 1.25
    assert score["truth_join_after_prediction_only"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_both_status",
        "missing_peak_rss_status",
        "missing_peak_cuda_status",
        "measured_zero_rss",
    ),
)
def test_scorer_rejects_missing_resource_status_or_measured_zero(tmp_path, mutation) -> None:
    root = _write_prediction(tmp_path / "prediction")
    receipt_path = root / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "missing_both_status":
        del receipt["resources"]["peak_rss_status"]
        del receipt["resources"]["peak_cuda_status"]
    elif mutation.startswith("missing_"):
        del receipt["resources"][mutation.removeprefix("missing_")]
    else:
        receipt["resources"]["peak_rss_bytes"] = 0
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="resource"):
        score_marc_ot_predictions(root, _truth(tmp_path / "truth.json"))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"status": "PARTIAL"}, "complete"),
        ({"query_truth_opened": True}, "truth-last"),
        ({"support_state_frozen_before_query": False}, "support"),
        ({"capsule_id": "wrong"}, "binding"),
    ],
)
def test_scorer_rejects_incomplete_or_unbound_receipts(tmp_path, updates, message) -> None:
    root = _write_prediction(tmp_path / "prediction", receipt_updates=updates)
    with pytest.raises(ValueError, match=message):
        score_marc_ot_predictions(root, _truth(tmp_path / "truth.json"))


def test_paired_score_reports_help_harm_and_da1_minus_da0(tmp_path) -> None:
    control = score_marc_ot_predictions(
        _write_prediction(tmp_path / "control", arm="R0", predictions=(0, 1, 1, 1)),
        _truth(tmp_path / "truth.json"),
    )
    candidate = score_marc_ot_predictions(
        _write_prediction(tmp_path / "candidate", arm="R8", predictions=(0, 0, 0, 1)),
        tmp_path / "truth.json",
    )
    paired = compare_marc_ot_score_rows(control, candidate)

    assert paired["comparison_state"] == "DA1_REG0-DA0_REG0"
    assert paired["control_arm"] == "R0"
    assert paired["candidate_arm"] == "R8"
    p3 = paired["probes"]["P3_OLD_D92"]
    assert p3["help_count"] == 1
    assert p3["harm_count"] == 1
    assert p3["net_help_minus_harm"] == 0
    assert p3["balanced_accuracy_delta_pp"] == pytest.approx(0.0)


def test_scorer_rejects_missing_prediction_member(tmp_path) -> None:
    root = _write_prediction(tmp_path / "prediction")
    np.savez_compressed(root / "predictions.npz", query_tokens=np.asarray(["q0"]))
    with pytest.raises(ValueError, match="incomplete"):
        score_marc_ot_predictions(root, _truth(tmp_path / "truth.json"))


def test_stage2b_score_ignores_unreferenced_target_new_truth_rows(tmp_path) -> None:
    root = _write_prediction(tmp_path / "prediction")
    truth_path = _truth(tmp_path / "truth.json")
    payload = json.loads(truth_path.read_text(encoding="utf-8"))
    payload["rows"].append(
        {
            "query_token": "new-only-token",
            "true_class_index": 2,
            "evaluation_role": "target_new",
        }
    )
    truth_path.write_text(json.dumps(payload), encoding="utf-8")
    score = score_marc_ot_predictions(root, truth_path)
    assert score["query_rows"] == 4
    assert score["registration_state"] == "REG0"


def test_stage2b_score_filters_predicted_target_new_rows_from_reg0_metrics(tmp_path) -> None:
    root = _write_prediction(tmp_path / "prediction")
    with np.load(root / "predictions.npz", allow_pickle=False) as source:
        members = {name: np.asarray(source[name]) for name in source.files}
    members["query_tokens"] = np.append(members["query_tokens"], "q4")
    for prefix in ("p1", "p2", "p3"):
        members[f"{prefix}_predictions"] = np.append(
            members[f"{prefix}_predictions"], 0
        )
        members[f"{prefix}_logits"] = np.vstack(
            (members[f"{prefix}_logits"], np.log([0.9, 0.1]))
        )
    np.savez_compressed(root / "predictions.npz", **members)
    receipt_path = root / "prediction_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["query_rows"] = 5
    receipt["expected_query_tokens"].append("q4")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    truth_path = _truth(tmp_path / "truth.json")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["rows"].append(
        {
            "query_token": "q4",
            "true_class_index": 2,
            "evaluation_role": "target_new",
        }
    )
    truth_path.write_text(json.dumps(truth), encoding="utf-8")

    score = score_marc_ot_predictions(root, truth_path)

    assert score["query_rows"] == 4
    assert score["total_query_rows"] == 5
    assert score["ignored_non_old_query_rows"] == 1
    assert score["query_tokens"] == ["q0", "q1", "q2", "q3"]


def test_paired_score_rejects_stored_metric_that_disagrees_with_pairing(tmp_path) -> None:
    control = score_marc_ot_predictions(
        _write_prediction(tmp_path / "control", arm="R0"),
        _truth(tmp_path / "truth.json"),
    )
    candidate = score_marc_ot_predictions(
        _write_prediction(tmp_path / "candidate", arm="R8"),
        tmp_path / "truth.json",
    )
    tampered = copy.deepcopy(candidate)
    tampered["probes"]["P3_OLD_D92"]["accuracy"] = 1.0
    with pytest.raises(ValueError, match="disagrees"):
        compare_marc_ot_score_rows(control, tampered)
