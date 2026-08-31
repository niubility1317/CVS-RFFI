from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

import numpy as np
import pytest

from cvsrffi.stage2_wiser_scoring import compare_wiser_score_rows, score_wiser_predictions


def _logits_for_predictions(predictions: np.ndarray, *, margin: float = 4.0) -> np.ndarray:
    logits = np.zeros((len(predictions), 6), dtype=np.float64)
    logits[np.arange(len(predictions)), predictions] = margin
    return logits


def _write_detailed_inputs(
    tmp_path: Path,
    *,
    arm: str = "B0",
    p1: np.ndarray | None = None,
    p2: np.ndarray | None = None,
    p3: np.ndarray | None = None,
    logits_margin: float = 4.0,
) -> tuple[Path, Path, Path, np.ndarray]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    tokens = np.asarray([f"q{index}" for index in range(12)])
    truth_values = np.repeat(np.arange(6), 2)
    p1 = truth_values.copy() if p1 is None else p1
    p2 = truth_values.copy() if p2 is None else p2
    p3 = truth_values.copy() if p3 is None else p3
    features = np.eye(6, dtype=np.float32)[truth_values]
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_tokens=tokens,
        p1_predictions=p1,
        p1_logits=_logits_for_predictions(p1, margin=logits_margin),
        p2_predictions=p2,
        p2_logits=_logits_for_predictions(p2, margin=logits_margin),
        p3_predictions=p3,
        p3_logits=_logits_for_predictions(p3, margin=logits_margin),
        query_z_id=features,
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_COMPLETE",
                "outer_key": "rx_3_19__seed_713102__k_10__new_5",
                "capsule_id": "capsule-v1",
                "split_id": "split-v1",
                "arm": arm,
                "receiver": "rx_3_19",
                "scenario": "leo_clear_weak",
                "query_rows": 12,
                "expected_query_tokens": tokens.tolist(),
                "query_truth_opened": False,
                "query_role_opened": False,
                "support_state_frozen_before_query": True,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "receiver": "rx_3_19",
                "rows": [
                    {"query_token": token, "true_class_index": int(class_id)}
                    for token, class_id in zip(tokens.tolist(), truth_values.tolist())
                ],
            }
        ),
        encoding="utf-8",
    )
    return predictions, receipt, truth, truth_values


def test_truth_last_scorer_aligns_tokens_and_reports_three_probes(tmp_path: Path) -> None:
    tokens = np.asarray([f"q{index}" for index in range(12)])
    truth_values = np.repeat(np.arange(6), 2)
    exact = truth_values.copy()
    p2 = exact.copy()
    p2[0] = 1
    features = np.eye(6, dtype=np.float32)[truth_values]
    features[1::2] += 0.01
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_tokens=tokens,
        p1_predictions=exact,
        p2_predictions=p2,
        p3_predictions=exact,
        query_z_id=features,
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_COMPLETE",
                "arm": "A",
                "receiver": "rx",
                "scenario": "leo_clear_weak",
                "query_rows": 12,
                "expected_query_tokens": tokens.tolist(),
                "query_truth_opened": False,
                "query_role_opened": False,
                "support_state_frozen_before_query": True,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "receiver": "rx",
                "rows": [
                    {"query_token": token, "true_class_index": int(class_id)}
                    for token, class_id in zip(tokens.tolist(), truth_values.tolist())
                ],
            }
        ),
        encoding="utf-8",
    )

    result = score_wiser_predictions(predictions, receipt, truth)

    assert result["status"] == "ANALYZED"
    assert result["truth_join_after_prediction_only"] is True
    assert result["probes"]["P1_SOURCE_HEAD"]["balanced_accuracy"] == 1.0
    assert result["probes"]["P2_SOURCE_PROTOTYPE"]["balanced_accuracy"] == pytest.approx(11 / 12)
    assert result["probes"]["P3_OLD_D92"]["floor"] == 1.0
    assert result["geometry"]["within_trace"] > 0.0
    assert result["geometry"]["between_within_ratio"] > 0.0


def test_truth_last_detailed_score_reports_absolute_metrics_nll_and_binding(
    tmp_path: Path,
) -> None:
    p2 = np.repeat(np.arange(6), 2)
    p2[0] = 1
    predictions, receipt, truth, _ = _write_detailed_inputs(tmp_path, p2=p2)

    result = score_wiser_predictions(predictions, receipt, truth)
    p2_metrics = result["probes"]["P2_SOURCE_PROTOTYPE"]

    assert result["schema"] == "cvs.phase2.wiser_rf.truth_last_score.v2"
    assert result["outer_key"] == "rx_3_19__seed_713102__k_10__new_5"
    assert result["capsule_id"] == "capsule-v1"
    assert result["split_id"] == "split-v1"
    assert result["query_tokens"] == [f"q{index}" for index in range(12)]
    assert p2_metrics["accuracy"] == pytest.approx(11 / 12)
    assert p2_metrics["balanced_accuracy"] == pytest.approx(11 / 12)
    assert p2_metrics["floor"] == pytest.approx(0.5)
    assert p2_metrics["per_class_accuracy"]["0"] == pytest.approx(0.5)
    assert p2_metrics["nll"] == pytest.approx(
        (11 * np.log1p(5 * np.exp(-4.0)) + np.log(np.exp(4.0) + 5) - 0.0) / 12
    )
    assert result["per_class_query_rows"] == {str(class_id): 2 for class_id in range(6)}
    assert result["pairing_payload"]["truth"] == np.repeat(np.arange(6), 2).tolist()
    assert result["pairing_payload"]["predictions"]["P3_OLD_D92"] == np.repeat(
        np.arange(6), 2
    ).tolist()


def test_truth_last_reg0_scores_old_roles_after_full_mixed_query_closure(
    tmp_path: Path,
) -> None:
    tokens = np.asarray([f"q{index}" for index in range(18)])
    truth_values = np.asarray(
        [0, 6, 0, 1, 7, 1, 2, 8, 2, 3, 9, 3, 4, 10, 4, 5, 6, 5],
        dtype=np.int64,
    )
    roles = np.asarray(
        ["target_old" if class_id < 6 else "target_new" for class_id in truth_values]
    )
    predictions_values = np.where(truth_values < 6, truth_values, 0)
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_tokens=tokens,
        p1_predictions=predictions_values,
        p1_logits=_logits_for_predictions(predictions_values),
        p2_predictions=predictions_values,
        p2_logits=_logits_for_predictions(predictions_values),
        p3_predictions=predictions_values,
        p3_logits=_logits_for_predictions(predictions_values),
        query_z_id=np.eye(6, dtype=np.float32)[predictions_values],
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_COMPLETE",
                "outer_key": "rx_3_19__seed_713102__k_10__new_5",
                "capsule_id": "capsule-v1",
                "split_id": "split-v1",
                "arm": "N0",
                "receiver": "rx_3_19",
                "scenario": "leo_clear_weak",
                "query_rows": 18,
                "expected_query_tokens": tokens.tolist(),
                "query_truth_opened": False,
                "query_role_opened": False,
                "support_state_frozen_before_query": True,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "receiver": "rx_3_19",
                "rows": [
                    {
                        "query_token": token,
                        "true_class_index": int(class_id),
                        "evaluation_role": role,
                    }
                    for token, class_id, role in zip(
                        tokens.tolist(), truth_values.tolist(), roles.tolist()
                    )
                ],
            }
        ),
        encoding="utf-8",
    )

    result = score_wiser_predictions(predictions, receipt, truth)

    expected_old_tokens = tokens[roles == "target_old"].tolist()
    assert result["total_query_rows"] == 18
    assert result["query_rows"] == 12
    assert result["old_query_rows"] == 12
    assert result["ignored_non_old_query_rows"] == 6
    assert result["scored_evaluation_role"] == "target_old"
    assert result["registration_state"] == "REG0"
    assert result["query_tokens"] == expected_old_tokens
    assert result["pairing_payload"]["query_tokens"] == expected_old_tokens
    assert result["per_class_query_rows"] == {str(class_id): 2 for class_id in range(6)}
    assert result["probes"]["P3_OLD_D92"]["balanced_accuracy"] == 1.0


def test_truth_last_detailed_score_uses_stable_nll_for_extreme_finite_logits(tmp_path: Path) -> None:
    predictions, receipt, truth, expected_truth = _write_detailed_inputs(
        tmp_path, logits_margin=1.0
    )
    with np.load(predictions, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    extreme = np.full((12, 6), -1.0e300, dtype=np.float64)
    extreme[np.arange(12), expected_truth] = 1.0e300
    values["p3_logits"] = extreme
    np.savez_compressed(predictions, **values)

    result = score_wiser_predictions(predictions, receipt, truth)

    assert result["probes"]["P3_OLD_D92"]["nll"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("member", "replacement", "message"),
    [
        ("p1_logits", np.zeros((12, 5), dtype=np.float64), "logit geometry"),
        ("p2_logits", np.full((12, 6), np.nan), "logits must be finite"),
        ("p3_predictions", np.full(12, 7, dtype=np.int64), "prediction index"),
    ],
)
def test_truth_last_detailed_score_rejects_invalid_probe_evidence(
    tmp_path: Path, member: str, replacement: np.ndarray, message: str
) -> None:
    predictions, receipt, truth, _ = _write_detailed_inputs(tmp_path)
    with np.load(predictions, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    values[member] = replacement
    np.savez_compressed(predictions, **values)

    with pytest.raises(ValueError, match=message):
        score_wiser_predictions(predictions, receipt, truth)


def test_truth_last_detailed_score_rejects_prediction_argmax_mismatch(tmp_path: Path) -> None:
    predictions, receipt, truth, _ = _write_detailed_inputs(tmp_path)
    with np.load(predictions, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files}
    values["p3_predictions"] = np.roll(values["p3_predictions"], 1)
    np.savez_compressed(predictions, **values)

    with pytest.raises(ValueError, match="argmax"):
        score_wiser_predictions(predictions, receipt, truth)


def test_truth_last_detailed_score_rejects_partial_logit_evidence(tmp_path: Path) -> None:
    predictions, receipt, truth, _ = _write_detailed_inputs(tmp_path)
    with np.load(predictions, allow_pickle=False) as source:
        values = {name: source[name] for name in source.files if name != "p2_logits"}
    np.savez_compressed(predictions, **values)

    with pytest.raises(ValueError, match="logits are incomplete"):
        score_wiser_predictions(predictions, receipt, truth)


def test_truth_last_detailed_score_rejects_duplicate_truth_tokens(tmp_path: Path) -> None:
    predictions, receipt, truth, _ = _write_detailed_inputs(tmp_path)
    truth_payload = json.loads(truth.read_text(encoding="utf-8"))
    truth_payload["rows"].append({"query_token": "q0", "true_class_index": 0})
    truth.write_text(json.dumps(truth_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate truth token"):
        score_wiser_predictions(predictions, receipt, truth)


def test_truth_last_detailed_score_rejects_missing_truth_class(tmp_path: Path) -> None:
    predictions, receipt, truth, _ = _write_detailed_inputs(tmp_path)
    truth_payload = json.loads(truth.read_text(encoding="utf-8"))
    for row in truth_payload["rows"]:
        if row["true_class_index"] == 5:
            row["true_class_index"] = 4
    truth.write_text(json.dumps(truth_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="six-old-class coverage"):
        score_wiser_predictions(predictions, receipt, truth)


def test_truth_last_detailed_score_rejects_out_of_range_truth_index(tmp_path: Path) -> None:
    predictions, receipt, truth, _ = _write_detailed_inputs(tmp_path)
    truth_payload = json.loads(truth.read_text(encoding="utf-8"))
    truth_payload["rows"][0]["true_class_index"] = 6
    truth.write_text(json.dumps(truth_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="truth index"):
        score_wiser_predictions(predictions, receipt, truth)


def test_paired_comparison_reports_pp_help_harm_and_p3_aliases(tmp_path: Path) -> None:
    truth = np.repeat(np.arange(6), 2)
    control_p3 = truth.copy()
    candidate_p3 = truth.copy()
    control_p3[[0, 2, 4]] = [1, 2, 3]
    candidate_p3[[0, 2, 6]] = [0, 2, 5]
    control = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "control", arm="B0", p3=control_p3)[:3])
    candidate = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "candidate", arm="N1", p3=candidate_p3)[:3])

    comparison = compare_wiser_score_rows(control, candidate)

    assert comparison["comparison_state"] == "DA1_REG0-DA0_REG0"
    assert comparison["accuracy_delta_pp"] == pytest.approx(8.333333, abs=1.0e-5)
    assert comparison["help_count"] == 2
    assert comparison["harm_count"] == 1
    assert comparison["unchanged_count"] == 9
    assert comparison["net_help_minus_harm"] == 1
    assert comparison["probes"]["P3_OLD_D92"]["nll_delta"] < 0.0
    assert comparison["probes"]["P3_OLD_D92"]["control_metrics"]["accuracy"] == pytest.approx(0.75)


def test_paired_comparison_separates_neutral_flips_and_same_predictions(tmp_path: Path) -> None:
    truth = np.repeat(np.arange(6), 2)
    control_p3 = truth.copy()
    candidate_p3 = truth.copy()
    control_p3[0] = 1
    candidate_p3[0] = 2
    control_p3[2] = 3
    candidate_p3[2] = 3
    control = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "control", arm="B0", p3=control_p3)[:3])
    candidate = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "candidate", arm="N2", p3=candidate_p3)[:3])

    result = compare_wiser_score_rows(control, candidate)["probes"]["P3_OLD_D92"]

    assert result["neutral_flip_count"] == 1
    assert result["same_prediction_count"] == 11
    assert result["prediction_flip_count"] == 1


def test_paired_comparison_rejects_same_bogus_class_registry_on_both_rows(
    tmp_path: Path,
) -> None:
    control = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "control", arm="B0")[:3])
    candidate = deepcopy(control)
    candidate["arm"] = "N1"
    control["class_registry"] = ["6", "7", "8", "9", "10", "11"]
    candidate["class_registry"] = ["6", "7", "8", "9", "10", "11"]

    with pytest.raises(ValueError, match="class registry"):
        compare_wiser_score_rows(control, candidate)


def test_paired_comparison_rejects_stored_hard_metrics_that_disagree_with_pairing(
    tmp_path: Path,
) -> None:
    truth = np.repeat(np.arange(6), 2)
    control_p3 = truth.copy()
    candidate_p3 = truth.copy()
    control_p3[0] = 1
    control = score_wiser_predictions(
        *_write_detailed_inputs(tmp_path / "control", arm="B0", p3=control_p3)[:3]
    )
    candidate = score_wiser_predictions(
        *_write_detailed_inputs(tmp_path / "candidate", arm="N1", p3=candidate_p3)[:3]
    )
    candidate["probes"]["P3_OLD_D92"]["accuracy"] = control["probes"]["P3_OLD_D92"]["accuracy"]

    with pytest.raises(ValueError, match="stored metric"):
        compare_wiser_score_rows(control, candidate)


def test_paired_comparison_rejects_corrupted_per_query_nll_evidence(tmp_path: Path) -> None:
    control = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "control", arm="B0")[:3])
    candidate = deepcopy(control)
    candidate["arm"] = "N1"
    candidate["pairing_payload"]["nll_contributions"]["P3_OLD_D92"][0] += 1.0

    with pytest.raises(ValueError, match="NLL evidence"):
        compare_wiser_score_rows(control, candidate)


@pytest.mark.parametrize(
    "field",
    [
        "query_tokens",
        "truth",
        "true_class_indices",
        "outer_key",
        "capsule_id",
        "split_id",
        "receiver",
        "scenario",
        "query_rows",
        "class_registry",
        "per_class_query_rows",
    ],
)
def test_paired_comparison_rejects_binding_mismatch(tmp_path: Path, field: str) -> None:
    control = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "control", arm="B0")[:3])
    candidate = deepcopy(control)
    candidate["arm"] = "N1"
    if field == "query_tokens":
        candidate["query_tokens"] = list(reversed(candidate["query_tokens"]))
    elif field == "truth":
        candidate["pairing_payload"]["truth"][0] = 1
    elif field == "true_class_indices":
        candidate["pairing_payload"][field][0] = 1
    elif field == "per_class_query_rows":
        candidate[field]["0"] = 1
    elif field == "query_rows":
        candidate[field] = 11
    else:
        candidate[field] = f"wrong-{field}"

    message = "class registry" if field == "class_registry" else "pairing binding"
    with pytest.raises(ValueError, match=message):
        compare_wiser_score_rows(control, candidate)


def test_paired_comparison_rejects_probe_registry_and_legacy_rows(tmp_path: Path) -> None:
    control = score_wiser_predictions(*_write_detailed_inputs(tmp_path / "control", arm="B0")[:3])
    candidate = deepcopy(control)
    candidate["arm"] = "N1"
    candidate["probes"].pop("P2_SOURCE_PROTOTYPE")
    with pytest.raises(ValueError, match="probe registry"):
        compare_wiser_score_rows(control, candidate)

    legacy_predictions = tmp_path / "legacy.npz"
    np.savez_compressed(
        legacy_predictions,
        query_tokens=np.asarray([f"q{index}" for index in range(12)]),
        p1_predictions=np.repeat(np.arange(6), 2),
        p2_predictions=np.repeat(np.arange(6), 2),
        p3_predictions=np.repeat(np.arange(6), 2),
        query_z_id=np.eye(6, dtype=np.float32)[np.repeat(np.arange(6), 2)],
    )
    _, receipt, truth, _ = _write_detailed_inputs(tmp_path / "legacy-input", arm="N1")
    legacy = score_wiser_predictions(legacy_predictions, receipt, truth)
    with pytest.raises(ValueError, match="detailed pairing"):
        compare_wiser_score_rows(control, legacy)


def test_truth_last_scorer_rejects_truncated_prediction_registry(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.npz"
    np.savez_compressed(
        predictions,
        query_tokens=np.asarray(["q0"]),
        p1_predictions=np.asarray([0]),
        p2_predictions=np.asarray([0]),
        p3_predictions=np.asarray([0]),
        query_z_id=np.zeros((1, 160), dtype=np.float32),
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PREDICTIONS_COMPLETE",
                "arm": "A",
                "receiver": "rx",
                "scenario": "leo_clear_weak",
                "query_rows": 2,
                "expected_query_tokens": ["q0", "q1"],
                "query_truth_opened": False,
                "query_role_opened": False,
                "support_state_frozen_before_query": True,
            }
        ),
        encoding="utf-8",
    )
    truth = tmp_path / "truth.json"
    truth.write_text(
        json.dumps(
            {
                "receiver": "rx",
                "rows": [
                    {"query_token": "q0", "true_class_index": 0},
                    {"query_token": "q1", "true_class_index": 1},
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="token join"):
        score_wiser_predictions(predictions, receipt, truth)
