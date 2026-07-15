from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from paper_reproduction.scripts.summarize_cvs_stage2c_locked_matrix import (
    K10_OLD_TARGET,
    _formal_row_content_sha256,
    clustered_paired_bootstrap,
    matched_k5_drop_summary,
    recompute_formal_metrics,
    validate_nested_protocol,
)


RECEIVERS = ("r1", "r2")
SEEDS = (11, 12)
SCENARIOS = ("leo_clear_weak", "leo_rain_weak")
NEW_COUNTS = (1, 2)
K_VALUES = (1, 2)
OLD_LABELS = ("old0", "old1")


def test_locked_old_accuracy_target_matches_current_project_protocol() -> None:
    assert K10_OLD_TARGET == 0.92


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ids_hash(values: list[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _sample(role: str, label: str, receiver: str, kind: str, index: int) -> str:
    return f"{role}|{label}|{receiver}|day0|eq1|{kind}{index}"


def _seal_row(row: dict[str, object]) -> None:
    row["formal_row_content_sha256"] = _formal_row_content_sha256(row)


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    global_hashes = {
        field: _hash(field)
        for field in (
            "candidate_lock_sha256",
            "locked_candidate_sha256",
            "checkpoint_sha256",
            "adapter_state_sha256",
            "adapter_manifest_sha256",
            "source_validation_manifest_sha256",
            "source_feature_statistics_sha256",
            "locked_head_selected_sha256",
            "tta_thresholds_sha256",
        )
    }
    for receiver in RECEIVERS:
        for seed in SEEDS:
            for scenario in SCENARIOS:
                cache_token = f"{receiver}:{seed}:{scenario}"
                for new_count in NEW_COUNTS:
                    new_labels = tuple(f"new{i}" for i in range(new_count))
                    labels = (*OLD_LABELS, *new_labels)
                    for k_shot in K_VALUES:
                        support: list[str] = []
                        query: list[str] = []
                        for label in labels:
                            role = "target_old" if label in OLD_LABELS else "target_new"
                            support.extend(
                                _sample(role, label, receiver, "s", index)
                                for index in range(k_shot)
                            )
                            query.extend(
                                _sample(role, label, receiver, "q", index)
                                for index in range(2)
                            )
                        key_token = f"{receiver}:{seed}:{scenario}:{new_count}:{k_shot}"
                        support_overlay_ids = [
                            f"support-overlay:{key_token}:{index}"
                            for index in range(len(support))
                        ]
                        support_iq_hashes = [
                            _hash(f"support-iq:{key_token}:{index}")
                            for index in range(len(support))
                        ]
                        query_overlay_ids = [
                            f"query-overlay:{key_token}:{index}"
                            for index in range(len(query))
                        ]
                        query_iq_hashes = [
                            _hash(f"query-iq:{key_token}:{index}")
                            for index in range(len(query))
                        ]
                        head_state_sha256 = _hash(
                            f"head:{receiver}:{seed}:{new_count}:{k_shot}"
                        )
                        cell_predictions: list[dict[str, object]] = []
                        for index, query_id in enumerate(query):
                            role, truth, *_rest = query_id.split("|")
                            is_old = role == "target_old"
                            prediction = truth
                            record: dict[str, object] = {
                                "candidate_id": "locked",
                                "candidate_lock_sha256": global_hashes[
                                    "candidate_lock_sha256"
                                ],
                                "receiver": receiver,
                                "seed": seed,
                                "scenario": scenario,
                                "new_class_count": new_count,
                                "k_shot": k_shot,
                                "query_id": query_id,
                                "evaluation_role": role,
                                "truth": truth,
                                "prediction": prediction,
                                "view_budget": (1, 3, 5)[index % 3],
                                "candidate_correct": 1,
                                "old_before_prediction": truth if is_old else "",
                                "old_before_correct": 1 if is_old else "",
                                "identity_before_prediction": truth if is_old else "",
                                "identity_before_correct": 1 if is_old else "",
                                "identity_after_prediction": truth if is_old else "",
                                "identity_after_correct": 1 if is_old else "",
                                "direct_prediction": (
                                    "old1" if is_old else ""
                                ),
                                "direct_correct": (
                                    int(truth == "old1") if is_old else ""
                                ),
                                "overlay_id": query_overlay_ids[index],
                                "post_channel_iq_sha256": query_iq_hashes[index],
                                "symmetric_locked_head_state_sha256": (
                                    head_state_sha256
                                ),
                            }
                            cell_predictions.append(record)
                        budgets = np.asarray(
                            [int(value["view_budget"]) for value in cell_predictions]
                        )
                        mean_views = float(np.mean(budgets))
                        p95_views = float(np.percentile(budgets, 95, method="higher"))
                        backbone_macs = 1000
                        head_macs = 100
                        row: dict[str, object] = {
                            "candidate_id": "locked",
                            **global_hashes,
                            "receiver": receiver,
                            "seed": seed,
                            "scenario": scenario,
                            "new_class_count": new_count,
                            "k_shot": k_shot,
                            "registered_class_count": len(labels),
                            "query_per_tx": 2,
                            "support_ids_json": json.dumps(support),
                            "query_ids_json": json.dumps(query),
                            "support_ids_sha256": _ids_hash(support),
                            "query_ids_sha256": _ids_hash(query),
                            "support_overlay_ids_sha256": _ids_hash(
                                support_overlay_ids
                            ),
                            "query_overlay_ids_sha256": _ids_hash(query_overlay_ids),
                            "support_post_channel_iq_sha256_root": _ids_hash(
                                support_iq_hashes
                            ),
                            "query_post_channel_iq_sha256_root": _ids_hash(
                                query_iq_hashes
                            ),
                            "support_overlay_ids_json": json.dumps(
                                support_overlay_ids
                            ),
                            "support_post_channel_iq_sha256_json": json.dumps(
                                support_iq_hashes
                            ),
                            "symmetric_locked_head_state_sha256": (
                                head_state_sha256
                            ),
                            "satellite_seeds_json": json.dumps(
                                [seed * 10 + SCENARIOS.index(scenario)]
                            ),
                            "leo_weak_cache_sha256": _hash(f"cache:{cache_token}"),
                            "leo_weak_cache_manifest_sha256": _hash(
                                f"cache-manifest:{cache_token}"
                            ),
                            "leo_weak_cache_set_manifest_sha256": _hash(
                                f"cache-set:{receiver}:{seed}"
                            ),
                            "leo_weak_cache_build_spec_sha256": _hash(
                                f"cache-spec:{receiver}:{seed}"
                            ),
                            "stage2_config_content_sha256": _hash(
                                f"config:{key_token}"
                            ),
                            "old_tx_labels_json": json.dumps(OLD_LABELS),
                            "new_tx_labels_json": json.dumps(new_labels),
                            "phase2_sample_view_policy": "leo_weak_only_no_clean_access",
                            "clean_sample_access": False,
                            "clean_derived_signal_access": False,
                            "support_query_view": "leo_weak_only",
                            "clean_support_query_rows": 0,
                            "head_mode": "symmetric_locked",
                            "old_new_role_oracle_used": False,
                            "class_quota_used": False,
                            "query_fit_used": False,
                            "query_batch_state_required": False,
                            "all_five_views_materialized_before_gate": False,
                            "old_acc_before_increment": 1.0,
                            "old_acc_after_increment": 1.0,
                            "average_forgetting": 0.0,
                            "old_adaptation_gain": 0.0,
                            "min_old_class_acc": 1.0,
                            "seen_new_acc": 1.0,
                            "min_new_class_acc": 1.0,
                            "h_old_new": 1.0,
                            "identity_average_forgetting": 0.0,
                            "identity_old_acc_before_increment": 1.0,
                            "identity_old_acc_after_increment": 1.0,
                            "direct_adv3b02_old_acc": 0.5,
                            "delta_vs_direct_adv3b02": 0.5,
                            "mean_backbone_forward_count": mean_views,
                            "p95_backbone_forward_count": p95_views,
                            "view1_trigger_rate": float(np.mean(budgets == 1)),
                            "view3_trigger_rate": float(np.mean(budgets == 3)),
                            "view5_trigger_rate": float(np.mean(budgets == 5)),
                            "worst_case_backbone_forward_count": 5,
                            "profiled_backbone_macs_per_forward": backbone_macs,
                            "support_head_macs_per_view": head_macs,
                            "mean_profiled_macs_per_query_excluding_fft_and_view_transform": int(
                                round((backbone_macs + head_macs) * mean_views)
                            ),
                            "mac_coverage": "executed layers plus support head",
                            "fft96_and_receive_transform_macs_included": False,
                            "deployment_query_latency_ms_per_sample": 1.0,
                            "deployment_end_to_end_latency_ms_per_query_including_enrollment": 2.0,
                            "peak_cuda_memory_bytes": 1024,
                            "host_peak_working_set_bytes": 2048,
                            "persistent_state_bytes": 4096,
                            "adapter_trainable_parameters": 512,
                            "adapter_epochs": 12,
                            "adapter_optimizer_steps": 24,
                            "resource_tier": "extreme_light",
                            "preferred_parameter_ratio": 0.01,
                            "preferred_epoch_ratio": 0.6,
                            "preferred_state_ratio": 0.02,
                        }
                        _seal_row(row)
                        for record in cell_predictions:
                            record["formal_row_content_sha256"] = row[
                                "formal_row_content_sha256"
                            ]
                        rows.append(row)
                        predictions.extend(cell_predictions)
    return rows, predictions


def _validate(rows: list[dict[str, object]]) -> dict[str, object]:
    return validate_nested_protocol(
        rows,
        expected_receivers=RECEIVERS,
        expected_scenarios=SCENARIOS,
        expected_new_counts=NEW_COUNTS,
        expected_k=K_VALUES,
        expected_seeds=SEEDS,
        expected_query_per_tx=2,
        minimum_seeds=2,
    )


def test_strict_formal_evidence_accepts_complete_nested_matrix_and_recomputes() -> None:
    rows, predictions = _fixture()
    audit = _validate(rows)
    rebuilt = recompute_formal_metrics(rows, predictions)
    assert audit["row_count"] == 32
    assert audit["artifact_hash_binding_pass"] is True
    assert audit["forbidden_oracle_quota_query_fit_pass"] is True
    assert len(rebuilt) == len(rows)
    assert rebuilt[0]["old_acc_after_increment"] == pytest.approx(1.0)
    assert rebuilt[0]["direct_adv3b02_old_acc"] == pytest.approx(0.5)


def test_nested_protocol_rejects_query_drift() -> None:
    rows, _predictions = _fixture()
    row = rows[-1]
    query = json.loads(str(row["query_ids_json"]))
    query[-1] = query[-1].replace("q1", "q99")
    row["query_ids_json"] = json.dumps(query)
    row["query_ids_sha256"] = _ids_hash(query)
    _seal_row(row)
    with pytest.raises(ValueError, match="query IDs drift"):
        _validate(rows)


def test_nested_protocol_rejects_set_nested_but_not_ordered_prefix() -> None:
    rows, _predictions = _fixture()
    for row in rows:
        if row["receiver"] == "r1" and row["seed"] == 11 and row["k_shot"] == 2:
            support = json.loads(str(row["support_ids_json"]))
            support[0], support[1] = support[1], support[0]
            row["support_ids_json"] = json.dumps(support)
            row["support_ids_sha256"] = _ids_hash(support)
            _seal_row(row)
    with pytest.raises(ValueError, match="ordered per-TX prefix"):
        _validate(rows)


@pytest.mark.parametrize(
    "field", ("old_new_role_oracle_used", "class_quota_used", "query_fit_used")
)
def test_nested_protocol_rejects_forbidden_role_quota_or_query_fit(field: str) -> None:
    rows, _predictions = _fixture()
    rows[0][field] = True
    _seal_row(rows[0])
    with pytest.raises(ValueError, match="forbidden formal policy"):
        _validate(rows)


def test_recompute_rejects_tampered_summary_metric() -> None:
    rows, predictions = _fixture()
    rows[0]["old_acc_after_increment"] = 0.25
    with pytest.raises(ValueError, match="summary disagrees with predictions"):
        recompute_formal_metrics(rows, predictions)


def test_recompute_rejects_missing_old_before_prediction() -> None:
    rows, predictions = _fixture()
    old_prediction = next(
        value for value in predictions if value["evaluation_role"] == "target_old"
    )
    old_prediction["old_before_prediction"] = ""
    with pytest.raises(ValueError, match="old-before prediction evidence"):
        recompute_formal_metrics(rows, predictions)


def test_recompute_rejects_prediction_without_same_lock() -> None:
    rows, predictions = _fixture()
    predictions[0]["candidate_lock_sha256"] = _hash("different lock")
    with pytest.raises(ValueError, match="candidate-lock mismatch"):
        recompute_formal_metrics(rows, predictions)


def test_recompute_rejects_duplicate_prediction_row() -> None:
    rows, predictions = _fixture()
    predictions.append(deepcopy(predictions[0]))
    with pytest.raises(ValueError, match="duplicate/empty formal prediction"):
        recompute_formal_metrics(rows, predictions)


def test_recompute_rejects_mac_summary_not_bound_to_view_budgets() -> None:
    rows, predictions = _fixture()
    rows[0]["mean_profiled_macs_per_query_excluding_fft_and_view_transform"] = 1
    with pytest.raises(ValueError, match="MAC count disagrees"):
        recompute_formal_metrics(rows, predictions)


def test_clustered_bootstrap_recomputes_candidate_correctness() -> None:
    _rows, predictions = _fixture()
    k1_old = [
        value
        for value in predictions
        if value["k_shot"] == 1 and value["evaluation_role"] == "target_old"
    ]
    for value in k1_old:
        value["candidate_correct"] = 0  # ignored: prediction/truth are authoritative
    result = clustered_paired_bootstrap(k1_old, repetitions=1000, seed=7)
    assert result["delta"] == pytest.approx(0.5)
    assert result["ci95_lower"] > 0.0
    assert result["cluster_count"] == 4


def test_k5_drop_gate_uses_worst_exactly_matched_cell_not_mean() -> None:
    rows = []
    for receiver, k5_value in (("r1", 0.89), ("r2", 0.97)):
        for k_shot, value in ((5, k5_value), (10, 0.95)):
            rows.append(
                {
                    "receiver": receiver,
                    "seed": 1,
                    "scenario": "leo_clear_weak",
                    "new_class_count": 5,
                    "k_shot": k_shot,
                    "seen_new_acc": value,
                }
            )
    result = matched_k5_drop_summary(
        rows, new_class_count=5, metric="seen_new_acc"
    )
    assert result["mean_drop"] == pytest.approx(0.02)
    assert result["max_drop"] == pytest.approx(0.06)
