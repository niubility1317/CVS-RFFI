#!/usr/bin/env python
"""Audit a locked Stage2-C matrix and compute cross-K forgetting gates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from paper_reproduction.scripts.build_cvs_stage2c_candidate_lock import (
    FORMAL_CONFIRMATION_SEEDS,
    FORMAL_K,
    FORMAL_NEW_COUNTS,
    FORMAL_RECEIVERS,
)


FORMAL_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
NEW_ACCURACY_TARGET = {5: 0.92, 10: 0.90, 20: 0.86}
K10_OLD_TARGET = 0.92
K10_MIN_OLD_CLASS_TARGET = 0.88
K5_MAX_DROP = 0.03

GLOBAL_ARTIFACT_HASH_FIELDS = (
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
ROW_PROVENANCE_HASH_FIELDS = (
    "leo_weak_cache_set_manifest_sha256",
    "leo_weak_cache_build_spec_sha256",
    "stage2_config_content_sha256",
    "leo_weak_cache_sha256",
    "leo_weak_cache_manifest_sha256",
    "support_ids_sha256",
    "query_ids_sha256",
    "support_overlay_ids_sha256",
    "query_overlay_ids_sha256",
    "support_post_channel_iq_sha256_root",
    "query_post_channel_iq_sha256_root",
    "symmetric_locked_head_state_sha256",
    "formal_row_content_sha256",
)
FORBIDDEN_TRUE_FIELDS = (
    "clean_sample_access",
    "clean_derived_signal_access",
    "old_new_role_oracle_used",
    "class_quota_used",
    "query_fit_used",
    "query_batch_state_required",
    "all_five_views_materialized_before_gate",
)
REQUIRED_RESOURCE_FIELDS = (
    "profiled_backbone_macs_per_forward",
    "support_head_macs_per_view",
    "mean_profiled_macs_per_query_excluding_fft_and_view_transform",
    "deployment_query_latency_ms_per_sample",
    "deployment_end_to_end_latency_ms_per_query_including_enrollment",
    "peak_cuda_memory_bytes",
    "host_peak_working_set_bytes",
    "persistent_state_bytes",
    "adapter_trainable_parameters",
    "adapter_epochs",
    "adapter_optimizer_steps",
    "preferred_parameter_ratio",
    "preferred_epoch_ratio",
    "preferred_state_ratio",
)
SUMMARY_METRIC_FIELDS = (
    "old_acc_before_increment",
    "old_acc_after_increment",
    "average_forgetting",
    "old_adaptation_gain",
    "min_old_class_acc",
    "seen_new_acc",
    "min_new_class_acc",
    "h_old_new",
    "identity_average_forgetting",
    "identity_old_acc_before_increment",
    "identity_old_acc_after_increment",
    "direct_adv3b02_old_acc",
    "delta_vs_direct_adv3b02",
    "mean_backbone_forward_count",
    "p95_backbone_forward_count",
    "view1_trigger_rate",
    "view3_trigger_rate",
    "view5_trigger_rate",
)


def _required(row: dict[str, Any], key: str) -> Any:
    if key not in row or row[key] is None or (
        isinstance(row[key], str) and not row[key].strip()
    ):
        raise ValueError(f"missing required field {key}")
    return row[key]


def _row_key(row: dict[str, Any]) -> tuple[str, int, str, int, int]:
    return (
        str(row["receiver"]),
        int(row["seed"]),
        str(row["scenario"]),
        int(row["new_class_count"]),
        int(row["k_shot"]),
    )


def _bool(row: dict[str, Any], key: str) -> bool:
    if key not in row:
        raise ValueError(f"missing required field {key}")
    value = row[key]
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"invalid boolean {key}={value!r}")


def _sha256(value: Any, key: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"invalid SHA256 field {key}")
    return digest


def _ids_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _formal_row_content_sha256(row: dict[str, Any]) -> str:
    payload = {
        str(key): str(value)
        for key, value in row.items()
        if str(key) != "formal_row_content_sha256"
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_parts(sample_id: str) -> tuple[str, str, str, str, str, str]:
    parts = tuple(str(sample_id).split("|"))
    if len(parts) != 6 or any(not value for value in parts):
        raise ValueError(f"invalid physical sample ID: {sample_id!r}")
    return parts  # type: ignore[return-value]


def _exact_binary(value: Any, key: str) -> int:
    normalized = str(value).strip()
    if normalized not in {"0", "1"}:
        raise ValueError(f"{key} must be binary, got {value!r}")
    return int(normalized)


def _assert_close(actual: float, declared: Any, key: str) -> None:
    expected = float(declared)
    if not np.isfinite(expected) or not np.isclose(
        float(actual), expected, rtol=1.0e-9, atol=1.0e-12
    ):
        raise ValueError(
            f"formal row summary disagrees with predictions for {key}: "
            f"declared={declared!r}, recomputed={actual!r}"
        )


def _float(row: dict[str, Any], key: str) -> float:
    value = float(_required(row, key))
    if not np.isfinite(value):
        raise ValueError(f"non-finite metric {key}")
    return value


def _ids(row: dict[str, Any], key: str) -> tuple[str, ...]:
    payload = _required(row, key)
    values = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{key} must be a JSON list")
    result = tuple(str(value) for value in values)
    if len(result) != len(set(result)):
        raise ValueError(f"duplicate IDs in {key}")
    return result


def _json_list(row: dict[str, Any], key: str) -> list[Any]:
    payload = _required(row, key)
    values = json.loads(payload) if isinstance(payload, str) else payload
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a JSON list")
    return values


def validate_nested_protocol(
    rows: Sequence[dict[str, Any]],
    *,
    expected_receivers: Sequence[str] = FORMAL_RECEIVERS,
    expected_scenarios: Sequence[str] = FORMAL_SCENARIOS,
    expected_new_counts: Sequence[int] = FORMAL_NEW_COUNTS,
    expected_k: Sequence[int] = FORMAL_K,
    expected_seeds: Sequence[int] = FORMAL_CONFIRMATION_SEEDS,
    expected_query_per_tx: int = 20,
    minimum_seeds: int = 5,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("locked matrix is empty")
    candidate_ids = {str(_required(row, "candidate_id")) for row in rows}
    lock_hashes = {
        _sha256(_required(row, "candidate_lock_sha256"), "candidate_lock_sha256")
        for row in rows
    }
    if len(candidate_ids) != 1 or len(lock_hashes) != 1:
        raise ValueError("matrix mixes candidates or candidate locks")
    if not next(iter(candidate_ids)).strip():
        raise ValueError("candidate_id is empty")
    for field in GLOBAL_ARTIFACT_HASH_FIELDS:
        values = {_sha256(_required(row, field), field) for row in rows}
        if len(values) != 1:
            raise ValueError(f"global artifact hash drifts across matrix: {field}")
    receivers = sorted({str(row["receiver"]) for row in rows})
    scenarios = sorted({str(row["scenario"]) for row in rows})
    new_counts = sorted({int(row["new_class_count"]) for row in rows})
    k_values = sorted({int(row["k_shot"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    if set(receivers) != set(str(value) for value in expected_receivers):
        raise ValueError("target receiver coverage mismatch")
    if set(scenarios) != set(str(value) for value in expected_scenarios):
        raise ValueError("leo_weak scenario coverage mismatch")
    if new_counts != sorted(int(value) for value in expected_new_counts):
        raise ValueError("new-class-count coverage mismatch")
    if k_values != sorted(int(value) for value in expected_k):
        raise ValueError("K coverage mismatch")
    if len(seeds) < int(minimum_seeds):
        raise ValueError("insufficient independent confirmation seeds")
    if tuple(seeds) != tuple(sorted(int(value) for value in expected_seeds)):
        raise ValueError("formal confirmation seed set mismatch")
    index: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}
    reference_old_labels: tuple[str, ...] | None = None
    reference_new_labels: dict[int, tuple[str, ...]] = {}
    for row in rows:
        key = _row_key(row)
        if key in index:
            raise ValueError(f"duplicate formal matrix row: {key}")
        index[key] = row
        for field in ROW_PROVENANCE_HASH_FIELDS:
            _sha256(_required(row, field), field)
        if _formal_row_content_sha256(row) != str(row["formal_row_content_sha256"]):
            raise ValueError(f"formal row content digest mismatch: {key}")
        for field in FORBIDDEN_TRUE_FIELDS:
            if _bool(row, field):
                raise ValueError(f"forbidden formal policy enabled: {field}")
        if str(row.get("phase2_sample_view_policy", "")) != (
            "leo_weak_only_no_clean_access"
        ):
            raise ValueError("formal matrix violates leo_weak-only sample policy")
        if str(row.get("support_query_view", "")) != "leo_weak_only" or int(
            row.get("clean_support_query_rows", -1)
        ) != 0:
            raise ValueError("formal matrix contains non-leo_weak or clean support/query")
        if str(row.get("head_mode", "")) != "symmetric_locked":
            raise ValueError("formal matrix head is not symmetric_locked")
        support = _ids(row, "support_ids_json")
        query = _ids(row, "query_ids_json")
        if _ids_sha256(support) != _sha256(row["support_ids_sha256"], "support_ids_sha256"):
            raise ValueError(f"support ID hash mismatch: {key}")
        if _ids_sha256(query) != _sha256(row["query_ids_sha256"], "query_ids_sha256"):
            raise ValueError(f"query ID hash mismatch: {key}")
        class_count = int(row["registered_class_count"])
        query_per_tx = int(row["query_per_tx"])
        if query_per_tx != int(expected_query_per_tx):
            raise ValueError(f"query_per_tx differs from the formal lock: {key}")
        old_labels = _ids(row, "old_tx_labels_json")
        new_labels = _ids(row, "new_tx_labels_json")
        if reference_old_labels is None:
            reference_old_labels = old_labels
        elif old_labels != reference_old_labels:
            raise ValueError("target-old TX labels drift across matrix rows")
        new_count = int(row["new_class_count"])
        if len(new_labels) != new_count or set(old_labels) & set(new_labels):
            raise ValueError("invalid old/new TX class split")
        if new_count not in reference_new_labels:
            reference_new_labels[new_count] = new_labels
        elif new_labels != reference_new_labels[new_count]:
            raise ValueError("target-new TX labels drift at fixed class count")
        if class_count != len(old_labels) + len(new_labels):
            raise ValueError("registered class count differs from locked TX labels")
        if len(support) != class_count * int(row["k_shot"]):
            raise ValueError(f"support cardinality drift: {key}")
        if len(query) != class_count * query_per_tx:
            raise ValueError(f"query cardinality drift: {key}")
        if set(support) & set(query):
            raise ValueError(f"support/query overlap: {key}")
        support_overlay_ids = [
            str(value) for value in _json_list(row, "support_overlay_ids_json")
        ]
        support_iq_hashes = [
            _sha256(value, "support_post_channel_iq_sha256")
            for value in _json_list(row, "support_post_channel_iq_sha256_json")
        ]
        if len(support_overlay_ids) != len(support) or any(
            not value for value in support_overlay_ids
        ):
            raise ValueError(f"support overlay evidence cardinality drift: {key}")
        if len(support_iq_hashes) != len(support):
            raise ValueError(f"support IQ-hash evidence cardinality drift: {key}")
        if _ids_sha256(support_overlay_ids) != str(
            row["support_overlay_ids_sha256"]
        ):
            raise ValueError(f"support overlay root mismatch: {key}")
        if _ids_sha256(support_iq_hashes) != str(
            row["support_post_channel_iq_sha256_root"]
        ):
            raise ValueError(f"support IQ-hash root mismatch: {key}")
        support_by_label: dict[str, list[str]] = defaultdict(list)
        query_by_label: dict[str, list[str]] = defaultdict(list)
        old_set, new_set = set(old_labels), set(new_labels)
        for sample_id in support:
            role, label, receiver, _day, _eq, _signal = _sample_parts(sample_id)
            expected_role = "target_old" if label in old_set else "target_new"
            if label not in old_set | new_set or role != expected_role or receiver != key[0]:
                raise ValueError(f"support physical identity violates row split: {sample_id}")
            support_by_label[label].append(sample_id)
        for sample_id in query:
            role, label, receiver, _day, _eq, _signal = _sample_parts(sample_id)
            expected_role = "target_old" if label in old_set else "target_new"
            if label not in old_set | new_set or role != expected_role or receiver != key[0]:
                raise ValueError(f"query physical identity violates row split: {sample_id}")
            query_by_label[label].append(sample_id)
        for label in (*old_labels, *new_labels):
            if len(support_by_label[label]) != int(row["k_shot"]):
                raise ValueError(f"per-TX support cardinality drift for {label}: {key}")
            if len(query_by_label[label]) != query_per_tx:
                raise ValueError(f"per-TX query cardinality drift for {label}: {key}")
        satellite_seeds = _json_list(row, "satellite_seeds_json")
        if not satellite_seeds:
            raise ValueError(f"missing satellite seed evidence: {key}")
        if len({int(value) for value in satellite_seeds}) != len(satellite_seeds):
            raise ValueError(f"duplicate satellite seeds: {key}")
        expected_satellite_seed = int(key[1]) * 10 + list(expected_scenarios).index(
            key[2]
        )
        if [int(value) for value in satellite_seeds] != [expected_satellite_seed]:
            raise ValueError(f"satellite seed differs from receiver/seed lock: {key}")
        for field in SUMMARY_METRIC_FIELDS:
            _float(row, field)
        for field in REQUIRED_RESOURCE_FIELDS:
            value = _float(row, field)
            if value < 0.0:
                raise ValueError(f"negative resource evidence {field}: {key}")
        if not str(row.get("resource_tier", "")).strip():
            raise ValueError(f"missing resource_tier: {key}")
        if int(row.get("worst_case_backbone_forward_count", -1)) != 5:
            raise ValueError(f"invalid worst-case view budget: {key}")
        if _bool(row, "fft96_and_receive_transform_macs_included"):
            raise ValueError("formal MAC total falsely includes unprofiled FFT/view transforms")
        if not str(row.get("mac_coverage", "")).strip():
            raise ValueError(f"missing MAC coverage declaration: {key}")
    expected_count = (
        len(expected_receivers)
        * len(seeds)
        * len(expected_scenarios)
        * len(expected_new_counts)
        * len(expected_k)
    )
    if len(index) != expected_count:
        raise ValueError(f"matrix is incomplete: {len(index)}!={expected_count}")
    ordered_counts = sorted(int(value) for value in expected_new_counts)
    for lower, upper in zip(ordered_counts, ordered_counts[1:]):
        if reference_new_labels[upper][:lower] != reference_new_labels[lower]:
            raise ValueError(f"new-{lower} TX labels are not a prefix of new-{upper}")

    # Physical identities are scenario-locked.  For every class, the ordered
    # K support list must be an exact prefix, rather than merely a set subset.
    for receiver in expected_receivers:
        for seed in seeds:
            for new_count in expected_new_counts:
                reference_query: tuple[str, ...] | None = None
                support_by_k: dict[int, tuple[str, ...]] = {}
                for scenario in expected_scenarios:
                    for k_shot in expected_k:
                        row = index[
                            (
                                str(receiver),
                                int(seed),
                                str(scenario),
                                int(new_count),
                                int(k_shot),
                            )
                        ]
                        query = _ids(row, "query_ids_json")
                        if reference_query is None:
                            reference_query = query
                        elif query != reference_query:
                            raise ValueError("query IDs drift across K/scenario")
                        support = _ids(row, "support_ids_json")
                        if scenario == expected_scenarios[0]:
                            support_by_k[int(k_shot)] = support
                        elif support != support_by_k[int(k_shot)]:
                            raise ValueError("support IDs drift across scenarios")
                ordered_k = sorted(int(value) for value in expected_k)
                for lower, upper in zip(ordered_k, ordered_k[1:]):
                    lower_by_tx: dict[str, list[str]] = defaultdict(list)
                    upper_by_tx: dict[str, list[str]] = defaultdict(list)
                    for sample_id in support_by_k[lower]:
                        lower_by_tx[_sample_parts(sample_id)[1]].append(sample_id)
                    for sample_id in support_by_k[upper]:
                        upper_by_tx[_sample_parts(sample_id)[1]].append(sample_id)
                    if any(
                        upper_by_tx[label][: len(values)] != values
                        for label, values in lower_by_tx.items()
                    ):
                        raise ValueError(
                            f"support K{lower} is not an ordered per-TX prefix of K{upper}"
                        )

    # A receiver/seed/scenario cache is immutable across K and new-count.  The
    # selected IDs grow, but the sealed cache artifacts and satellite seeds do not.
    for receiver in expected_receivers:
        for seed in seeds:
            for scenario in expected_scenarios:
                cells = [
                    index[(str(receiver), int(seed), str(scenario), int(count), int(k))]
                    for count in expected_new_counts
                    for k in expected_k
                ]
                for field in (
                    "leo_weak_cache_set_manifest_sha256",
                    "leo_weak_cache_sha256",
                    "leo_weak_cache_manifest_sha256",
                    "satellite_seeds_json",
                ):
                    if len({str(row[field]) for row in cells}) != 1:
                        raise ValueError(f"sealed cache provenance drifts across K/new-count: {field}")
    cache_set_hashes = {
        str(index[(str(receiver), int(seed), str(expected_scenarios[0]), int(expected_new_counts[0]), int(expected_k[0]))]["leo_weak_cache_set_manifest_sha256"])
        for receiver in expected_receivers
        for seed in seeds
    }
    if len(cache_set_hashes) != len(expected_receivers) * len(seeds):
        raise ValueError("target cache-set hashes are not distinct per receiver/seed")
    scenario_cache_hashes = {
        str(index[(str(receiver), int(seed), str(scenario), int(expected_new_counts[0]), int(expected_k[0]))]["leo_weak_cache_sha256"])
        for receiver in expected_receivers
        for seed in seeds
        for scenario in expected_scenarios
    }
    if len(scenario_cache_hashes) != (
        len(expected_receivers) * len(seeds) * len(expected_scenarios)
    ):
        raise ValueError("scenario cache hashes are not distinct per receiver/seed/scenario")

    # Existing classes retain the exact same ordered support/query identities
    # when the registered target-new prefix expands from 5 to 10 to 20.
    for receiver in expected_receivers:
        for seed in seeds:
            for scenario in expected_scenarios:
                for k_shot in expected_k:
                    previous_support: dict[str, list[str]] | None = None
                    previous_query: dict[str, list[str]] | None = None
                    for new_count in ordered_counts:
                        row = index[(str(receiver), int(seed), str(scenario), int(new_count), int(k_shot))]
                        support_groups: dict[str, list[str]] = defaultdict(list)
                        query_groups: dict[str, list[str]] = defaultdict(list)
                        for value in _ids(row, "support_ids_json"):
                            support_groups[_sample_parts(value)[1]].append(value)
                        for value in _ids(row, "query_ids_json"):
                            query_groups[_sample_parts(value)[1]].append(value)
                        if previous_support is not None:
                            for label, values in previous_support.items():
                                if support_groups[label] != values:
                                    raise ValueError("support IDs drift for retained TX across new-count")
                            for label, values in (previous_query or {}).items():
                                if query_groups[label] != values:
                                    raise ValueError("query IDs drift for retained TX across new-count")
                        previous_support = support_groups
                        previous_query = query_groups
    return {
        "candidate_id": next(iter(candidate_ids)),
        "candidate_lock_sha256": next(iter(lock_hashes)),
        "row_count": int(len(rows)),
        "receivers": receivers,
        "seeds": seeds,
        "query_per_tx": int(expected_query_per_tx),
        "scenarios": scenarios,
        "new_class_counts": new_counts,
        "k_values": k_values,
        "target_old_tx_labels": list(reference_old_labels or ()),
        "nested_target_new_tx_labels": {
            str(count): list(reference_new_labels[count]) for count in ordered_counts
        },
        "clean_support_query_rows": 0,
        "nested_support_pass": True,
        "query_identity_lock_pass": True,
        "artifact_hash_binding_pass": True,
        "forbidden_oracle_quota_query_fit_pass": True,
        "resource_evidence_present": True,
    }


def recompute_formal_metrics(
    rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return metric rows rebuilt exclusively from same-cell predictions.

    The declared summary columns are retained only as a tamper/corruption
    cross-check.  Every metric used by promotion gates is overwritten by the
    value derived from ``formal_predictions.csv``.
    """

    if not prediction_rows:
        raise ValueError("formal_predictions is empty")
    row_index = {_row_key(row): row for row in rows}
    if len(row_index) != len(rows):
        raise ValueError("duplicate formal row before prediction linkage")
    candidate_id = str(rows[0]["candidate_id"])
    candidate_lock = _sha256(rows[0]["candidate_lock_sha256"], "candidate_lock_sha256")
    grouped: dict[tuple[str, int, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    seen_prediction_ids: set[tuple[tuple[str, int, str, int, int], str]] = set()
    for prediction in prediction_rows:
        key = _row_key(prediction)
        if key not in row_index:
            raise ValueError(f"formal prediction has no same-source formal row: {key}")
        if str(prediction.get("candidate_id", "")) != candidate_id:
            raise ValueError(f"formal prediction candidate mismatch: {key}")
        if _sha256(
            prediction.get("candidate_lock_sha256", ""), "candidate_lock_sha256"
        ) != candidate_lock:
            raise ValueError(f"formal prediction candidate-lock mismatch: {key}")
        query_id = str(prediction.get("query_id", ""))
        identity = (key, query_id)
        if not query_id or identity in seen_prediction_ids:
            raise ValueError(f"duplicate/empty formal prediction query ID: {identity}")
        seen_prediction_ids.add(identity)
        grouped[key].append(prediction)
    if set(grouped) != set(row_index):
        missing = sorted(set(row_index) - set(grouped))
        raise ValueError(f"formal prediction coverage is incomplete: {missing[:3]}")

    rebuilt: list[dict[str, Any]] = []
    for row in rows:
        key = _row_key(row)
        predictions = grouped[key]
        query_ids = _ids(row, "query_ids_json")
        actual_ids = tuple(str(value["query_id"]) for value in predictions)
        if actual_ids != query_ids:
            raise ValueError(f"formal prediction order/coverage differs from query IDs: {key}")
        row_digest = _sha256(
            row["formal_row_content_sha256"], "formal_row_content_sha256"
        )
        head_state_digest = _sha256(
            row["symmetric_locked_head_state_sha256"],
            "symmetric_locked_head_state_sha256",
        )
        if any(
            _sha256(
                prediction.get("formal_row_content_sha256", ""),
                "formal_row_content_sha256",
            )
            != row_digest
            or _sha256(
                prediction.get("symmetric_locked_head_state_sha256", ""),
                "symmetric_locked_head_state_sha256",
            )
            != head_state_digest
            for prediction in predictions
        ):
            raise ValueError(f"formal prediction artifact binding mismatch: {key}")
        query_overlay_ids = [str(value.get("overlay_id", "")) for value in predictions]
        query_iq_hashes = [
            _sha256(value.get("post_channel_iq_sha256", ""), "post_channel_iq_sha256")
            for value in predictions
        ]
        if any(not value for value in query_overlay_ids) or _ids_sha256(
            query_overlay_ids
        ) != str(row["query_overlay_ids_sha256"]):
            raise ValueError(f"query overlay root mismatch: {key}")
        if _ids_sha256(query_iq_hashes) != str(
            row["query_post_channel_iq_sha256_root"]
        ):
            raise ValueError(f"query IQ-hash root mismatch: {key}")
        old_labels = set(_ids(row, "old_tx_labels_json"))
        new_labels = set(_ids(row, "new_tx_labels_json"))
        old_correct: list[int] = []
        old_before_correct: list[int] = []
        new_correct: list[int] = []
        identity_before_correct: list[int] = []
        identity_after_correct: list[int] = []
        direct_correct: list[int] = []
        budgets: list[int] = []
        candidate_by_class: dict[str, list[int]] = defaultdict(list)
        for prediction in predictions:
            query_id = str(prediction["query_id"])
            physical_role, physical_truth, receiver, _day, _eq, _signal = _sample_parts(
                query_id
            )
            truth = str(prediction.get("truth", ""))
            predicted = str(prediction.get("prediction", ""))
            if truth != physical_truth or receiver != key[0]:
                raise ValueError(f"prediction truth/receiver differs from query ID: {query_id}")
            if truth in old_labels:
                expected_role = "target_old"
            elif truth in new_labels:
                expected_role = "target_new"
            else:
                raise ValueError(f"prediction truth is outside locked labels: {truth}")
            if physical_role != expected_role or str(
                prediction.get("evaluation_role", "")
            ) != expected_role:
                raise ValueError(f"prediction role differs from locked TX split: {query_id}")
            if predicted not in old_labels | new_labels:
                raise ValueError(f"prediction is outside registered labels: {predicted}")
            correct = int(predicted == truth)
            if _exact_binary(prediction.get("candidate_correct", ""), "candidate_correct") != correct:
                raise ValueError(f"candidate_correct disagrees with prediction: {query_id}")
            budget = int(prediction.get("view_budget", -1))
            if budget not in (1, 3, 5):
                raise ValueError(f"invalid adaptive view budget {budget}: {query_id}")
            budgets.append(budget)
            candidate_by_class[truth].append(correct)
            if expected_role == "target_old":
                old_before = str(prediction.get("old_before_prediction", ""))
                identity_before = str(prediction.get("identity_before_prediction", ""))
                identity_after = str(prediction.get("identity_after_prediction", ""))
                direct_prediction = str(prediction.get("direct_prediction", ""))
                if old_before not in old_labels or identity_before not in old_labels:
                    raise ValueError(f"old-before prediction evidence is missing/invalid: {query_id}")
                if identity_after not in old_labels | new_labels:
                    raise ValueError(f"identity-after prediction evidence is missing/invalid: {query_id}")
                if direct_prediction not in old_labels:
                    raise ValueError(f"direct prediction evidence is missing/invalid: {query_id}")
                before_correct = int(old_before == truth)
                identity_before_hit = int(identity_before == truth)
                identity_after_hit = int(identity_after == truth)
                if _exact_binary(prediction.get("old_before_correct", ""), "old_before_correct") != before_correct:
                    raise ValueError(f"old_before_correct disagrees with prediction: {query_id}")
                if _exact_binary(prediction.get("identity_before_correct", ""), "identity_before_correct") != identity_before_hit:
                    raise ValueError(f"identity_before_correct disagrees with prediction: {query_id}")
                if _exact_binary(prediction.get("identity_after_correct", ""), "identity_after_correct") != identity_after_hit:
                    raise ValueError(f"identity_after_correct disagrees with prediction: {query_id}")
                old_correct.append(correct)
                old_before_correct.append(before_correct)
                identity_before_correct.append(identity_before_hit)
                identity_after_correct.append(identity_after_hit)
                direct_hit = int(direct_prediction == truth)
                if _exact_binary(
                    prediction.get("direct_correct", ""), "direct_correct"
                ) != direct_hit:
                    raise ValueError(f"direct_correct disagrees with prediction: {query_id}")
                direct_correct.append(direct_hit)
            else:
                for field in (
                    "old_before_prediction",
                    "old_before_correct",
                    "identity_before_prediction",
                    "identity_before_correct",
                    "identity_after_prediction",
                    "identity_after_correct",
                    "direct_correct",
                    "direct_prediction",
                ):
                    if str(prediction.get(field, "")).strip():
                        raise ValueError(f"new-TX prediction carries old-only evidence {field}: {query_id}")
                new_correct.append(correct)
        if not old_correct or not new_correct:
            raise ValueError(f"formal prediction cell lacks old or new evidence: {key}")
        old_acc = float(np.mean(old_correct))
        old_before_acc = float(np.mean(old_before_correct))
        new_acc = float(np.mean(new_correct))
        identity_before_acc = float(np.mean(identity_before_correct))
        identity_after_acc = float(np.mean(identity_after_correct))
        direct_acc = float(np.mean(direct_correct))
        class_acc = {
            label: float(np.mean(values)) for label, values in candidate_by_class.items()
        }
        min_old = min(class_acc[label] for label in old_labels)
        min_new = min(class_acc[label] for label in new_labels)
        harmonic = float(2.0 * old_acc * new_acc / max(old_acc + new_acc, 1.0e-12))
        budget_array = np.asarray(budgets, dtype=np.int64)
        view_mean = float(np.mean(budget_array))
        view_p95 = float(np.percentile(budget_array, 95, method="higher"))
        metrics = {
            "old_acc_before_increment": old_before_acc,
            "old_acc_after_increment": old_acc,
            "average_forgetting": old_before_acc - old_acc,
            "old_adaptation_gain": old_acc - old_before_acc,
            "min_old_class_acc": min_old,
            "seen_new_acc": new_acc,
            "min_new_class_acc": min_new,
            "h_old_new": harmonic,
            "identity_average_forgetting": identity_before_acc - identity_after_acc,
            "identity_old_acc_before_increment": identity_before_acc,
            "identity_old_acc_after_increment": identity_after_acc,
            "direct_adv3b02_old_acc": direct_acc,
            "delta_vs_direct_adv3b02": old_acc - direct_acc,
            "mean_backbone_forward_count": view_mean,
            "p95_backbone_forward_count": view_p95,
            "view1_trigger_rate": float(np.mean(budget_array == 1)),
            "view3_trigger_rate": float(np.mean(budget_array == 3)),
            "view5_trigger_rate": float(np.mean(budget_array == 5)),
        }
        for field, value in metrics.items():
            _assert_close(value, row[field], field)
        expected_macs = int(
            round(
                (
                    int(row["profiled_backbone_macs_per_forward"])
                    + int(row["support_head_macs_per_view"])
                )
                * view_mean
            )
        )
        if int(row["mean_profiled_macs_per_query_excluding_fft_and_view_transform"]) != expected_macs:
            raise ValueError(f"profiled mean MAC count disagrees with view budgets: {key}")
        rebuilt.append({**row, **metrics})
    return rebuilt


def clustered_paired_bootstrap(
    prediction_rows: Sequence[dict[str, Any]],
    *,
    repetitions: int = 10_000,
    seed: int = 20260715,
) -> dict[str, float]:
    clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in prediction_rows:
        if int(row["k_shot"]) != 1 or str(row["evaluation_role"]) != "target_old":
            continue
        clusters[(str(row["receiver"]), int(row["seed"]))].append(
            float(str(row["prediction"]) == str(row["truth"]))
            - float(_exact_binary(row["direct_correct"], "direct_correct"))
        )
    if len(clusters) < 2 or any(not values for values in clusters.values()):
        raise ValueError("K1 paired bootstrap requires at least two receiver-seed clusters")
    keys = sorted(clusters)
    values = [np.asarray(clusters[key], dtype=np.float64) for key in keys]
    observed = float(np.concatenate(values).mean())
    rng = np.random.default_rng(int(seed))
    samples = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        selected = rng.integers(0, len(values), size=len(values))
        samples[index] = float(
            np.concatenate([values[int(position)] for position in selected]).mean()
        )
    return {
        "delta": observed,
        "ci95_lower": float(np.quantile(samples, 0.025)),
        "ci95_upper": float(np.quantile(samples, 0.975)),
        "cluster_count": int(len(values)),
        "repetitions": int(repetitions),
    }


def matched_k5_drop_summary(
    rows: Sequence[dict[str, Any]],
    *,
    new_class_count: int,
    metric: str,
) -> dict[str, float | int]:
    """Compare K5 with the exact receiver/seed/scenario-matched K10 row."""

    matched: dict[tuple[str, int, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if int(row["new_class_count"]) != int(new_class_count):
            continue
        k_shot = int(row["k_shot"])
        if k_shot not in (5, 10):
            continue
        key = (str(row["receiver"]), int(row["seed"]), str(row["scenario"]))
        if k_shot in matched[key]:
            raise ValueError(f"duplicate matched K{k_shot} row: {key}")
        matched[key][k_shot] = row
    if not matched or any(set(pair) != {5, 10} for pair in matched.values()):
        raise ValueError(
            f"incomplete matched K5/K10 rows for new-{new_class_count} {metric}"
        )
    drops = np.asarray(
        [
            _float(pair[10], metric) - _float(pair[5], metric)
            for pair in matched.values()
        ],
        dtype=np.float64,
    )
    return {
        "pair_count": int(drops.size),
        "mean_drop": float(drops.mean()),
        "max_drop": float(drops.max()),
    }


def evaluate_gates(
    rows: Sequence[dict[str, Any]], prediction_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    by_k: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_receiver_k: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_new_k: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        k = int(row["k_shot"])
        receiver = str(row["receiver"])
        new_count = int(row["new_class_count"])
        by_k[k].append(row)
        by_receiver_k[(receiver, k)].append(row)
        by_new_k[(new_count, k)].append(row)
    summary_by_k: dict[str, Any] = {}
    for k in FORMAL_K:
        values = by_k[int(k)]
        forgetting = float(np.mean([_float(row, "average_forgetting") for row in values]))
        identity_forgetting = float(
            np.mean([_float(row, "identity_average_forgetting") for row in values])
        )
        summary_by_k[str(k)] = {
            "old_acc_before_increment": float(
                np.mean([_float(row, "old_acc_before_increment") for row in values])
            ),
            "old_acc_after_increment": float(
                np.mean([_float(row, "old_acc_after_increment") for row in values])
            ),
            "average_forgetting": forgetting,
            "old_adaptation_gain": -forgetting,
            "identity_average_forgetting": identity_forgetting,
            "forgetting_delta_vs_identity": forgetting - identity_forgetting,
        }
    k_forgetting = [summary_by_k[str(k)]["average_forgetting"] for k in FORMAL_K]
    paired = clustered_paired_bootstrap(prediction_rows)
    gates: dict[str, bool] = {
        "k1_forgetting_overall_nonpositive": summary_by_k["1"][
            "average_forgetting"
        ]
        <= 0.0,
        "k1_forgetting_each_receiver_nonpositive": all(
            np.mean(
                [
                    _float(row, "average_forgetting")
                    for row in by_receiver_k[(receiver, 1)]
                ]
            )
            <= 0.0
            for receiver in FORMAL_RECEIVERS
        ),
        "k5_k10_k20_forgetting_no_worse_than_identity": all(
            summary_by_k[str(k)]["forgetting_delta_vs_identity"] <= 0.0
            for k in (5, 10, 20)
        ),
        "k1_direct_delta_at_least_2pp": paired["delta"] >= 0.02,
        "k1_direct_delta_ci_lower_positive": paired["ci95_lower"] > 0.0,
        "k1_direct_delta_each_receiver_nonnegative": all(
            np.mean(
                [
                    _float(row, "old_acc_after_increment")
                    - _float(row, "direct_adv3b02_old_acc")
                    for row in by_receiver_k[(receiver, 1)]
                ]
            )
            >= 0.0
            for receiver in FORMAL_RECEIVERS
        ),
    }
    matched_k5_drops: dict[str, dict[str, dict[str, float | int]]] = {}
    for new_count, target in NEW_ACCURACY_TARGET.items():
        k10 = by_new_k[(int(new_count), 10)]
        gates[f"k10_old_acc_new{new_count}"] = float(
            np.mean([_float(row, "old_acc_after_increment") for row in k10])
        ) >= K10_OLD_TARGET
        gates[f"k10_min_old_class_new{new_count}"] = float(
            np.min([_float(row, "min_old_class_acc") for row in k10])
        ) >= K10_MIN_OLD_CLASS_TARGET
        gates[f"k10_seen_new_acc_new{new_count}"] = float(
            np.mean([_float(row, "seen_new_acc") for row in k10])
        ) >= float(target)
        matched_k5_drops[str(new_count)] = {}
        for metric in (
            "old_acc_after_increment",
            "min_old_class_acc",
            "seen_new_acc",
            "h_old_new",
        ):
            drop_summary = matched_k5_drop_summary(
                rows,
                new_class_count=int(new_count),
                metric=metric,
            )
            matched_k5_drops[str(new_count)][metric] = drop_summary
            gates[f"k5_drop_{metric}_new{new_count}"] = (
                float(drop_summary["max_drop"]) <= K5_MAX_DROP
            )
    return {
        "summary_by_k": summary_by_k,
        "worst_K_forgetting": float(max(k_forgetting)),
        "mean_positive_forgetting": float(
            np.mean([max(value, 0.0) for value in k_forgetting])
        ),
        "k1_paired_vs_direct": paired,
        "k5_matched_drop_summary": matched_k5_drops,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "promotion_pass": all(gates.values()),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--row_csv", type=Path, required=True)
    parser.add_argument("--prediction_csv", type=Path, required=True)
    parser.add_argument("--out_json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = _read_csv(args.row_csv)
    predictions = _read_csv(args.prediction_csv)
    protocol = validate_nested_protocol(rows)
    recomputed_rows = recompute_formal_metrics(rows, predictions)
    result = {
        "schema": "cvs_stage2c_locked_cross_k_summary_v2",
        "protocol": protocol,
        "metric_source": "formal_predictions_recomputed_and_formal_rows_cross_checked",
        "formal_evidence_files": {
            "formal_rows_sha256": _file_sha256(args.row_csv),
            "formal_predictions_sha256": _file_sha256(args.prediction_csv),
        },
        **evaluate_gates(recomputed_rows, predictions),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return 0 if result["promotion_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
