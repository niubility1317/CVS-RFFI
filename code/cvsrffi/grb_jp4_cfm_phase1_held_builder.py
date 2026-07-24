"""Deterministic source-only aggregate builder for the GRB-JP4 held falsifier."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi.phase1_grb_jp4_cfm_bundle import (
    AGGREGATION_RECEIPT_SCHEMA,
    CLASS_COUNT,
    FEATURE_DIM,
    MARGIN_RECEIPT_SCHEMA,
    METHOD_ID,
    METHOD_LOCK_SCHEMA,
    PROTOCOL_SCHEMA,
    RECEIVER_DAY_MEAN_SCHEMA,
    canonical_array_sha256,
)
from cvsrffi.stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.phase1.grb_jp4_cfm.held_source_aggregate_builder.v1"
K_VALUES = (1, 5, 10)


class GRBJP4HeldBuilderError(ValueError):
    """Raised when source-only Phase1 aggregation is not reproducible."""


def _canon(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canon(value)).hexdigest()


def _registry(value: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(item) for item in value)
    if (
        len(result) != CLASS_COUNT
        or len(set(result)) != CLASS_COUNT
        or any(not item for item in result)
    ):
        raise GRBJP4HeldBuilderError("builder requires six opaque class handles")
    return result


def build_phase1_qknn_locks() -> dict[int, Phase1ZIDStudentTLock]:
    """Create the three frozen K-specific qKNN locks before held scoring."""

    result = {}
    for k_shot in K_VALUES:
        preimage = {
            "schema": SCHEMA,
            "scope": "phase1_receiver_lodo_pseudoepisode",
            "K": k_shot,
            "target_access": False,
            "query_rows_used_for_fit": 0,
        }
        result[k_shot] = Phase1ZIDStudentTLock(
            k_shot,
            3.0,
            FEATURE_DIM,
            1.0,
            0.2,
            2.0,
            0.5,
            2.0,
            1.0,
            _sha({"kind": "lodo", **preimage}),
            _sha({"kind": "quantization", **preimage}),
        )
    return result


def build_phase1_method_lock(
    *,
    checkpoint_sha256: str,
    class_handle_binding_sha256: str,
    qknn_locks: Mapping[int, Phase1ZIDStudentTLock],
) -> dict[str, Any]:
    if set(qknn_locks) != set(K_VALUES) or any(
        type(qknn_locks[k]) is not Phase1ZIDStudentTLock
        or qknn_locks[k].active_k != k
        for k in K_VALUES
    ):
        raise GRBJP4HeldBuilderError("method lock requires exact K1/K5/K10 qKNN locks")
    return {
        "schema": METHOD_LOCK_SCHEMA,
        "method_id": METHOD_ID,
        "candidate_id": METHOD_ID,
        "protocol_schema": PROTOCOL_SCHEMA,
        "feature_schema": "ADV3B02:z_id:unit_l2:160:v1",
        "checkpoint_sha256": str(checkpoint_sha256),
        "class_handle_binding_sha256": str(class_handle_binding_sha256),
        "qknn_lock_sha256_by_k": {
            str(k): qknn_locks[k].lock_digest for k in K_VALUES
        },
        "rank": 4,
        "old_class_count": 6,
        "allowed_k": [1, 5, 10],
        "ground_old_multiprototype_enabled": True,
        "ground_old_multiprototype_max_per_class": 3,
        "ground_old_multiprototype_min_physical_samples": 2,
        "ground_old_multiprototype_old_classes_only": True,
        "ground_prototypes_enter_qknn_bank": False,
        "ground_prototypes_generate_logits": False,
        "ground_prototypes_add_k": False,
        "ground_component_phase2_mutable": False,
        "delta_tau_source": "phase1_receiver_lodo_correct_held_pseudoquery_only",
        "active_set_steps": 2,
        "ridge_fraction": 0.01,
        "theta_box_abs": 1.0,
        "trust_divisor_squared": 160,
        "g_denominator": 4,
        "target25_release_authorized": False,
        "query_fit_access": False,
        "query_rows_used_for_fit": 0,
    }


def _validate_tap_archive(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "z_id",
        "hidden",
        "pre_relu",
        "joint_weight",
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "class_ids",
        "observation_ids",
    }
    if set(arrays) != required:
        raise GRBJP4HeldBuilderError("tap archive member allowlist drift")
    result = {name: np.asarray(value) for name, value in arrays.items()}
    count = len(result["labels"])
    for name, width in (("z_id", 160), ("hidden", 320), ("pre_relu", 160)):
        value = result[name]
        if (
            value.dtype != np.float32
            or value.shape != (count, width)
            or not np.isfinite(value).all()
        ):
            raise GRBJP4HeldBuilderError(f"tap {name} contract drift")
    if (
        result["joint_weight"].dtype != np.float32
        or result["joint_weight"].shape != (160, 320)
        or not np.isfinite(result["joint_weight"]).all()
    ):
        raise GRBJP4HeldBuilderError("tap joint weight contract drift")
    for name in (
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "observation_ids",
    ):
        value = result[name]
        if (
            value.ndim != 1
            or len(value) != count
            or any(not item for item in value.astype(str).tolist())
        ):
            raise GRBJP4HeldBuilderError(f"tap metadata drift: {name}")
    classes = _registry(result["class_ids"].astype(str).tolist())
    if (
        set(result["labels"].astype(str).tolist()) != set(classes)
        or len(set(result["physical_ids"].astype(str).tolist())) != count
        or len(set(result["observation_ids"].astype(str).tolist())) != count
    ):
        raise GRBJP4HeldBuilderError("tap class/physical closure drift")
    return result


def _unit_mean(rows: np.ndarray) -> np.ndarray:
    center = np.asarray(rows, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(center))
    if not np.isfinite(norm) or norm <= 1.0e-12:
        raise GRBJP4HeldBuilderError("ground aggregate has zero/nonfinite mean")
    return center / norm


def _prototype_records(
    arrays: Mapping[str, np.ndarray], classes: tuple[str, ...]
) -> list[dict[str, Any]]:
    labels = arrays["labels"].astype(str)
    receivers = arrays["receiver_ids"].astype(str)
    days = arrays["day_ids"].astype(str)
    physical = arrays["physical_ids"].astype(str)
    z_id = arrays["z_id"]
    records = []
    for class_id in classes:
        class_mask = labels == class_id
        domain_keys = sorted(
            set(
                f"{receiver}\0{day}"
                for receiver, day in zip(receivers[class_mask], days[class_mask])
            )
        )
        if len(domain_keys) < 3:
            raise GRBJP4HeldBuilderError("each class needs at least three source domains")
        buckets = [domain_keys[index::3] for index in range(3)]
        prototypes = []
        for prototype_index, bucket in enumerate(buckets):
            mask = class_mask & np.asarray(
                [
                    f"{receiver}\0{day}" in bucket
                    for receiver, day in zip(receivers, days)
                ]
            )
            indices = np.flatnonzero(mask)
            if len(indices) < 2:
                raise GRBJP4HeldBuilderError("prototype bucket lacks two physical samples")
            vector = _unit_mean(z_id[indices])
            local = z_id[indices].astype(np.float64)
            local /= np.linalg.norm(local, axis=1, keepdims=True)
            radius = float(
                np.sqrt(np.mean(np.maximum(2.0 * (1.0 - local @ vector), 0.0)))
            )
            commitment = _sha(sorted(physical[indices].tolist()))
            prototypes.append(
                {
                    "vector": vector,
                    "aggregation_receipt": {
                        "schema": AGGREGATION_RECEIPT_SCHEMA,
                        "class_handle": class_id,
                        "prototype_index": prototype_index,
                        "distinct_physical_sample_count": int(len(indices)),
                        "physical_sample_commitment_sha256": commitment,
                        "prototype_sha256": canonical_array_sha256(
                            np.asarray(vector, dtype=np.float64)
                        ),
                        "aggregation_radius": radius,
                        "phase1_before_target_access": True,
                        "multi_physical_aggregation": True,
                        "member_ids_included": False,
                        "sample_features_included": False,
                        "source_path_included": False,
                    },
                }
            )
        records.append({"class_handle": class_id, "prototypes": prototypes})
    return records


def _receiver_day_means(
    arrays: Mapping[str, np.ndarray], classes: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = arrays["labels"].astype(str)
    receivers = arrays["receiver_ids"].astype(str)
    days = arrays["day_ids"].astype(str)
    domains = sorted(
        set(f"{receiver}\0{day}" for receiver, day in zip(receivers, days))
    )
    means = np.zeros((len(classes), len(domains), FEATURE_DIM), dtype=np.float64)
    mask = np.zeros((len(classes), len(domains)), dtype=np.bool_)
    counts = np.zeros((len(classes), len(domains)), dtype=np.int64)
    pre = arrays["pre_relu"].astype(np.float64)
    row_domain = np.asarray(
        [f"{receiver}\0{day}" for receiver, day in zip(receivers, days)]
    )
    for class_index, class_id in enumerate(classes):
        for domain_index, domain in enumerate(domains):
            indices = np.flatnonzero((labels == class_id) & (row_domain == domain))
            if len(indices) < 2:
                continue
            means[class_index, domain_index] = pre[indices].mean(axis=0)
            mask[class_index, domain_index] = True
            counts[class_index, domain_index] = len(indices)
        if int(mask[class_index].sum()) < 2:
            raise GRBJP4HeldBuilderError("class lacks two receiver/day means")
    return means, mask, counts


def _margin_receipt(
    arrays: Mapping[str, np.ndarray],
    classes: tuple[str, ...],
    locks: Mapping[int, Phase1ZIDStudentTLock],
) -> dict[str, Any]:
    labels = arrays["labels"].astype(str)
    receivers = arrays["receiver_ids"].astype(str)
    physical = arrays["physical_ids"].astype(str)
    z_id = arrays["z_id"]
    evidence = []
    margins: list[float] = []
    for k_shot in K_VALUES:
        config = locks[k_shot]
        metric = identity_shared_psd_metric(config=config)
        for held_receiver in sorted(set(receivers.tolist())):
            support_indices: list[int] = []
            query_indices: list[int] = []
            for class_id in classes:
                local = np.flatnonzero(
                    (receivers == held_receiver) & (labels == class_id)
                ).tolist()
                ordered = sorted(
                    local,
                    key=lambda index: _sha(
                        {
                            "schema": SCHEMA,
                            "receiver": held_receiver,
                            "K": k_shot,
                            "physical_id": physical[index],
                        }
                    ),
                )
                if len(ordered) <= k_shot:
                    raise GRBJP4HeldBuilderError("LODO cell lacks support plus query")
                support_indices.extend(ordered[:k_shot])
                query_indices.extend(ordered[k_shot:])
            support = np.asarray(support_indices, dtype=np.int64)
            query = np.asarray(query_indices, dtype=np.int64)
            bank = build_typed_zid_support_bank(
                z_id[support].astype(np.float32),
                labels[support].tolist(),
                classes,
                config=config,
            )
            logits = score_zid_student_t_logits(
                bank, z_id[query].astype(np.float32), metric=metric
            ).astype(np.float64)
            winners = np.argmax(logits, axis=1)
            true_indices = np.asarray(
                [classes.index(value) for value in labels[query]], dtype=np.int64
            )
            correct = winners == true_indices
            for row, class_index in zip(logits[correct], true_indices[correct]):
                other = np.delete(row, class_index)
                maximum = float(np.max(other))
                logsumexp_other = maximum + math.log(
                    float(np.sum(np.exp(other - maximum)))
                )
                margins.append(float(row[class_index] - logsumexp_other))
            evidence.append(
                {
                    "K": k_shot,
                    "held_receiver": held_receiver,
                    "support_root_sha256": _sha(
                        sorted(physical[support].tolist())
                    ),
                    "query_root_sha256": _sha(sorted(physical[query].tolist())),
                    "query_rows": int(len(query)),
                    "correct_rows": int(np.sum(correct)),
                }
            )
    margin_array = np.asarray(margins, dtype=np.float64)
    if margin_array.size < 2 or not np.isfinite(margin_array).all():
        raise GRBJP4HeldBuilderError("Phase1 LODO margin evidence is insufficient")
    return {
        "schema": MARGIN_RECEIPT_SCHEMA,
        "target_accessed": False,
        "receiver_lodo": True,
        "pseudo_support_query_physical_id_disjoint": True,
        "correct_predictions_only": True,
        "target_query_truth_used": False,
        "margin_definition": "top1_minus_logsumexp_other_raw_qknn_score",
        "margin_evidence_sha256": _sha(evidence),
        "margins": margin_array,
    }


def build_source_aggregate(
    tap_archive: Mapping[str, np.ndarray],
    *,
    qknn_locks: Mapping[int, Phase1ZIDStudentTLock],
) -> dict[str, Any]:
    arrays = _validate_tap_archive(tap_archive)
    classes = _registry(arrays["class_ids"].astype(str).tolist())
    if set(qknn_locks) != set(K_VALUES):
        raise GRBJP4HeldBuilderError("source aggregate requires K1/K5/K10 locks")
    means, mask, counts = _receiver_day_means(arrays, classes)
    return {
        "feature_key": "z_id",
        "protocol_schema": PROTOCOL_SCHEMA,
        "ground_multiprototypes": _prototype_records(arrays, classes),
        "receiver_day_mean_schema": RECEIVER_DAY_MEAN_SCHEMA,
        "receiver_day_means": means,
        "receiver_day_mask": mask,
        "receiver_day_physical_counts": counts,
        "phase1_qknn_margin_receipt": _margin_receipt(
            arrays, classes, qknn_locks
        ),
    }


__all__ = [
    "GRBJP4HeldBuilderError",
    "K_VALUES",
    "SCHEMA",
    "build_phase1_method_lock",
    "build_phase1_qknn_locks",
    "build_source_aggregate",
]
