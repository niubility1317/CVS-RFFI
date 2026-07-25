"""Matched source-held execution primitives for D103-R2.

Package preparation, truth-free prediction, and truth-side scoring are kept as
separate callable boundaries.  The predictor never accepts query labels.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .phase1_rb_metabias4_bundle import (
    Phase1RBMetaBias4Bundle,
    apply_metabias4,
    infer_metabias4_coefficient,
)
from .rxid_metabias4_bundle import (
    RXIDMetaBias4Bundle,
    build_rxid_metabias4_bundle,
)
from .stage2_rb_metabias4_qknn import (
    baseline_zid_from_pre_relu,
    build_d102_baseline_bank,
)
from .stage2_rxid_metabias4 import (
    K1IdentifiabilityReceipt,
    audit_d103_int8,
    audit_d103_resources,
    fit_d103_stage2_state,
    predict_d103_class_indices,
    solve_d103_support_coefficient,
)
from .stage2_zid_student_t_qknn import (
    Phase1ZIDStudentTLock,
    audit_int8_margin,
    build_typed_zid_support_bank,
    identity_shared_psd_metric,
    normalize_zid_rows,
    score_zid_student_t_logits,
)


SCHEMA = "cvs.d103_r2.rxid_crossreceiver.held_execution.v1"
K_VALUES = (1, 5, 10)


class D103HeldExecutionError(ValueError):
    pass


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(header + b"\0" + array.tobytes()).hexdigest()


def validate_teacher_fit_manifest(
    manifest: Mapping[str, Any],
    teacher_arrays: Mapping[str, np.ndarray],
    *,
    expected_outer_spec: Mapping[str, str | None],
    checkpoint_sha256: str,
    runtime_sha256: str,
    teacher_archive_sha256: str,
) -> dict[str, Any]:
    """Fail closed on fit identity, access ledger, and teacher aggregation."""

    expected_manifest_keys = {
        "schema",
        "candidate_id",
        "checkpoint_sha256",
        "runtime_sha256",
        "status",
        "performance_metrics_computed",
        "target_access",
        "formal_query_access",
        "source_val_rows_used_for_training",
        "completed_meta_steps",
        "fit_elapsed_seconds",
        "peak_cuda_memory_bytes",
        "outer_spec",
        "input_sha256",
        "teacher_archive",
        "aggregation_receipt",
        "access_receipt",
        "step_receipts_sha256",
    }
    teacher_archive = manifest.get("teacher_archive")
    if (
        set(manifest) != expected_manifest_keys
        or manifest.get("schema")
        != "cvs.d103_r2.rxid_crossreceiver.phase1_fit.v1"
        or manifest.get("candidate_id")
        != "D103-R2-RXID-CROSSRECEIVER-MB4"
        or manifest.get("status")
        != "PHASE1_FIT_COMPLETE_GROUND_TEACHER_NOT_DEPLOYMENT"
        or manifest.get("performance_metrics_computed") is not False
        or manifest.get("target_access") is not False
        or manifest.get("formal_query_access") is not False
        or manifest.get("source_val_rows_used_for_training") != 0
        or manifest.get("completed_meta_steps") != 400
        or manifest.get("checkpoint_sha256") != checkpoint_sha256
        or manifest.get("runtime_sha256") != runtime_sha256
        or manifest.get("outer_spec") != dict(expected_outer_spec)
        or not np.isfinite(float(manifest.get("fit_elapsed_seconds", -1.0)))
        or float(manifest.get("fit_elapsed_seconds", -1.0)) < 0.0
        or isinstance(manifest.get("peak_cuda_memory_bytes"), bool)
        or int(manifest.get("peak_cuda_memory_bytes", -1)) < 0
        or not isinstance(teacher_archive, Mapping)
        or set(teacher_archive)
        != {"name", "sha256", "ground_only_fp32", "phase2_eligible"}
        or teacher_archive.get("name")
        != "teacher_arrays_fp32_ground_only.npz"
        or teacher_archive.get("sha256") != teacher_archive_sha256
        or teacher_archive.get("ground_only_fp32") is not True
        or teacher_archive.get("phase2_eligible") is not False
        or len(str(manifest.get("step_receipts_sha256", ""))) != 64
        or any(
            ch not in "0123456789abcdef"
            for ch in str(manifest.get("step_receipts_sha256", ""))
        )
    ):
        raise D103HeldExecutionError("fit manifest identity/access drift")
    access = manifest.get("access_receipt")
    if not isinstance(access, Mapping):
        raise D103HeldExecutionError("fit access receipt missing")
    access_body = {key: value for key, value in access.items() if key != "receipt_sha256"}
    events = access.get("events")
    if (
        set(access)
        != {
            "schema",
            "candidate_id",
            "events",
            "denied_attempts",
            "source_val_array_access",
            "target_access",
            "formal_query_access",
            "performance_selection_access",
            "receipt_sha256",
        }
        or access.get("schema")
        != "cvs.d103_r2.rxid_metabias4.phase1_trainer.v1.access_ledger"
        or access.get("candidate_id") != "D103-R2-RXID-CROSSRECEIVER-MB4"
        or access.get("denied_attempts") != 0
        or access.get("source_val_array_access") is not False
        or access.get("target_access") is not False
        or access.get("formal_query_access") is not False
        or access.get("performance_selection_access") is not False
        or not isinstance(events, list)
        or not events
        or access.get("receipt_sha256") != canonical_sha256(access_body)
    ):
        raise D103HeldExecutionError("fit access ledger drift")
    allowed_event_fields = {
        ("L_s", "fold_mask"): {"receiver_ids", "day_ids", "tx_labels"},
        ("U_s", "fold_mask"): {"receiver_ids", "day_ids"},
        ("L_s", "tx_projector"): {
            "z_dom",
            "receiver_ids",
            "day_ids",
            "tx_labels",
            "physical_ids",
        },
        ("L_s", "tx_mmd"): {
            "z_dom",
            "receiver_ids",
            "day_ids",
            "tx_labels",
            "physical_ids",
        },
        ("L_s", "class_balanced_bank"): {
            "z_dom",
            "receiver_ids",
            "day_ids",
            "tx_labels",
            "physical_ids",
        },
        ("L_s", "metabias_meta"): {
            "z_dom",
            "pre_relu",
            "receiver_ids",
            "day_ids",
            "tx_labels",
            "physical_ids",
        },
        ("L_s", "receiver_day_self_supervision"): {
            "z_dom",
            "receiver_ids",
            "day_ids",
            "tx_labels",
            "physical_ids",
        },
        ("U_s", "receiver_day_self_supervision"): {
            "z_dom",
            "receiver_ids",
            "day_ids",
            "physical_ids",
        },
        ("L_s", "vicreg"): {"z_dom"},
        ("U_s", "vicreg"): {"z_dom"},
        ("L_s", "final_teacher_aggregation"): {
            "z_dom",
            "receiver_ids",
            "day_ids",
            "tx_labels",
            "physical_ids",
        },
    }
    for ordinal, event in enumerate(events):
        key = (event.get("role"), event.get("operation")) if isinstance(
            event, Mapping
        ) else (None, None)
        if (
            not isinstance(event, Mapping)
            or set(event)
            != {"ordinal", "role", "operation", "fields", "row_count"}
            or event.get("ordinal") != ordinal
            or key not in allowed_event_fields
            or not isinstance(event.get("fields"), list)
            or not event.get("fields")
            or len(event["fields"]) != len(set(event["fields"]))
            or event["fields"] != sorted(event["fields"])
            or set(event["fields"]) != allowed_event_fields.get(key, set())
            or isinstance(event.get("row_count"), bool)
            or int(event.get("row_count", -1)) < 0
        ):
            raise D103HeldExecutionError("fit access event closure drift")
    aggregation = manifest.get("aggregation_receipt")
    if not isinstance(aggregation, Mapping):
        raise D103HeldExecutionError("fit aggregation receipt missing")
    aggregation_keys = {
        "schema",
        "candidate_id",
        "completed_meta_steps",
        "eligible_labeled_rows",
        "unlabeled_rows_used",
        "source_val_rows_used",
        "bank_cell_count",
        "registered_class_count",
        "class_cell_count",
        "minimum_physical_samples_per_class_cell",
        "all_eligible_labeled_physical_rows_used",
        "aggregation_order",
        "class_weight",
        "array_shapes",
        "array_sha256",
        "contains_receiver_values",
        "contains_day_values",
        "contains_class_values",
        "contains_physical_ids",
        "contains_optimizer",
    }
    expected_arrays = {
        "U": _array_sha256(teacher_arrays["U"]),
        "B": _array_sha256(teacher_arrays["B"]),
        "bank_g": _array_sha256(teacher_arrays["bank_g"]),
        "bank_t": _array_sha256(teacher_arrays["bank_t"]),
        "bank_precision": _array_sha256(teacher_arrays["bank_precision"]),
        "bank_sigma": _array_sha256(teacher_arrays["bank_sigma"]),
    }
    expected_shapes = {
        name: list(np.asarray(teacher_arrays[name]).shape)
        for name in expected_arrays
    }
    registered_class_count = int(
        aggregation.get("registered_class_count", 0)
    )
    if (
        set(aggregation) != aggregation_keys
        or aggregation.get("schema")
        != "cvs.d103_r2.rxid_metabias4.phase1_trainer.v1.teacher_aggregation_receipt"
        or aggregation.get("candidate_id")
        != "D103-R2-RXID-CROSSRECEIVER-MB4"
        or aggregation.get("completed_meta_steps") != 400
        or int(aggregation.get("eligible_labeled_rows", 0)) <= 0
        or aggregation.get("unlabeled_rows_used") != 0
        or aggregation.get("source_val_rows_used") != 0
        or int(aggregation.get("bank_cell_count", 0)) <= 0
        or registered_class_count <= 0
        or int(aggregation.get("class_cell_count", 0)) <= 0
        or int(
            aggregation.get(
                "minimum_physical_samples_per_class_cell", 0
            )
        )
        <= 0
        or aggregation.get("all_eligible_labeled_physical_rows_used")
        is not True
        or aggregation.get("aggregation_order")
        != "physical_mean_within_class_cell_then_equal_mean_over_classes"
        or not np.isclose(
            float(aggregation.get("class_weight", -1.0)),
            1.0 / registered_class_count,
            rtol=0.0,
            atol=1.0e-15,
        )
        or aggregation.get("array_shapes") != expected_shapes
        or aggregation.get("contains_optimizer") is not False
        or aggregation.get("contains_receiver_values") is not False
        or aggregation.get("contains_day_values") is not False
        or aggregation.get("contains_class_values") is not False
        or aggregation.get("contains_physical_ids") is not False
        or aggregation.get("array_sha256") != expected_arrays
    ):
        raise D103HeldExecutionError("fit aggregation/teacher binding drift")
    inputs = manifest.get("input_sha256")
    if (
        not isinstance(inputs, Mapping)
        or set(inputs)
        != {"labeled_archive", "unlabeled_archive", "source_val_seal"}
        or any(
            len(str(value)) != 64
            or any(ch not in "0123456789abcdef" for ch in str(value))
            for value in inputs.values()
        )
    ):
        raise D103HeldExecutionError("fit input SHA closure drift")
    return {
        "access_receipt_sha256": str(access["receipt_sha256"]),
        "input_sha256": dict(inputs),
        "outer_spec": dict(expected_outer_spec),
    }


def package_id(held_receiver: str, k_shot: int) -> str:
    return hashlib.sha256(
        f"{held_receiver}\0{k_shot}".encode("utf-8")
    ).hexdigest()[:20]


def build_receiver_package_indices(
    receiver_ids: np.ndarray,
    labels: np.ndarray,
    physical_ids: np.ndarray,
    *,
    held_receiver: str,
    registered_classes: Sequence[str],
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray]:
    receivers = np.asarray(receiver_ids).astype(str)
    truth = np.asarray(labels).astype(str)
    physical = np.asarray(physical_ids).astype(str)
    classes = tuple(str(value) for value in registered_classes)
    if (
        k_shot not in K_VALUES
        or len(receivers) != len(truth)
        or len(truth) != len(physical)
        or np.unique(physical).size != len(physical)
        or len(classes) != 6
        or len(set(classes)) != 6
    ):
        raise D103HeldExecutionError("held package metadata closure drift")
    support: list[int] = []
    query: list[int] = []
    for class_id in classes:
        local = np.flatnonzero(
            (receivers == str(held_receiver)) & (truth == class_id)
        )
        ranked = sorted(local.astype(int).tolist(), key=lambda row: physical[row])
        if len(ranked) <= k_shot:
            raise D103HeldExecutionError(
                f"held receiver/class lacks K+query rows: {held_receiver}/{class_id}"
            )
        support.extend(ranked[:k_shot])
        query.extend(ranked[k_shot:])
    support_rows = np.asarray(support, dtype=np.int64)
    query_rows = np.asarray(query, dtype=np.int64)
    if (
        np.intersect1d(physical[support_rows], physical[query_rows]).size
        or set(truth[support_rows].tolist()) != set(classes)
        or set(truth[query_rows].tolist()) != set(classes)
    ):
        raise D103HeldExecutionError("support/query physical or class closure drift")
    return support_rows, query_rows


def frozen_qknn(k_shot: int) -> Phase1ZIDStudentTLock:
    if k_shot not in K_VALUES:
        raise D103HeldExecutionError("unsupported K")
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
        phase1_lodo_receipt_sha256=canonical_sha256(
            {"schema": SCHEMA, "lock": "phase1_lodo"}
        ),
        quantization_margin_audit_sha256=canonical_sha256(
            {"schema": SCHEMA, "lock": "int8_margin"}
        ),
    )


def compile_teacher_bundle(
    teacher_arrays: Mapping[str, np.ndarray],
    fit_manifest: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    runtime_sha256: str,
    method_lock_sha256: str,
    training_receipt_sha256: str,
    tx_probe_receipt_sha256: str,
    tx_probe_mean: float,
    tx_probe_max: float,
) -> RXIDMetaBias4Bundle:
    required = {"U", "B", "bank_g", "bank_t", "bank_precision", "bank_sigma"}
    if set(teacher_arrays) != required:
        raise D103HeldExecutionError("teacher array member closure drift")
    aggregation = fit_manifest.get("aggregation_receipt")
    if not isinstance(aggregation, Mapping):
        raise D103HeldExecutionError("fit aggregation receipt missing")
    cells = int(aggregation.get("bank_cell_count", 0))
    classes = int(aggregation.get("registered_class_count", 0))
    minimum = int(aggregation.get("minimum_physical_samples_per_class_cell", 0))
    if cells < 2 or classes < 2 or minimum < 2:
        raise D103HeldExecutionError("fit aggregation counts drift")
    return build_rxid_metabias4_bundle(
        np.asarray(teacher_arrays["U"], dtype=np.float32),
        np.asarray(teacher_arrays["B"], dtype=np.float32),
        np.asarray(teacher_arrays["bank_g"], dtype=np.float32),
        np.asarray(teacher_arrays["bank_t"], dtype=np.float32),
        np.asarray(teacher_arrays["bank_precision"], dtype=np.float32),
        np.asarray(teacher_arrays["bank_sigma"], dtype=np.float32),
        cell_min_physical_count=np.full(cells, minimum, dtype=np.int16),
        cell_class_count=np.full(cells, classes, dtype=np.int16),
        checkpoint_sha256=checkpoint_sha256,
        runtime_sha256=runtime_sha256,
        method_lock_sha256=method_lock_sha256,
        training_receipt_sha256=training_receipt_sha256,
        nested_receipt_sha256=canonical_sha256(
            {
                "schema": SCHEMA,
                "outer_spec": fit_manifest.get("outer_spec"),
                "candidate_id": fit_manifest.get("candidate_id"),
            }
        ),
        tx_probe_receipt_sha256=tx_probe_receipt_sha256,
        aggregation_receipt_sha256=canonical_sha256(dict(aggregation)),
        quantization_receipt_sha256=canonical_sha256(
            {
                "schema": SCHEMA,
                "mode": "frozen_int8_fp16_compile_then_margin_audit",
            }
        ),
        tx_probe_mean_balanced_accuracy=tx_probe_mean,
        tx_probe_max_balanced_accuracy=tx_probe_max,
    )


def encode_domain_rows(
    bundle: RXIDMetaBias4Bundle, z_dom: np.ndarray
) -> np.ndarray:
    rows = np.asarray(z_dom)
    if (
        rows.dtype != np.float32
        or rows.ndim != 2
        or rows.shape[1] != 160
        or not np.isfinite(rows).all()
    ):
        raise D103HeldExecutionError("domain rows must be finite float32 [N,160]")
    encoded = rows.astype(np.float64) @ bundle.decode_u().astype(np.float64).T
    norms = np.linalg.norm(encoded, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise D103HeldExecutionError("encoded domain row has zero norm")
    return np.asarray(encoded / norms, dtype=np.float32)


def _shift(bundle: RXIDMetaBias4Bundle, coefficient: np.ndarray) -> np.ndarray:
    return np.asarray(
        bundle.decode_b().astype(np.float64)
        @ np.asarray(coefficient, dtype=np.float64),
        dtype=np.float64,
    )


def leave_day_stability(
    outer_bundle: RXIDMetaBias4Bundle,
    day_bundles: Sequence[RXIDMetaBias4Bundle],
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
) -> dict[str, Any]:
    if len(day_bundles) != 4:
        raise D103HeldExecutionError("leave-day evidence requires exactly 4 bundles")
    outer = solve_d103_support_coefficient(
        outer_bundle,
        support_zdom,
        support_labels,
        registered_classes,
        active_k=1,
    )
    outer_shift = _shift(outer_bundle, outer.coefficient_fp16)
    outer_norm = float(np.linalg.norm(outer_shift))
    norms: list[float] = []
    cosines: list[float] = []
    for bundle in day_bundles:
        solution = solve_d103_support_coefficient(
            bundle,
            support_zdom,
            support_labels,
            registered_classes,
            active_k=1,
        )
        shift = _shift(bundle, solution.coefficient_fp16)
        norm = float(np.linalg.norm(shift))
        norms.append(norm)
        cosine = (
            float(np.dot(outer_shift, shift) / (outer_norm * norm))
            if outer_norm > 0.0 and norm > 0.0
            else -1.0
        )
        cosines.append(float(np.clip(cosine, -1.0, 1.0)))
    return {
        "outer_shift_norm": outer_norm,
        "day_shift_norms": norms,
        "direction_cosines": cosines,
        "direction_cosine_median": float(np.median(cosines)),
        "actual_160d_shift_used": True,
        "query_rows_used": 0,
    }


def _d103_int8_receipt(
    bundle: RXIDMetaBias4Bundle,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    registered_classes: Sequence[str],
    qknn: Phase1ZIDStudentTLock,
    direction_cosine: float,
) -> tuple[K1IdentifiabilityReceipt, dict[str, Any]]:
    solution = solve_d103_support_coefficient(
        bundle,
        support_zdom,
        support_labels,
        registered_classes,
        active_k=1,
    )
    support_shifted = normalize_zid_rows(
        np.maximum(
            np.asarray(support_pre_relu, dtype=np.float64)
            + _shift(bundle, solution.coefficient_fp16)[None, :],
            0.0,
        ).astype(np.float32)
    )
    bank = build_typed_zid_support_bank(
        support_shifted,
        support_labels,
        registered_classes,
        config=qknn,
    )
    metric = identity_shared_psd_metric(config=qknn)
    audit = dict(
        audit_int8_margin(
            bank,
            support_shifted,
            support_labels,
            support_shifted,
            metric=metric,
        )
    )
    receipt_values = {
        "view_top1_agreement": float(audit["top1_agreement"]),
        "large_margin_flip_count": int(audit["margin_sign_flip_count"]),
        "independent_direction_cosine_median": float(direction_cosine),
        "independent_episode_count": 4,
        "query_rows_used_for_fit": 0,
    }
    receipt_body = {
        **receipt_values,
        "evidence_rows": "support_only_no_held_query",
    }
    audit["evidence_rows"] = "support_only_no_held_query"
    audit["held_query_rows_used"] = 0
    return (
        K1IdentifiabilityReceipt(
            **receipt_values,
            receipt_sha256=canonical_sha256(
                {"schema": SCHEMA, **receipt_body}
            ),
        ),
        audit,
    )


def predict_matched_row(
    *,
    held_receiver: str,
    held_class: str | None,
    k_shot: int,
    support_pre_relu: np.ndarray,
    support_zdom: np.ndarray,
    support_labels: Sequence[str],
    query_pre_relu: np.ndarray,
    query_physical_ids: Sequence[str],
    registered_classes: Sequence[str],
    d102_bundle: Phase1RBMetaBias4Bundle,
    d103_outer_bundle: RXIDMetaBias4Bundle,
    d103_day_bundles: Sequence[RXIDMetaBias4Bundle],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Produce one truth-free M0/D102/D103 prediction artifact."""

    qknn = frozen_qknn(k_shot)
    classes = tuple(str(value) for value in registered_classes)
    labels = tuple(str(value) for value in support_labels)
    physical = tuple(str(value) for value in query_physical_ids)
    if len(physical) != len(query_pre_relu):
        raise D103HeldExecutionError("query physical/pre-ReLU alignment drift")

    base_bank, base_metric, _ = build_d102_baseline_bank(
        support_pre_relu, labels, classes, qknn_config=qknn
    )
    base_query = baseline_zid_from_pre_relu(query_pre_relu)
    base_index = np.argmax(
        score_zid_student_t_logits(base_bank, base_query, metric=base_metric),
        axis=1,
    )

    d102_coefficient, d102_audit = infer_metabias4_coefficient(
        d102_bundle, support_zdom, labels
    )
    d102_support = apply_metabias4(
        d102_bundle, support_pre_relu, d102_coefficient
    ).astype(np.float32)
    d102_query = apply_metabias4(
        d102_bundle, query_pre_relu, d102_coefficient
    ).astype(np.float32)
    d102_bank = build_typed_zid_support_bank(
        d102_support, labels, classes, config=qknn
    )
    d102_metric = identity_shared_psd_metric(config=qknn)
    d102_index = np.argmax(
        score_zid_student_t_logits(d102_bank, d102_query, metric=d102_metric),
        axis=1,
    )

    stability = None
    k1_receipt = None
    raw_int8: dict[str, Any] | None = None
    if k_shot == 1:
        stability = leave_day_stability(
            d103_outer_bundle,
            d103_day_bundles,
            support_zdom,
            labels,
            classes,
        )
        k1_receipt, raw_int8 = _d103_int8_receipt(
            d103_outer_bundle,
            support_pre_relu,
            support_zdom,
            labels,
            classes,
            qknn,
            float(stability["direction_cosine_median"]),
        )
    state = fit_d103_stage2_state(
        d103_outer_bundle,
        np.asarray(support_pre_relu, dtype=np.float32),
        np.asarray(support_zdom, dtype=np.float32),
        labels,
        classes,
        qknn_config=qknn,
        stage="S_C",
        support_receipt_sha256=canonical_sha256(
            {
                "schema": SCHEMA,
                "held_receiver": held_receiver,
                "held_class": held_class,
                "K": k_shot,
                "support_physical_ids_not_persisted": True,
            }
        ),
        k1_identifiability_receipt=k1_receipt,
    )
    d103_index = predict_d103_class_indices(state, query_pre_relu)
    int8 = audit_d103_int8(
        state, support_pre_relu, labels, query_pre_relu
    )
    resources = audit_d103_resources(state)
    artifact = {
        "schema": SCHEMA + ".prediction",
        "held_receiver": held_receiver,
        "held_class": held_class,
        "K": k_shot,
        "registered_classes": list(classes),
        "query_physical_ids": list(physical),
        "m0_predictions": [classes[int(index)] for index in base_index],
        "d102_predictions": [classes[int(index)] for index in d102_index],
        "d103_predictions": [classes[int(index)] for index in d103_index],
        "d102_bundle_content_root_sha256": d102_bundle.content_root_sha256,
        "d103_bundle_content_root_sha256": d103_outer_bundle.content_root_sha256,
        "d103_state_receipt_sha256": state.state_receipt_sha256,
        "d102_fit_audit": d102_audit,
        "d103_fit_audit": dict(state.fit_audit),
        "int8_audit": int8,
        "resource_audit": resources,
        "k1_view_audit": raw_int8,
        "k1_stability": stability,
        "query_truth_present": False,
        "query_rows_used_for_fit": 0,
        "all_registered_classes_compete": True,
    }
    artifact["prediction_receipt_sha256"] = canonical_sha256(artifact)
    return artifact, stability


def _metrics(truth: Sequence[str], predicted: Sequence[str]) -> tuple[float, float]:
    truth_array = np.asarray(truth).astype(str)
    predicted_array = np.asarray(predicted).astype(str)
    classes = sorted(set(truth_array.tolist()))
    accuracies = [
        float(np.mean(predicted_array[truth_array == class_id] == class_id))
        for class_id in classes
    ]
    return float(np.mean(accuracies)), float(np.min(accuracies))


def score_prediction_artifact(
    prediction: Mapping[str, Any],
    truth_labels: Sequence[str],
) -> dict[str, Any]:
    truth = tuple(str(value) for value in truth_labels)
    n = len(prediction.get("query_physical_ids", ()))
    if (
        prediction.get("query_truth_present") is not False
        or len(truth) != n
        or any(
            len(prediction.get(field, ())) != n
            for field in ("m0_predictions", "d102_predictions", "d103_predictions")
        )
    ):
        raise D103HeldExecutionError("truth-side scoring alignment drift")
    base_ba, base_floor = _metrics(truth, prediction["m0_predictions"])
    d102_ba, d102_floor = _metrics(truth, prediction["d102_predictions"])
    d103_ba, d103_floor = _metrics(truth, prediction["d103_predictions"])
    base_correct = np.asarray(prediction["m0_predictions"]).astype(str) == np.asarray(
        truth
    )
    adapted_correct = np.asarray(prediction["d103_predictions"]).astype(
        str
    ) == np.asarray(truth)
    audit = dict(prediction["d103_fit_audit"])
    int8 = dict(prediction["int8_audit"])
    k1_view = prediction.get("k1_view_audit")
    stability = prediction.get("k1_stability")
    if int(prediction["K"]) == 1:
        if not isinstance(k1_view, Mapping) or not isinstance(stability, Mapping):
            raise D103HeldExecutionError("K1 evidence missing from prediction artifact")
        view_agreement = float(k1_view["top1_agreement"])
        view_flips = int(k1_view["margin_sign_flip_count"])
        direction_cosine = float(stability["direction_cosine_median"])
    else:
        view_agreement = float(int8["top1_agreement"])
        view_flips = int(int8["large_margin_flip_count"])
        direction_cosine = 1.0
    return {
        "held_receiver": prediction["held_receiver"],
        "held_class": prediction["held_class"],
        "K": int(prediction["K"]),
        "base_ba": base_ba,
        "adapted_ba": d103_ba,
        "base_floor": base_floor,
        "adapted_floor": d103_floor,
        "wrong_to_correct": int(np.sum(~base_correct & adapted_correct)),
        "correct_to_wrong": int(np.sum(base_correct & ~adapted_correct)),
        "joint_score_d102": (d102_ba + d102_floor) / 2.0,
        "joint_score_d103": (d103_ba + d103_floor) / 2.0,
        "d102_comparator_status": (
            "DIAGNOSTIC_REJECTED_D102_COMPARATOR_NON_PROMOTABLE"
        ),
        "d102_bundle_content_root_sha256": prediction[
            "d102_bundle_content_root_sha256"
        ],
        "active": audit["status"] == "ACTIVE",
        "information_rank": int(audit["data_information_rank"]),
        "min_singular_value": float(audit["data_minimum_singular_value"]),
        "condition_number": float(audit["system_condition_number"]),
        "prior_fraction": float(audit["prior_fraction"]),
        "coefficient_norm": float(audit["coefficient_norm"]),
        "view_top1_agreement": view_agreement,
        "view_large_margin_flip_count": view_flips,
        "direction_cosine_median": direction_cosine,
        "k1_receipt_evidence_scope": audit.get(
            "k1_receipt_evidence_scope"
        ),
    }


__all__ = [
    "D103HeldExecutionError",
    "K_VALUES",
    "SCHEMA",
    "build_receiver_package_indices",
    "canonical_sha256",
    "compile_teacher_bundle",
    "encode_domain_rows",
    "frozen_qknn",
    "leave_day_stability",
    "package_id",
    "predict_matched_row",
    "score_prediction_artifact",
    "sha256_file",
    "validate_teacher_fit_manifest",
]
