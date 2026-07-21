"""Matched D81/D99/D100 query evaluation on one sealed Phase2 row.

This is a development evaluator for wireless-signal classification.  It reuses
the already validated p2_min_v1 enrollment/apply packages, fits every state from
registered support only, writes predictions before any truth join, and never
updates state from query rows.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from cvsrffi import stage2_d99_ra_cgtmk_d81 as d99
from cvsrffi import stage2_d100_ra_cgspr_lgf as d100
from cvsrffi import stage2_d99_d100_phase1_lodo as lodo
from cvsrffi.somph_diagnostic_bundle_loader import load_verified_somph_predictor_bundle
from cvsrffi.somph_predictor_bundle import FORMAL_LEO_WEAK_SCENARIOS
from cvsrffi.stage2_d42_unified_shrinkage_lda import (
    D42UnifiedShrinkageLDAConfig,
    fit_d42_unified_shrinkage_lda,
    score_d42_unified_shrinkage_lda,
)
from cvsrffi.stage2_diag_cosine_exploration import (
    _descriptor,
    _device,
    _output_root,
    _sha256_file,
    _validate_matched_packages,
    _write_json_new,
    _write_npz_new,
    forward_zid160,
    registered_feature,
)
from cvsrffi.stage2_d81_query_evaluation import (
    D81_ALLOWED_SKLEARN_RUNTIME_VERSIONS,
    _audit_fit,
    _registered_handles,
)
from cvsrffi.stage2_predictor_runtime import load_torchscript_backbone_same_fd


SCHEMA = "cvs.phase2.d99_d100.matched_query_evaluation.v1"
CANDIDATES = ("d81_ground_nuisance_cauchy_center", "d99_ra_cgtmk_d81", "d100_ra_cgspr_lgf")
ALLOWED_K = (1, 5, 10, 20)
ALLOWED_NEW_COUNTS = (2, 5, 10, 20)
_LOCKED_PARAMETER_FIELDS = {
    "eta",
    "student_nu",
    "kernel_volume_gamma",
    "shared_h0",
    "scale_prior_strength",
    "scale_min_ratio",
    "scale_max_ratio",
    "d99_temperature",
    "lambda0",
    "ridge_temperature",
    "alpha",
}


class D99D100QueryEvaluationError(ValueError):
    """Raised when matched row geometry or a frozen evaluation input drifts."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def typed_class_binding_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a real D19 v1 binding into the typed v2 logit-column contract."""

    schema = payload.get("schema")
    if schema == "cvs.phase2.d20_adv3b02_class_binding.v2":
        return dict(payload)
    if schema != "cvs.phase2.d19_adv3b02_class_binding.v1":
        raise D99D100QueryEvaluationError("D19 class binding schema drift")
    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != 6:
        raise D99D100QueryEvaluationError("D19 v1 class binding entries drift")
    typed_entries = []
    for row in entries:
        if not isinstance(row, Mapping) or "direct_logit_index" in row:
            raise D99D100QueryEvaluationError("D19 v1 class binding row drift")
        typed = dict(row)
        typed["direct_logit_index"] = int(row.get("class_index", -1))
        typed_entries.append(typed)
    return {
        **dict(payload),
        "schema": "cvs.phase2.d20_adv3b02_class_binding.v2",
        "entries": typed_entries,
    }


def locked_parameters_from_lodo(
    receipt: Mapping[str, Any], *, k_shot: int
) -> Mapping[str, float]:
    """Read one K-specific parameter row selected using Phase1-only evidence."""

    if (
        receipt.get("schema") != lodo.SCHEMA
        or receipt.get("status") not in {lodo.STATUS_DIAGNOSTIC, lodo.STATUS_FORMAL}
        or not lodo.verify_receipt(receipt)
    ):
        raise D99D100QueryEvaluationError("Phase1 LODO schema drift")
    formal = receipt.get("status") == lodo.STATUS_FORMAL
    if (
        receipt.get("formal_phase1_lock") is not formal
        or receipt.get("canonical_lock_artifact_write_allowed") is not formal
        or receipt.get("formal_authority_status")
        != (lodo.STATUS_FORMAL if formal else lodo.STATUS_BLOCKED)
    ):
        raise D99D100QueryEvaluationError("Phase1 LODO status contract drift")
    protocol = receipt.get("protocol_audit")
    if (
        not isinstance(protocol, Mapping)
        or protocol.get("phase1_only") is not True
        or int(protocol.get("target_rows_used", -1)) != 0
        or int(protocol.get("query_rows_used_for_selection", -1)) != 0
        or protocol.get("clean_or_raw_iq_used") is not False
        or protocol.get("class_specific_hyperparameters") is not False
    ):
        raise D99D100QueryEvaluationError("Phase1-only parameter selection audit drift")
    try:
        selected = dict(receipt["locked_parameters_by_k"][str(int(k_shot))])
        selection = dict(receipt["selected_by_k"][str(int(k_shot))]["selected"])
    except (KeyError, TypeError, ValueError) as exc:
        raise D99D100QueryEvaluationError("K-specific LODO parameters are missing") from exc
    if set(selected) != _LOCKED_PARAMETER_FIELDS:
        raise D99D100QueryEvaluationError("LODO locked parameter registry drift")
    converted = {name: float(value) for name, value in selected.items()}
    if not all(np.isfinite(value) for value in converted.values()):
        raise D99D100QueryEvaluationError("LODO locked parameter is non-finite")
    if not 0.0 <= converted["eta"] <= 1.0 or not 0.0 <= converted["alpha"] <= 1.0:
        raise D99D100QueryEvaluationError("LODO fusion weight drift")
    if dict(selection.get("effective_parameters", {})) != selected:
        raise D99D100QueryEvaluationError("LODO effective parameter binding drift")
    guard = selection.get("guard")
    forced = selection.get("alpha_forced_zero")
    if not isinstance(guard, Mapping) or not isinstance(forced, bool):
        raise D99D100QueryEvaluationError("LODO alpha guard is missing")
    if converted["alpha"] > 0.0 and (
        forced
        or guard.get("bidirectional_rescue_nonzero") is not True
        or guard.get(
            "every_receiver_pseudo_new_pair_floor_old_new_h_non_decreasing"
        )
        is not True
        or int(guard.get("degraded_pair_count", -1)) != 0
    ):
        raise D99D100QueryEvaluationError("unsafe nonzero alpha escaped Phase1 guard")
    if forced and converted["alpha"] != 0.0:
        raise D99D100QueryEvaluationError("forced alpha-zero guard drift")
    return MappingProxyType(converted)


def class_binding_maps(
    payload: Mapping[str, Any],
    *,
    payload_sha256: str,
    payload_bytes: bytes | None = None,
    checkpoint_sha256: str,
    old_handles: Sequence[str],
    target_old_tx_labels: Sequence[str],
) -> tuple[Mapping[str, str], Mapping[str, str]]:
    """Bind fixed Phase1 TX order to the current row's opaque old handles."""

    claimed_sha = str(payload_sha256).lower()
    raw = _canonical_bytes(payload) if payload_bytes is None else bytes(payload_bytes)
    if hashlib.sha256(raw).hexdigest() != claimed_sha:
        raise D99D100QueryEvaluationError("class binding bytes/SHA drift")
    try:
        raw_payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise D99D100QueryEvaluationError("class binding raw JSON drift") from exc
    if (
        not isinstance(raw_payload, Mapping)
        or typed_class_binding_payload(raw_payload) != dict(payload)
    ):
        raise D99D100QueryEvaluationError("class binding raw/typed semantic drift")
    entries = payload.get("entries")
    if (
        payload.get("schema") != "cvs.phase2.d20_adv3b02_class_binding.v2"
        or payload.get("checkpoint_sha256") != checkpoint_sha256
        or not isinstance(entries, list)
        or len(entries) != 6
    ):
        raise D99D100QueryEvaluationError("class binding schema/checkpoint drift")
    expected_handles = tuple(str(value) for value in old_handles)
    locked_target_tx = tuple(str(value) for value in target_old_tx_labels)
    if (
        len(expected_handles) != 6
        or len(set(expected_handles)) != 6
        or len(locked_target_tx) != 6
        or len(set(locked_target_tx)) != 6
    ):
        raise D99D100QueryEvaluationError("old handle registry drift")
    ordered = sorted(entries, key=lambda row: int(row.get("class_index", -1)))
    if [int(row.get("class_index", -1)) for row in ordered] != list(range(6)):
        raise D99D100QueryEvaluationError("class binding class index drift")
    phase1_tx_order: list[str] = []
    historical_handles: list[str] = []
    for index, row in enumerate(ordered):
        tx = str(row.get("phase1_tx", ""))
        historical_handle = str(row.get("registered_class_handle", ""))
        if (
            int(row.get("direct_logit_index", -1)) != index
            or not tx
            or not historical_handle
            or tx in phase1_tx_order
            or historical_handle in historical_handles
        ):
            raise D99D100QueryEvaluationError("class binding Phase1 registry drift")
        phase1_tx_order.append(tx)
        historical_handles.append(historical_handle)
    if tuple(phase1_tx_order) != locked_target_tx:
        raise D99D100QueryEvaluationError("class binding target-old TX order drift")
    tx_to_handle = dict(zip(locked_target_tx, expected_handles, strict=True))
    handle_to_tx = {handle: tx for tx, handle in tx_to_handle.items()}
    return MappingProxyType(tx_to_handle), MappingProxyType(handle_to_tx)


def _array_receipt(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "nbytes": int(array.nbytes),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _d81_state_receipt(state: Any) -> str:
    payload = {
        "schema": str(state.schema),
        "classes": list(state.classes),
        "old_class_count": int(state.old_class_count),
        "covariance_policy": str(state.covariance_policy),
        "log_diag_fp32": _array_receipt(state.log_diag_fp32),
        "coef1_qint8": _array_receipt(state.coef1_qint8),
        "coef2_qint8": _array_receipt(state.coef2_qint8),
        "scale1_fp16": _array_receipt(state.scale1_fp16),
        "scale2_fp16": _array_receipt(state.scale2_fp16),
        "intercept_fp16": _array_receipt(state.intercept_fp16),
    }
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def validate_lodo_input_binding(
    receipt: Mapping[str, Any],
    *,
    bundle: d99.Phase1GroundAggregateBundle,
    ground_manifest_sha256: str,
    base_d99_lock: d99.Phase1D99Lock,
    d81_ground_manifest_sha256: str,
    d81_ground_component_sha256: str,
    checkpoint_sha256: str,
) -> None:
    """Bind the selected Phase1 row to the exact current model inputs."""

    if not lodo.verify_receipt(receipt):
        raise D99D100QueryEvaluationError("Phase1 LODO receipt fixed-point drift")
    ground = receipt.get("ground", {})
    scorer = receipt.get("d81_scorer", {})
    scorer_receipt = scorer.get("receipt", {}) if isinstance(scorer, Mapping) else {}
    if (
        len(bundle.domain_ids) != 7
        or len(set(bundle.domain_ids)) != 7
        or ground.get("domain_ids") != list(bundle.domain_ids)
        or ground.get("bundle_sha256") != bundle.bundle_sha256
        or ground.get("aggregation_receipt_sha256")
        != bundle.aggregation_receipt.receipt_sha256
        or ground.get("release_manifest_sha256") != str(ground_manifest_sha256).lower()
        or len(dict(ground.get("receiver_domain_map", {}))) != 7
        or set(dict(ground.get("receiver_domain_map", {})).values())
        != set(bundle.domain_ids)
        or receipt.get("base_d99_lock_digest") != base_d99_lock.lock_digest
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or scorer_receipt.get("phase1_checkpoint_sha256") != checkpoint_sha256
        or scorer_receipt.get("ground_manifest_sha256")
        != str(d81_ground_manifest_sha256).lower()
        or scorer_receipt.get("ground_component_npz_sha256")
        != str(d81_ground_component_sha256).lower()
    ):
        raise D99D100QueryEvaluationError("LODO/current input binding drift")


def _require_cross_state_lock(
    before_enrollment: Mapping[str, Any],
    before_apply: Mapping[str, Any],
    after_enrollment: Mapping[str, Any],
    after_apply: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    for field in (
        "receiver",
        "seed",
        "k_shot",
        "phase1_checkpoint_sha256",
        "feature_runtime_sha256",
        "method_lock_sha256",
        "row_handle",
        "row_manifest_sha256",
    ):
        left = before_apply.get(field, before_enrollment.get(field))
        right = after_apply.get(field, after_enrollment.get(field))
        if left != right:
            raise D99D100QueryEvaluationError(f"before/after {field} drift")
    old_classes = _registered_handles(before_enrollment)
    classes = _registered_handles(after_enrollment)
    k_shot = int(after_enrollment.get("k_shot", -1))
    if (
        classes[: len(old_classes)] != old_classes
        or len(old_classes) != 6
        or len(classes) - len(old_classes) not in ALLOWED_NEW_COUNTS
        or k_shot not in ALLOWED_K
    ):
        raise D99D100QueryEvaluationError("matched class/K registry drift")
    return old_classes, classes, k_shot


def _support_features(
    payload: Mapping[str, np.ndarray],
    *,
    model: Any,
    runtime_device: Any,
    class_handles: tuple[str, ...],
    k_shot: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ranks = np.asarray(payload["support_rank_within_class"], dtype=np.int64)
    indices = np.asarray(payload["support_class_indices"], dtype=np.int64)
    tokens = np.asarray(payload["support_tokens"]).astype(str)
    mask = ranks < int(k_shot)
    if (
        ranks.shape != indices.shape
        or tokens.shape != ranks.shape
        or ranks.ndim != 1
        or int(np.sum(mask)) != len(class_handles) * int(k_shot)
        or len(set(tokens[mask].tolist())) != int(np.sum(mask))
        or int(indices[mask].min()) != 0
        or int(indices[mask].max()) != len(class_handles) - 1
    ):
        raise D99D100QueryEvaluationError("support assignment/token drift")
    iq = np.asarray(payload["support_leo_weak_iq"], dtype=np.float32)[mask]
    zid = forward_zid160(model, iq, device=runtime_device, batch_size=64)
    features = registered_feature(iq, zid)
    labels = np.asarray(class_handles, dtype=str)[indices[mask]]
    return features, labels, tokens[mask], indices[mask]


def _query_features(
    payload: Mapping[str, np.ndarray], *, model: Any, runtime_device: Any
) -> tuple[np.ndarray, np.ndarray]:
    iq = np.asarray(payload["query_leo_weak_iq"], dtype=np.float32)
    tokens = np.asarray(payload["query_tokens"]).astype(str)
    if iq.ndim < 2 or tokens.ndim != 1 or len(iq) != len(tokens) or len(iq) == 0:
        raise D99D100QueryEvaluationError("query payload drift")
    zid = forward_zid160(model, iq, device=runtime_device, batch_size=1)
    return registered_feature(iq, zid), tokens


def _d99_config(
    base: d99.Phase1D99Lock,
    parameters: Mapping[str, float],
    *,
    active_k: int,
) -> d99.Phase1D99Lock:
    k_shot = int(active_k)
    if k_shot not in ALLOWED_K:
        raise D99D100QueryEvaluationError("active K is not supported")
    updates: dict[str, Any] = {
        "student_nu": parameters["student_nu"],
        "kernel_volume_gamma": parameters["kernel_volume_gamma"],
        "shared_h0": parameters["shared_h0"],
        "scale_prior_strength": parameters["scale_prior_strength"],
        "scale_min_ratio": parameters["scale_min_ratio"],
        "scale_max_ratio": parameters["scale_max_ratio"],
        f"eta_k{k_shot}": parameters["eta"],
    }
    return replace(base, **updates)


def _d100_config(
    config99: d99.Phase1D99Lock,
    parameters: Mapping[str, float],
    *,
    active_k: int,
    phase1_lodo_receipt_sha256: str,
    phase2_authority_sha256: str,
) -> d100.Phase1D100Lock:
    k_shot = int(active_k)
    if k_shot not in ALLOWED_K:
        raise D99D100QueryEvaluationError("active K is not supported")
    lodo_sha = str(phase1_lodo_receipt_sha256).lower()
    authority_sha = str(phase2_authority_sha256).lower()
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in (lodo_sha, authority_sha)
    ):
        raise D99D100QueryEvaluationError("LODO receipt SHA drift")
    values: dict[str, Any] = {}
    for k in ALLOWED_K:
        row = (
            (
                parameters["lambda0"],
                parameters["ridge_temperature"],
                parameters["d99_temperature"],
                parameters["alpha"],
            )
            if k == k_shot
            else (1.0, 1.0, 1.0, 0.0)
        )
        values[f"lambda_k{k}"] = row[0]
        values[f"temperature_k{k}"] = row[1]
        values[f"d99_temperature_k{k}"] = row[2]
        values[f"alpha_k{k}"] = row[3]
    return d100.Phase1D100Lock(
        **values,
        d99_phase1_lock_digest=config99.lock_digest,
        phase1_lodo_rescue_receipt_sha256=lodo_sha,
        external_phase2_authority_sha256=authority_sha,
        quantization_margin_audit_sha256=config99.quantization_margin_audit_sha256,
    )


def _active_k_configs(
    base: d99.Phase1D99Lock,
    receipt: Mapping[str, Any],
    *,
    active_k: int,
    phase2_authority_sha256: str,
) -> tuple[Mapping[str, float], d99.Phase1D99Lock, d100.Phase1D100Lock]:
    parameters = locked_parameters_from_lodo(receipt, k_shot=int(active_k))
    config99 = _d99_config(base, parameters, active_k=int(active_k))
    config100 = _d100_config(
        config99,
        parameters,
        active_k=int(active_k),
        phase1_lodo_receipt_sha256=str(receipt.get("receipt_sha256", "")),
        phase2_authority_sha256=phase2_authority_sha256,
    )
    return parameters, config99, config100


def _validate_active_k_state_binding(
    bank: d99.TypedINT8MetricKernelBank,
    config100: d100.Phase1D100Lock,
    parameters: Mapping[str, float],
    *,
    active_k: int,
) -> None:
    expected = (
        float(parameters["lambda0"]),
        float(parameters["ridge_temperature"]),
        float(parameters["d99_temperature"]),
        float(parameters["alpha"]),
    )
    if int(bank.metric.k_shot) != int(active_k):
        raise D99D100QueryEvaluationError("D99 bank active-K binding drift")
    if config100.values_for_k(int(active_k)) != expected:
        raise D99D100QueryEvaluationError("D100 active-K parameter binding drift")


def _publish_prediction(
    root: Path,
    *,
    tokens: Sequence[np.ndarray],
    scenarios: Sequence[np.ndarray],
    predictions: Sequence[np.ndarray],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=False)
    destination = _output_root(root)
    prediction_sha = _write_npz_new(
        destination / "prediction_artifact.npz",
        query_tokens=np.concatenate(tokens).astype(str),
        scenarios=np.concatenate(scenarios).astype(str),
        predicted_class_handles=np.concatenate(predictions).astype(str),
    )
    audit_sha = _write_json_new(destination / "evaluation_audit.json", dict(audit))
    return {
        "output_root": str(destination),
        "prediction_path": str(destination / "prediction_artifact.npz"),
        "prediction_artifact_sha256": prediction_sha,
        "evaluation_audit_sha256": audit_sha,
    }


def run_d99_d100_query_evaluation(
    *,
    before_enrollment_package_root: str | Path,
    before_enrollment_seal_path: str | Path,
    before_enrollment_seal_sha256: str,
    before_apply_package_root: str | Path,
    before_apply_seal_path: str | Path,
    before_apply_seal_sha256: str,
    after_enrollment_package_root: str | Path,
    after_enrollment_seal_path: str | Path,
    after_enrollment_seal_sha256: str,
    after_apply_package_root: str | Path,
    after_apply_seal_path: str | Path,
    after_apply_seal_sha256: str,
    d81_ground_component_dir: str | Path,
    d81_ground_manifest_sha256: str,
    d99_ground_bundle: d99.Phase1GroundAggregateBundle,
    d99_ground_manifest_sha256: str,
    base_d99_config: d99.Phase1D99Lock,
    phase1_lodo_receipt: Mapping[str, Any],
    class_binding_payload: Mapping[str, Any],
    class_binding_bytes: bytes,
    class_binding_sha256: str,
    class_binding_source_schema: str,
    target_old_tx_labels: Sequence[str],
    phase2_authority_sha256: str,
    output_root: str | Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate matched D81/D99/D100 before/after streams without query truth."""

    from scripts import probe_d81_ground_nuisance_cauchy_center as probe
    from cvsrffi import stage2_d42_unified_shrinkage_lda as d42

    loaders = []
    for package, seal, expected in (
        (before_enrollment_package_root, before_enrollment_seal_path, before_enrollment_seal_sha256),
        (before_apply_package_root, before_apply_seal_path, before_apply_seal_sha256),
        (after_enrollment_package_root, after_enrollment_seal_path, after_enrollment_seal_sha256),
        (after_apply_package_root, after_apply_seal_path, after_apply_seal_sha256),
    ):
        loaders.append(
            load_verified_somph_predictor_bundle(
                package, detached_seal_path=seal, expected_seal_sha256=str(expected).lower()
            )
        )
    (before_support, before_manifest, before_enrollment_audit), (
        before_query,
        before_apply,
        before_apply_audit,
    ), (after_support, after_manifest, after_enrollment_audit), (
        after_query,
        after_apply,
        after_apply_audit,
    ) = loaders
    _validate_matched_packages(before_manifest, before_apply)
    _validate_matched_packages(after_manifest, after_apply)
    old_classes, classes, k_shot = _require_cross_state_lock(
        before_manifest, before_apply, after_manifest, after_apply
    )
    receipt_sha = str(phase1_lodo_receipt.get("receipt_sha256", "")).lower()
    tx_to_handle, handle_to_tx = class_binding_maps(
        class_binding_payload,
        payload_sha256=class_binding_sha256,
        payload_bytes=class_binding_bytes,
        checkpoint_sha256=str(after_manifest["phase1_checkpoint_sha256"]),
        old_handles=old_classes,
        target_old_tx_labels=target_old_tx_labels,
    )
    raw_binding_payload = json.loads(class_binding_bytes.decode("utf-8"))
    if raw_binding_payload.get("schema") != class_binding_source_schema:
        raise D99D100QueryEvaluationError("class binding source schema audit drift")
    old_internal = tuple(handle_to_tx[value] for value in old_classes)
    internal_classes = old_internal + classes[len(old_classes) :]
    parameters, config99, config100 = _active_k_configs(
        base_d99_config,
        phase1_lodo_receipt,
        active_k=k_shot,
        phase2_authority_sha256=phase2_authority_sha256,
    )
    if tuple(d99_ground_bundle.ground_old_registry) != old_internal:
        raise D99D100QueryEvaluationError("D99 ground-old registry does not match row old classes")
    if (
        d99_ground_bundle.bundle_sha256 != config99.ground_bundle_receipt_sha256
        or d99_ground_bundle.aggregation_receipt.receipt_sha256
        != config99.ground_aggregation_receipt_sha256
    ):
        raise D99D100QueryEvaluationError("D99 ground bundle/base lock receipt drift")
    ground99 = d99.build_ground_geometry(d99_ground_bundle, config=config99)
    runtime_device = _device(device)
    model = load_torchscript_backbone_same_fd(
        after_enrollment_package_root,
        _descriptor(after_manifest, "feature_runtime"),
        device=runtime_device,
    )
    basis, spectral_weights, ground81_audit = probe.load_ground_basis(
        Path(d81_ground_component_dir), str(d81_ground_manifest_sha256), 288
    )
    validate_lodo_input_binding(
        phase1_lodo_receipt,
        bundle=d99_ground_bundle,
        ground_manifest_sha256=d99_ground_manifest_sha256,
        base_d99_lock=base_d99_config,
        d81_ground_manifest_sha256=d81_ground_manifest_sha256,
        d81_ground_component_sha256=ground81_audit["component_npz_sha256"],
        checkpoint_sha256=str(after_manifest["phase1_checkpoint_sha256"]),
    )
    d81_fit, call_records, transform_records = probe.build_d81_fit(
        d42, basis, spectral_weights, ground81_audit
    )
    sklearn_version = str(d42.sklearn.__version__)
    if sklearn_version not in D81_ALLOWED_SKLEARN_RUNTIME_VERSIONS:
        raise D99D100QueryEvaluationError(f"unsupported D81 sklearn runtime {sklearn_version}")

    streams: dict[str, dict[str, dict[str, list[np.ndarray]]]] = {
        candidate: {
            state: {"tokens": [], "scenarios": [], "predictions": []}
            for state in ("before", "after")
        }
        for candidate in CANDIDATES
    }
    scenario_audits: list[dict[str, Any]] = []
    original_fit = d42._fit_equal_prior_lda
    original_version = d42.SKLEARN_RUNTIME_VERSION
    d81_score_seconds = 0.0
    canonical_score_seconds = 0.0
    d81_scored_query_count = 0
    canonical_scored_query_count = 0
    try:
        d42._fit_equal_prior_lda = d81_fit
        d42.SKLEARN_RUNTIME_VERSION = sklearn_version
        for scenario_index, scenario in enumerate(FORMAL_LEO_WEAK_SCENARIOS):
            before_query_tokens = np.asarray(
                before_query[scenario]["query_tokens"]
            ).astype(str)
            after_query_tokens = np.asarray(after_query[scenario]["query_tokens"]).astype(str)
            if (
                len(set(before_query_tokens.tolist())) != len(before_query_tokens)
                or len(set(after_query_tokens.tolist())) != len(after_query_tokens)
                or len(after_query_tokens) * len(old_classes)
                != len(before_query_tokens) * len(classes)
                or not set(before_query_tokens.tolist()).issubset(
                    set(after_query_tokens.tolist())
                )
            ):
                raise D99D100QueryEvaluationError(
                    "before query tokens must be the exact old-query subset of after"
                )
            old_x, old_y, old_ids, old_indices = _support_features(
                before_support[scenario],
                model=model,
                runtime_device=runtime_device,
                class_handles=old_classes,
                k_shot=k_shot,
            )
            all_x, all_y, all_ids, all_indices = _support_features(
                after_support[scenario],
                model=model,
                runtime_device=runtime_device,
                class_handles=classes,
                k_shot=k_shot,
            )
            old_mask = all_indices < len(old_classes)
            if (
                not np.array_equal(all_ids[old_mask], old_ids)
                or not np.array_equal(all_y[old_mask], old_y)
                or not np.allclose(all_x[old_mask], old_x, atol=1e-6, rtol=0.0)
            ):
                raise D99D100QueryEvaluationError("before/after old support is not identical")
            result81 = fit_d42_unified_shrinkage_lda(
                old_x,
                old_y,
                old_classes,
                all_x[~old_mask],
                all_y[~old_mask],
                classes[len(old_classes) :],
                seed=int(after_manifest["seed"]) + scenario_index,
                device=runtime_device,
                config=D42UnifiedShrinkageLDAConfig(sklearn_runtime_version=sklearn_version),
            )
            audit81 = _audit_fit(
                result81,
                scenario=scenario,
                k_shot=k_shot,
                old_count=len(old_classes),
                class_count=len(classes),
            )
            state_rows = {
                "before": (
                    old_x,
                    np.asarray([handle_to_tx[value] for value in old_y], dtype=str),
                    old_ids,
                    old_internal,
                    old_classes,
                    result81.before_state,
                    before_query[scenario],
                ),
                "after": (
                    all_x,
                    np.asarray(
                        [handle_to_tx.get(value, value) for value in all_y], dtype=str
                    ),
                    all_ids,
                    internal_classes,
                    classes,
                    result81.state,
                    after_query[scenario],
                ),
            }
            state_audit: dict[str, Any] = {}
            for state, (
                support_x,
                support_y,
                physical_ids,
                registry,
                output_registry,
                state81,
                query_payload,
            ) in state_rows.items():
                metric = d99.fit_support_metric(
                    ground99,
                    support_x,
                    support_y,
                    physical_ids,
                    registry,
                    old_internal,
                    config=config99,
                )
                bank = d99.build_typed_support_bank(
                    metric,
                    support_x,
                    support_y,
                    physical_ids,
                    registry,
                    config=config99,
                )
                _validate_active_k_state_binding(
                    bank, config100, parameters, active_k=k_shot
                )
                state100 = d100.build_simplex_ridge_state(bank, config=config100)
                query_x, tokens = _query_features(
                    query_payload, model=model, runtime_device=runtime_device
                )
                started = time.perf_counter()
                d81_scores = score_d42_unified_shrinkage_lda(state81, query_x)
                d81_score_seconds += time.perf_counter() - started
                d81_scored_query_count += len(query_x)
                typed_d81 = d100.bind_typed_d81_logits(
                    d81_scores,
                    query_x,
                    registry,
                    k_shot,
                    source_schema=state81.schema,
                    source_receipt_sha256=_d81_state_receipt(state81),
                )
                started = time.perf_counter()
                fused = d100.canonical_fuse_typed_d81_d99_d100(
                    state100,
                    bank,
                    typed_d81,
                    query_x,
                    evaluate_complementarity_branch=False,
                )
                canonical_score_seconds += time.perf_counter() - started
                canonical_scored_query_count += len(query_x)
                internal_predictions = {
                    CANDIDATES[0]: np.asarray(output_registry, dtype=str)[
                        np.argmax(fused.d81_probability_fp32, axis=1)
                    ],
                    CANDIDATES[1]: np.asarray(registry, dtype=str)[
                        np.argmax(fused.d99_probability_fp32, axis=1)
                    ],
                    CANDIDATES[2]: fused.prediction,
                }
                for candidate, predictions in internal_predictions.items():
                    if candidate != CANDIDATES[0]:
                        predictions = np.asarray(
                            [tx_to_handle.get(value, value) for value in predictions],
                            dtype=str,
                        )
                    streams[candidate][state]["tokens"].append(tokens)
                    streams[candidate][state]["scenarios"].append(
                        np.asarray([scenario] * len(tokens), dtype=str)
                    )
                    streams[candidate][state]["predictions"].append(predictions)
                ground99_bytes = int(
                    sum(
                        value.nbytes
                        for value in (
                            d99_ground_bundle.codes_qint8,
                            d99_ground_bundle.scales_fp16,
                            d99_ground_bundle.domain_class_mask,
                            d99_ground_bundle.physical_sample_count_floor_uint16,
                        )
                    )
                )
                d81_state_bytes = int(state81.persistent_state_bytes)
                d99_bytes = int(bank.resource_audit["logical_runtime_numeric_state_bytes"])
                d100_bytes = int(state100.resource_audit["numeric_logical_state_bytes"])
                shared_bytes = int(ground81_audit["ground_int8_component_logical_state_bytes"])
                state_audit[state] = {
                    "registered_class_count": len(registry),
                    "support_count": len(support_x),
                    "ground_coverage_rho": float(metric.ground_coverage_rho),
                    "ground_weight": float(metric.ground_weight),
                    "target_weight": float(metric.target_weight),
                    "target_rank": int(metric.target_basis_fp32.shape[1]),
                    "combined_rank": int(metric.metric_basis_fp32.shape[1]),
                    "d99_deployment_status": bank.deployment_status,
                    "d99_quantization_audit": dict(bank.quantization_audit),
                    "d100_quantization_audit": dict(state100.quantization_audit),
                    "canonical_fusion_audit": dict(fused.audit),
                    "typed_d81_batch_receipt_sha256": typed_d81.batch_receipt_sha256,
                    "d81_int8_vs_fp32_support_argmax_change_count": int(
                        result81.geometry_audit[
                            "int8_vs_fp32_before_support_argmax_change_count"
                            if state == "before"
                            else "int8_vs_fp32_final_support_argmax_change_count"
                        ]
                    ),
                    "d81_support_score_max_abs_error": float(
                        result81.geometry_audit[
                            "before_support_score_max_abs_error"
                            if state == "before"
                            else "final_support_score_max_abs_error"
                        ]
                    ),
                    "d81_state_bytes": d81_state_bytes,
                    "d81_ground_bytes": shared_bytes,
                    "d99_ground_bundle_numeric_bytes": ground99_bytes,
                    "d99_bank_numeric_bytes": d99_bytes,
                    "d99_bank_wire_bytes": int(bank.resource_audit["actual_serialized_runtime_artifact_bytes"]),
                    "d100_state_numeric_bytes": d100_bytes,
                    "d100_state_wire_bytes": int(state100.resource_audit["actual_serialized_state_bytes"]),
                    "d99_known_persistent_numeric_bytes": shared_bytes + ground99_bytes + d81_state_bytes + d99_bytes,
                    "d100_known_persistent_numeric_bytes": shared_bytes + ground99_bytes + d81_state_bytes + d99_bytes + d100_bytes,
                    "complete_serialized_state_total_available": False,
                    "serialized_state_total_status": (
                        "KNOWN_D99_D100_WIRES_D81_AND_GROUND_SERIALIZATION_NOT_CLOSED"
                    ),
                    "d99_known_numeric_state_below_256kib": bool(
                        shared_bytes + ground99_bytes + d81_state_bytes + d99_bytes
                        <= 256 * 1024
                    ),
                    "d100_known_numeric_state_below_256kib": bool(
                        shared_bytes
                        + ground99_bytes
                        + d81_state_bytes
                        + d99_bytes
                        + d100_bytes
                        <= 256 * 1024
                    ),
                    "d81_query_macs": int(result81.resource_audit["estimated_macs_per_query"]),
                    "d99_incremental_query_macs": int(bank.resource_audit["query_mac_upper_bound"]),
                    "d100_incremental_query_macs": int(state100.resource_audit["d100_incremental_query_mac_upper_bound_per_sample"]),
                    "d99_total_query_mac_upper_bound": int(
                        result81.resource_audit["estimated_macs_per_query"]
                        + bank.resource_audit["query_mac_upper_bound"]
                    ),
                    "d100_total_query_mac_upper_bound": int(
                        result81.resource_audit["estimated_macs_per_query"]
                        + state100.resource_audit[
                            "combined_query_mac_upper_bound_per_sample"
                        ]
                    ),
                    "d99_fit_peak_transient_bytes_upper_bound": int(
                        bank.resource_audit["peak_transient_bytes_upper_bound"]
                    ),
                    "d81_fit_peak_cuda_memory_bytes": int(
                        result81.resource_audit["peak_cuda_memory_bytes"]
                    ),
                    "complete_combined_fit_peak_available": False,
                    "trainable_parameters": int(result81.resource_audit["trainable_parameters"]),
                    "adaptation_epochs": int(result81.resource_audit["adaptation_epochs"]),
                    "optimizer_steps": int(result81.resource_audit["optimizer_steps"]),
                    "query_state_updates": 0,
                    "query_batch_dependency": False,
                }
            scenario_audits.append(
                {"scenario": scenario, "d81": audit81, "states": state_audit}
            )
    finally:
        d42._fit_equal_prior_lda = original_fit
        d42.SKLEARN_RUNTIME_VERSION = original_version
    if _sha256_file(Path(d81_ground_component_dir) / probe.d66.NPZ_NAME) != ground81_audit[
        "component_npz_sha256"
    ]:
        raise D99D100QueryEvaluationError("D81 ground component changed during evaluation")

    output = Path(output_root)
    if output.exists() and (not output.is_dir() or output.is_symlink() or any(output.iterdir())):
        raise D99D100QueryEvaluationError("evaluation output must be absent or an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    published: dict[str, Any] = {}
    common_audit = {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_NARROW_NONFORMAL",
        "receiver": after_manifest["receiver"],
        "seed": after_manifest["seed"],
        "k_shot": k_shot,
        "new_class_count": len(classes) - len(old_classes),
        "phase1_locked_parameters": dict(parameters),
        "phase1_lodo_receipt_sha256": receipt_sha,
        "class_binding_sha256": str(class_binding_sha256).lower(),
        "class_binding_source_schema": str(class_binding_source_schema),
        "class_binding_typed_schema": str(class_binding_payload.get("schema", "")),
        "class_binding_typed_payload_sha256": hashlib.sha256(
            _canonical_bytes(class_binding_payload)
        ).hexdigest(),
        "old_internal_tx_registry": list(old_internal),
        "old_output_handle_registry": list(old_classes),
        "phase1_d99_config": asdict(config99),
        "phase1_d100_config": asdict(config100),
        "phase1_parameter_lock_scope": {
            "parameter_lock_scope": "ACTIVE_K_ONLY",
            "active_k": int(k_shot),
            "inactive_k": [int(k) for k in ALLOWED_K if int(k) != int(k_shot)],
            "inactive_d99_eta_source": "BASE_LOCK_VALUE_UNMODIFIED_UNUSED_THIS_ROW",
            "inactive_d100_fields": "DATA_INDEPENDENT_INERT_PLACEHOLDER_UNUSED_THIS_ROW",
            "inactive_d100_placeholder": {
                "lambda": 1.0,
                "temperature": 1.0,
                "d99_temperature": 1.0,
                "alpha": 0.0,
            },
        },
        "query_truth_present_in_predictor": False,
        "query_truth_used_for_fit": False,
        "query_state_updates": 0,
        "query_decision_policy": "per_sample_all_registered_classes",
        "query_batch_dependency": False,
        "same_formula_all_registered_classes": True,
        "support_only_fit": True,
        "single_leo_weak_observation_only": True,
        "d81_ground_component_update_access": False,
        "d99_ground_bundle_update_access": False,
        "d81_ground_fit_count": len(call_records),
        "d81_transform_count": len(transform_records),
        "scenario_audits": scenario_audits,
    }
    for candidate in CANDIDATES:
        published[candidate] = {}
        for state in ("before", "after"):
            audit = {
                **common_audit,
                "candidate": candidate,
                "registration_state": state,
                "shared_d81_score_seconds": float(d81_score_seconds),
                "canonical_d99_d100_score_seconds": float(canonical_score_seconds),
                "shared_d81_scored_query_count": int(d81_scored_query_count),
                "canonical_d99_d100_scored_query_count": int(
                    canonical_scored_query_count
                ),
                "shared_d81_seconds_per_query": float(
                    d81_score_seconds / max(d81_scored_query_count, 1)
                ),
                "canonical_d99_d100_seconds_per_query": float(
                    canonical_score_seconds / max(canonical_scored_query_count, 1)
                ),
                "latency_measurement_scope": (
                    "aggregate_all_states_scenarios_shared_d81_and_canonical_increment"
                ),
                "preopen_audit": {
                    "before_enrollment": before_enrollment_audit,
                    "before_apply": before_apply_audit,
                    "after_enrollment": after_enrollment_audit,
                    "after_apply": after_apply_audit,
                },
            }
            published[candidate][state] = _publish_prediction(
                output / candidate / state,
                tokens=streams[candidate][state]["tokens"],
                scenarios=streams[candidate][state]["scenarios"],
                predictions=streams[candidate][state]["predictions"],
                audit=audit,
            )
    return {
        "schema": SCHEMA,
        "status": "DEVELOPMENT_NARROW_PREDICTIONS_COMPLETE",
        "receiver": after_manifest["receiver"],
        "seed": after_manifest["seed"],
        "k_shot": k_shot,
        "new_class_count": len(classes) - len(old_classes),
        "candidates": published,
        "phase1_locked_parameters": dict(parameters),
        "truth_join_performed": False,
    }


__all__ = [
    "ALLOWED_K",
    "ALLOWED_NEW_COUNTS",
    "CANDIDATES",
    "D99D100QueryEvaluationError",
    "locked_parameters_from_lodo",
    "run_d99_d100_query_evaluation",
    "typed_class_binding_payload",
]
