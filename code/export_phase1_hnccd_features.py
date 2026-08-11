#!/usr/bin/env python
"""Export sealed L/V/proxy features for frozen P1-HNCCD postfreeze scoring.

The ICMT exporter is used only as the frozen source-split and forward kernel.
This facade owns HNCCD identity, reopens the actual terminal receipt before
every export, and persists a HNCCD-only sealed manifest.  It never fits a
proxy, reads a query, opens another LEO view, or accepts a training-log metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from cvsrffi import phase1_hnccd as _hnccd
import export_phase1_icmt_features as _icmt


EXPECTED_TRAINING_RUN_ID = "phase1_hnccd12_20260811_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.hnccd_lv_export.v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.hnccd_receipt.v1"
EXPECTED_RECEIPT_METHOD = "P1_HNCCD"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_HNCCD12$")
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_HNCCD_LAMBDA = float(_hnccd.FROZEN_HNCCD_LAMBDA)
FROZEN_HNCCD_BATCH_SIZE = int(_hnccd.FROZEN_HNCCD_BATCH_SIZE)
FROZEN_HNCCD_FEATURE_DIM = int(_hnccd.FROZEN_HNCCD_FEATURE_DIM)
FROZEN_HNCCD_LOCAL_CLASS_COUNT = len(_hnccd.FROZEN_HNCCD_CLASS_IDS)
FROZEN_HNCCD_SOURCE_RECEIVER_COUNT = int(_hnccd.FROZEN_HNCCD_SOURCE_RECEIVER_COUNT)
FROZEN_HNCCD_GLOBAL_DENOMINATOR = int(_hnccd.FROZEN_HNCCD_TERM_DIVISOR)
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400
SOURCE_RECEIVER_PROVENANCE = "SOURCE_SPLIT_RECEIPT_ORDERED_SOURCE_RECEIVERS_PHYSICAL_ID_BOUND_L_ONLY"
LEGACY_IDENTITY_PREFIXES = ("icmt_", "hscf_", "rcmmc_", "rcat_", "recte_", "rcrmd_", "cagm_")
RAW_RECEIVER_TOKEN_FIELDS = frozenset({"source_receiver_ids", "frozen_source_receiver_ids"})


class HNCCDSplitExportError(RuntimeError):
    """Raised when a frozen HNCCD source export cannot prove its binding."""


def _require_no_raw_receiver_tokens(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            if str(raw_key).lower() in RAW_RECEIVER_TOKEN_FIELDS:
                raise HNCCDSplitExportError(f"raw source receiver token leaked into {label}: {raw_key}")
            _require_no_raw_receiver_tokens(raw_value, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _require_no_raw_receiver_tokens(item, label=label)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    values = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise HNCCDSplitExportError(f"{field} must be non-empty and duplicate-free")
    return values


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise HNCCDSplitExportError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise HNCCDSplitExportError(f"{name} drifted: expected={expected} observed={observed}")


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    observed = mapping.get(field)
    if type(observed) is not bool or observed is not expected:
        raise HNCCDSplitExportError(f"{field} drifted: expected literal {expected!r}")


def _require_sha256(value: Any, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
        raise HNCCDSplitExportError(f"{field} must be a lowercase SHA256")


def _validate_none_or_zero(values: Any, *, field: str, expected_count: float | None, allow_absent: bool) -> None:
    if not isinstance(values, Mapping):
        raise HNCCDSplitExportError(f"HNCCD VJP audit lacks {field}")
    try:
        count = float(values.get("parameter_count", float("nan")))
        none_count = float(values.get("none_parameters", float("nan")))
        zero_count = float(values.get("zero_parameters", float("nan")))
        nonzero_count = float(values.get("nonzero_parameters", float("nan")))
    except (TypeError, ValueError) as exc:
        raise HNCCDSplitExportError(f"HNCCD VJP audit {field} is malformed") from exc
    absent = values.get("parameter_absent") is True
    if (
        (expected_count is not None and count != expected_count)
        or not all(math.isfinite(item) and item >= 0.0 for item in (count, none_count, zero_count, nonzero_count))
        or none_count + zero_count != count
        or nonzero_count != 0.0
        or values.get("none_or_zero_expected") is not True
        or (count == 0.0 and not (allow_absent and absent))
        or (count > 0.0 and absent)
    ):
        raise HNCCDSplitExportError(f"HNCCD VJP audit {field} must be None-or-zero")


def _validate_hnccd_gradient_audits(receipt: Mapping[str, Any]) -> None:
    scenarios = tuple(_hnccd.FROZEN_HNCCD_SCENARIOS)
    if receipt.get("hnccd_gradient_audit_completed") is not True:
        raise HNCCDSplitExportError("HNCCD G receipt lacks completed per-scene VJP audits")
    audits = receipt.get("hnccd_gradient_audit_scenes")
    scenes = receipt.get("hnccd_scenes")
    positives = receipt.get("hnccd_scene_positive_batches")
    if (
        not isinstance(audits, Mapping)
        or set(audits) != set(scenarios)
        or not isinstance(scenes, Mapping)
        or set(scenes) != set(scenarios)
        or not isinstance(positives, Mapping)
        or set(positives) != set(scenarios)
    ):
        raise HNCCDSplitExportError("HNCCD G receipt lacks clear/low/rain VJP and positive coverage")
    for scenario in scenarios:
        if int(positives.get(scenario, 0)) <= 0:
            raise HNCCDSplitExportError(f"HNCCD {scenario} lacks a positive auxiliary batch")
        audit = audits[scenario]
        if not isinstance(audit, Mapping) or (
            audit.get("raw_unscaled") is not True
            or audit.get("diagnostic_only") is not True
            or audit.get("touches_amp_optimizer_rng") is not False
            or audit.get("clean_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
            or audit.get("head_bias_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
            or audit.get("exact_head_weight_source") != "model.id_backbone.cls_head.head.weight"
        ):
            raise HNCCDSplitExportError(f"HNCCD {scenario} raw VJP semantics drifted")
        for group in ("feat_joint_leo", "shared_encoder", "head_weight"):
            evidence = audit.get(group)
            if not isinstance(evidence, Mapping):
                raise HNCCDSplitExportError(f"HNCCD {scenario} VJP lacks {group}")
            try:
                count = float(evidence["parameter_count"])
                norm = float(evidence["norm"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HNCCDSplitExportError(f"HNCCD {scenario} VJP {group} is malformed") from exc
            if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
                raise HNCCDSplitExportError(f"HNCCD {scenario} VJP {group} is zero/nonfinite")
        _validate_none_or_zero(audit.get("clean_feat_joint"), field="clean_feat_joint", expected_count=1.0, allow_absent=False)
        _validate_none_or_zero(audit.get("head_bias"), field="head_bias", expected_count=None, allow_absent=True)


def _validate_hnccd_cells(receipt: Mapping[str, Any], *, arm: str) -> None:
    scenarios = tuple(_hnccd.FROZEN_HNCCD_SCENARIOS)
    common = receipt.get("hnccd_common_cells")
    if not isinstance(common, Mapping) or set(common) != set(scenarios):
        raise HNCCDSplitExportError("HNCCD receipt lacks three-scene common coverage")
    for scenario in scenarios:
        cells = common.get(scenario)
        if not isinstance(cells, Mapping) or len(cells) != FROZEN_HNCCD_GLOBAL_DENOMINATOR:
            raise HNCCDSplitExportError("HNCCD common coverage is not fixed 28 cells per scene")
    if arm == "C":
        forbidden = (
            "hnccd_batches", "hnccd_total_rows", "hnccd_positive_c_cells", "hnccd_positive_c_batches",
            "hnccd_insufficient_cells", "hnccd_optimizer_step_attempts", "hnccd_effective_optimizer_steps",
        )
        if any(int(receipt.get(field, 0)) != 0 for field in forbidden):
            raise HNCCDSplitExportError("HNCCD C receipt has nonzero auxiliary evidence")
        if any(bool(receipt.get(field)) for field in ("hnccd_scenes", "hnccd_g_batch_aux", "hnccd_gradient_audit_scenes")):
            raise HNCCDSplitExportError("HNCCD C receipt must keep G-only fields N/A-or-zero")
        if receipt.get("hnccd_gradient_audit_completed") is not False:
            raise HNCCDSplitExportError("HNCCD C receipt must not mark a G-only VJP audit complete")
        return
    scenes = receipt.get("hnccd_scenes")
    positive = receipt.get("hnccd_scene_positive_batches")
    if not isinstance(scenes, Mapping) or set(scenes) != set(scenarios):
        raise HNCCDSplitExportError("HNCCD G receipt lacks three-scene auxiliary coverage")
    if not isinstance(positive, Mapping) or set(positive) != set(scenarios):
        raise HNCCDSplitExportError("HNCCD G receipt lacks per-scene positive evidence")
    for scenario in scenarios:
        if int(positive.get(scenario, 0)) <= 0:
            raise HNCCDSplitExportError(f"HNCCD G receipt lacks positive cell evidence for {scenario}")
        cells = scenes.get(scenario)
        if not isinstance(cells, Mapping) or len(cells) != FROZEN_HNCCD_GLOBAL_DENOMINATOR:
            raise HNCCDSplitExportError("HNCCD G coverage is not fixed 28 cells per scene")


def validate_hnccd_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    source_tx_ids: Sequence[str] | None = None,
    known_validation_tx_ids: Sequence[str] | None = None,
    proxy_unknown_tx_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reopen a scalar-only raw HNCCD terminal receipt and fail closed on drift."""

    if arm not in {"C", "G"} or not isinstance(receipt, Mapping):
        raise HNCCDSplitExportError("checkpoint lacks a valid HNCCD C/G receipt")
    supplied_roles = (source_tx_ids, known_validation_tx_ids, proxy_unknown_tx_ids)
    if any(value is not None for value in supplied_roles):
        if any(value is None for value in supplied_roles):
            raise HNCCDSplitExportError("HNCCD TX role validation requires source, held and proxy roles together")
        source = tuple(str(item) for item in source_tx_ids or ())
        held = tuple(str(item) for item in known_validation_tx_ids or ())
        proxy = tuple(str(item) for item in proxy_unknown_tx_ids or ())
        if len(source) != 4 or len(held) != 1 or len(proxy) != 1 or len(set(source + held + proxy)) != 6:
            raise HNCCDSplitExportError("HNCCD local4/held/proxy TX role binding drifted")
    frozen = dict(receipt)
    _require_no_raw_receiver_tokens(frozen, label="HNCCD terminal receipt")
    legacy = [str(field) for field in frozen if str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES)]
    if legacy:
        raise HNCCDSplitExportError("HNCCD receipt contains historical method identity: " + ",".join(sorted(legacy)))
    if str(frozen.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA or str(frozen.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise HNCCDSplitExportError("HNCCD terminal receipt identity drifted")
    _require_bool(frozen, "frozen_mode", True)
    _require_bool(frozen, "enabled", arm == "G")
    _require_close("HNCCD receipt lambda", frozen.get("lambda"), FROZEN_HNCCD_LAMBDA if arm == "G" else 0.0)
    expected_text = {
        "loss_rule": "SOURCE_L_ORDERED_RECEIVER_SLOT_BY_LOCAL4_LEO_HEAD_NULL_CROSS_COVARIANCE_DECORRELATION_TOTALIZED_L2_feat_joint",
        "loss_formula": "Q=W^T chol(WW^T)^(-T);h=Q^T u;b=u-Qh;C_rc=(H-Hbar)^T(B-Bbar)/n_rc;if_n_lt_2_C=0;L=sum_rc||C_rc||F^2/28",
        "z_id_key": "feat_joint",
        "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "clean_feature_detached": "NOT_READ_BY_HNCCD_AUXILIARY",
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "receipt_payload": "SCALARS_COUNTS_AND_SHA_ONLY_NO_IQ_FEATURE_COVARIANCE_OR_RECEIVER_TOKEN",
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "aux_gradient_scope": "LEO_feat_joint_SHARED_ENCODER_EXACT_HEAD_WEIGHT_FINITE_NONZERO;CLEAN_AND_HEAD_BIAS_NONE_OR_ZERO",
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_SOURCE_SPLIT_RECEIPT_ORDERED_TOKEN_ONLY",
        "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
        "source_receiver_provenance": SOURCE_RECEIVER_PROVENANCE,
        "exact_head_weight_path": "model.id_backbone.cls_head.head.weight",
        "head_null_basis_rule": "FP32_DIFFERENTIABLE_CHOLESKY_WWT_AND_TRIANGULAR_SOLVE_Q_EQ_WT_LINVTRANSPOSE_NO_PINV_EPSILON_OR_FALLBACK",
    }
    for field, expected in expected_text.items():
        if frozen.get(field) != expected:
            raise HNCCDSplitExportError(f"HNCCD receipt {field} drifted")
    for field, expected in (
        ("loss_global_denominator", FROZEN_HNCCD_GLOBAL_DENOMINATOR),
        ("local_class_count", FROZEN_HNCCD_LOCAL_CLASS_COUNT),
        ("fixed_batch_size", FROZEN_HNCCD_BATCH_SIZE),
        ("fixed_local_class_count", FROZEN_HNCCD_LOCAL_CLASS_COUNT),
        ("frozen_batch_size", FROZEN_HNCCD_BATCH_SIZE),
        ("frozen_feature_dim", FROZEN_HNCCD_FEATURE_DIM),
        ("frozen_source_receiver_count", FROZEN_HNCCD_SOURCE_RECEIVER_COUNT),
        ("source_receiver_count", FROZEN_HNCCD_SOURCE_RECEIVER_COUNT),
        ("local_data_class_count", FROZEN_HNCCD_LOCAL_CLASS_COUNT),
        ("checkpoint_head_class_count", FROZEN_HNCCD_LOCAL_CLASS_COUNT),
        ("live_head_class_count", FROZEN_HNCCD_LOCAL_CLASS_COUNT),
    ):
        if type(frozen.get(field)) is not int or int(frozen[field]) != expected:
            raise HNCCDSplitExportError(f"HNCCD receipt {field} drifted")
    if list(frozen.get("exact_head_weight_shape", [])) != [FROZEN_HNCCD_LOCAL_CLASS_COUNT, FROZEN_HNCCD_FEATURE_DIM]:
        raise HNCCDSplitExportError("HNCCD receipt exact head shape drifted")
    for field in (
        "baseline_sha256", "initial_checkpoint_sha256", "source_partition_sha256",
        "class_order_binding_sha256", "source_labeled_indices_sha256",
        "source_split_manifest_sha256", "source_receiver_order_sha256",
        "source_receiver_ids_sha256", "optimizer_initial_state_sha256", "common_batch_sequence_sha256",
    ):
        _require_sha256(frozen.get(field), field=f"HNCCD receipt {field}")
    if (
        frozen.get("warm_start_mode") != "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP"
        or frozen.get("checkpoint_role") != EXPECTED_CHECKPOINT_ROLE
        or frozen.get("strict_model_keys") is not True
        or frozen.get("missing_model_keys") != []
        or frozen.get("unexpected_model_keys") != []
        or frozen.get("optimizer_state_restored") is not False
        or frozen.get("rng_state_restored") is not False
        or frozen.get("optimizer_type") != "AdamW"
        or frozen.get("optimizer_initial_state_empty") is not True
        or frozen.get("common_l_base_head_input_path_verified") is not True
        or frozen.get("resource_selection_feedback") is not False
    ):
        raise HNCCDSplitExportError("HNCCD warm-start/AdamW/common-head/resource receipt drifted")
    for field in (
        "uses_new_forward", "uses_resampling", "uses_day_labels", "uses_domain_labels",
        "uses_target_rows", "uses_proxy_rows", "uses_held_rows", "uses_unlabeled_rows",
        "uses_ema_or_state", "uses_threshold", "uses_cross_sample_pairing", "uses_cross_receiver_pairing",
    ):
        _require_bool(frozen, field, False)
    _require_bool(frozen, "uses_rx_labels", True)
    try:
        validated = _hnccd.validate_hnccd_terminal_receipt(frozen)
    except (_hnccd.HNCCDConfigurationError, _hnccd.HNCCDRuntimeError) as exc:
        raise HNCCDSplitExportError(f"HNCCD terminal receipt revalidation failed: {exc}") from exc
    if validated.get("hnccd_terminal_contract_passed") is not True:
        raise HNCCDSplitExportError("HNCCD terminal receipt did not pass its raw contract")
    _validate_hnccd_cells(validated, arm=arm)
    if arm == "G":
        _validate_hnccd_gradient_audits(validated)
    return dict(validated)


def _source_split_binding(checkpoint: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    split_info = checkpoint.get("split_info")
    if not isinstance(split_info, Mapping):
        raise HNCCDSplitExportError("HNCCD checkpoint lacks split_info")
    source = split_info.get("source_split_receipt")
    partition = split_info.get("tx_partition_receipt")
    if not isinstance(source, Mapping) or not isinstance(partition, Mapping):
        raise HNCCDSplitExportError("HNCCD checkpoint lacks source split/partition receipt")
    try:
        tokens = _hnccd.resolve_hnccd_source_receiver_tokens(source)
    except _hnccd.HNCCDConfigurationError as exc:
        raise HNCCDSplitExportError(f"HNCCD source receiver receipt is invalid: {exc}") from exc
    expected_receiver_sha = _canonical_json_sha256([int(value) for value in tokens])
    if (
        receipt.get("source_receiver_order_sha256") != expected_receiver_sha
        or receipt.get("source_receiver_ids_sha256") != expected_receiver_sha
        or int(receipt.get("source_receiver_count", 0)) != len(tokens)
    ):
        raise HNCCDSplitExportError("HNCCD ordered source receiver SHA/count drifted")
    for source_field, receipt_field in (
        ("labeled_indices_sha256", "source_labeled_indices_sha256"),
        ("split_manifest_sha256", "source_split_manifest_sha256"),
    ):
        if str(source.get(source_field, "")) != str(receipt.get(receipt_field, "")):
            raise HNCCDSplitExportError(f"HNCCD source split binding drifted: {source_field}")
    if str(partition.get("partition_sha256", "")) != str(receipt.get("source_partition_sha256", "")):
        raise HNCCDSplitExportError("HNCCD source partition SHA drifted")


def validate_hnccd_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Validate original final-only checkpoint arguments and raw HNCCD receipt."""

    if (
        str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE
        or str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION
    ):
        raise HNCCDSplitExportError("checkpoint must be final-only")
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise HNCCDSplitExportError("checkpoint must contain model and args mappings")
    args = checkpoint["args"]
    expected_text = {
        "split_mode": "tx_rx_day_1_6_3",
        "model_variant": "lite_d",
        "id_feature_key": "feat_joint",
        "phase1_source_train_tx_ids": ",".join(source_tx_ids),
        "phase1_source_known_validation_tx_ids": ",".join(known_validation_tx_ids),
        "phase1_source_proxy_unknown_tx_ids": ",".join(proxy_unknown_tx_ids),
        "checkpoint_selection": EXPECTED_CHECKPOINT_SELECTION,
    }
    for field, expected in expected_text.items():
        if str(args.get(field, "")) != expected:
            raise HNCCDSplitExportError(f"checkpoint arg {field} drifted")
    for field, expected in (("labeled_ratio", 0.07), ("unlabeled_ratio", 0.63), ("source_val_ratio", 0.30)):
        _require_close(f"checkpoint arg {field}", args.get(field), expected)
    if int(args.get("seed", -1)) != 7281105:
        raise HNCCDSplitExportError("checkpoint seed drifted")
    match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if match is None:
        raise HNCCDSplitExportError("checkpoint candidate is not frozen HNCCD F1..F6 C/G")
    arm = match.group(2)
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name or str(checkpoint.get("candidate_id", "")) != checkpoint_path.parent.name:
        raise HNCCDSplitExportError("checkpoint candidate does not bind parent directory")
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID or str(checkpoint.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID:
        raise HNCCDSplitExportError("checkpoint run_id does not bind HNCCD training root")
    _require_bool(args, "phase1_hnccd_frozen_mode", True)
    _require_bool(args, "phase1_hnccd_enabled", arm == "G")
    _require_close("checkpoint lambda_hnccd", args.get("lambda_hnccd"), FROZEN_HNCCD_LAMBDA if arm == "G" else 0.0)
    try:
        _hnccd.validate_hnccd_args(SimpleNamespace(**dict(args)))
    except _hnccd.HNCCDConfigurationError as exc:
        raise HNCCDSplitExportError(f"checkpoint HNCCD frozen-arg contract drifted: {exc}") from exc
    receipt = validate_hnccd_terminal_receipt(
        checkpoint.get("hnccd_receipt", {}),
        arm=arm,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    if tuple(str(item) for item in receipt.get("local_tx_class_order", ())) != tuple(str(item) for item in source_tx_ids):
        raise HNCCDSplitExportError("HNCCD receipt local TX order drifted")
    if tuple(str(item) for item in receipt.get("checkpoint_train_tx_class_order", ())) != tuple(str(item) for item in source_tx_ids):
        raise HNCCDSplitExportError("HNCCD receipt checkpoint TX order drifted")
    _source_split_binding(checkpoint, receipt)
    return args, receipt, arm


def _base_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Compatibility adapter used only in-memory by the frozen fair exporter."""

    args, receipt, _ = validate_hnccd_training_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    if isinstance(checkpoint, dict):
        checkpoint["icmt_receipt"] = dict(receipt)
    return args


@contextmanager
def _patched_signed_export() -> Iterator[None]:
    """Redirect generic source reconstruction to HNCCD only for this call."""

    saved = {
        "EXPECTED_TRAINING_RUN_ID": _icmt.EXPECTED_TRAINING_RUN_ID,
        "EXPECTED_RECEIPT_SCHEMA": _icmt.EXPECTED_RECEIPT_SCHEMA,
        "EXPECTED_RECEIPT_METHOD": _icmt.EXPECTED_RECEIPT_METHOD,
        "EXPECTED_CANDIDATE_PATTERN": _icmt.EXPECTED_CANDIDATE_PATTERN,
        "_validate_checkpoint_contract": _icmt._validate_checkpoint_contract,
    }
    _icmt.EXPECTED_TRAINING_RUN_ID = EXPECTED_TRAINING_RUN_ID
    _icmt.EXPECTED_RECEIPT_SCHEMA = EXPECTED_RECEIPT_SCHEMA
    _icmt.EXPECTED_RECEIPT_METHOD = EXPECTED_RECEIPT_METHOD
    _icmt.EXPECTED_CANDIDATE_PATTERN = EXPECTED_CANDIDATE_PATTERN
    _icmt._validate_checkpoint_contract = _base_checkpoint_contract
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_icmt, name, value)


def _coerce_frozen_proxy_args(args: argparse.Namespace) -> argparse.Namespace:
    result = argparse.Namespace(**vars(args))
    fixed = {
        "proxy_days": ",".join(FROZEN_PROXY_DAYS),
        "proxy_rxs": ",".join(FROZEN_PROXY_RXS),
        "max_proxy_samples_per_tx": FROZEN_PROXY_MAX_SAMPLES_PER_TX,
    }
    for field, expected in fixed.items():
        observed = getattr(result, field, expected)
        if observed != expected:
            raise HNCCDSplitExportError(f"{field} is frozen and may not be changed")
        setattr(result, field, expected)
    return result


def _sanitize_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            lower = key.lower()
            if lower.startswith(LEGACY_IDENTITY_PREFIXES) or lower in {"source_receivers", "target_receivers", *RAW_RECEIVER_TOKEN_FIELDS}:
                continue
            result[key] = _sanitize_mapping(raw_value)
        return result
    if isinstance(value, list):
        return [_sanitize_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_mapping(item) for item in value]
    return value


def _assert_hnccd_only_identity(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key).lower()
            if key.startswith(LEGACY_IDENTITY_PREFIXES):
                raise HNCCDSplitExportError(f"legacy identity leaked into {label}: {raw_key}")
            if key in {"source_receivers", "target_receivers", *RAW_RECEIVER_TOKEN_FIELDS}:
                raise HNCCDSplitExportError(f"raw source receiver token leaked into {label}: {raw_key}")
            _assert_hnccd_only_identity(raw_value, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_hnccd_only_identity(item, label=label)


def _atomic_rewrite_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["manifest_json"] = np.asarray(json.dumps(dict(manifest), ensure_ascii=True))
    temporary = path.with_name(path.name + ".hnccd-manifest.tmp")
    if temporary.exists():
        raise HNCCDSplitExportError(f"refusing to overwrite temporary export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Export sealed source L/V/proxy features under the HNCCD frozen contract."""

    frozen_args = _coerce_frozen_proxy_args(args)
    checkpoint_path = Path(frozen_args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise HNCCDSplitExportError(f"missing final checkpoint: {checkpoint_path}")
    source_tx_ids = _parse_csv(frozen_args.source_tx_ids, field="source_tx_ids")
    known_tx_ids = _parse_csv(frozen_args.known_validation_tx_ids, field="known_validation_tx_ids")
    proxy_tx_ids = _parse_csv(frozen_args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise HNCCDSplitExportError("HNCCD export requires local4 plus one held and one proxy TX")
    if set(source_tx_ids) & set(known_tx_ids) or set(source_tx_ids) & set(proxy_tx_ids) or set(known_tx_ids) & set(proxy_tx_ids):
        raise HNCCDSplitExportError("source/held/proxy TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise HNCCDSplitExportError("checkpoint payload must be a mapping")
    _, receipt, arm = validate_hnccd_training_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_tx_ids,
        proxy_unknown_tx_ids=proxy_tx_ids,
    )
    try:
        with _patched_signed_export():
            base_result = _icmt.export(frozen_args)
    except _icmt.ICMTSplitExportError as exc:
        raise HNCCDSplitExportError(str(exc)) from exc
    manifest = dict(_sanitize_mapping(base_result["manifest"]))
    common_scenes = receipt.get("common_scenario_batches", {})
    manifest.update(
        {
            "schema": EXPECTED_LV_EXPORT_SCHEMA,
            "method": EXPECTED_RECEIPT_METHOD,
            "training_run_contract": EXPECTED_TRAINING_RUN_ID,
            "candidate_id": checkpoint_path.parent.name,
            "hnccd_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
            "hnccd_receipt_sha256": _canonical_json_sha256(dict(checkpoint["hnccd_receipt"])),
            "hnccd_terminal_contract": str(receipt["hnccd_terminal_contract"]),
            "hnccd_terminal_contract_passed": True,
            "hnccd_enabled": arm == "G",
            "hnccd_lambda": FROZEN_HNCCD_LAMBDA if arm == "G" else 0.0,
            "hnccd_frozen_batch_size": FROZEN_HNCCD_BATCH_SIZE,
            "hnccd_feature_dim": FROZEN_HNCCD_FEATURE_DIM,
            "hnccd_local_class_count": FROZEN_HNCCD_LOCAL_CLASS_COUNT,
            "hnccd_loss_global_denominator": FROZEN_HNCCD_GLOBAL_DENOMINATOR,
            "hnccd_fixed_batch_size": FROZEN_HNCCD_BATCH_SIZE,
            "hnccd_fixed_feature_dim": FROZEN_HNCCD_FEATURE_DIM,
            "hnccd_fixed_local_class_count": FROZEN_HNCCD_LOCAL_CLASS_COUNT,
            "hnccd_fixed_cells_per_scene": FROZEN_HNCCD_GLOBAL_DENOMINATOR,
            "hnccd_source_receiver_count": FROZEN_HNCCD_SOURCE_RECEIVER_COUNT,
            "hnccd_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
            "hnccd_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
            "hnccd_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "hnccd_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "hnccd_source_partition_sha256": str(receipt["source_partition_sha256"]),
            "hnccd_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
            "hnccd_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
            "hnccd_common_scenario_batches": {str(key): int(value) for key, value in dict(common_scenes).items()},
            "hnccd_common_cells_sha256": _canonical_json_sha256(receipt.get("hnccd_common_cells", {})),
            "hnccd_g_scenes_sha256": _canonical_json_sha256(receipt.get("hnccd_scenes", {})) if arm == "G" else "",
            "hnccd_clean_head_bias_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
            "hnccd_leo_encoder_head_weight_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
            "hnccd_common_physical_order_bound": True,
            "hnccd_common_scene_cycle_bound": True,
            "hnccd_raw_vjp_per_scene_required": True,
            "hnccd_leo_encoder_head_weight_vjp_finite_nonzero": True,
            "hnccd_clean_head_bias_vjp_na_none_or_zero": True,
            "proxy_selection_frozen_not_cli_tunable": True,
        }
    )
    _assert_hnccd_only_identity(manifest, label="HNCCD L/V manifest")
    output_path = Path(base_result["out_npz"]).resolve()
    _atomic_rewrite_manifest(output_path, manifest)
    return {"out_npz": str(output_path), "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--wisig_pkl", required=True)
    parser.add_argument("--out_npz", required=True)
    parser.add_argument("--source_tx_ids", required=True)
    parser.add_argument("--known_validation_tx_ids", required=True)
    parser.add_argument("--proxy_unknown_tx_ids", required=True)
    parser.add_argument("--expected-wisig-sha256", required=True)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = export(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
