#!/usr/bin/env python
"""Export sealed L/V/proxy features for frozen P1-RCMMC postfreeze scoring.

The ICMT exporter below is a fixed source-split/forward compatibility kernel,
not an RCMMC identity.  This facade reopens the raw RCMMC terminal receipt
before each forward, exports only L/V/fixed-proxy rows, and rewrites the
persisted manifest to RCMMC-only identity fields.
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

from cvsrffi import phase1_rcmmc as _rcmmc
import export_phase1_icmt_features as _icmt


EXPECTED_TRAINING_RUN_ID = "phase1_rcmmc12_20260811_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.rcmmc_lv_export.v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.rcmmc_receipt.v1"
EXPECTED_RECEIPT_METHOD = "P1_RCMMC"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_RCMMC12$")
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_RCMMC_LAMBDA = float(_rcmmc.FROZEN_RCMMC_LAMBDA)
FROZEN_RCMMC_BATCH_SIZE = int(_rcmmc.FROZEN_RCMMC_BATCH_SIZE)
FROZEN_RCMMC_FEATURE_DIM = int(_rcmmc.FROZEN_RCMMC_FEATURE_DIM)
FROZEN_RCMMC_LOCAL_CLASS_COUNT = len(_rcmmc.FROZEN_RCMMC_CLASS_IDS)
FROZEN_RCMMC_SOURCE_RECEIVER_COUNT = int(_rcmmc.FROZEN_RCMMC_SOURCE_RECEIVER_COUNT)
FROZEN_RCMMC_GLOBAL_DENOMINATOR = int(_rcmmc.FROZEN_RCMMC_TERM_DIVISOR)
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400
SOURCE_RECEIVER_PROVENANCE = "SOURCE_SPLIT_RECEIPT_ORDERED_SOURCE_RECEIVERS_PHYSICAL_ID_BOUND_L_ONLY"
LEGACY_IDENTITY_PREFIXES = ("icmt_", "hscf_", "rcat_", "recte_", "rcrmd_", "cagm_")
RAW_RECEIVER_TOKEN_FIELDS = frozenset({"source_receiver_ids", "frozen_source_receiver_ids"})


class RCMMCSplitExportError(RuntimeError):
    """Raised when a frozen RCMMC source export cannot prove its binding."""


def _require_no_raw_receiver_tokens(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            if str(raw_key).lower() in RAW_RECEIVER_TOKEN_FIELDS:
                raise RCMMCSplitExportError(f"raw source receiver token leaked into {label}: {raw_key}")
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
        raise RCMMCSplitExportError(f"{field} must be non-empty and duplicate-free")
    return values


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise RCMMCSplitExportError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise RCMMCSplitExportError(f"{name} drifted: expected={expected} observed={observed}")


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    observed = mapping.get(field)
    if type(observed) is not bool or observed is not expected:
        raise RCMMCSplitExportError(f"{field} drifted: expected literal {expected!r}")


def _require_sha256(value: Any, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
        raise RCMMCSplitExportError(f"{field} must be a lowercase SHA256")


def _validate_none_or_zero(values: Any, *, field: str, expected_count: float | None) -> None:
    if not isinstance(values, Mapping):
        raise RCMMCSplitExportError(f"RCMMC VJP audit lacks {field}")
    try:
        count = float(values.get("parameter_count", float("nan")))
        none_count = float(values.get("none_parameters", float("nan")))
        zero_count = float(values.get("zero_parameters", float("nan")))
        nonzero_count = float(values.get("nonzero_parameters", float("nan")))
    except (TypeError, ValueError) as exc:
        raise RCMMCSplitExportError(f"RCMMC VJP audit {field} is malformed") from exc
    if (
        count <= 0.0
        or (expected_count is not None and count != expected_count)
        or not all(math.isfinite(v) and v >= 0.0 for v in (none_count, zero_count, nonzero_count))
        or none_count + zero_count != count
        or nonzero_count != 0.0
        or values.get("none_or_zero_expected") is not True
    ):
        raise RCMMCSplitExportError(f"RCMMC VJP audit {field} must be None-or-zero")


def _validate_rcmmc_gradient_audit(receipt: Mapping[str, Any]) -> None:
    if receipt.get("rcmmc_gradient_audit_completed") is not True:
        raise RCMMCSplitExportError("RCMMC G receipt lacks completed four-argument VJP audit")
    audit = receipt.get("rcmmc_gradient_audit")
    if not isinstance(audit, Mapping):
        raise RCMMCSplitExportError("RCMMC G receipt lacks four-argument VJP payload")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("clean_feat_joint_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        or audit.get("exact_head_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
    ):
        raise RCMMCSplitExportError("RCMMC raw VJP semantics drifted")
    for scope in ("feat_joint_leo", "shared_encoder"):
        evidence = audit.get(scope)
        if not isinstance(evidence, Mapping):
            raise RCMMCSplitExportError(f"RCMMC VJP lacks {scope}")
        try:
            count = float(evidence["parameter_count"])
            norm = float(evidence["norm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RCMMCSplitExportError(f"RCMMC VJP {scope} is malformed") from exc
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise RCMMCSplitExportError(f"RCMMC VJP {scope} is zero/nonfinite")
    _validate_none_or_zero(audit.get("clean_feat_joint"), field="clean_feat_joint", expected_count=1.0)
    _validate_none_or_zero(audit.get("classifier_head"), field="classifier_head", expected_count=None)


def _validate_rcmmc_cells(receipt: Mapping[str, Any], *, arm: str) -> None:
    scenarios = tuple(_rcmmc.FROZEN_RCMMC_SCENARIOS)
    common = receipt.get("rcmmc_common_cells")
    if not isinstance(common, Mapping) or set(common) != set(scenarios):
        raise RCMMCSplitExportError("RCMMC receipt lacks three-scene common coverage")
    for scene in scenarios:
        cells = common.get(scene)
        if not isinstance(cells, Mapping) or len(cells) != FROZEN_RCMMC_GLOBAL_DENOMINATOR:
            raise RCMMCSplitExportError("RCMMC common coverage is not fixed 28 cells per scene")
    if arm == "C":
        forbidden = ("rcmmc_batches", "rcmmc_total_rows", "rcmmc_positive_d_cells", "rcmmc_positive_d_batches")
        if any(int(receipt.get(field, 0)) != 0 for field in forbidden):
            raise RCMMCSplitExportError("RCMMC C receipt has nonzero auxiliary counters")
        if any(bool(receipt.get(field)) for field in ("rcmmc_scenes", "rcmmc_g_batch_aux", "rcmmc_gradient_audit")):
            raise RCMMCSplitExportError("RCMMC C receipt must keep G-only fields N/A-or-zero")
        return
    scenes = receipt.get("rcmmc_scenes")
    positive = receipt.get("rcmmc_scene_positive_batches")
    if not isinstance(scenes, Mapping) or set(scenes) != set(scenarios):
        raise RCMMCSplitExportError("RCMMC G receipt lacks three-scene 84-cell coverage")
    if not isinstance(positive, Mapping) or set(positive) != set(scenarios):
        raise RCMMCSplitExportError("RCMMC G receipt lacks per-scene positive-D evidence")
    for scene in scenarios:
        if int(positive.get(scene, 0)) <= 0:
            raise RCMMCSplitExportError(f"RCMMC G receipt lacks positive-D batch for {scene}")
        cells = scenes.get(scene)
        if not isinstance(cells, Mapping) or len(cells) != FROZEN_RCMMC_GLOBAL_DENOMINATOR:
            raise RCMMCSplitExportError("RCMMC G coverage is not fixed 28 cells per scene")


def validate_rcmmc_terminal_receipt(
    receipt: Mapping[str, Any], *, arm: str,
    source_tx_ids: Sequence[str] | None = None,
    known_validation_tx_ids: Sequence[str] | None = None,
    proxy_unknown_tx_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Reopen a scalar-only raw RCMMC terminal receipt, fail closed on drift."""

    if arm not in {"C", "G"} or not isinstance(receipt, Mapping):
        raise RCMMCSplitExportError("checkpoint lacks a valid RCMMC C/G receipt")
    supplied_roles = (source_tx_ids, known_validation_tx_ids, proxy_unknown_tx_ids)
    if any(value is not None for value in supplied_roles):
        if any(value is None for value in supplied_roles):
            raise RCMMCSplitExportError("RCMMC TX role validation requires source, held and proxy roles together")
        source = tuple(str(item) for item in source_tx_ids or ())
        held = tuple(str(item) for item in known_validation_tx_ids or ())
        proxy = tuple(str(item) for item in proxy_unknown_tx_ids or ())
        if len(source) != 4 or len(held) != 1 or len(proxy) != 1 or len(set(source + held + proxy)) != 6:
            raise RCMMCSplitExportError("RCMMC local4/held/proxy TX role binding drifted")
    frozen = dict(receipt)
    _require_no_raw_receiver_tokens(frozen, label="RCMMC terminal receipt")
    legacy = [
        str(field)
        for field in frozen
        if str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES)
        and str(field) != "rcat_relation"
    ]
    if legacy:
        raise RCMMCSplitExportError("RCMMC receipt contains historical method identity: " + ",".join(sorted(legacy)))
    if str(frozen.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA or str(frozen.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise RCMMCSplitExportError("RCMMC terminal receipt identity drifted")
    _require_bool(frozen, "frozen_mode", True)
    _require_bool(frozen, "enabled", arm == "G")
    _require_close("RCMMC receipt lambda", frozen.get("lambda"), FROZEN_RCMMC_LAMBDA if arm == "G" else 0.0)
    expected_text = {
        "loss_rule": "SOURCE_L_ORDERED_RECEIVER_SLOT_BY_LOCAL4_MOMENT_MATRIX_CONGRUENCE_STOPGRAD_CLEAN_TO_LEO_TOTALIZED_L2_feat_joint",
        "loss_formula": "D_rc=2||mu_L-sg(mu_C)||2^2+||Q_L-sg(Q_C)||F^2;L=sum_rc(A_rc*D_rc)/28",
        "z_id_key": "feat_joint",
        "training_accumulation_dtype": "float32_OUTSIDE_AMP",
        "clean_feature_detached": True,
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "receipt_payload": "SCALARS_COUNTS_AND_SHA_ONLY_NO_IQ_FEATURE_MOMENT_MATRIX_OR_RECEIVER_TOKEN",
        "rcat_relation": "RCAT_ZERO_IMPLIES_RCMMC_ZERO_STRICT_RELAXATION_NOT_BIDIRECTIONALLY_INCOMPARABLE",
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "aux_gradient_scope": "LEO_feat_joint_AND_SHARED_ENCODER_FINITE_NONZERO;EXACT_HEAD_AUX_VJP_NA_NONE_OR_ZERO",
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_SOURCE_SPLIT_RECEIPT_ORDERED_TOKEN_ONLY",
        "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
        "source_receiver_provenance": SOURCE_RECEIVER_PROVENANCE,
    }
    for field, expected in expected_text.items():
        if frozen.get(field) != expected:
            raise RCMMCSplitExportError(f"RCMMC receipt {field} drifted")
    for field, expected in (
        ("loss_global_denominator", FROZEN_RCMMC_GLOBAL_DENOMINATOR),
        ("local_class_count", FROZEN_RCMMC_LOCAL_CLASS_COUNT),
        ("frozen_batch_size", FROZEN_RCMMC_BATCH_SIZE),
        ("frozen_feature_dim", FROZEN_RCMMC_FEATURE_DIM),
        ("frozen_source_receiver_count", FROZEN_RCMMC_SOURCE_RECEIVER_COUNT),
        ("source_receiver_count", FROZEN_RCMMC_SOURCE_RECEIVER_COUNT),
        ("local_data_class_count", FROZEN_RCMMC_LOCAL_CLASS_COUNT),
        ("checkpoint_head_class_count", FROZEN_RCMMC_LOCAL_CLASS_COUNT),
        ("live_head_class_count", FROZEN_RCMMC_LOCAL_CLASS_COUNT),
    ):
        if type(frozen.get(field)) is not int or int(frozen[field]) != expected:
            raise RCMMCSplitExportError(f"RCMMC receipt {field} drifted")
    for field in (
        "baseline_sha256", "initial_checkpoint_sha256", "source_partition_sha256",
        "class_order_binding_sha256", "source_labeled_indices_sha256",
        "source_split_manifest_sha256", "source_receiver_order_sha256",
        "source_receiver_ids_sha256", "optimizer_initial_state_sha256",
        "common_batch_sequence_sha256",
    ):
        _require_sha256(frozen.get(field), field=f"RCMMC receipt {field}")
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
    ):
        raise RCMMCSplitExportError("RCMMC warm-start/AdamW/common-head receipt drifted")
    for field in (
        "uses_new_forward", "uses_resampling", "uses_day_labels", "uses_domain_labels",
        "uses_target_rows", "uses_proxy_rows", "uses_held_rows", "uses_unlabeled_rows",
        "uses_ema_or_state", "uses_threshold", "uses_cross_sample_pairing", "uses_cross_receiver_pairing",
    ):
        _require_bool(frozen, field, False)
    _require_bool(frozen, "uses_rx_labels", True)
    try:
        validated = _rcmmc.validate_rcmmc_terminal_receipt(frozen)
    except (_rcmmc.RCMMCConfigurationError, _rcmmc.RCMMCRuntimeError) as exc:
        raise RCMMCSplitExportError(f"RCMMC terminal receipt revalidation failed: {exc}") from exc
    if validated.get("rcmmc_terminal_contract_passed") is not True:
        raise RCMMCSplitExportError("RCMMC terminal receipt did not pass its raw contract")
    _validate_rcmmc_cells(validated, arm=arm)
    if arm == "G":
        _validate_rcmmc_gradient_audit(validated)
    return dict(validated)


def _source_split_binding(checkpoint: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    split_info = checkpoint.get("split_info")
    if not isinstance(split_info, Mapping):
        raise RCMMCSplitExportError("RCMMC checkpoint lacks split_info")
    source = split_info.get("source_split_receipt")
    partition = split_info.get("tx_partition_receipt")
    if not isinstance(source, Mapping) or not isinstance(partition, Mapping):
        raise RCMMCSplitExportError("RCMMC checkpoint lacks source split/partition receipt")
    try:
        tokens = _rcmmc.resolve_rcmmc_source_receiver_tokens(source)
    except _rcmmc.RCMMCConfigurationError as exc:
        raise RCMMCSplitExportError(f"RCMMC source receiver receipt is invalid: {exc}") from exc
    expected_receiver_sha = _canonical_json_sha256([int(value) for value in tokens])
    if (
        receipt.get("source_receiver_order_sha256") != expected_receiver_sha
        or receipt.get("source_receiver_ids_sha256") != expected_receiver_sha
        or int(receipt.get("source_receiver_count", 0)) != len(tokens)
    ):
        raise RCMMCSplitExportError("RCMMC ordered source receiver SHA/count drifted")
    pairs = (
        ("labeled_indices_sha256", "source_labeled_indices_sha256"),
        ("split_manifest_sha256", "source_split_manifest_sha256"),
    )
    for source_field, receipt_field in pairs:
        if str(source.get(source_field, "")) != str(receipt.get(receipt_field, "")):
            raise RCMMCSplitExportError(f"RCMMC source split binding drifted: {source_field}")
    if str(partition.get("partition_sha256", "")) != str(receipt.get("source_partition_sha256", "")):
        raise RCMMCSplitExportError("RCMMC source partition SHA drifted")


def validate_rcmmc_training_checkpoint(
    checkpoint: Mapping[str, Any], *, checkpoint_path: Path,
    source_tx_ids: Sequence[str], known_validation_tx_ids: Sequence[str], proxy_unknown_tx_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Validate original checkpoint arguments and raw RCMMC receipt only."""

    if str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE or str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise RCMMCSplitExportError("checkpoint must be final-only")
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise RCMMCSplitExportError("checkpoint must contain model and args mappings")
    args = checkpoint["args"]
    expected_text = {
        "split_mode": "tx_rx_day_1_6_3", "model_variant": "lite_d", "id_feature_key": "feat_joint",
        "phase1_source_train_tx_ids": ",".join(source_tx_ids),
        "phase1_source_known_validation_tx_ids": ",".join(known_validation_tx_ids),
        "phase1_source_proxy_unknown_tx_ids": ",".join(proxy_unknown_tx_ids),
        "checkpoint_selection": EXPECTED_CHECKPOINT_SELECTION,
    }
    for field, expected in expected_text.items():
        if str(args.get(field, "")) != expected:
            raise RCMMCSplitExportError(f"checkpoint arg {field} drifted")
    for field, expected in (("labeled_ratio", 0.07), ("unlabeled_ratio", 0.63), ("source_val_ratio", 0.30)):
        _require_close(f"checkpoint arg {field}", args.get(field), expected)
    if int(args.get("seed", -1)) != 7281105:
        raise RCMMCSplitExportError("checkpoint seed drifted")
    match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if match is None:
        raise RCMMCSplitExportError("checkpoint candidate is not frozen RCMMC F1..F6 C/G")
    arm = match.group(2)
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name or str(checkpoint.get("candidate_id", "")) != checkpoint_path.parent.name:
        raise RCMMCSplitExportError("checkpoint candidate does not bind parent directory")
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID or str(checkpoint.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID:
        raise RCMMCSplitExportError("checkpoint run_id does not bind RCMMC training root")
    _require_bool(args, "phase1_rcmmc_frozen_mode", True)
    _require_bool(args, "phase1_rcmmc_enabled", arm == "G")
    _require_close("checkpoint lambda_rcmmc", args.get("lambda_rcmmc"), FROZEN_RCMMC_LAMBDA if arm == "G" else 0.0)
    try:
        _rcmmc.validate_rcmmc_args(SimpleNamespace(**dict(args)))
    except _rcmmc.RCMMCConfigurationError as exc:
        raise RCMMCSplitExportError(f"checkpoint RCMMC frozen-arg contract drifted: {exc}") from exc
    receipt = validate_rcmmc_terminal_receipt(checkpoint.get("rcmmc_receipt", {}), arm=arm)
    _source_split_binding(checkpoint, receipt)
    return args, receipt, arm


def _base_checkpoint_contract(
    checkpoint: Mapping[str, Any], *, checkpoint_path: Path,
    source_tx_ids: Sequence[str], known_validation_tx_ids: Sequence[str], proxy_unknown_tx_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Compatibility adapter used only in-memory by the signed fair exporter."""

    args, receipt, _ = validate_rcmmc_training_checkpoint(
        checkpoint, checkpoint_path=checkpoint_path, source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids, proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    if isinstance(checkpoint, dict):
        checkpoint["icmt_receipt"] = dict(receipt)
    return args


@contextmanager
def _patched_signed_export() -> Iterator[None]:
    """Redirect the generic source reconstruction only for this RCMMC call."""

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
    result.proxy_days = ",".join(FROZEN_PROXY_DAYS)
    result.proxy_rxs = ",".join(FROZEN_PROXY_RXS)
    result.max_proxy_samples_per_tx = FROZEN_PROXY_MAX_SAMPLES_PER_TX
    return result


def _sanitize_mapping(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            lower = key.lower()
            if lower.startswith(LEGACY_IDENTITY_PREFIXES):
                continue
            if lower in {"source_receivers", "target_receivers", "source_receiver_ids", "frozen_source_receiver_ids"}:
                continue
            result[key] = _sanitize_mapping(raw_value)
        return result
    if isinstance(value, list):
        return [_sanitize_mapping(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_mapping(item) for item in value]
    return value


def _assert_rcmmc_only_identity(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key).lower()
            if key.startswith(LEGACY_IDENTITY_PREFIXES):
                raise RCMMCSplitExportError(f"legacy identity leaked into {label}: {raw_key}")
            if key in {"source_receivers", "target_receivers", "source_receiver_ids", "frozen_source_receiver_ids"}:
                raise RCMMCSplitExportError(f"raw source receiver token leaked into {label}: {raw_key}")
            _assert_rcmmc_only_identity(raw_value, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_rcmmc_only_identity(item, label=label)


def _atomic_rewrite_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["manifest_json"] = np.asarray(json.dumps(dict(manifest), ensure_ascii=True))
    temporary = path.with_name(path.name + ".rcmmc-manifest.tmp")
    if temporary.exists():
        raise RCMMCSplitExportError(f"refusing to overwrite temporary export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Run frozen source reconstruction and persist RCMMC-only L/V identity."""

    frozen_args = _coerce_frozen_proxy_args(args)
    checkpoint_path = Path(frozen_args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise RCMMCSplitExportError(f"missing final checkpoint: {checkpoint_path}")
    source_tx_ids = _parse_csv(frozen_args.source_tx_ids, field="source_tx_ids")
    known_tx_ids = _parse_csv(frozen_args.known_validation_tx_ids, field="known_validation_tx_ids")
    proxy_tx_ids = _parse_csv(frozen_args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise RCMMCSplitExportError("RCMMC export requires local4 plus one held and one proxy TX")
    if set(source_tx_ids) & set(known_tx_ids) or set(source_tx_ids) & set(proxy_tx_ids) or set(known_tx_ids) & set(proxy_tx_ids):
        raise RCMMCSplitExportError("source/held/proxy TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RCMMCSplitExportError("checkpoint payload must be a mapping")
    _, receipt, arm = validate_rcmmc_training_checkpoint(
        checkpoint, checkpoint_path=checkpoint_path, source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_tx_ids, proxy_unknown_tx_ids=proxy_tx_ids,
    )
    try:
        with _patched_signed_export():
            base_result = _icmt.export(frozen_args)
    except _icmt.ICMTSplitExportError as exc:
        raise RCMMCSplitExportError(str(exc)) from exc
    manifest = dict(_sanitize_mapping(base_result["manifest"]))
    common_scenes = receipt.get("common_scenario_batches", {})
    manifest.update(
        {
            "schema": EXPECTED_LV_EXPORT_SCHEMA,
            "method": EXPECTED_RECEIPT_METHOD,
            "training_run_contract": EXPECTED_TRAINING_RUN_ID,
            "candidate_id": checkpoint_path.parent.name,
            "rcmmc_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
            "rcmmc_receipt_sha256": _canonical_json_sha256(dict(checkpoint["rcmmc_receipt"])),
            "rcmmc_terminal_contract": str(receipt["rcmmc_terminal_contract"]),
            "rcmmc_terminal_contract_passed": True,
            "rcmmc_enabled": arm == "G",
            "rcmmc_lambda": FROZEN_RCMMC_LAMBDA if arm == "G" else 0.0,
            "rcmmc_frozen_batch_size": FROZEN_RCMMC_BATCH_SIZE,
            "rcmmc_feature_dim": FROZEN_RCMMC_FEATURE_DIM,
            "rcmmc_local_class_count": FROZEN_RCMMC_LOCAL_CLASS_COUNT,
            "rcmmc_loss_global_denominator": FROZEN_RCMMC_GLOBAL_DENOMINATOR,
            "rcmmc_fixed_batch_size": FROZEN_RCMMC_BATCH_SIZE,
            "rcmmc_fixed_feature_dim": FROZEN_RCMMC_FEATURE_DIM,
            "rcmmc_fixed_local_class_count": FROZEN_RCMMC_LOCAL_CLASS_COUNT,
            "rcmmc_fixed_cells_per_scene": FROZEN_RCMMC_GLOBAL_DENOMINATOR,
            "rcmmc_source_receiver_count": FROZEN_RCMMC_SOURCE_RECEIVER_COUNT,
            "rcmmc_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
            "rcmmc_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
            "rcmmc_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "rcmmc_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "rcmmc_source_partition_sha256": str(receipt["source_partition_sha256"]),
            "rcmmc_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
            "rcmmc_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
            "rcmmc_common_scenario_batches": {str(key): int(value) for key, value in dict(common_scenes).items()},
            "rcmmc_common_cells_sha256": _canonical_json_sha256(receipt.get("rcmmc_common_cells", {})),
            "rcmmc_g_scenes_sha256": _canonical_json_sha256(receipt.get("rcmmc_scenes", {})) if arm == "G" else "",
            "rcmmc_clean_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if arm == "G" else "N_A",
            "rcmmc_leo_encoder_aux_vjp": "FINITE_NONZERO_REQUIRED" if arm == "G" else "N_A",
            "rcmmc_common_physical_order_bound": True,
            "rcmmc_common_scene_cycle_bound": True,
            "rcmmc_raw_vjp_required": True,
            "rcmmc_leo_encoder_vjp_finite_nonzero": True,
            "rcmmc_clean_head_vjp_na_none_or_zero": True,
            "proxy_selection_frozen_not_cli_tunable": True,
        }
    )
    _assert_rcmmc_only_identity(manifest, label="RCMMC L/V manifest")
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
