#!/usr/bin/env python
"""Export sealed L/V/proxy features for frozen P1-RCAT postfreeze scoring.

The forwarding and split reconstruction are delegated to the signed ICMT-v2
export kernel. Before that kernel can inspect a checkpoint, this facade
revalidates the original RCAT final-only checkpoint and raw terminal receipt.
It forwards L, V and the fixed 400-row source proxy only; U is reconstructed
and hash-checked but receives no loader, forward or persisted feature row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from cvsrffi import phase1_rcat as _rcat
import export_phase1_icmt_features as _icmt


EXPECTED_TRAINING_RUN_ID = "phase1_rcat12_20260810_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.rcat_lv_export.v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.rcat_receipt.v1"
EXPECTED_RECEIPT_METHOD = "P1_RCAT"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_RCAT12$")
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_RCAT_LAMBDA = 0.02
FROZEN_SOURCE_RECEIVER_IDS = tuple(int(value) for value in _rcat.FROZEN_RCAT_SOURCE_RECEIVER_IDS)
FROZEN_SOURCE_RECEIVER_COUNT = len(FROZEN_SOURCE_RECEIVER_IDS)
FROZEN_CELLS_PER_SCENE = 28
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400
SOURCE_RECEIVER_PROVENANCE = "SOURCE_SPLIT_RECEIPT_source_receivers_PHYSICAL_ID_BOUND_L_ONLY"


class RCATSplitExportError(RuntimeError):
    """Raised when an RCAT final-only source export cannot prove its binding."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise RCATSplitExportError(f"{field} must be non-empty and duplicate-free")
    return items


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise RCATSplitExportError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise RCATSplitExportError(
            f"{name} drifted: expected={expected} observed={observed}"
        )


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    value = mapping.get(field)
    if type(value) is not bool or value is not expected:
        raise RCATSplitExportError(
            f"{field} drifted: expected literal {expected!r}, got {value!r}"
        )


def _require_sha256(value: Any, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
        raise RCATSplitExportError(f"{field} must be a lowercase SHA256")


def _validate_rcat_gradient_audit(receipt: Mapping[str, Any]) -> None:
    audit = receipt.get("rcat_gradient_audit")
    if not isinstance(audit, Mapping):
        raise RCATSplitExportError("RCAT G receipt lacks raw auxiliary VJP audit")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
        or audit.get("exact_head_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        or audit.get("common_l_base_head_input_path") != "LIVE_AND_BOUND_SEPARATELY"
    ):
        raise RCATSplitExportError("RCAT VJP audit is not raw state-free diagnostic evidence")
    for scope in ("feat_joint_leo", "shared_encoder"):
        values = audit.get(scope)
        if not isinstance(values, Mapping):
            raise RCATSplitExportError(f"RCAT VJP {scope} scope is malformed")
        try:
            count = float(values["parameter_count"])
            norm = float(values["norm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RCATSplitExportError(f"RCAT VJP {scope} lacks numeric evidence") from exc
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise RCATSplitExportError(f"RCAT VJP {scope} is zero or non-finite")
    head = audit.get("classifier_head")
    if not isinstance(head, Mapping):
        raise RCATSplitExportError("RCAT exact-head auxiliary VJP scope is malformed")
    try:
        parameter_count = float(head["parameter_count"])
        none_parameters = float(head["none_parameters"])
        zero_parameters = float(head["zero_parameters"])
        nonzero_parameters = float(head["nonzero_parameters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RCATSplitExportError("RCAT exact-head auxiliary VJP lacks numeric evidence") from exc
    head_counts = (parameter_count, none_parameters, zero_parameters, nonzero_parameters)
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in head_counts)
        or parameter_count <= 0.0
        or none_parameters + zero_parameters != parameter_count
        or nonzero_parameters != 0.0
        or head.get("none_or_zero_expected") is not True
    ):
        raise RCATSplitExportError("RCAT exact-head auxiliary VJP must be None-or-zero")


def _validate_source_receiver_binding(receipt: Mapping[str, Any]) -> None:
    for field, expected in (
        ("frozen_source_receiver_ids", list(FROZEN_SOURCE_RECEIVER_IDS)),
        ("source_receiver_ids", list(FROZEN_SOURCE_RECEIVER_IDS)),
    ):
        value = receipt.get(field)
        if type(value) is not list or tuple(value) != tuple(expected) or any(
            type(item) is not int for item in value
        ):
            raise RCATSplitExportError(f"RCAT receipt {field} drifted")
    for field, expected in (
        ("source_receiver_count", FROZEN_SOURCE_RECEIVER_COUNT),
        ("frozen_cells_per_scene", FROZEN_CELLS_PER_SCENE),
    ):
        if type(receipt.get(field)) is not int or receipt.get(field) != expected:
            raise RCATSplitExportError(f"RCAT receipt {field} drifted")
    if receipt.get("source_receiver_provenance") != SOURCE_RECEIVER_PROVENANCE:
        raise RCATSplitExportError("RCAT source receiver provenance drifted")
    expected_sha = _canonical_json_sha256(list(FROZEN_SOURCE_RECEIVER_IDS))
    if receipt.get("source_receiver_ids_sha256") != expected_sha:
        raise RCATSplitExportError("RCAT source receiver SHA256 drifted")


def validate_rcat_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Re-run RCAT terminal closure and inspect raw arm-specific evidence."""

    if not isinstance(receipt, Mapping):
        raise RCATSplitExportError("checkpoint lacks an RCAT terminal receipt")
    frozen = dict(receipt)
    if str(frozen.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA:
        raise RCATSplitExportError("RCAT terminal receipt schema drifted")
    if str(frozen.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise RCATSplitExportError("RCAT terminal receipt method drifted")
    _require_bool(frozen, "frozen_mode", True)
    expected_enabled = arm == "G"
    _require_bool(frozen, "enabled", expected_enabled)
    _require_close(
        "RCAT receipt lambda",
        frozen.get("lambda"),
        FROZEN_RCAT_LAMBDA if expected_enabled else 0.0,
    )
    if (
        type(frozen.get("loss_global_denominator")) is not int
        or frozen.get("loss_global_denominator") != FROZEN_CELLS_PER_SCENE
    ):
        raise RCATSplitExportError("RCAT fixed 1/28 denominator receipt drifted")
    expected_semantics = {
        "loss_rule": "SOURCE_L_RX_BY_LOCAL4_EQUAL_WEIGHT_STOPGRAD_CLEAN_TO_LEO_TOTALIZED_L2_feat_joint",
        "loss_formula": "T(z)=z/||z||2_if_norm_gt_0_else_0;q=||T(z_leo)-sg(T(z_clean))||2;g_rc=0_if_n_rc=0_else_mean_Irc(q);L=sum_rc(g_rc)/28",
        "z_id_key": "feat_joint",
        "feature_dimension_contract": "RAW_ENCODER_feat_joint_EXACT_HEAD_INPUT_DIMENSION_BOUND",
        "totalized_l2_rule": "T(z)=z/||z||2_IF_norm_gt_0_ELSE_0",
        "training_accumulation_dtype": "float32",
        "postfreeze_totalized_l2_dtype": "float64_SAME_PIECEWISE_RULE_NOT_BYTE_IDENTICAL",
        "same_physical_pairing": "SAME_SOURCE_L_PHYSICAL_ROW_COMMON_CLEAN_AND_SINGLE_LEO_FORWARD",
        "common_sat_kl": "sg(clean_tx_logits)_TO_leo_tx_logits",
        "head_input_path": "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)",
        "aux_gradient_scope": "LEO_feat_joint_AND_SHARED_ENCODER_FINITE_NONZERO;EXACT_HEAD_AUX_VJP_NA_NONE_OR_ZERO",
        "rx_permission": "SOURCE_KNOWN_TRAIN_L_PHYSICAL_ID_BOUND_rx_i_ONLY",
        "no_day_assertion": "day_i_NOT_READ_BY_RCAT",
        "amp_contract": "COMMON_TRAINER_AMP_ENABLED",
    }
    for field, expected in expected_semantics.items():
        if frozen.get(field) != expected:
            raise RCATSplitExportError(f"RCAT receipt {field} drifted")
    if frozen.get("clean_feature_detached") is not True:
        raise RCATSplitExportError("RCAT clean feature must remain stop-gradient")
    if frozen.get("common_l_base_head_input_path_verified") is not True:
        raise RCATSplitExportError("RCAT common L_base exact-head input path is not live")
    _require_close("RCAT common lambda_sat_cons", frozen.get("common_lambda_sat_cons"), 0.10)
    expected_bool_permissions = {
        "uses_new_forward": False,
        "uses_resampling": False,
        "uses_rx_labels": True,
        "uses_day_labels": False,
        "uses_domain_labels": False,
        "uses_target_rows": False,
        "uses_proxy_rows": False,
        "uses_held_rows": False,
        "uses_unlabeled_rows": False,
        "uses_ema_or_state": False,
        "uses_threshold": False,
        "uses_gradient_projection": False,
        "uses_cross_sample_pairing": False,
        "uses_cross_receiver_pairing": False,
    }
    for field, expected in expected_bool_permissions.items():
        _require_bool(frozen, field, expected)
    if frozen.get("rx_metadata_allowlist") != ["rx_i"]:
        raise RCATSplitExportError("RCAT RX metadata allowlist drifted")
    _validate_source_receiver_binding(frozen)
    if str(frozen.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise RCATSplitExportError("RCAT receipt checkpoint role is not training_final_only")
    if tuple(str(item) for item in frozen.get("source_train_tx", [])) != tuple(source_tx_ids):
        raise RCATSplitExportError("RCAT receipt source train TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_known_validation_tx", [])) != tuple(
        known_validation_tx_ids
    ):
        raise RCATSplitExportError("RCAT receipt known-validation TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_proxy_unknown_tx", [])) != tuple(
        proxy_unknown_tx_ids
    ):
        raise RCATSplitExportError("RCAT receipt proxy-unknown TX binding drifted")
    if tuple(str(item) for item in frozen.get("local_tx_class_order", [])) != tuple(source_tx_ids):
        raise RCATSplitExportError("RCAT receipt local TX/head order drifted")
    if tuple(str(item) for item in frozen.get("checkpoint_train_tx_class_order", [])) != tuple(
        source_tx_ids
    ):
        raise RCATSplitExportError("RCAT receipt checkpoint TX/head order drifted")
    if tuple(int(item) for item in frozen.get("local_to_head_class_ids", [])) != (0, 1, 2, 3):
        raise RCATSplitExportError("RCAT receipt local-to-head class order drifted")
    for field in (
        "baseline_sha256",
        "initial_checkpoint_sha256",
        "class_order_binding_sha256",
        "source_labeled_indices_sha256",
        "source_split_manifest_sha256",
        "source_receiver_ids_sha256",
        "optimizer_initial_state_sha256",
        "common_batch_sequence_sha256",
    ):
        _require_sha256(frozen.get(field), field=f"RCAT receipt {field}")
    if (
        frozen.get("optimizer_state_restored") is not False
        or frozen.get("rng_state_restored") is not False
        or frozen.get("optimizer_initial_state_empty") is not True
        or frozen.get("optimizer_type") != "AdamW"
        or frozen.get("warm_start_mode") != "MODEL_WEIGHTS_ONLY_NEW_ADAMW_AMP"
        or frozen.get("strict_model_keys") is not True
        or frozen.get("missing_model_keys") != []
        or frozen.get("unexpected_model_keys") != []
    ):
        raise RCATSplitExportError("RCAT receipt does not prove a new AdamW/RNG state")
    try:
        validated = _rcat.validate_rcat_terminal_receipt(frozen)
    except (_rcat.RCATConfigurationError, _rcat.RCATRuntimeError) as exc:
        raise RCATSplitExportError(f"RCAT terminal receipt revalidation failed: {exc}") from exc
    if validated.get("rcat_terminal_contract_passed") is not True:
        raise RCATSplitExportError("RCAT terminal receipt did not pass its raw contract")
    _validate_source_receiver_binding(validated)
    common_cells = validated.get("rcat_common_cells")
    if (
        not isinstance(common_cells, Mapping)
        or set(common_cells) != set(_rcat.FROZEN_RCAT_SCENARIOS)
        or any(
            not isinstance(common_cells[scene], Mapping)
            or len(common_cells[scene]) != FROZEN_CELLS_PER_SCENE
            for scene in _rcat.FROZEN_RCAT_SCENARIOS
        )
    ):
        raise RCATSplitExportError("RCAT common 28-cell/three-scene receipt drifted")
    if expected_enabled:
        if int(validated.get("rcat_positive_q", 0)) <= 0:
            raise RCATSplitExportError("RCAT G receipt lacks positive-q evidence")
        if validated.get("rcat_gradient_audit_completed") is not True:
            raise RCATSplitExportError("RCAT G receipt lacks completed VJP audit")
        _validate_rcat_gradient_audit(validated)
        scenes = validated.get("rcat_scenes")
        if (
            not isinstance(scenes, Mapping)
            or set(scenes) != set(_rcat.FROZEN_RCAT_SCENARIOS)
            or any(
                not isinstance(scenes[scene], Mapping)
                or len(scenes[scene]) != FROZEN_CELLS_PER_SCENE
                for scene in _rcat.FROZEN_RCAT_SCENARIOS
            )
        ):
            raise RCATSplitExportError("RCAT G receipt lacks 84-cell auxiliary coverage")
    else:
        zero_fields = ("rcat_batches", "rcat_total_rows", "rcat_positive_q")
        if any(int(validated.get(field, -1)) != 0 for field in zero_fields):
            raise RCATSplitExportError("RCAT C receipt contains non-zero auxiliary evidence")
        if abs(float(validated.get("rcat_loss_sum", float("nan")))) > 1e-12:
            raise RCATSplitExportError("RCAT C receipt contains non-zero auxiliary loss")
        if any(
            bool(validated.get(field))
            for field in ("rcat_scenes", "rcat_g_batch_aux", "rcat_gradient_audit")
        ) or validated.get("rcat_gradient_audit_completed") is not False:
            raise RCATSplitExportError("RCAT C receipt must keep G-only fields N/A-or-zero")
    return dict(validated)


def validate_rcat_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Validate the original checkpoint, args and raw RCAT terminal receipt."""

    if str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise RCATSplitExportError("checkpoint_role must be training_final_only")
    if str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise RCATSplitExportError("checkpoint_selection must be final_only")
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise RCATSplitExportError("checkpoint must contain model and args mappings")
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
            raise RCATSplitExportError(
                f"checkpoint arg {field} drifted: expected={expected} observed={args.get(field)}"
            )
    for field, expected in (
        ("labeled_ratio", 0.07),
        ("unlabeled_ratio", 0.63),
        ("source_val_ratio", 0.30),
    ):
        _require_close(f"checkpoint arg {field}", args.get(field), expected)
    if int(args.get("seed", -1)) != 7281105:
        raise RCATSplitExportError("checkpoint seed must remain 7281105")
    _require_bool(args, "phase1_rcat_frozen_mode", True)
    candidate_match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if candidate_match is None:
        raise RCATSplitExportError("checkpoint candidate is not a frozen F1..F6 C/G RCAT12 arm")
    arm = candidate_match.group(2)
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name or str(
        checkpoint.get("candidate_id", "")
    ) != checkpoint_path.parent.name:
        raise RCATSplitExportError("checkpoint candidate_id does not bind parent arm directory")
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID or str(
        checkpoint.get("run_id", "")
    ) != EXPECTED_TRAINING_RUN_ID:
        raise RCATSplitExportError("checkpoint run_id does not bind frozen RCAT training root")
    expected_enabled = arm == "G"
    _require_bool(args, "phase1_rcat_enabled", expected_enabled)
    _require_close(
        "checkpoint arg lambda_rcat",
        args.get("lambda_rcat"),
        FROZEN_RCAT_LAMBDA if expected_enabled else 0.0,
    )
    receipt = validate_rcat_terminal_receipt(
        checkpoint.get("rcat_receipt", {}),
        arm=arm,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    return args, receipt, arm


def _base_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> Mapping[str, Any]:
    """Compatibility adapter consumed only inside signed source split mechanics."""

    args, receipt, _ = validate_rcat_training_checkpoint(
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
    """Temporarily bind the generic source exporter to RCAT identities."""

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
            raise RCATSplitExportError(f"{field} is frozen and may not be changed")
        setattr(result, field, expected)
    return result


def _atomic_rewrite_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["manifest_json"] = np.asarray(json.dumps(dict(manifest), ensure_ascii=True))
    temporary = path.with_name(path.name + ".rcat-manifest.tmp")
    if temporary.exists():
        raise RCATSplitExportError(f"refusing to overwrite temporary export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Run signed source reconstruction with RCAT-only checkpoint identity."""

    frozen_args = _coerce_frozen_proxy_args(args)
    checkpoint_path = Path(frozen_args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise RCATSplitExportError(f"missing final checkpoint: {checkpoint_path}")
    source_tx_ids = _parse_csv(frozen_args.source_tx_ids, field="source_tx_ids")
    known_tx_ids = _parse_csv(
        frozen_args.known_validation_tx_ids, field="known_validation_tx_ids"
    )
    proxy_tx_ids = _parse_csv(frozen_args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise RCATSplitExportError("P1-RCAT export requires local4 plus one held and one proxy TX")
    roles = (set(source_tx_ids), set(known_tx_ids), set(proxy_tx_ids))
    if any(roles[left] & roles[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RCATSplitExportError("source/known-validation/proxy TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RCATSplitExportError("checkpoint payload must be a mapping")
    _, receipt, arm = validate_rcat_training_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_tx_ids,
        proxy_unknown_tx_ids=proxy_tx_ids,
    )
    with _patched_signed_export():
        base_result = _icmt.export(frozen_args)
    manifest = dict(base_result["manifest"])
    for field in (
        "icmt_receipt_schema",
        "icmt_enabled",
        "icmt_source_labeled_indices_sha256",
        "icmt_source_split_manifest_sha256",
    ):
        manifest.pop(field, None)
    manifest.update(
        {
            "schema": EXPECTED_LV_EXPORT_SCHEMA,
            "method": EXPECTED_RECEIPT_METHOD,
            "training_run_contract": EXPECTED_TRAINING_RUN_ID,
            "rcat_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
            "rcat_enabled": arm == "G",
            "rcat_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "rcat_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "rcat_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
            "rcat_source_receiver_ids": list(FROZEN_SOURCE_RECEIVER_IDS),
            "rcat_source_receiver_count": FROZEN_SOURCE_RECEIVER_COUNT,
            "rcat_source_receiver_provenance": SOURCE_RECEIVER_PROVENANCE,
            "rcat_frozen_cells_per_scene": FROZEN_CELLS_PER_SCENE,
            "rcat_receipt_sha256": _canonical_json_sha256(dict(checkpoint["rcat_receipt"])),
            "rcat_terminal_contract": str(receipt["rcat_terminal_contract"]),
            "rcat_terminal_contract_passed": True,
            "rcat_lambda": FROZEN_RCAT_LAMBDA if arm == "G" else 0.0,
            "rcat_loss_global_denominator": FROZEN_CELLS_PER_SCENE,
            "rcat_totalized_l2_rule": str(receipt["totalized_l2_rule"]),
            "rcat_training_accumulation_dtype": str(receipt["training_accumulation_dtype"]),
            "rcat_postfreeze_totalized_l2_dtype": str(receipt["postfreeze_totalized_l2_dtype"]),
            "rcat_clean_feature_detached": True,
            "rcat_common_l_base_head_input_path_verified": True,
            "rcat_exact_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED",
            "rcat_common_physical_rx_class_scene_nrc_bound": True,
            "rcat_batch_order_bound": True,
            "proxy_selection_frozen_not_cli_tunable": True,
        }
    )
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
