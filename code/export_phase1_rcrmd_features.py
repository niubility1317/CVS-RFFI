#!/usr/bin/env python
"""Export sealed L/V/proxy features for frozen P1-RCRMD postfreeze scoring.

The forwarding and split reconstruction are delegated to the signed ICMT-v2
export kernel. Before that kernel can inspect a checkpoint, this facade
revalidates the original RCRMD final-only checkpoint and raw terminal receipt.
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

from cvsrffi import phase1_rcrmd as _rcrmd
import export_phase1_icmt_features as _icmt


EXPECTED_TRAINING_RUN_ID = "phase1_rcrmd12_20260810_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.rcrmd_lv_export.v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.rcrmd_receipt.v1"
EXPECTED_RECEIPT_METHOD = "P1_RCRMD"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_RCRMD12$")
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_RCRMD_LAMBDA = 0.02
FROZEN_SOURCE_RECEIVER_IDS = tuple(int(value) for value in _rcrmd.FROZEN_RCRMD_SOURCE_RECEIVER_IDS)
FROZEN_SOURCE_RECEIVER_COUNT = len(FROZEN_SOURCE_RECEIVER_IDS)
FROZEN_CELLS_PER_SCENE = 28
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400
SOURCE_RECEIVER_PROVENANCE = "SOURCE_SPLIT_RECEIPT_source_receivers_PHYSICAL_ID_BOUND_L_ONLY"


class RCRMDSplitExportError(RuntimeError):
    """Raised when an RCRMD final-only source export cannot prove its binding."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise RCRMDSplitExportError(f"{field} must be non-empty and duplicate-free")
    return items


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise RCRMDSplitExportError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise RCRMDSplitExportError(
            f"{name} drifted: expected={expected} observed={observed}"
        )


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    value = mapping.get(field)
    if type(value) is not bool or value is not expected:
        raise RCRMDSplitExportError(
            f"{field} drifted: expected literal {expected!r}, got {value!r}"
        )


def _require_sha256(value: Any, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
        raise RCRMDSplitExportError(f"{field} must be a lowercase SHA256")


def _validate_rcrmd_gradient_audit(receipt: Mapping[str, Any]) -> None:
    audit = receipt.get("rcrmd_gradient_audit")
    if not isinstance(audit, Mapping):
        raise RCRMDSplitExportError("RCRMD G receipt lacks raw auxiliary VJP audit")
    if (
        audit.get("raw_unscaled") is not True
        or audit.get("diagnostic_only") is not True
        or audit.get("touches_amp_optimizer_rng") is not False
    ):
        raise RCRMDSplitExportError("RCRMD VJP audit is not raw state-free diagnostic evidence")
    for scope in ("shared_encoder", "classifier_head"):
        values = audit.get(scope)
        if not isinstance(values, Mapping):
            raise RCRMDSplitExportError(f"RCRMD VJP {scope} scope is malformed")
        try:
            count = float(values["parameter_count"])
            norm = float(values["norm"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RCRMDSplitExportError(f"RCRMD VJP {scope} lacks numeric evidence") from exc
        if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
            raise RCRMDSplitExportError(f"RCRMD VJP {scope} is zero or non-finite")


def _validate_source_receiver_binding(receipt: Mapping[str, Any]) -> None:
    for field, expected in (
        ("frozen_source_receiver_ids", list(FROZEN_SOURCE_RECEIVER_IDS)),
        ("source_receiver_ids", list(FROZEN_SOURCE_RECEIVER_IDS)),
    ):
        value = receipt.get(field)
        if type(value) is not list or tuple(value) != tuple(expected) or any(
            type(item) is not int for item in value
        ):
            raise RCRMDSplitExportError(f"RCRMD receipt {field} drifted")
    for field, expected in (
        ("frozen_source_receiver_count", FROZEN_SOURCE_RECEIVER_COUNT),
        ("source_receiver_count", FROZEN_SOURCE_RECEIVER_COUNT),
        ("frozen_cells_per_scene", FROZEN_CELLS_PER_SCENE),
    ):
        if type(receipt.get(field)) is not int or receipt.get(field) != expected:
            raise RCRMDSplitExportError(f"RCRMD receipt {field} drifted")
    if receipt.get("source_receiver_provenance") != SOURCE_RECEIVER_PROVENANCE:
        raise RCRMDSplitExportError("RCRMD source receiver provenance drifted")
    expected_sha = _canonical_json_sha256(list(FROZEN_SOURCE_RECEIVER_IDS))
    if receipt.get("source_receiver_ids_sha256") != expected_sha:
        raise RCRMDSplitExportError("RCRMD source receiver SHA256 drifted")


def validate_rcrmd_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Re-run RCRMD terminal closure and inspect raw arm-specific evidence."""

    if not isinstance(receipt, Mapping):
        raise RCRMDSplitExportError("checkpoint lacks an RCRMD terminal receipt")
    frozen = dict(receipt)
    if str(frozen.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA:
        raise RCRMDSplitExportError("RCRMD terminal receipt schema drifted")
    if str(frozen.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise RCRMDSplitExportError("RCRMD terminal receipt method drifted")
    _require_bool(frozen, "frozen_mode", True)
    expected_enabled = arm == "G"
    _require_bool(frozen, "enabled", expected_enabled)
    _require_close(
        "RCRMD receipt lambda",
        frozen.get("lambda"),
        FROZEN_RCRMD_LAMBDA if expected_enabled else 0.0,
    )
    if frozen.get("loss_global_denominator") != "4_TIMES_FIXED_SOURCE_RECEIVER_COUNT":
        raise RCRMDSplitExportError("RCRMD fixed 1/28 denominator receipt drifted")
    if frozen.get("clean_margin_detached") is not True:
        raise RCRMDSplitExportError("RCRMD clean margin must remain stop-gradient")
    _validate_source_receiver_binding(frozen)
    if str(frozen.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise RCRMDSplitExportError("RCRMD receipt checkpoint role is not training_final_only")
    if tuple(str(item) for item in frozen.get("source_train_tx", [])) != tuple(source_tx_ids):
        raise RCRMDSplitExportError("RCRMD receipt source train TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_known_validation_tx", [])) != tuple(
        known_validation_tx_ids
    ):
        raise RCRMDSplitExportError("RCRMD receipt known-validation TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_proxy_unknown_tx", [])) != tuple(
        proxy_unknown_tx_ids
    ):
        raise RCRMDSplitExportError("RCRMD receipt proxy-unknown TX binding drifted")
    if tuple(str(item) for item in frozen.get("local_tx_class_order", [])) != tuple(source_tx_ids):
        raise RCRMDSplitExportError("RCRMD receipt local TX/head order drifted")
    if tuple(str(item) for item in frozen.get("checkpoint_train_tx_class_order", [])) != tuple(
        source_tx_ids
    ):
        raise RCRMDSplitExportError("RCRMD receipt checkpoint TX/head order drifted")
    if tuple(int(item) for item in frozen.get("local_to_head_class_ids", [])) != (0, 1, 2, 3):
        raise RCRMDSplitExportError("RCRMD receipt local-to-head class order drifted")
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
        _require_sha256(frozen.get(field), field=f"RCRMD receipt {field}")
    if (
        frozen.get("optimizer_state_restored") is not False
        or frozen.get("rng_state_restored") is not False
        or frozen.get("optimizer_initial_state_empty") is not True
        or frozen.get("optimizer_type") != "AdamW"
    ):
        raise RCRMDSplitExportError("RCRMD receipt does not prove a new AdamW/RNG state")
    try:
        validated = _rcrmd.validate_rcrmd_terminal_receipt(frozen)
    except (_rcrmd.RCRMDConfigurationError, _rcrmd.RCRMDRuntimeError) as exc:
        raise RCRMDSplitExportError(f"RCRMD terminal receipt revalidation failed: {exc}") from exc
    if validated.get("rcrmd_terminal_contract_passed") is not True:
        raise RCRMDSplitExportError("RCRMD terminal receipt did not pass its raw contract")
    _validate_source_receiver_binding(validated)
    common_cells = validated.get("rcrmd_common_cells")
    if (
        not isinstance(common_cells, Mapping)
        or set(common_cells) != set(_rcrmd.FROZEN_RCRMD_SCENARIOS)
        or any(
            not isinstance(common_cells[scene], Mapping)
            or len(common_cells[scene]) != FROZEN_CELLS_PER_SCENE
            for scene in _rcrmd.FROZEN_RCRMD_SCENARIOS
        )
    ):
        raise RCRMDSplitExportError("RCRMD common 28-cell/three-scene receipt drifted")
    if expected_enabled:
        if validated.get("rcrmd_gradient_audit_completed") is not True:
            raise RCRMDSplitExportError("RCRMD G receipt lacks completed VJP audit")
        _validate_rcrmd_gradient_audit(validated)
        scenes = validated.get("rcrmd_scenes")
        if (
            not isinstance(scenes, Mapping)
            or set(scenes) != set(_rcrmd.FROZEN_RCRMD_SCENARIOS)
            or any(
                not isinstance(scenes[scene], Mapping)
                or len(scenes[scene]) != FROZEN_CELLS_PER_SCENE
                for scene in _rcrmd.FROZEN_RCRMD_SCENARIOS
            )
        ):
            raise RCRMDSplitExportError("RCRMD G receipt lacks 84-cell auxiliary coverage")
    else:
        zero_fields = ("rcrmd_batches", "rcrmd_total_rows", "rcrmd_active_q")
        if any(int(validated.get(field, -1)) != 0 for field in zero_fields):
            raise RCRMDSplitExportError("RCRMD C receipt contains non-zero auxiliary evidence")
        if abs(float(validated.get("rcrmd_loss_sum", float("nan")))) > 1e-12:
            raise RCRMDSplitExportError("RCRMD C receipt contains non-zero auxiliary loss")
        if any(
            bool(validated.get(field))
            for field in ("rcrmd_scenes", "rcrmd_g_batch_aux", "rcrmd_gradient_audit")
        ) or validated.get("rcrmd_gradient_audit_completed") is not False:
            raise RCRMDSplitExportError("RCRMD C receipt must keep G-only fields N/A-or-zero")
    return dict(validated)


def validate_rcrmd_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Validate the original checkpoint, args and raw RCRMD terminal receipt."""

    if str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise RCRMDSplitExportError("checkpoint_role must be training_final_only")
    if str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise RCRMDSplitExportError("checkpoint_selection must be final_only")
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise RCRMDSplitExportError("checkpoint must contain model and args mappings")
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
            raise RCRMDSplitExportError(
                f"checkpoint arg {field} drifted: expected={expected} observed={args.get(field)}"
            )
    for field, expected in (
        ("labeled_ratio", 0.07),
        ("unlabeled_ratio", 0.63),
        ("source_val_ratio", 0.30),
    ):
        _require_close(f"checkpoint arg {field}", args.get(field), expected)
    if int(args.get("seed", -1)) != 7281105:
        raise RCRMDSplitExportError("checkpoint seed must remain 7281105")
    _require_bool(args, "phase1_rcrmd_frozen_mode", True)
    candidate_match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if candidate_match is None:
        raise RCRMDSplitExportError("checkpoint candidate is not a frozen F1..F6 C/G RCRMD12 arm")
    arm = candidate_match.group(2)
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name or str(
        checkpoint.get("candidate_id", "")
    ) != checkpoint_path.parent.name:
        raise RCRMDSplitExportError("checkpoint candidate_id does not bind parent arm directory")
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID or str(
        checkpoint.get("run_id", "")
    ) != EXPECTED_TRAINING_RUN_ID:
        raise RCRMDSplitExportError("checkpoint run_id does not bind frozen RCRMD training root")
    expected_enabled = arm == "G"
    _require_bool(args, "phase1_rcrmd_enabled", expected_enabled)
    _require_close(
        "checkpoint arg lambda_rcrmd",
        args.get("lambda_rcrmd"),
        FROZEN_RCRMD_LAMBDA if expected_enabled else 0.0,
    )
    receipt = validate_rcrmd_terminal_receipt(
        checkpoint.get("rcrmd_receipt", {}),
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

    args, receipt, _ = validate_rcrmd_training_checkpoint(
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
    """Temporarily bind the generic source exporter to RCRMD identities."""

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
            raise RCRMDSplitExportError(f"{field} is frozen and may not be changed")
        setattr(result, field, expected)
    return result


def _atomic_rewrite_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["manifest_json"] = np.asarray(json.dumps(dict(manifest), ensure_ascii=True))
    temporary = path.with_name(path.name + ".rcrmd-manifest.tmp")
    if temporary.exists():
        raise RCRMDSplitExportError(f"refusing to overwrite temporary export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Run signed source reconstruction with RCRMD-only checkpoint identity."""

    frozen_args = _coerce_frozen_proxy_args(args)
    checkpoint_path = Path(frozen_args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise RCRMDSplitExportError(f"missing final checkpoint: {checkpoint_path}")
    source_tx_ids = _parse_csv(frozen_args.source_tx_ids, field="source_tx_ids")
    known_tx_ids = _parse_csv(
        frozen_args.known_validation_tx_ids, field="known_validation_tx_ids"
    )
    proxy_tx_ids = _parse_csv(frozen_args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise RCRMDSplitExportError("P1-RCRMD export requires local4 plus one held and one proxy TX")
    roles = (set(source_tx_ids), set(known_tx_ids), set(proxy_tx_ids))
    if any(roles[left] & roles[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RCRMDSplitExportError("source/known-validation/proxy TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RCRMDSplitExportError("checkpoint payload must be a mapping")
    _, receipt, arm = validate_rcrmd_training_checkpoint(
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
            "rcrmd_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
            "rcrmd_enabled": arm == "G",
            "rcrmd_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "rcrmd_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "rcrmd_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
            "rcrmd_source_receiver_ids": list(FROZEN_SOURCE_RECEIVER_IDS),
            "rcrmd_source_receiver_count": FROZEN_SOURCE_RECEIVER_COUNT,
            "rcrmd_source_receiver_provenance": SOURCE_RECEIVER_PROVENANCE,
            "rcrmd_frozen_cells_per_scene": FROZEN_CELLS_PER_SCENE,
            "rcrmd_receipt_sha256": _canonical_json_sha256(dict(checkpoint["rcrmd_receipt"])),
            "rcrmd_terminal_contract": str(receipt["rcrmd_terminal_contract"]),
            "rcrmd_terminal_contract_passed": True,
            "rcrmd_lambda": FROZEN_RCRMD_LAMBDA if arm == "G" else 0.0,
            "rcrmd_loss_global_denominator": "4_TIMES_FIXED_SOURCE_RECEIVER_COUNT",
            "rcrmd_common_physical_rx_class_scene_nrc_bound": True,
            "rcrmd_batch_order_bound": True,
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
