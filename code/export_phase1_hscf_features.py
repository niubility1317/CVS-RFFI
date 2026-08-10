#!/usr/bin/env python
"""Export sealed L/V/proxy features for frozen P1-HSCF postfreeze scoring.

The signed ICMT exporter is a source-split/forward compatibility kernel only.
This facade owns HSCF identity and reopens the raw HSCF terminal receipt before
every source row is exported.  It never iterates, forwards, fits or persists U,
and it preserves all L/V/proxy rows while fixing the 400-row proxy selection.
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

from cvsrffi import phase1_hscf as _hscf
import export_phase1_icmt_features as _icmt


EXPECTED_TRAINING_RUN_ID = "phase1_hscf12_20260811_v2"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.hscf_lv_export.v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.hscf_receipt.v1"
EXPECTED_RECEIPT_METHOD = "P1_HSCF"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_HSCF12$")
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_HSCF_LAMBDA = float(_hscf.FROZEN_HSCF_LAMBDA)
FROZEN_HSCF_BATCH_SIZE = int(_hscf.FROZEN_HSCF_BATCH_SIZE)
FROZEN_HSCF_LOCAL_CLASS_COUNT = int(_hscf.FROZEN_HSCF_CLASS_COUNT)
FROZEN_HSCF_GLOBAL_DENOMINATOR = int(_hscf.FROZEN_HSCF_GLOBAL_DENOMINATOR)
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400
SOURCE_L_PROVENANCE = "SOURCE_SPLIT_RECEIPT_L_PHYSICAL_ORDER_ONLY_RX_DAY_NOT_READ"
LEGACY_IDENTITY_PREFIXES = ("icmt_", "rcat_", "rcrmd_", "cagm_", "recte_")


class HSCFSplitExportError(RuntimeError):
    """Raised when a HSCF final-only source export cannot prove its binding."""


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise HSCFSplitExportError(f"{field} must be non-empty and duplicate-free")
    return items


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise HSCFSplitExportError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise HSCFSplitExportError(f"{name} drifted: expected={expected} observed={observed}")


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    value = mapping.get(field)
    if type(value) is not bool or value is not expected:
        raise HSCFSplitExportError(f"{field} drifted: expected literal {expected!r}, got {value!r}")


def _require_sha256(value: Any, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
        raise HSCFSplitExportError(f"{field} must be a lowercase SHA256")


def _validate_none_or_zero_audit(values: Any, *, field: str, allow_absent: bool) -> None:
    if not isinstance(values, Mapping):
        raise HSCFSplitExportError(f"HSCF audit lacks {field}")
    try:
        count = float(values.get("parameter_count", float("nan")))
        none_count = float(values.get("none_parameters", float("nan")))
        zero_count = float(values.get("zero_parameters", float("nan")))
        nonzero_count = float(values.get("nonzero_parameters", float("nan")))
    except (TypeError, ValueError) as exc:
        raise HSCFSplitExportError(f"HSCF audit {field} is malformed") from exc
    if (
        not all(math.isfinite(value) and value >= 0.0 for value in (count, none_count, zero_count, nonzero_count))
        or none_count + zero_count != count
        or nonzero_count != 0.0
        or values.get("none_or_zero_expected") is not True
        or (not allow_absent and count <= 0.0)
    ):
        raise HSCFSplitExportError(f"HSCF audit {field} must be None/zero")


def _validate_hscf_gradient_audits(receipt: Mapping[str, Any]) -> None:
    if receipt.get("hscf_gradient_audit_completed") is not True:
        raise HSCFSplitExportError("HSCF G receipt lacks completed per-scene VJP audits")
    audits = receipt.get("hscf_gradient_audit_scenes")
    scenes = receipt.get("hscf_scenes")
    expected = set(_hscf.FROZEN_HSCF_SCENARIOS)
    if not isinstance(audits, Mapping) or set(audits) != expected:
        raise HSCFSplitExportError("HSCF G receipt lacks clear/low/rain VJP audits")
    if not isinstance(scenes, Mapping) or set(scenes) != expected:
        raise HSCFSplitExportError("HSCF G receipt lacks clear/low/rain positive coverage")
    for scenario in _hscf.FROZEN_HSCF_SCENARIOS:
        scene = scenes[scenario]
        audit = audits[scenario]
        if not isinstance(scene, Mapping) or int(scene.get("positive_batches", 0)) <= 0:
            raise HSCFSplitExportError(f"HSCF {scenario} lacks a positive auxiliary batch")
        if not isinstance(audit, Mapping) or (
            audit.get("raw_unscaled") is not True
            or audit.get("diagnostic_only") is not True
            or audit.get("touches_amp_optimizer_rng") is not False
            or audit.get("clean_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
            or audit.get("head_bias_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        ):
            raise HSCFSplitExportError(f"HSCF {scenario} VJP audit semantics drifted")
        for group in ("leo_raw_logits", "shared_encoder", "head_weight"):
            evidence = audit.get(group)
            if not isinstance(evidence, Mapping):
                raise HSCFSplitExportError(f"HSCF {scenario} VJP lacks {group}")
            try:
                count = float(evidence["parameter_count"])
                norm = float(evidence["norm"])
            except (KeyError, TypeError, ValueError) as exc:
                raise HSCFSplitExportError(f"HSCF {scenario} VJP {group} is malformed") from exc
            if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
                raise HSCFSplitExportError(f"HSCF {scenario} VJP {group} is zero/nonfinite")
        _validate_none_or_zero_audit(audit.get("clean_raw_logits"), field="clean_raw_logits", allow_absent=False)
        _validate_none_or_zero_audit(audit.get("head_bias"), field="head_bias", allow_absent=True)


def validate_hscf_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Revalidate raw HSCF terminal evidence before any sealed export."""

    if not isinstance(receipt, Mapping):
        raise HSCFSplitExportError("checkpoint lacks a HSCF terminal receipt")
    frozen = dict(receipt)
    if str(frozen.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA:
        raise HSCFSplitExportError("HSCF terminal receipt schema drifted")
    if str(frozen.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise HSCFSplitExportError("HSCF terminal receipt method drifted")
    _require_bool(frozen, "frozen_mode", True)
    expected_enabled = arm == "G"
    _require_bool(frozen, "enabled", expected_enabled)
    _require_close("HSCF receipt lambda", frozen.get("lambda"), FROZEN_HSCF_LAMBDA if expected_enabled else 0.0)
    if str(frozen.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise HSCFSplitExportError("HSCF receipt checkpoint role is not training_final_only")
    if tuple(str(item) for item in frozen.get("source_train_tx", [])) != tuple(source_tx_ids):
        raise HSCFSplitExportError("HSCF receipt source train TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_known_validation_tx", [])) != tuple(known_validation_tx_ids):
        raise HSCFSplitExportError("HSCF receipt known-validation TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_proxy_unknown_tx", [])) != tuple(proxy_unknown_tx_ids):
        raise HSCFSplitExportError("HSCF receipt proxy-unknown TX binding drifted")
    if tuple(str(item) for item in frozen.get("local_tx_class_order", [])) != tuple(source_tx_ids):
        raise HSCFSplitExportError("HSCF receipt local TX/head order drifted")
    if tuple(str(item) for item in frozen.get("checkpoint_train_tx_class_order", [])) != tuple(source_tx_ids):
        raise HSCFSplitExportError("HSCF receipt checkpoint TX/head order drifted")
    if tuple(int(item) for item in frozen.get("local_to_head_class_ids", [])) != tuple(_hscf.FROZEN_HSCF_CLASS_IDS):
        raise HSCFSplitExportError("HSCF receipt local-to-head class order drifted")
    for field in (
        "baseline_sha256",
        "initial_checkpoint_sha256",
        "class_order_binding_sha256",
        "source_partition_sha256",
        "source_labeled_indices_sha256",
        "source_split_manifest_sha256",
        "optimizer_initial_state_sha256",
        "common_batch_sequence_sha256",
    ):
        _require_sha256(frozen.get(field), field=f"HSCF receipt {field}")
    if frozen.get("source_labeled_provenance") != SOURCE_L_PROVENANCE:
        raise HSCFSplitExportError("HSCF source-L physical-order provenance drifted")
    if (
        frozen.get("optimizer_state_restored") is not False
        or frozen.get("rng_state_restored") is not False
        or frozen.get("optimizer_initial_state_empty") is not True
        or frozen.get("optimizer_type") != "AdamW"
    ):
        raise HSCFSplitExportError("HSCF receipt does not prove a new AdamW/RNG state")
    if frozen.get("common_l_base_head_input_path_verified") is not True:
        raise HSCFSplitExportError("HSCF common live L_base exact-head path is not verified")
    for field, expected in (
        ("fixed_batch_size", FROZEN_HSCF_BATCH_SIZE),
        ("fixed_local_class_count", FROZEN_HSCF_LOCAL_CLASS_COUNT),
        ("loss_global_denominator", FROZEN_HSCF_GLOBAL_DENOMINATOR),
    ):
        if type(frozen.get(field)) is not int or frozen.get(field) != expected:
            raise HSCFSplitExportError(f"HSCF receipt {field} drifted")
    if frozen.get("z_id_key") != "feat_joint" or frozen.get("head_input_path") != "model_output.tx_logits_from_id_backbone.cls_head.head(feat_joint)":
        raise HSCFSplitExportError("HSCF receipt feat_joint/exact-head path drifted")
    try:
        validated = _hscf.validate_hscf_terminal_receipt(frozen)
    except (_hscf.HSCFConfigurationError, _hscf.HSCFRuntimeError) as exc:
        raise HSCFSplitExportError(f"HSCF terminal receipt revalidation failed: {exc}") from exc
    if validated.get("hscf_terminal_contract_passed") is not True:
        raise HSCFSplitExportError("HSCF terminal receipt did not pass its raw contract")
    if expected_enabled:
        _validate_hscf_gradient_audits(validated)
    else:
        zero_fields = ("hscf_batches", "hscf_total_rows", "hscf_positive_batches", "hscf_positive_components")
        if any(int(validated.get(field, -1)) != 0 for field in zero_fields):
            raise HSCFSplitExportError("HSCF C receipt contains non-zero auxiliary evidence")
        if abs(float(validated.get("hscf_loss_sum", float("nan")))) > 1e-12:
            raise HSCFSplitExportError("HSCF C receipt contains non-zero auxiliary loss")
        if any(bool(validated.get(field)) for field in ("hscf_scenes", "hscf_g_batch_aux", "hscf_gradient_audit_scenes")) or validated.get("hscf_gradient_audit_completed") is not False:
            raise HSCFSplitExportError("HSCF C receipt must keep G-only fields N/A-or-zero")
    return dict(validated)


def validate_hscf_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Validate original checkpoint args and raw HSCF terminal receipt."""

    if str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise HSCFSplitExportError("checkpoint_role must be training_final_only")
    if str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise HSCFSplitExportError("checkpoint_selection must be final_only")
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise HSCFSplitExportError("checkpoint must contain model and args mappings")
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
            raise HSCFSplitExportError(f"checkpoint arg {field} drifted: expected={expected} observed={args.get(field)}")
    for field, expected in (("labeled_ratio", 0.07), ("unlabeled_ratio", 0.63), ("source_val_ratio", 0.30)):
        _require_close(f"checkpoint arg {field}", args.get(field), expected)
    if int(args.get("seed", -1)) != 7281105:
        raise HSCFSplitExportError("checkpoint seed must remain 7281105")
    candidate_match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if candidate_match is None:
        raise HSCFSplitExportError("checkpoint candidate is not a frozen F1..F6 C/G HSCF12 arm")
    arm = candidate_match.group(2)
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name or str(checkpoint.get("candidate_id", "")) != checkpoint_path.parent.name:
        raise HSCFSplitExportError("checkpoint candidate_id does not bind parent arm directory")
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID or str(checkpoint.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID:
        raise HSCFSplitExportError("checkpoint run_id does not bind frozen HSCF training root")
    _require_bool(args, "phase1_hscf_frozen_mode", True)
    expected_enabled = arm == "G"
    _require_bool(args, "phase1_hscf_enabled", expected_enabled)
    _require_close("checkpoint arg lambda_hscf", args.get("lambda_hscf"), FROZEN_HSCF_LAMBDA if expected_enabled else 0.0)
    try:
        _hscf.validate_hscf_args(SimpleNamespace(**dict(args)))
    except _hscf.HSCFConfigurationError as exc:
        raise HSCFSplitExportError(f"checkpoint HSCF frozen-arg contract drifted: {exc}") from exc
    receipt = validate_hscf_terminal_receipt(
        checkpoint.get("hscf_receipt", {}),
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
    """Compatibility adapter used only inside the signed source export kernel."""

    args, receipt, _ = validate_hscf_training_checkpoint(
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
    """Point signed source reconstruction at HSCF only for this process."""

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
            raise HSCFSplitExportError(f"{field} is frozen and may not be changed")
        setattr(result, field, expected)
    return result


def _atomic_rewrite_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["manifest_json"] = np.asarray(json.dumps(dict(manifest), ensure_ascii=True))
    temporary = path.with_name(path.name + ".hscf-manifest.tmp")
    if temporary.exists():
        raise HSCFSplitExportError(f"refusing to overwrite temporary export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def _drop_legacy_identity_fields(manifest: dict[str, Any]) -> None:
    for field in tuple(manifest):
        if str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES):
            manifest.pop(field)
    if any(str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES) for field in manifest):
        raise HSCFSplitExportError("legacy method identity leaked into HSCF manifest")


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Run signed source reconstruction with HSCF-only checkpoint identity."""

    frozen_args = _coerce_frozen_proxy_args(args)
    checkpoint_path = Path(frozen_args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise HSCFSplitExportError(f"missing final checkpoint: {checkpoint_path}")
    source_tx_ids = _parse_csv(frozen_args.source_tx_ids, field="source_tx_ids")
    known_tx_ids = _parse_csv(frozen_args.known_validation_tx_ids, field="known_validation_tx_ids")
    proxy_tx_ids = _parse_csv(frozen_args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise HSCFSplitExportError("P1-HSCF export requires local4 plus one held and one proxy TX")
    roles = (set(source_tx_ids), set(known_tx_ids), set(proxy_tx_ids))
    if any(roles[left] & roles[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise HSCFSplitExportError("source/known-validation/proxy TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise HSCFSplitExportError("checkpoint payload must be a mapping")
    _, receipt, arm = validate_hscf_training_checkpoint(
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
        raise HSCFSplitExportError(str(exc)) from exc
    manifest = dict(base_result["manifest"])
    _drop_legacy_identity_fields(manifest)
    manifest.update(
        {
            "schema": EXPECTED_LV_EXPORT_SCHEMA,
            "method": EXPECTED_RECEIPT_METHOD,
            "training_run_contract": EXPECTED_TRAINING_RUN_ID,
            "candidate_id": checkpoint_path.parent.name,
            "hscf_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
            "hscf_enabled": arm == "G",
            "hscf_source_partition_sha256": str(receipt["source_partition_sha256"]),
            "hscf_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "hscf_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "hscf_source_labeled_provenance": SOURCE_L_PROVENANCE,
            "hscf_receipt_sha256": _canonical_json_sha256(dict(checkpoint["hscf_receipt"])),
            "hscf_terminal_contract": str(receipt["hscf_terminal_contract"]),
            "hscf_terminal_contract_passed": True,
            "hscf_lambda": FROZEN_HSCF_LAMBDA if arm == "G" else 0.0,
            "hscf_fixed_batch_size": FROZEN_HSCF_BATCH_SIZE,
            "hscf_fixed_local_class_count": FROZEN_HSCF_LOCAL_CLASS_COUNT,
            "hscf_loss_global_denominator": FROZEN_HSCF_GLOBAL_DENOMINATOR,
            "hscf_common_physical_order_bound": True,
            "hscf_common_scene_cycle_bound": True,
            "hscf_raw_vjp_per_scene_required": True,
            "hscf_exact_head_weight_vjp_nonzero_required": True,
            "hscf_head_bias_aux_vjp_na_none_or_zero": True,
            "proxy_selection_frozen_not_cli_tunable": True,
        }
    )
    _drop_legacy_identity_fields(manifest)
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
