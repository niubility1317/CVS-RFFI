#!/usr/bin/env python
"""Export sealed L/V/proxy features for frozen P1-RECTE postfreeze scoring.

The signed ICMT-v2 exporter is used only as a compatibility implementation of
the frozen source split and feature forward.  This facade owns all RECTE
identity and revalidates the raw RECTE terminal receipt before any source row
is exported.  It never iterates, forwards, persists or fits U; it preserves
every L/V/proxy row and freezes the source-only 400-row proxy selection.
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

from cvsrffi import phase1_recte as _recte
import export_phase1_icmt_features as _icmt


EXPECTED_TRAINING_RUN_ID = "phase1_recte12_20260810_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.recte_lv_export.v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.recte_receipt.v1"
EXPECTED_RECEIPT_METHOD = "P1_RECTE"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_RECTE12$")
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_RECTE_LAMBDA = 0.02
FROZEN_SOURCE_RECEIVER_IDS = tuple(int(value) for value in _recte.FROZEN_RECTE_SOURCE_RECEIVER_IDS)
FROZEN_SOURCE_RECEIVER_COUNT = len(FROZEN_SOURCE_RECEIVER_IDS)
FROZEN_CELLS_PER_SCENE = int(_recte.FROZEN_RECTE_CELL_COUNT)
FROZEN_UNORDERED_PAIR_COUNT = int(_recte.FROZEN_RECTE_PAIR_DENOMINATOR)
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400
SOURCE_RECEIVER_PROVENANCE = "SOURCE_SPLIT_RECEIPT_source_receivers_PHYSICAL_ID_BOUND_L_ONLY"
LEGACY_IDENTITY_PREFIXES = ("icmt_", "rcat_", "rcrmd_", "cagm_")


class RECTESplitExportError(RuntimeError):
    """Raised when a RECTE final-only source export cannot prove its binding."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise RECTESplitExportError(f"{field} must be non-empty and duplicate-free")
    return items


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise RECTESplitExportError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise RECTESplitExportError(
            f"{name} drifted: expected={expected} observed={observed}"
        )


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    value = mapping.get(field)
    if type(value) is not bool or value is not expected:
        raise RECTESplitExportError(
            f"{field} drifted: expected literal {expected!r}, got {value!r}"
        )


def _require_sha256(value: Any, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
        raise RECTESplitExportError(f"{field} must be a lowercase SHA256")


def _validate_source_receiver_binding(receipt: Mapping[str, Any]) -> None:
    for field, expected in (
        ("frozen_source_receiver_ids", list(FROZEN_SOURCE_RECEIVER_IDS)),
        ("source_receiver_ids", list(FROZEN_SOURCE_RECEIVER_IDS)),
    ):
        value = receipt.get(field)
        if type(value) is not list or tuple(value) != tuple(expected) or any(
            type(item) is not int for item in value
        ):
            raise RECTESplitExportError(f"RECTE receipt {field} drifted")
    for field, expected in (
        ("frozen_source_receiver_count", FROZEN_SOURCE_RECEIVER_COUNT),
        ("source_receiver_count", FROZEN_SOURCE_RECEIVER_COUNT),
        ("frozen_cells_per_scene", FROZEN_CELLS_PER_SCENE),
        ("loss_global_denominator", FROZEN_UNORDERED_PAIR_COUNT),
        ("fixed_unordered_pair_count", FROZEN_UNORDERED_PAIR_COUNT),
    ):
        if type(receipt.get(field)) is not int or receipt.get(field) != expected:
            raise RECTESplitExportError(f"RECTE receipt {field} drifted")
    if receipt.get("source_receiver_provenance") != SOURCE_RECEIVER_PROVENANCE:
        raise RECTESplitExportError("RECTE source receiver provenance drifted")
    expected_sha = _canonical_json_sha256(list(FROZEN_SOURCE_RECEIVER_IDS))
    if receipt.get("source_receiver_ids_sha256") != expected_sha:
        raise RECTESplitExportError("RECTE source receiver SHA256 drifted")


def _validate_recte_gradient_audits(receipt: Mapping[str, Any]) -> None:
    if receipt.get("recte_gradient_audit_completed") is not True:
        raise RECTESplitExportError("RECTE G receipt lacks completed per-scene VJP audits")
    audits = receipt.get("recte_gradient_audit_scenes")
    if not isinstance(audits, Mapping) or set(audits) != set(_recte.FROZEN_RECTE_SCENARIOS):
        raise RECTESplitExportError("RECTE G receipt lacks clear/low/rain VJP audits")
    scenes = receipt.get("recte_scenes")
    if not isinstance(scenes, Mapping) or set(scenes) != set(_recte.FROZEN_RECTE_SCENARIOS):
        raise RECTESplitExportError("RECTE G receipt lacks clear/low/rain tail coverage")
    for scenario in _recte.FROZEN_RECTE_SCENARIOS:
        scene = scenes[scenario]
        audit = audits[scenario]
        if not isinstance(scene, Mapping) or int(scene.get("positive_tail_pair_count", 0)) <= 0:
            raise RECTESplitExportError(f"RECTE {scenario} lacks a positive-tail pair")
        if not isinstance(audit, Mapping) or (
            audit.get("raw_unscaled") is not True
            or audit.get("diagnostic_only") is not True
            or audit.get("touches_amp_optimizer_rng") is not False
            or audit.get("exact_head_aux_vjp") != "N_A_NONE_OR_ZERO_EXPECTED"
        ):
            raise RECTESplitExportError(f"RECTE {scenario} VJP audit semantics drifted")
        for group in ("feat_joint_leo", "shared_encoder"):
            evidence = audit.get(group)
            if not isinstance(evidence, Mapping):
                raise RECTESplitExportError(f"RECTE {scenario} VJP lacks {group}")
            try:
                count = float(evidence["parameter_count"])
                norm = float(evidence["norm"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RECTESplitExportError(f"RECTE {scenario} VJP {group} is malformed") from exc
            if count <= 0.0 or not math.isfinite(norm) or norm <= 0.0:
                raise RECTESplitExportError(f"RECTE {scenario} VJP {group} is zero/nonfinite")
        head = audit.get("classifier_head")
        if not isinstance(head, Mapping):
            raise RECTESplitExportError(f"RECTE {scenario} VJP lacks classifier-head evidence")
        try:
            count = float(head["parameter_count"])
            none_count = float(head["none_parameters"])
            zero_count = float(head["zero_parameters"])
            nonzero_count = float(head["nonzero_parameters"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RECTESplitExportError(f"RECTE {scenario} classifier-head VJP is malformed") from exc
        if (
            count <= 0.0
            or not all(math.isfinite(value) and value >= 0.0 for value in (none_count, zero_count, nonzero_count))
            or none_count + zero_count != count
            or nonzero_count != 0.0
            or head.get("none_or_zero_expected") is not True
        ):
            raise RECTESplitExportError(
                f"RECTE {scenario} classifier-head auxiliary VJP must be None/zero"
            )


def validate_recte_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Revalidate raw RECTE terminal evidence before sealed export."""

    if not isinstance(receipt, Mapping):
        raise RECTESplitExportError("checkpoint lacks a RECTE terminal receipt")
    frozen = dict(receipt)
    if str(frozen.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA:
        raise RECTESplitExportError("RECTE terminal receipt schema drifted")
    if str(frozen.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise RECTESplitExportError("RECTE terminal receipt method drifted")
    _require_bool(frozen, "frozen_mode", True)
    expected_enabled = arm == "G"
    _require_bool(frozen, "enabled", expected_enabled)
    _require_close(
        "RECTE receipt lambda",
        frozen.get("lambda"),
        FROZEN_RECTE_LAMBDA if expected_enabled else 0.0,
    )
    _validate_source_receiver_binding(frozen)
    if str(frozen.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise RECTESplitExportError("RECTE receipt checkpoint role is not training_final_only")
    if tuple(str(item) for item in frozen.get("source_train_tx", [])) != tuple(source_tx_ids):
        raise RECTESplitExportError("RECTE receipt source train TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_known_validation_tx", [])) != tuple(known_validation_tx_ids):
        raise RECTESplitExportError("RECTE receipt known-validation TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_proxy_unknown_tx", [])) != tuple(proxy_unknown_tx_ids):
        raise RECTESplitExportError("RECTE receipt proxy-unknown TX binding drifted")
    if tuple(str(item) for item in frozen.get("local_tx_class_order", [])) != tuple(source_tx_ids):
        raise RECTESplitExportError("RECTE receipt local TX/head order drifted")
    if tuple(str(item) for item in frozen.get("checkpoint_train_tx_class_order", [])) != tuple(source_tx_ids):
        raise RECTESplitExportError("RECTE receipt checkpoint TX/head order drifted")
    if tuple(int(item) for item in frozen.get("local_to_head_class_ids", [])) != (0, 1, 2, 3):
        raise RECTESplitExportError("RECTE receipt local-to-head class order drifted")
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
        _require_sha256(frozen.get(field), field=f"RECTE receipt {field}")
    if (
        frozen.get("optimizer_state_restored") is not False
        or frozen.get("rng_state_restored") is not False
        or frozen.get("optimizer_initial_state_empty") is not True
        or frozen.get("optimizer_type") != "AdamW"
    ):
        raise RECTESplitExportError("RECTE receipt does not prove a new AdamW/RNG state")
    if frozen.get("common_l_base_head_input_path_verified") is not True:
        raise RECTESplitExportError("RECTE common live L_base exact-head path is not verified")
    try:
        validated = _recte.validate_recte_terminal_receipt(frozen)
    except (_recte.RECTEConfigurationError, _recte.RECTERuntimeError) as exc:
        raise RECTESplitExportError(f"RECTE terminal receipt revalidation failed: {exc}") from exc
    if validated.get("recte_terminal_contract_passed") is not True:
        raise RECTESplitExportError("RECTE terminal receipt did not pass its raw contract")
    _validate_source_receiver_binding(validated)
    common_cells = validated.get("recte_common_cells")
    if (
        not isinstance(common_cells, Mapping)
        or set(common_cells) != set(_recte.FROZEN_RECTE_SCENARIOS)
        or any(
            not isinstance(common_cells[scene], Mapping)
            or len(common_cells[scene]) != FROZEN_CELLS_PER_SCENE
            for scene in _recte.FROZEN_RECTE_SCENARIOS
        )
    ):
        raise RECTESplitExportError("RECTE common 28-cell/three-scene receipt drifted")
    if expected_enabled:
        _validate_recte_gradient_audits(validated)
    else:
        zero_fields = (
            "recte_batches",
            "recte_total_rows",
            "recte_occupied_unordered_pair_count",
            "recte_positive_tail_pair_count",
            "recte_functional_head_readout_count",
        )
        if any(int(validated.get(field, -1)) != 0 for field in zero_fields):
            raise RECTESplitExportError("RECTE C receipt contains non-zero auxiliary evidence")
        if abs(float(validated.get("recte_loss_sum", float("nan")))) > 1e-12:
            raise RECTESplitExportError("RECTE C receipt contains non-zero auxiliary loss")
        if any(
            bool(validated.get(field))
            for field in (
                "recte_scenes",
                "recte_g_batch_aux",
                "recte_gradient_audit_scenes",
            )
        ) or validated.get("recte_gradient_audit_completed") is not False:
            raise RECTESplitExportError("RECTE C receipt must keep G-only fields N/A-or-zero")
    return dict(validated)


def validate_recte_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Validate the original checkpoint, args and raw RECTE terminal receipt."""

    if str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise RECTESplitExportError("checkpoint_role must be training_final_only")
    if str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise RECTESplitExportError("checkpoint_selection must be final_only")
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise RECTESplitExportError("checkpoint must contain model and args mappings")
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
            raise RECTESplitExportError(
                f"checkpoint arg {field} drifted: expected={expected} observed={args.get(field)}"
            )
    for field, expected in (
        ("labeled_ratio", 0.07),
        ("unlabeled_ratio", 0.63),
        ("source_val_ratio", 0.30),
    ):
        _require_close(f"checkpoint arg {field}", args.get(field), expected)
    if int(args.get("seed", -1)) != 7281105:
        raise RECTESplitExportError("checkpoint seed must remain 7281105")
    _require_bool(args, "phase1_recte_frozen_mode", True)
    candidate_match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if candidate_match is None:
        raise RECTESplitExportError("checkpoint candidate is not a frozen F1..F6 C/G RECTE12 arm")
    arm = candidate_match.group(2)
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name or str(
        checkpoint.get("candidate_id", "")
    ) != checkpoint_path.parent.name:
        raise RECTESplitExportError("checkpoint candidate_id does not bind parent arm directory")
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID or str(
        checkpoint.get("run_id", "")
    ) != EXPECTED_TRAINING_RUN_ID:
        raise RECTESplitExportError("checkpoint run_id does not bind frozen RECTE training root")
    expected_enabled = arm == "G"
    _require_bool(args, "phase1_recte_enabled", expected_enabled)
    _require_close(
        "checkpoint arg lambda_recte",
        args.get("lambda_recte"),
        FROZEN_RECTE_LAMBDA if expected_enabled else 0.0,
    )
    receipt = validate_recte_terminal_receipt(
        checkpoint.get("recte_receipt", {}),
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
    """Compatibility adapter used only inside the signed frozen export kernel."""

    args, receipt, _ = validate_recte_training_checkpoint(
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
    """Bind signed source reconstruction to RECTE only for this process."""

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
            raise RECTESplitExportError(f"{field} is frozen and may not be changed")
        setattr(result, field, expected)
    return result


def _atomic_rewrite_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["manifest_json"] = np.asarray(json.dumps(dict(manifest), ensure_ascii=True))
    temporary = path.with_name(path.name + ".recte-manifest.tmp")
    if temporary.exists():
        raise RECTESplitExportError(f"refusing to overwrite temporary export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def _drop_legacy_identity_fields(manifest: dict[str, Any]) -> None:
    for field in tuple(manifest):
        if str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES):
            manifest.pop(field)
    leaked = [
        str(field)
        for field in manifest
        if str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES)
    ]
    if leaked:
        raise RECTESplitExportError("legacy method identity leaked into RECTE manifest")


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Run signed source reconstruction with a RECTE-only checkpoint identity."""

    frozen_args = _coerce_frozen_proxy_args(args)
    checkpoint_path = Path(frozen_args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise RECTESplitExportError(f"missing final checkpoint: {checkpoint_path}")
    source_tx_ids = _parse_csv(frozen_args.source_tx_ids, field="source_tx_ids")
    known_tx_ids = _parse_csv(
        frozen_args.known_validation_tx_ids, field="known_validation_tx_ids"
    )
    proxy_tx_ids = _parse_csv(frozen_args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise RECTESplitExportError("P1-RECTE export requires local4 plus one held and one proxy TX")
    roles = (set(source_tx_ids), set(known_tx_ids), set(proxy_tx_ids))
    if any(roles[left] & roles[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise RECTESplitExportError("source/known-validation/proxy TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RECTESplitExportError("checkpoint payload must be a mapping")
    _, receipt, arm = validate_recte_training_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_tx_ids,
        proxy_unknown_tx_ids=proxy_tx_ids,
    )
    with _patched_signed_export():
        base_result = _icmt.export(frozen_args)
    manifest = dict(base_result["manifest"])
    _drop_legacy_identity_fields(manifest)
    manifest.update(
        {
            "schema": EXPECTED_LV_EXPORT_SCHEMA,
            "method": EXPECTED_RECEIPT_METHOD,
            "training_run_contract": EXPECTED_TRAINING_RUN_ID,
            "candidate_id": checkpoint_path.parent.name,
            "recte_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
            "recte_enabled": arm == "G",
            "recte_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "recte_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "recte_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
            "recte_source_receiver_ids": list(FROZEN_SOURCE_RECEIVER_IDS),
            "recte_source_receiver_count": FROZEN_SOURCE_RECEIVER_COUNT,
            "recte_source_receiver_provenance": SOURCE_RECEIVER_PROVENANCE,
            "recte_frozen_cells_per_scene": FROZEN_CELLS_PER_SCENE,
            "recte_fixed_unordered_pair_count": FROZEN_UNORDERED_PAIR_COUNT,
            "recte_receipt_sha256": _canonical_json_sha256(dict(checkpoint["recte_receipt"])),
            "recte_terminal_contract": str(receipt["recte_terminal_contract"]),
            "recte_terminal_contract_passed": True,
            "recte_lambda": FROZEN_RECTE_LAMBDA if arm == "G" else 0.0,
            "recte_loss_global_denominator": FROZEN_UNORDERED_PAIR_COUNT,
            "recte_common_physical_rx_class_scene_nrc_bound": True,
            "recte_batch_order_bound": True,
            "recte_functional_logits_live_equality_required": True,
            "recte_exact_head_aux_vjp_na_none_or_zero": True,
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
