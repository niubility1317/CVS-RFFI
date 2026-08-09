#!/usr/bin/env python
"""Export the sealed L/V/proxy rows for frozen P1-CAGM postfreeze scoring.

This is deliberately a CAGM-specific facade over the signed ICMT-v2 source
split reconstruction.  It shares only the data-free export mechanics: all
training identity, receipt, arm, CAGM-loss and terminal checks below bind to
the original P1-CAGM ``training_final_only`` checkpoint before any feature is
forwarded.  U is hash-checked by the reconstructed split and is never given a
loader, a forward, or a persisted feature row.
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

from cvsrffi import phase1_cagm as _cagm
import export_phase1_icmt_features as _icmt


EXPECTED_TRAINING_RUN_ID = "phase1_cagm12_20260810_v2"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.cagm_lv_export.v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.cagm_receipt.v2"
EXPECTED_RECEIPT_METHOD = "P1_CAGM"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_CAGM12$")
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_CAGM_LAMBDA = 0.02
FROZEN_CAGM_DIVISOR = 10
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400


class CAGMSplitExportError(RuntimeError):
    """Raised when a CAGM final-only source export cannot prove its binding."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise CAGMSplitExportError(f"{field} must be non-empty and duplicate-free")
    return items


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise CAGMSplitExportError(f"{name} must be numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise CAGMSplitExportError(
            f"{name} drifted: expected={expected} observed={observed}"
        )


def _require_bool(mapping: Mapping[str, Any], field: str, expected: bool) -> None:
    value = mapping.get(field)
    if type(value) is not bool or value is not expected:
        raise CAGMSplitExportError(
            f"{field} drifted: expected literal {expected!r}, got {value!r}"
        )


def _require_sha256(value: Any, *, field: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{64}", str(value or "").lower()):
        raise CAGMSplitExportError(f"{field} must be a lowercase SHA256")


def _validate_cagm_gradient_audit(receipt: Mapping[str, Any]) -> None:
    audit = receipt.get("cagm_gradient_audit")
    if not isinstance(audit, Mapping):
        raise CAGMSplitExportError("CAGM G receipt lacks raw auxiliary VJP audit")
    if audit.get("raw_unscaled") is not True or audit.get("diagnostic_only") is not True:
        raise CAGMSplitExportError("CAGM VJP audit is not raw-unscaled diagnostic-only")
    encoder = audit.get("shared_encoder")
    head = audit.get("classifier_head")
    if not isinstance(encoder, Mapping) or not isinstance(head, Mapping):
        raise CAGMSplitExportError("CAGM VJP audit scope is malformed")
    try:
        encoder_count = float(encoder["parameter_count"])
        encoder_norm = float(encoder["norm"])
        head_count = float(head["parameter_count"])
        head_none = float(head["none_parameters"])
        head_zero = float(head["zero_parameters"])
        head_nonzero = float(head["nonzero_parameters"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CAGMSplitExportError("CAGM VJP audit lacks numeric evidence") from exc
    if (
        encoder_count <= 0.0
        or not math.isfinite(encoder_norm)
        or encoder_norm <= 0.0
        or head_count <= 0.0
        or not all(math.isfinite(value) and value >= 0.0 for value in (head_none, head_zero, head_nonzero))
        or head_none + head_zero != head_count
        or head_nonzero != 0.0
        or head.get("none_or_zero_expected") is not True
    ):
        raise CAGMSplitExportError("CAGM raw encoder/head VJP contract drifted")


def validate_cagm_terminal_receipt(
    receipt: Mapping[str, Any],
    *,
    arm: str,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Re-run the CAGM terminal validator and inspect arm-specific raw evidence."""

    if not isinstance(receipt, Mapping):
        raise CAGMSplitExportError("checkpoint lacks a CAGM terminal receipt")
    frozen = dict(receipt)
    if str(frozen.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA:
        raise CAGMSplitExportError("CAGM terminal receipt schema drifted")
    if str(frozen.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise CAGMSplitExportError("CAGM terminal receipt method drifted")
    if frozen.get("frozen_mode") is not True:
        raise CAGMSplitExportError("CAGM terminal receipt is not frozen")
    expected_enabled = arm == "G"
    if frozen.get("enabled") is not expected_enabled:
        raise CAGMSplitExportError("CAGM receipt enabled flag does not bind arm")
    _require_close(
        "CAGM receipt lambda",
        frozen.get("lambda"),
        FROZEN_CAGM_LAMBDA if expected_enabled else 0.0,
    )
    if int(frozen.get("loss_divisor", -1)) != FROZEN_CAGM_DIVISOR:
        raise CAGMSplitExportError("CAGM receipt loss divisor must remain 10")
    if frozen.get("clean_statistics_detached") is not True:
        raise CAGMSplitExportError("CAGM receipt does not detach clean statistics")
    _require_bool(frozen, "joint_zero_mask_aux_only", expected_enabled)
    if type(frozen.get("optimizer_type")) is not str or frozen.get("optimizer_type") != "AdamW":
        raise CAGMSplitExportError("CAGM receipt optimizer_type must be literal AdamW")
    if str(frozen.get("z_id_key", "")) != "feat_joint":
        raise CAGMSplitExportError("CAGM receipt z_id key is not feat_joint")
    if str(frozen.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise CAGMSplitExportError("CAGM receipt checkpoint role is not training_final_only")
    if tuple(str(item) for item in frozen.get("source_train_tx", [])) != tuple(source_tx_ids):
        raise CAGMSplitExportError("CAGM receipt source train TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_known_validation_tx", [])) != tuple(
        known_validation_tx_ids
    ):
        raise CAGMSplitExportError("CAGM receipt known-validation TX binding drifted")
    if tuple(str(item) for item in frozen.get("source_proxy_unknown_tx", [])) != tuple(
        proxy_unknown_tx_ids
    ):
        raise CAGMSplitExportError("CAGM receipt proxy-unknown TX binding drifted")
    if tuple(str(item) for item in frozen.get("local_tx_class_order", [])) != tuple(source_tx_ids):
        raise CAGMSplitExportError("CAGM receipt local TX/head order drifted")
    if tuple(int(item) for item in frozen.get("local_to_head_class_ids", [])) != (0, 1, 2, 3):
        raise CAGMSplitExportError("CAGM receipt local-to-head class order drifted")
    for field in (
        "baseline_sha256",
        "initial_checkpoint_sha256",
        "class_order_binding_sha256",
        "source_labeled_indices_sha256",
        "source_split_manifest_sha256",
        "optimizer_initial_state_sha256",
        "common_batch_sequence_sha256",
    ):
        _require_sha256(frozen.get(field), field=f"CAGM receipt {field}")
    if (
        frozen.get("optimizer_state_restored") is not False
        or frozen.get("rng_state_restored") is not False
        or frozen.get("optimizer_initial_state_empty") is not True
    ):
        raise CAGMSplitExportError("CAGM receipt does not prove a new AdamW/RNG state")
    try:
        validated = _cagm.validate_cagm_terminal_receipt(frozen)
    except (_cagm.CAGMConfigurationError, _cagm.CAGMRuntimeError) as exc:
        raise CAGMSplitExportError(f"CAGM terminal receipt revalidation failed: {exc}") from exc
    if validated.get("cagm_terminal_contract_passed") is not True:
        raise CAGMSplitExportError("CAGM terminal receipt did not pass its raw contract")
    _require_bool(validated, "joint_zero_mask_aux_only", expected_enabled)
    if expected_enabled:
        if validated.get("cagm_gradient_audit_completed") is not True:
            raise CAGMSplitExportError("CAGM G receipt lacks completed VJP audit")
        _validate_cagm_gradient_audit(validated)
        # The terminal validator independently closes all three scenes and 4+6
        # terms.  Require their raw maps to be present before accepting it.
        if set(dict(validated.get("cagm_scenes", {}))) != set(_cagm.FROZEN_CAGM_SCENARIOS):
            raise CAGMSplitExportError("CAGM receipt lacks three-scene raw evidence")
        if len(dict(validated.get("cagm_radius_terms", {}))) != 4 or len(
            dict(validated.get("cagm_gram_terms", {}))
        ) != 6:
            raise CAGMSplitExportError("CAGM receipt lacks 4-radius/6-Gram coverage")
    else:
        zero_fields = (
            "cagm_batches",
            "cagm_total_rows",
            "cagm_valid_rows",
            "cagm_clean_zero_rows",
            "cagm_leo_zero_rows",
            "cagm_union_zero_rows",
            "cagm_both_zero_rows",
        )
        if any(int(validated.get(field, -1)) != 0 for field in zero_fields):
            raise CAGMSplitExportError("CAGM C receipt contains non-zero auxiliary evidence")
        if any(dict(validated.get(field, {})) for field in ("cagm_scenes", "cagm_radius_terms", "cagm_gram_terms")):
            raise CAGMSplitExportError("CAGM C receipt retains G-only term maps")
        if validated.get("cagm_gradient_audit_completed") is not False:
            raise CAGMSplitExportError("CAGM C receipt must keep VJP audit N/A")
    return dict(validated)


def validate_cagm_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> tuple[Mapping[str, Any], dict[str, Any], str]:
    """Validate the original checkpoint, args and terminal receipt, not a summary."""

    if str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise CAGMSplitExportError("checkpoint_role must be training_final_only")
    if str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise CAGMSplitExportError("checkpoint_selection must be final_only")
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise CAGMSplitExportError("checkpoint must contain model and args mappings")
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
            raise CAGMSplitExportError(
                f"checkpoint arg {field} drifted: expected={expected} observed={args.get(field)}"
            )
    for field, expected in (("labeled_ratio", 0.07), ("unlabeled_ratio", 0.63), ("source_val_ratio", 0.30)):
        _require_close(f"checkpoint arg {field}", args.get(field), expected)
    if int(args.get("seed", -1)) != 7281105:
        raise CAGMSplitExportError("checkpoint seed must remain 7281105")
    _require_bool(args, "phase1_cagm_frozen_mode", True)
    candidate_match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if candidate_match is None:
        raise CAGMSplitExportError("checkpoint candidate is not a frozen F1..F6 C/G CAGM12 arm")
    arm = candidate_match.group(2)
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name or str(
        checkpoint.get("candidate_id", "")
    ) != checkpoint_path.parent.name:
        raise CAGMSplitExportError("checkpoint candidate_id does not bind parent arm directory")
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID or str(
        checkpoint.get("run_id", "")
    ) != EXPECTED_TRAINING_RUN_ID:
        raise CAGMSplitExportError("checkpoint run_id does not bind frozen CAGM training root")
    expected_enabled = arm == "G"
    _require_bool(args, "phase1_cagm_enabled", expected_enabled)
    _require_close(
        "checkpoint arg lambda_cagm",
        args.get("lambda_cagm"),
        FROZEN_CAGM_LAMBDA if expected_enabled else 0.0,
    )
    receipt = validate_cagm_terminal_receipt(
        checkpoint.get("cagm_receipt", {}),
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
    """Adapter consumed only inside the signed ICMT split-export mechanics."""

    args, receipt, _ = validate_cagm_training_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    # This is an in-memory compatibility alias only.  The source checkpoint and
    # emitted CAGM manifest retain their genuine CAGM names after export.
    if isinstance(checkpoint, dict):
        checkpoint["icmt_receipt"] = dict(receipt)
    return args


@contextmanager
def _patched_signed_export() -> Iterator[None]:
    """Temporarily point the signed generic split machinery at CAGM identities."""

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
            raise CAGMSplitExportError(f"{field} is frozen and may not be changed")
        setattr(result, field, expected)
    return result


def _atomic_rewrite_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    with np.load(path, allow_pickle=False) as data:
        payload = {name: np.asarray(data[name]) for name in data.files}
    payload["manifest_json"] = np.asarray(json.dumps(dict(manifest), ensure_ascii=True))
    temporary = path.with_name(path.name + ".cagm-manifest.tmp")
    if temporary.exists():
        raise CAGMSplitExportError(f"refusing to overwrite temporary export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **payload)
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Run the signed split reconstruction with CAGM-only checkpoint identity."""

    frozen_args = _coerce_frozen_proxy_args(args)
    checkpoint_path = Path(frozen_args.ckpt).resolve()
    if not checkpoint_path.is_file():
        raise CAGMSplitExportError(f"missing final checkpoint: {checkpoint_path}")
    source_tx_ids = _parse_csv(frozen_args.source_tx_ids, field="source_tx_ids")
    known_tx_ids = _parse_csv(
        frozen_args.known_validation_tx_ids, field="known_validation_tx_ids"
    )
    proxy_tx_ids = _parse_csv(frozen_args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise CAGMSplitExportError("P1-CAGM export requires local4 plus one held and one proxy TX")
    roles = (set(source_tx_ids), set(known_tx_ids), set(proxy_tx_ids))
    if any(roles[left] & roles[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise CAGMSplitExportError("source/known-validation/proxy TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise CAGMSplitExportError("checkpoint payload must be a mapping")
    _, receipt, arm = validate_cagm_training_checkpoint(
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
            "cagm_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
            "cagm_enabled": arm == "G",
            "cagm_source_labeled_indices_sha256": str(
                receipt["source_labeled_indices_sha256"]
            ),
            "cagm_source_split_manifest_sha256": str(
                receipt["source_split_manifest_sha256"]
            ),
            "cagm_receipt_sha256": _canonical_json_sha256(dict(checkpoint["cagm_receipt"])),
            "cagm_terminal_contract": str(receipt["cagm_terminal_contract"]),
            "cagm_terminal_contract_passed": True,
            "cagm_loss_divisor": FROZEN_CAGM_DIVISOR,
            "cagm_clean_statistics_detached": True,
            "cagm_joint_zero_mask_aux_only": receipt[
                "joint_zero_mask_aux_only"
            ],
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
