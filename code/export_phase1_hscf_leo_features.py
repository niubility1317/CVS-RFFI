#!/usr/bin/env python
"""Export and bind the frozen source-only LEO slice for P1-HSCF.

The signed generic forwarder is used only for stable source reconstruction.
This facade owns HSCF identity, reopens its raw receipt, and seals current
ManySig, physical-key and per-scenario TX/RX/day evidence.  It never touches
target/held/proxy fitting or creates a second LEO observation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

import export_phase1_hscf_features as _hscf_export
import export_phase1_icmt_leo_features as _icmt_leo


EXPECTED_TRAINING_RUN_LEAF = "phase1_hscf12_20260811_v2"
EXPECTED_BINDING_SCHEMA = "cvs.phase1.hscf_leo_binding.v1"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
FROZEN_WISIG_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
EXPECTED_SOURCE_DAYS = ("2021_03_01", "2021_03_08")
EXPECTED_SOURCE_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
EXPECTED_SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
EXPECTED_EQUALIZED = "1"
EXPECTED_DOMAIN = "rx_day"
EXPECTED_OUT_LEN = 256
EXPECTED_MAX_PER_COMBO = 0
EXPECTED_MAX_PER_TX = 400
EXPECTED_EXPORT_SEED = 7281105
EXPECTED_SOURCE_SAT_SEED = 7281718
EXPECTED_BATCH_SIZE = 32
EXPECTED_CHANNEL_VIEW = "satellite"
EXPECTED_TTA_POLICY = "none"
EXPECTED_STAR_GROUND_IMPL = "simplified_leo_residual"
LEGACY_IDENTITY_PREFIXES = ("icmt_", "rcat_", "rcrmd_", "cagm_", "recte_")


class HSCFLEOBindingError(RuntimeError):
    """Raised when an HSCF LEO export cannot prove its frozen source binding."""


def _translate(error: BaseException) -> HSCFLEOBindingError:
    return HSCFLEOBindingError(str(error))


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    try:
        return _icmt_leo._parse_csv(value, field=field)
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise _translate(exc) from exc


def _drop_legacy_identity_fields(binding: dict[str, Any]) -> None:
    for field in tuple(binding):
        if str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES):
            binding.pop(field)
    if any(str(field).lower().startswith(LEGACY_IDENTITY_PREFIXES) for field in binding):
        raise HSCFLEOBindingError("legacy method identity leaked into HSCF LEO binding")


def _validate_frozen_args(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.fold_index)
    arm = str(args.arm).upper()
    if fold not in range(1, 7) or arm not in {"C", "G"}:
        raise HSCFLEOBindingError("fold/arm must be F1..F6 and C/G")
    candidate = f"F{fold}{arm}_HSCF12"
    if str(args.candidate_id) != candidate:
        raise HSCFLEOBindingError(f"candidate_id must be {candidate}")
    training_root = Path(args.training_run_root).resolve()
    if training_root.name != EXPECTED_TRAINING_RUN_LEAF or not training_root.is_dir():
        raise HSCFLEOBindingError(f"training root must be existing {EXPECTED_TRAINING_RUN_LEAF}")
    postfreeze_root = Path(args.postfreeze_output_root).resolve()
    if training_root == postfreeze_root or not postfreeze_root.is_dir():
        raise HSCFLEOBindingError("postfreeze root must exist and differ from training root")
    checkpoint = Path(args.ckpt).resolve()
    expected_checkpoint = (training_root / candidate / "final_ssdg.pth").resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise HSCFLEOBindingError("checkpoint path does not bind frozen HSCF candidate")
    candidate_dir = (postfreeze_root / candidate).resolve()
    try:
        out_npz = _icmt_leo._require_under_root(args.out_npz, candidate_dir, label="LEO NPZ")
        binding_json = _icmt_leo._require_under_root(args.binding_json, candidate_dir, label="LEO binding JSON")
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise _translate(exc) from exc
    if out_npz != (candidate_dir / "source_leo_final_only.npz").resolve():
        raise HSCFLEOBindingError("LEO NPZ path does not match frozen candidate layout")
    if binding_json != (candidate_dir / "source_leo_binding.json").resolve():
        raise HSCFLEOBindingError("LEO binding path does not match frozen candidate layout")
    source_tx_ids = _parse_csv(args.source_tx_ids, field="source_tx_ids")
    if len(source_tx_ids) != 4:
        raise HSCFLEOBindingError("P1-HSCF LEO binding requires local4 source TX")
    for field, expected in (
        ("source_days", EXPECTED_SOURCE_DAYS),
        ("source_rxs", EXPECTED_SOURCE_RXS),
        ("source_sat_scenarios", EXPECTED_SCENARIOS),
    ):
        try:
            observed = _icmt_leo._require_exact_tuple(getattr(args, field), expected, field=field)
        except _icmt_leo.ICMTLEOBindingError as exc:
            raise _translate(exc) from exc
        if observed != expected:
            raise HSCFLEOBindingError(f"{field} drifted")
    for field, expected in (
        ("source_sat_seed", EXPECTED_SOURCE_SAT_SEED),
        ("seed", EXPECTED_EXPORT_SEED),
        ("max_samples_per_tx", EXPECTED_MAX_PER_TX),
        ("batch_size", EXPECTED_BATCH_SIZE),
    ):
        if int(getattr(args, field)) != expected:
            raise HSCFLEOBindingError(f"{field} must equal frozen value {expected}")
    for field, expected in (
        ("feature_name", "z_id"),
        ("source_channel_view", EXPECTED_CHANNEL_VIEW),
        ("satellite_tta_policy", EXPECTED_TTA_POLICY),
        ("star_ground_channel_impl", EXPECTED_STAR_GROUND_IMPL),
        ("wisig_equalized", EXPECTED_EQUALIZED),
        ("wisig_domain", EXPECTED_DOMAIN),
    ):
        if str(getattr(args, field)) != expected:
            raise HSCFLEOBindingError(f"{field} must equal frozen value {expected}")
    if args.source_only_export is not True:
        raise HSCFLEOBindingError("P1-HSCF LEO export must be source-only")
    if int(args.wisig_out_len) != EXPECTED_OUT_LEN or int(args.max_samples_per_combo) != EXPECTED_MAX_PER_COMBO:
        raise HSCFLEOBindingError("frozen LEO shape or combo selection drifted")
    dataset = Path(args.wisig_pkl).resolve()
    if not dataset.is_file():
        raise HSCFLEOBindingError(f"missing frozen ManySig dataset: {dataset}")
    if str(args.expected_wisig_sha256).lower() != FROZEN_WISIG_SHA256:
        raise HSCFLEOBindingError("expected WiSig SHA256 does not equal frozen value")
    dataset_sha256 = _icmt_leo._sha256_file(dataset)
    if dataset_sha256 != FROZEN_WISIG_SHA256:
        raise HSCFLEOBindingError("ManySig input bytes do not match frozen SHA256")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise HSCFLEOBindingError("checkpoint payload must be a mapping")
    raw_receipt = checkpoint_payload.get("hscf_receipt", {})
    if not isinstance(raw_receipt, Mapping):
        raise HSCFLEOBindingError("checkpoint lacks raw HSCF receipt")
    _, receipt, observed_arm = _hscf_export.validate_hscf_training_checkpoint(
        checkpoint_payload,
        checkpoint_path=checkpoint,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=(
            str(getattr(args, "known_validation_tx_id", "")),
        )
        if str(getattr(args, "known_validation_tx_id", ""))
        else tuple(str(item) for item in raw_receipt.get("source_known_validation_tx", [])),
        proxy_unknown_tx_ids=(
            str(getattr(args, "proxy_unknown_tx_id", "")),
        )
        if str(getattr(args, "proxy_unknown_tx_id", ""))
        else tuple(str(item) for item in raw_receipt.get("source_proxy_unknown_tx", [])),
    )
    if observed_arm != arm:
        raise HSCFLEOBindingError("HSCF checkpoint receipt arm does not bind LEO arm")
    return {
        "fold": fold,
        "arm": arm,
        "candidate": candidate,
        "training_root": training_root,
        "postfreeze_root": postfreeze_root,
        "checkpoint": checkpoint,
        "candidate_dir": candidate_dir,
        "out_npz": out_npz,
        "binding_json": binding_json,
        "source_tx_ids": source_tx_ids,
        "dataset": dataset,
        "dataset_sha256": dataset_sha256,
        "hscf_receipt": receipt,
        "hscf_receipt_sha256": _hscf_export._canonical_json_sha256(dict(raw_receipt)),
    }


@contextmanager
def _patched_signed_binding() -> Iterator[None]:
    """Temporarily point signed LEO mechanics at HSCF identity."""

    saved = {
        "EXPECTED_TRAINING_RUN_LEAF": _icmt_leo.EXPECTED_TRAINING_RUN_LEAF,
        "EXPECTED_BINDING_SCHEMA": _icmt_leo.EXPECTED_BINDING_SCHEMA,
        "_validate_frozen_args": _icmt_leo._validate_frozen_args,
    }
    _icmt_leo.EXPECTED_TRAINING_RUN_LEAF = EXPECTED_TRAINING_RUN_LEAF
    _icmt_leo.EXPECTED_BINDING_SCHEMA = EXPECTED_BINDING_SCHEMA
    _icmt_leo._validate_frozen_args = _validate_frozen_args
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(_icmt_leo, name, value)


def build_binding_from_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Bind an existing LEO NPZ and attach raw HSCF terminal evidence."""

    try:
        with _patched_signed_binding():
            binding = _icmt_leo.build_binding_from_existing(args)
    except (_icmt_leo.ICMTLEOBindingError, _hscf_export.HSCFSplitExportError) as exc:
        raise _translate(exc) from exc
    frozen = _validate_frozen_args(args)
    receipt = frozen["hscf_receipt"]
    result = dict(binding)
    _drop_legacy_identity_fields(result)
    result.update(
        {
            "schema": EXPECTED_BINDING_SCHEMA,
            "method": "P1_HSCF",
            "hscf_receipt_schema": _hscf_export.EXPECTED_RECEIPT_SCHEMA,
            "hscf_receipt_sha256": frozen["hscf_receipt_sha256"],
            "hscf_terminal_contract": str(receipt["hscf_terminal_contract"]),
            "hscf_terminal_contract_passed": True,
            "hscf_lambda": _hscf_export.FROZEN_HSCF_LAMBDA if frozen["arm"] == "G" else 0.0,
            "hscf_fixed_batch_size": _hscf_export.FROZEN_HSCF_BATCH_SIZE,
            "hscf_fixed_local_class_count": _hscf_export.FROZEN_HSCF_LOCAL_CLASS_COUNT,
            "hscf_loss_global_denominator": _hscf_export.FROZEN_HSCF_GLOBAL_DENOMINATOR,
            "hscf_source_partition_sha256": str(receipt["source_partition_sha256"]),
            "hscf_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "hscf_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "hscf_source_labeled_provenance": _hscf_export.SOURCE_L_PROVENANCE,
            "hscf_common_physical_order_bound": True,
            "hscf_common_scene_cycle_bound": True,
            "hscf_raw_vjp_per_scene_required": True,
            "hscf_exact_head_weight_vjp_nonzero_required": True,
            "hscf_head_bias_aux_vjp_na_none_or_zero": True,
        }
    )
    _drop_legacy_identity_fields(result)
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise HSCFLEOBindingError(f"refusing to overwrite LEO binding: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise HSCFLEOBindingError(f"refusing to overwrite temporary LEO binding: {temporary}")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--wisig-pkl", "--wisig_pkl", dest="wisig_pkl", required=True)
    parser.add_argument("--out-npz", "--out_npz", dest="out_npz", required=True)
    parser.add_argument("--binding-json", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--postfreeze-output-root", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-tx-ids", "--source_tx_ids", dest="source_tx_ids", required=True)
    parser.add_argument("--feature-name", "--feature_name", dest="feature_name", default="z_id")
    parser.add_argument("--source_only_export", action="store_true")
    parser.add_argument("--source-channel-view", "--source_channel_view", dest="source_channel_view", default=EXPECTED_CHANNEL_VIEW)
    parser.add_argument("--source-days", "--source_days", dest="source_days", default=",".join(EXPECTED_SOURCE_DAYS))
    parser.add_argument("--source-rxs", "--source_rxs", dest="source_rxs", default=",".join(EXPECTED_SOURCE_RXS))
    parser.add_argument("--source-sat-scenarios", "--source_sat_scenarios", dest="source_sat_scenarios", default=",".join(EXPECTED_SCENARIOS))
    parser.add_argument("--source-sat-seed", "--source_sat_seed", dest="source_sat_seed", type=int, default=EXPECTED_SOURCE_SAT_SEED)
    parser.add_argument("--satellite-tta-policy", "--satellite_tta_policy", dest="satellite_tta_policy", default=EXPECTED_TTA_POLICY)
    parser.add_argument("--star-ground-channel-impl", "--star_ground_channel_impl", dest="star_ground_channel_impl", default=EXPECTED_STAR_GROUND_IMPL)
    parser.add_argument("--wisig-equalized", default=EXPECTED_EQUALIZED)
    parser.add_argument("--wisig-domain", default=EXPECTED_DOMAIN)
    parser.add_argument("--wisig-out-len", type=int, default=EXPECTED_OUT_LEN)
    parser.add_argument("--max-samples-per-combo", type=int, default=EXPECTED_MAX_PER_COMBO)
    parser.add_argument("--seed", type=int, default=EXPECTED_EXPORT_SEED)
    parser.add_argument("--max-samples-per-tx", "--max_samples_per_tx", dest="max_samples_per_tx", type=int, default=EXPECTED_MAX_PER_TX)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--expected-wisig-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frozen = _validate_frozen_args(args)
    if frozen["out_npz"].exists() or frozen["binding_json"].exists():
        raise HSCFLEOBindingError("refusing to overwrite HSCF LEO artifact")
    frozen["candidate_dir"].mkdir(parents=True, exist_ok=True)
    try:
        command = _icmt_leo._generic_export_command(args, frozen)
        subprocess.run(command, check=True)
        binding = build_binding_from_existing(args)
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise _translate(exc) from exc
    _atomic_write_json(frozen["binding_json"], binding)
    print(json.dumps({"out_npz": str(frozen["out_npz"]), "binding_json": str(frozen["binding_json"]), "binding_sha256": _icmt_leo._sha256_file(frozen["binding_json"])}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
