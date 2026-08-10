#!/usr/bin/env python
"""Export and bind the frozen source-only LEO slice for P1-RCMMC.

The generic LEO forwarder remains a stable source reconstruction kernel only.
This facade reopens a real RCMMC receipt, binds source physical TX/RX/day
coverage and a single fixed LEO view, and persists RCMMC-only binding identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

import export_phase1_rcmmc_features as _rcmmc_export
import export_phase1_icmt_leo_features as _icmt_leo


EXPECTED_TRAINING_RUN_LEAF = "phase1_rcmmc12_20260811_v1"
EXPECTED_BINDING_SCHEMA = "cvs.phase1.rcmmc_leo_binding.v1"
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
LEGACY_IDENTITY_PREFIXES = ("icmt_", "hscf_", "rcat_", "recte_", "rcrmd_", "cagm_")
RAW_RECEIVER_TOKEN_FIELDS = frozenset({"source_receiver_ids", "frozen_source_receiver_ids"})


class RCMMCLEOBindingError(RuntimeError):
    """Raised when an RCMMC LEO export cannot prove its frozen binding."""


def _translate(error: BaseException) -> RCMMCLEOBindingError:
    return RCMMCLEOBindingError(str(error))


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    try:
        return _icmt_leo._parse_csv(value, field=field)
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise _translate(exc) from exc


def _drop_legacy_identity_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key).lower()
            if key in RAW_RECEIVER_TOKEN_FIELDS:
                raise RCMMCLEOBindingError(f"raw source receiver token leaked into RCMMC LEO binding: {raw_key}")
            if key.startswith(LEGACY_IDENTITY_PREFIXES):
                continue
            result[str(raw_key)] = _drop_legacy_identity_fields(raw_value)
        return result
    if isinstance(value, list):
        return [_drop_legacy_identity_fields(item) for item in value]
    if isinstance(value, tuple):
        return [_drop_legacy_identity_fields(item) for item in value]
    return value


def _assert_rcmmc_only_identity(value: Any, *, label: str) -> None:
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            key = str(raw_key).lower()
            if key in RAW_RECEIVER_TOKEN_FIELDS:
                raise RCMMCLEOBindingError(f"raw source receiver token leaked into {label}: {raw_key}")
            if key.startswith(LEGACY_IDENTITY_PREFIXES):
                raise RCMMCLEOBindingError(f"legacy identity leaked into {label}: {raw_key}")
            _assert_rcmmc_only_identity(raw_value, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_rcmmc_only_identity(item, label=label)


def _validate_frozen_args(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.fold_index)
    arm = str(args.arm).upper()
    if fold not in range(1, 7) or arm not in {"C", "G"}:
        raise RCMMCLEOBindingError("fold/arm must be F1..F6 and C/G")
    candidate = f"F{fold}{arm}_RCMMC12"
    if str(args.candidate_id) != candidate:
        raise RCMMCLEOBindingError(f"candidate_id must be {candidate}")
    training_root = Path(args.training_run_root).resolve()
    if training_root.name != EXPECTED_TRAINING_RUN_LEAF or not training_root.is_dir():
        raise RCMMCLEOBindingError(f"training root must be existing {EXPECTED_TRAINING_RUN_LEAF}")
    postfreeze_root = Path(args.postfreeze_output_root).resolve()
    if training_root == postfreeze_root or not postfreeze_root.is_dir():
        raise RCMMCLEOBindingError("postfreeze root must exist and differ from training root")
    checkpoint = Path(args.ckpt).resolve()
    expected_checkpoint = (training_root / candidate / "final_ssdg.pth").resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise RCMMCLEOBindingError("checkpoint path does not bind frozen RCMMC candidate")
    candidate_dir = (postfreeze_root / candidate).resolve()
    try:
        out_npz = _icmt_leo._require_under_root(args.out_npz, candidate_dir, label="LEO NPZ")
        binding_json = _icmt_leo._require_under_root(args.binding_json, candidate_dir, label="LEO binding JSON")
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise _translate(exc) from exc
    if out_npz != (candidate_dir / "source_leo_final_only.npz").resolve() or binding_json != (candidate_dir / "source_leo_binding.json").resolve():
        raise RCMMCLEOBindingError("LEO artifact paths do not match frozen candidate layout")
    source_tx_ids = _parse_csv(args.source_tx_ids, field="source_tx_ids")
    if len(source_tx_ids) != 4:
        raise RCMMCLEOBindingError("RCMMC LEO binding requires local4 source TX")
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
            raise RCMMCLEOBindingError(f"{field} drifted")
    for field, expected in (
        ("source_sat_seed", EXPECTED_SOURCE_SAT_SEED), ("seed", EXPECTED_EXPORT_SEED),
        ("max_samples_per_tx", EXPECTED_MAX_PER_TX), ("batch_size", EXPECTED_BATCH_SIZE),
    ):
        if int(getattr(args, field)) != expected:
            raise RCMMCLEOBindingError(f"{field} must equal frozen value {expected}")
    for field, expected in (
        ("feature_name", "z_id"), ("source_channel_view", EXPECTED_CHANNEL_VIEW),
        ("satellite_tta_policy", EXPECTED_TTA_POLICY), ("star_ground_channel_impl", EXPECTED_STAR_GROUND_IMPL),
        ("wisig_equalized", EXPECTED_EQUALIZED), ("wisig_domain", EXPECTED_DOMAIN),
    ):
        if str(getattr(args, field)) != expected:
            raise RCMMCLEOBindingError(f"{field} must equal frozen value {expected}")
    if args.source_only_export is not True:
        raise RCMMCLEOBindingError("RCMMC LEO export must be source-only")
    if int(args.wisig_out_len) != EXPECTED_OUT_LEN or int(args.max_samples_per_combo) != EXPECTED_MAX_PER_COMBO:
        raise RCMMCLEOBindingError("frozen LEO shape/combo selection drifted")
    dataset = Path(args.wisig_pkl).resolve()
    if not dataset.is_file():
        raise RCMMCLEOBindingError(f"missing frozen ManySig dataset: {dataset}")
    if str(args.expected_wisig_sha256).lower() != FROZEN_WISIG_SHA256:
        raise RCMMCLEOBindingError("expected WiSig SHA256 does not equal frozen value")
    dataset_sha256 = _icmt_leo._sha256_file(dataset)
    if dataset_sha256 != FROZEN_WISIG_SHA256:
        raise RCMMCLEOBindingError("ManySig input bytes do not match frozen SHA256")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping) or not isinstance(checkpoint_payload.get("args"), Mapping):
        raise RCMMCLEOBindingError("checkpoint payload must contain args mapping")
    checkpoint_args = checkpoint_payload["args"]
    known_tx_ids = _parse_csv(checkpoint_args.get("phase1_source_known_validation_tx_ids"), field="checkpoint known validation TX")
    proxy_tx_ids = _parse_csv(checkpoint_args.get("phase1_source_proxy_unknown_tx_ids"), field="checkpoint proxy TX")
    if len(known_tx_ids) != 1 or len(proxy_tx_ids) != 1:
        raise RCMMCLEOBindingError("checkpoint held/proxy role cardinality drifted")
    _, receipt, observed_arm = _rcmmc_export.validate_rcmmc_training_checkpoint(
        checkpoint_payload, checkpoint_path=checkpoint, source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_tx_ids, proxy_unknown_tx_ids=proxy_tx_ids,
    )
    if observed_arm != arm:
        raise RCMMCLEOBindingError("RCMMC checkpoint receipt arm does not bind LEO arm")
    return {
        "fold": fold, "arm": arm, "candidate": candidate,
        "training_root": training_root, "postfreeze_root": postfreeze_root,
        "checkpoint": checkpoint, "candidate_dir": candidate_dir,
        "out_npz": out_npz, "binding_json": binding_json,
        "source_tx_ids": source_tx_ids, "dataset": dataset, "dataset_sha256": dataset_sha256,
        "rcmmc_receipt": receipt,
        "rcmmc_receipt_sha256": _rcmmc_export._canonical_json_sha256(dict(checkpoint_payload["rcmmc_receipt"])),
    }


@contextmanager
def _patched_signed_binding() -> Iterator[None]:
    """Temporarily redirect generic LEO mechanics to RCMMC root/schema."""

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
    """Bind a sealed single-view LEO NPZ and attach RCMMC-only evidence."""

    try:
        with _patched_signed_binding():
            binding = _icmt_leo.build_binding_from_existing(args)
    except (_icmt_leo.ICMTLEOBindingError, _rcmmc_export.RCMMCSplitExportError) as exc:
        raise _translate(exc) from exc
    frozen = _validate_frozen_args(args)
    receipt = frozen["rcmmc_receipt"]
    result = dict(_drop_legacy_identity_fields(binding))
    result.update(
        {
            "schema": EXPECTED_BINDING_SCHEMA,
            "method": "P1_RCMMC",
            "rcmmc_receipt_schema": _rcmmc_export.EXPECTED_RECEIPT_SCHEMA,
            "rcmmc_receipt_sha256": frozen["rcmmc_receipt_sha256"],
            "rcmmc_terminal_contract": str(receipt["rcmmc_terminal_contract"]),
            "rcmmc_terminal_contract_passed": True,
            "rcmmc_enabled": frozen["arm"] == "G",
            "rcmmc_lambda": _rcmmc_export.FROZEN_RCMMC_LAMBDA if frozen["arm"] == "G" else 0.0,
            "rcmmc_frozen_batch_size": _rcmmc_export.FROZEN_RCMMC_BATCH_SIZE,
            "rcmmc_feature_dim": _rcmmc_export.FROZEN_RCMMC_FEATURE_DIM,
            "rcmmc_local_class_count": _rcmmc_export.FROZEN_RCMMC_LOCAL_CLASS_COUNT,
            "rcmmc_loss_global_denominator": _rcmmc_export.FROZEN_RCMMC_GLOBAL_DENOMINATOR,
            "rcmmc_fixed_batch_size": _rcmmc_export.FROZEN_RCMMC_BATCH_SIZE,
            "rcmmc_fixed_feature_dim": _rcmmc_export.FROZEN_RCMMC_FEATURE_DIM,
            "rcmmc_fixed_local_class_count": _rcmmc_export.FROZEN_RCMMC_LOCAL_CLASS_COUNT,
            "rcmmc_fixed_cells_per_scene": _rcmmc_export.FROZEN_RCMMC_GLOBAL_DENOMINATOR,
            "rcmmc_source_receiver_count": _rcmmc_export.FROZEN_RCMMC_SOURCE_RECEIVER_COUNT,
            "rcmmc_source_receiver_order_sha256": str(receipt["source_receiver_order_sha256"]),
            "rcmmc_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
            "rcmmc_source_labeled_indices_sha256": str(receipt["source_labeled_indices_sha256"]),
            "rcmmc_source_split_manifest_sha256": str(receipt["source_split_manifest_sha256"]),
            "rcmmc_source_partition_sha256": str(receipt["source_partition_sha256"]),
            "rcmmc_class_order_binding_sha256": str(receipt["class_order_binding_sha256"]),
            "rcmmc_common_batch_sequence_sha256": str(receipt["common_batch_sequence_sha256"]),
            "rcmmc_common_scenario_batches": {str(key): int(value) for key, value in dict(receipt["common_scenario_batches"]).items()},
            "rcmmc_common_cells_sha256": _rcmmc_export._canonical_json_sha256(receipt.get("rcmmc_common_cells", {})),
            "rcmmc_g_scenes_sha256": _rcmmc_export._canonical_json_sha256(receipt.get("rcmmc_scenes", {})) if frozen["arm"] == "G" else "",
            "rcmmc_clean_head_aux_vjp": "N_A_NONE_OR_ZERO_EXPECTED" if frozen["arm"] == "G" else "N_A",
            "rcmmc_leo_encoder_aux_vjp": "FINITE_NONZERO_REQUIRED" if frozen["arm"] == "G" else "N_A",
            "rcmmc_single_leo_forward_bound": True,
            "rcmmc_physical_tx_rx_day_binding_required": True,
            "rcmmc_common_physical_order_bound": True,
            "rcmmc_common_scene_cycle_bound": True,
            "rcmmc_raw_vjp_required": True,
            "rcmmc_leo_encoder_vjp_finite_nonzero": True,
            "rcmmc_clean_head_vjp_na_none_or_zero": True,
        }
    )
    _assert_rcmmc_only_identity(result, label="RCMMC LEO binding")
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RCMMCLEOBindingError(f"refusing to overwrite LEO binding: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RCMMCLEOBindingError(f"refusing to overwrite temporary LEO binding: {temporary}")
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
        raise RCMMCLEOBindingError("refusing to overwrite RCMMC LEO artifact")
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
