#!/usr/bin/env python
"""Export and independently bind the frozen source-only LEO slice for RECTE.

The generic forwarding implementation is the signed compatibility kernel only.
This file owns RECTE candidate/root/receipt identity, replays the fixed ManySig
selection, and seals a RECTE sidecar with current NPZ, physical-key and full
scenario/TX/RX/day binding evidence.  No target, held, proxy-fit or second
LEO observation is permitted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

import export_phase1_recte_features as _recte_export
import export_phase1_icmt_leo_features as _icmt_leo


EXPECTED_TRAINING_RUN_LEAF = "phase1_recte12_20260810_v1"
EXPECTED_BINDING_SCHEMA = "cvs.phase1.recte_leo_binding.v1"
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
LEGACY_IDENTITY_PREFIXES = ("icmt_", "rcat_", "rcrmd_", "cagm_")


class RECTELEOBindingError(RuntimeError):
    """Raised when a RECTE LEO export cannot prove its frozen source binding."""


def _translate(error: BaseException) -> RECTELEOBindingError:
    return RECTELEOBindingError(str(error))


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
        raise RECTELEOBindingError("legacy method identity leaked into RECTE LEO binding")


def _validate_frozen_args(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.fold_index)
    arm = str(args.arm).upper()
    if fold not in range(1, 7) or arm not in {"C", "G"}:
        raise RECTELEOBindingError("fold/arm must be F1..F6 and C/G")
    candidate = f"F{fold}{arm}_RECTE12"
    if str(args.candidate_id) != candidate:
        raise RECTELEOBindingError(f"candidate_id must be {candidate}")
    training_root = Path(args.training_run_root).resolve()
    if training_root.name != EXPECTED_TRAINING_RUN_LEAF or not training_root.is_dir():
        raise RECTELEOBindingError(
            f"training root must be existing {EXPECTED_TRAINING_RUN_LEAF}"
        )
    postfreeze_root = Path(args.postfreeze_output_root).resolve()
    if training_root == postfreeze_root or not postfreeze_root.is_dir():
        raise RECTELEOBindingError("postfreeze root must exist and differ from training root")
    checkpoint = Path(args.ckpt).resolve()
    expected_checkpoint = (training_root / candidate / "final_ssdg.pth").resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise RECTELEOBindingError("checkpoint path does not bind frozen RECTE candidate")
    candidate_dir = (postfreeze_root / candidate).resolve()
    try:
        out_npz = _icmt_leo._require_under_root(args.out_npz, candidate_dir, label="LEO NPZ")
        binding_json = _icmt_leo._require_under_root(
            args.binding_json, candidate_dir, label="LEO binding JSON"
        )
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise _translate(exc) from exc
    if out_npz != (candidate_dir / "source_leo_final_only.npz").resolve():
        raise RECTELEOBindingError("LEO NPZ path does not match frozen candidate layout")
    if binding_json != (candidate_dir / "source_leo_binding.json").resolve():
        raise RECTELEOBindingError("LEO binding path does not match frozen candidate layout")
    source_tx_ids = _parse_csv(args.source_tx_ids, field="source_tx_ids")
    if len(source_tx_ids) != 4:
        raise RECTELEOBindingError("P1-RECTE LEO binding requires local4 source TX")
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
            raise RECTELEOBindingError(f"{field} drifted")
    for field, expected in (
        ("source_sat_seed", EXPECTED_SOURCE_SAT_SEED),
        ("seed", EXPECTED_EXPORT_SEED),
        ("max_samples_per_tx", EXPECTED_MAX_PER_TX),
        ("batch_size", EXPECTED_BATCH_SIZE),
    ):
        if int(getattr(args, field)) != expected:
            raise RECTELEOBindingError(f"{field} must equal frozen value {expected}")
    for field, expected in (
        ("feature_name", "z_id"),
        ("source_channel_view", EXPECTED_CHANNEL_VIEW),
        ("satellite_tta_policy", EXPECTED_TTA_POLICY),
        ("star_ground_channel_impl", EXPECTED_STAR_GROUND_IMPL),
        ("wisig_equalized", EXPECTED_EQUALIZED),
        ("wisig_domain", EXPECTED_DOMAIN),
    ):
        if str(getattr(args, field)) != expected:
            raise RECTELEOBindingError(f"{field} must equal frozen value {expected}")
    if args.source_only_export is not True:
        raise RECTELEOBindingError("P1-RECTE LEO export must be source-only")
    if int(args.wisig_out_len) != EXPECTED_OUT_LEN or int(args.max_samples_per_combo) != EXPECTED_MAX_PER_COMBO:
        raise RECTELEOBindingError("frozen LEO shape or combo selection drifted")
    dataset = Path(args.wisig_pkl).resolve()
    if not dataset.is_file():
        raise RECTELEOBindingError(f"missing frozen ManySig dataset: {dataset}")
    if str(args.expected_wisig_sha256).lower() != FROZEN_WISIG_SHA256:
        raise RECTELEOBindingError("expected WiSig SHA256 does not equal frozen value")
    dataset_sha256 = _icmt_leo._sha256_file(dataset)
    if dataset_sha256 != FROZEN_WISIG_SHA256:
        raise RECTELEOBindingError("ManySig input bytes do not match frozen SHA256")
    checkpoint_payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise RECTELEOBindingError("checkpoint payload must be a mapping")
    _, receipt, observed_arm = _recte_export.validate_recte_training_checkpoint(
        checkpoint_payload,
        checkpoint_path=checkpoint,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=(
            str(getattr(args, "known_validation_tx_id", "")),
        )
        if str(getattr(args, "known_validation_tx_id", ""))
        else tuple(
            str(item)
            for item in checkpoint_payload.get("recte_receipt", {}).get(
                "source_known_validation_tx", []
            )
        ),
        proxy_unknown_tx_ids=(
            str(getattr(args, "proxy_unknown_tx_id", "")),
        )
        if str(getattr(args, "proxy_unknown_tx_id", ""))
        else tuple(
            str(item)
            for item in checkpoint_payload.get("recte_receipt", {}).get(
                "source_proxy_unknown_tx", []
            )
        ),
    )
    if observed_arm != arm:
        raise RECTELEOBindingError("RECTE checkpoint receipt arm does not bind LEO arm")
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
        "recte_receipt": receipt,
        "recte_receipt_sha256": _recte_export._canonical_json_sha256(
            dict(checkpoint_payload["recte_receipt"])
        ),
    }


@contextmanager
def _patched_signed_binding() -> Iterator[None]:
    """Temporarily point signed LEO mechanics at RECTE identity."""

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
    """Bind an existing LEO NPZ and attach raw RECTE terminal evidence."""

    try:
        with _patched_signed_binding():
            binding = _icmt_leo.build_binding_from_existing(args)
    except (
        _icmt_leo.ICMTLEOBindingError,
        _recte_export.RECTESplitExportError,
    ) as exc:
        raise _translate(exc) from exc
    frozen = _validate_frozen_args(args)
    receipt = frozen["recte_receipt"]
    result = dict(binding)
    _drop_legacy_identity_fields(result)
    result.update(
        {
            "schema": EXPECTED_BINDING_SCHEMA,
            "method": "P1_RECTE",
            "recte_receipt_schema": _recte_export.EXPECTED_RECEIPT_SCHEMA,
            "recte_receipt_sha256": frozen["recte_receipt_sha256"],
            "recte_terminal_contract": str(receipt["recte_terminal_contract"]),
            "recte_terminal_contract_passed": True,
            "recte_lambda": _recte_export.FROZEN_RECTE_LAMBDA if frozen["arm"] == "G" else 0.0,
            "recte_loss_global_denominator": _recte_export.FROZEN_UNORDERED_PAIR_COUNT,
            "recte_fixed_unordered_pair_count": _recte_export.FROZEN_UNORDERED_PAIR_COUNT,
            "recte_source_receiver_ids": list(_recte_export.FROZEN_SOURCE_RECEIVER_IDS),
            "recte_source_receiver_count": _recte_export.FROZEN_SOURCE_RECEIVER_COUNT,
            "recte_source_receiver_ids_sha256": str(receipt["source_receiver_ids_sha256"]),
            "recte_source_receiver_provenance": _recte_export.SOURCE_RECEIVER_PROVENANCE,
            "recte_frozen_cells_per_scene": _recte_export.FROZEN_CELLS_PER_SCENE,
            "recte_common_physical_rx_class_scene_nrc_bound": True,
            "recte_batch_order_bound": True,
            "recte_functional_logits_live_equality_required": True,
            "recte_exact_head_aux_vjp_na_none_or_zero": True,
        }
    )
    _drop_legacy_identity_fields(result)
    return result


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise RECTELEOBindingError(f"refusing to overwrite LEO binding: {path}")
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise RECTELEOBindingError(f"refusing to overwrite temporary LEO binding: {temporary}")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
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
        raise RECTELEOBindingError("refusing to overwrite RECTE LEO artifact")
    frozen["candidate_dir"].mkdir(parents=True, exist_ok=True)
    try:
        command = _icmt_leo._generic_export_command(args, frozen)
        subprocess.run(command, check=True)
        binding = build_binding_from_existing(args)
    except _icmt_leo.ICMTLEOBindingError as exc:
        raise _translate(exc) from exc
    _atomic_write_json(frozen["binding_json"], binding)
    print(
        json.dumps(
            {
                "out_npz": str(frozen["out_npz"]),
                "binding_json": str(frozen["binding_json"]),
                "binding_sha256": _icmt_leo._sha256_file(frozen["binding_json"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
