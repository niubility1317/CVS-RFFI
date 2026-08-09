#!/usr/bin/env python
"""Export and bind the frozen source-only LEO slice for P1-ICMT.

This is one postfreeze step: it invokes the existing generic feature exporter
with the frozen ICMT arguments, then independently reconstructs the selected
ManySig physical rows and writes an immutable binding sidecar.  The generic
exporter remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from export_spaceborne_features import _build_wisig_dataset  # noqa: E402


FROZEN_WISIG_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
EXPECTED_TRAINING_RUN_LEAF = "phase1_icmt12_20260810_v1"
EXPECTED_BINDING_SCHEMA = "cvs.phase1.icmt_leo_binding.v1"
EXPECTED_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
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
EXPECTED_RUNTIME_VIEW = "single"


class ICMTLEOBindingError(RuntimeError):
    """Raised when an ICMT LEO export cannot be bound to frozen source bytes."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise ICMTLEOBindingError(f"{field} must be non-empty and duplicate-free")
    return items


def _require_exact_tuple(value: Any, expected: Sequence[str], *, field: str) -> tuple[str, ...]:
    observed = _parse_csv(value, field=field)
    if observed != tuple(expected):
        raise ICMTLEOBindingError(
            f"{field} drifted: expected={tuple(expected)} observed={observed}"
        )
    return observed


def _require_under_root(path: str | Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ICMTLEOBindingError(f"{label} must stay under {root}: {resolved}") from exc
    return resolved


def _load_npz_metadata(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise ICMTLEOBindingError(f"missing LEO NPZ: {source}")
    required = (
        "tx_ids",
        "rx_ids",
        "day_ids",
        "eq_ids",
        "sig_ids",
        "dataset_role",
        "channel_views",
        "sat_scenarios",
        "manifest_json",
    )
    with np.load(source, allow_pickle=False) as data:
        missing = [field for field in required if field not in data.files]
        if missing:
            raise ICMTLEOBindingError(f"LEO NPZ lacks fields: {','.join(missing)}")
        payload = {
            field: np.asarray(data[field]).reshape(-1)
            for field in required
            if field != "manifest_json"
        }
        try:
            manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
        except Exception as exc:
            raise ICMTLEOBindingError("LEO NPZ manifest is invalid") from exc
    lengths = {field: int(values.size) for field, values in payload.items()}
    if len(set(lengths.values())) != 1 or not lengths or next(iter(lengths.values())) <= 0:
        raise ICMTLEOBindingError(f"LEO NPZ metadata lengths do not close: {lengths}")
    if not isinstance(manifest, Mapping):
        raise ICMTLEOBindingError("LEO NPZ manifest must encode an object")
    payload["manifest"] = dict(manifest)
    return payload


def _physical_keys_from_payload(payload: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(
        "\x1f".join(str(payload[field][index]) for field in ("tx_ids", "rx_ids", "day_ids", "eq_ids", "sig_ids"))
        for index in range(len(payload["tx_ids"]))
    )


def _physical_keys_from_dataset(dataset: Any) -> tuple[str, ...]:
    eq_list = getattr(dataset, "eq_list", None)
    if eq_list is None and hasattr(dataset, "base"):
        eq_list = getattr(dataset.base, "eq_list", None)
    if eq_list is None:
        raise ICMTLEOBindingError("reconstructed source dataset lacks equalized labels")
    keys: list[str] = []
    for item in dataset.index:
        keys.append(
            "\x1f".join(
                (
                    str(dataset.tx_list[int(item.tx_i)]),
                    str(dataset.rx_list[int(item.rx_i)]),
                    str(dataset.day_list[int(item.day_i)]),
                    str(eq_list[int(item.eq_i)]),
                    str(int(item.sig_i)),
                )
            )
        )
    return tuple(keys)


def _physical_key_receipt(keys: Sequence[str]) -> dict[str, Any]:
    ordered = tuple(str(value) for value in keys)
    if not ordered or len(set(ordered)) != len(ordered):
        raise ICMTLEOBindingError("LEO physical keys must be non-empty and unique")
    return {
        "row_count": len(ordered),
        "unique_count": len(set(ordered)),
        "ordered_sha256": _canonical_json_sha256(list(ordered)),
        "set_sha256": _canonical_json_sha256(sorted(ordered)),
    }


def _scenario_coverage_receipt(
    payload: Mapping[str, Any],
    *,
    source_tx_ids: Sequence[str],
) -> dict[str, Any]:
    tx = np.asarray(payload["tx_ids"]).astype(str)
    rx = np.asarray(payload["rx_ids"]).astype(str)
    day = np.asarray(payload["day_ids"]).astype(str)
    eq = np.asarray(payload["eq_ids"]).astype(str)
    scenarios = np.asarray(payload["sat_scenarios"]).astype(str)
    if set(tx.tolist()) != set(source_tx_ids):
        raise ICMTLEOBindingError("LEO global source TX coverage drifted")
    if set(rx.tolist()) != set(EXPECTED_SOURCE_RXS):
        raise ICMTLEOBindingError("LEO global source RX coverage drifted")
    if set(day.tolist()) != set(EXPECTED_SOURCE_DAYS):
        raise ICMTLEOBindingError("LEO global source day coverage drifted")
    if set(eq.tolist()) != {EXPECTED_EQUALIZED}:
        raise ICMTLEOBindingError("LEO equalized selection drifted")
    if set(scenarios.tolist()) != set(EXPECTED_SCENARIOS):
        raise ICMTLEOBindingError("LEO scenario set drifted")
    keys = np.asarray(_physical_keys_from_payload(payload), dtype=object)
    by_scenario: dict[str, Any] = {}
    for scenario in EXPECTED_SCENARIOS:
        mask = scenarios == scenario
        if not np.any(mask):
            raise ICMTLEOBindingError(f"LEO scenario is empty: {scenario}")
        coverage = {
            "row_count": int(mask.sum()),
            "tx_ids": sorted(set(tx[mask].tolist())),
            "rx_ids": sorted(set(rx[mask].tolist())),
            "day_ids": sorted(set(day[mask].tolist())),
            "equalized_ids": sorted(set(eq[mask].tolist())),
            "physical_key_set_sha256": _canonical_json_sha256(sorted(keys[mask].tolist())),
        }
        if set(coverage["tx_ids"]) != set(source_tx_ids):
            raise ICMTLEOBindingError(f"LEO scenario lacks complete TX coverage: {scenario}")
        if set(coverage["rx_ids"]) != set(EXPECTED_SOURCE_RXS):
            raise ICMTLEOBindingError(f"LEO scenario lacks complete RX coverage: {scenario}")
        if set(coverage["day_ids"]) != set(EXPECTED_SOURCE_DAYS):
            raise ICMTLEOBindingError(f"LEO scenario lacks complete day coverage: {scenario}")
        if coverage["equalized_ids"] != [EXPECTED_EQUALIZED]:
            raise ICMTLEOBindingError(f"LEO scenario equalized coverage drifted: {scenario}")
        by_scenario[scenario] = coverage
    return {
        "global_tx_ids": sorted(set(tx.tolist())),
        "global_rx_ids": sorted(set(rx.tolist())),
        "global_day_ids": sorted(set(day.tolist())),
        "global_equalized_ids": sorted(set(eq.tolist())),
        "by_scenario": by_scenario,
        "all_scenarios_complete": True,
    }


def _validate_frozen_args(args: argparse.Namespace) -> dict[str, Any]:
    fold = int(args.fold_index)
    arm = str(args.arm).upper()
    if fold not in range(1, 7) or arm not in {"C", "G"}:
        raise ICMTLEOBindingError("fold/arm must be F1..F6 and C/G")
    candidate = f"F{fold}{arm}_ICMT12"
    if str(args.candidate_id) != candidate:
        raise ICMTLEOBindingError(f"candidate_id must be {candidate}")
    training_root = Path(args.training_run_root).resolve()
    if training_root.name != EXPECTED_TRAINING_RUN_LEAF or not training_root.is_dir():
        raise ICMTLEOBindingError(
            f"training root must be existing {EXPECTED_TRAINING_RUN_LEAF}"
        )
    postfreeze_root = Path(args.postfreeze_output_root).resolve()
    if training_root == postfreeze_root or not postfreeze_root.is_dir():
        raise ICMTLEOBindingError("postfreeze root must exist and differ from training root")
    checkpoint = Path(args.ckpt).resolve()
    expected_checkpoint = (training_root / candidate / "final_ssdg.pth").resolve()
    if checkpoint != expected_checkpoint or not checkpoint.is_file():
        raise ICMTLEOBindingError("checkpoint path does not bind frozen ICMT candidate")
    candidate_dir = (postfreeze_root / candidate).resolve()
    out_npz = _require_under_root(args.out_npz, candidate_dir, label="LEO NPZ")
    binding_json = _require_under_root(args.binding_json, candidate_dir, label="LEO binding JSON")
    if out_npz != (candidate_dir / "source_leo_final_only.npz").resolve():
        raise ICMTLEOBindingError("LEO NPZ path does not match frozen candidate layout")
    if binding_json != (candidate_dir / "source_leo_binding.json").resolve():
        raise ICMTLEOBindingError("LEO binding path does not match frozen candidate layout")
    source_tx_ids = _parse_csv(args.source_tx_ids, field="source_tx_ids")
    if len(source_tx_ids) != 4:
        raise ICMTLEOBindingError("P1-ICMT LEO binding requires local4 source TX")
    _require_exact_tuple(args.source_days, EXPECTED_SOURCE_DAYS, field="source_days")
    _require_exact_tuple(args.source_rxs, EXPECTED_SOURCE_RXS, field="source_rxs")
    _require_exact_tuple(args.source_sat_scenarios, EXPECTED_SCENARIOS, field="source_sat_scenarios")
    frozen_scalars = {
        "source_sat_seed": EXPECTED_SOURCE_SAT_SEED,
        "seed": EXPECTED_EXPORT_SEED,
        "max_samples_per_tx": EXPECTED_MAX_PER_TX,
        "batch_size": EXPECTED_BATCH_SIZE,
    }
    for field, expected in frozen_scalars.items():
        if int(getattr(args, field)) != expected:
            raise ICMTLEOBindingError(f"{field} must equal frozen value {expected}")
    frozen_text = {
        "feature_name": "z_id",
        "source_channel_view": EXPECTED_CHANNEL_VIEW,
        "satellite_tta_policy": EXPECTED_TTA_POLICY,
        "star_ground_channel_impl": EXPECTED_STAR_GROUND_IMPL,
        "wisig_equalized": EXPECTED_EQUALIZED,
        "wisig_domain": EXPECTED_DOMAIN,
    }
    for field, expected in frozen_text.items():
        if str(getattr(args, field)) != expected:
            raise ICMTLEOBindingError(f"{field} must equal frozen value {expected}")
    if args.source_only_export is not True:
        raise ICMTLEOBindingError("P1-ICMT LEO export must be source-only")
    if int(args.wisig_out_len) != EXPECTED_OUT_LEN:
        raise ICMTLEOBindingError(f"wisig_out_len must equal {EXPECTED_OUT_LEN}")
    if int(args.max_samples_per_combo) != EXPECTED_MAX_PER_COMBO:
        raise ICMTLEOBindingError(
            f"max_samples_per_combo must equal {EXPECTED_MAX_PER_COMBO}"
        )
    dataset = Path(args.wisig_pkl).resolve()
    if not dataset.is_file():
        raise ICMTLEOBindingError(f"missing frozen ManySig dataset: {dataset}")
    if str(args.expected_wisig_sha256).lower() != FROZEN_WISIG_SHA256:
        raise ICMTLEOBindingError("expected WiSig SHA256 does not equal frozen value")
    actual_dataset_sha = _sha256_file(dataset)
    if actual_dataset_sha != FROZEN_WISIG_SHA256:
        raise ICMTLEOBindingError("ManySig input bytes do not match frozen SHA256")
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
        "dataset_sha256": actual_dataset_sha,
    }


def _generic_export_command(args: argparse.Namespace, frozen: Mapping[str, Any]) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(CODE_ROOT / "export_spaceborne_features.py"),
        "--ckpt",
        str(frozen["checkpoint"]),
        "--wisig_pkl",
        str(frozen["dataset"]),
        "--out_npz",
        str(frozen["out_npz"]),
        "--feature_name",
        "z_id",
        "--source_only_export",
        "--source_tx_ids",
        ",".join(frozen["source_tx_ids"]),
        "--source_days",
        ",".join(EXPECTED_SOURCE_DAYS),
        "--source_rxs",
        ",".join(EXPECTED_SOURCE_RXS),
        "--source_channel_view",
        EXPECTED_CHANNEL_VIEW,
        "--source_sat_scenarios",
        ",".join(EXPECTED_SCENARIOS),
        "--source_sat_seed",
        str(EXPECTED_SOURCE_SAT_SEED),
        "--star_ground_channel_impl",
        EXPECTED_STAR_GROUND_IMPL,
        "--satellite_tta_policy",
        EXPECTED_TTA_POLICY,
        "--wisig_equalized",
        EXPECTED_EQUALIZED,
        "--wisig_domain",
        EXPECTED_DOMAIN,
        "--wisig_out_len",
        str(EXPECTED_OUT_LEN),
        "--max_samples_per_combo",
        str(EXPECTED_MAX_PER_COMBO),
        "--max_samples_per_tx",
        str(EXPECTED_MAX_PER_TX),
        "--batch_size",
        str(EXPECTED_BATCH_SIZE),
        "--seed",
        str(EXPECTED_EXPORT_SEED),
        "--device",
        str(args.device),
    ]


def build_binding_from_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Reconstruct and bind an already-created generic LEO NPZ."""

    frozen = _validate_frozen_args(args)
    payload = _load_npz_metadata(frozen["out_npz"])
    manifest = payload["manifest"]
    if set(np.asarray(payload["dataset_role"]).astype(str).tolist()) != {"source"}:
        raise ICMTLEOBindingError("LEO NPZ contains non-source roles")
    if set(np.asarray(payload["channel_views"]).astype(str).tolist()) != {EXPECTED_RUNTIME_VIEW}:
        raise ICMTLEOBindingError("LEO NPZ runtime view is not frozen single-view")
    if manifest.get("source_only_export") is not True:
        raise ICMTLEOBindingError("LEO manifest is not source-only")
    if str(manifest.get("feature_name", "")) != "z_id":
        raise ICMTLEOBindingError("LEO manifest feature is not z_id")
    if str(manifest.get("classification_head_contract", "")) != EXPECTED_HEAD_CONTRACT:
        raise ICMTLEOBindingError("LEO manifest head contract drifted")
    if Path(str(manifest.get("checkpoint", ""))).resolve() != frozen["checkpoint"]:
        raise ICMTLEOBindingError("LEO manifest checkpoint path drifted")
    checkpoint_sha = _sha256_file(frozen["checkpoint"])
    if str(manifest.get("source_checkpoint_sha256", "")) != checkpoint_sha:
        raise ICMTLEOBindingError("LEO manifest checkpoint SHA256 drifted")
    if tuple(str(item) for item in manifest.get("source_tx_ids", [])) != tuple(
        frozen["source_tx_ids"]
    ):
        raise ICMTLEOBindingError("LEO manifest source TX order drifted")
    source_info = manifest.get("source")
    if not isinstance(source_info, Mapping):
        raise ICMTLEOBindingError("LEO manifest lacks source selection info")
    if Path(str(source_info.get("pkl", ""))).resolve() != frozen["dataset"]:
        raise ICMTLEOBindingError("LEO manifest dataset path drifted")
    if str(source_info.get("days", "")) != ",".join(EXPECTED_SOURCE_DAYS):
        raise ICMTLEOBindingError("LEO manifest source days drifted")
    if str(source_info.get("rxs", "")) != ",".join(EXPECTED_SOURCE_RXS):
        raise ICMTLEOBindingError("LEO manifest source RXs drifted")

    rebuilt, rebuilt_info = _build_wisig_dataset(
        pkl_path=str(frozen["dataset"]),
        tx_spec=",".join(frozen["source_tx_ids"]),
        role="source",
        equalized=EXPECTED_EQUALIZED,
        out_len=EXPECTED_OUT_LEN,
        domain=EXPECTED_DOMAIN,
        days=",".join(EXPECTED_SOURCE_DAYS),
        rxs=",".join(EXPECTED_SOURCE_RXS),
        max_samples_per_combo=EXPECTED_MAX_PER_COMBO,
        max_samples_per_tx=EXPECTED_MAX_PER_TX,
        seed=EXPECTED_EXPORT_SEED,
    )
    reconstructed_keys = _physical_keys_from_dataset(rebuilt)
    exported_keys = _physical_keys_from_payload(payload)
    if reconstructed_keys != exported_keys:
        raise ICMTLEOBindingError("LEO exported physical rows do not equal reconstructed selection")
    expected_scenarios = tuple(
        EXPECTED_SCENARIOS[(index // EXPECTED_BATCH_SIZE) % len(EXPECTED_SCENARIOS)]
        for index in range(len(reconstructed_keys))
    )
    observed_scenarios = tuple(np.asarray(payload["sat_scenarios"]).astype(str).tolist())
    if observed_scenarios != expected_scenarios:
        raise ICMTLEOBindingError("LEO scenario assignment does not match frozen batch sequence")
    physical_receipt = _physical_key_receipt(exported_keys)
    coverage_receipt = _scenario_coverage_receipt(
        payload, source_tx_ids=frozen["source_tx_ids"]
    )

    import torch  # Lazy: pair-side validation never loads checkpoint weights.

    checkpoint = torch.load(frozen["checkpoint"], map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ICMTLEOBindingError("checkpoint payload must be a mapping")
    checkpoint_args = checkpoint.get("args")
    if not isinstance(checkpoint_args, Mapping):
        raise ICMTLEOBindingError("checkpoint lacks args mapping")
    if str(checkpoint.get("candidate_id", "")) != frozen["candidate"] or str(
        checkpoint_args.get("candidate_id", "")
    ) != frozen["candidate"]:
        raise ICMTLEOBindingError("checkpoint candidate binding drifted")
    if str(checkpoint.get("run_id", "")) != EXPECTED_TRAINING_RUN_LEAF or str(
        checkpoint_args.get("run_id", "")
    ) != EXPECTED_TRAINING_RUN_LEAF:
        raise ICMTLEOBindingError("checkpoint training run binding drifted")
    if str(checkpoint.get("checkpoint_role", "")) != "training_final_only":
        raise ICMTLEOBindingError("checkpoint role is not training_final_only")

    selection = {
        "source_tx_ids": list(frozen["source_tx_ids"]),
        "source_rx_ids": list(EXPECTED_SOURCE_RXS),
        "source_day_ids": list(EXPECTED_SOURCE_DAYS),
        "equalized": EXPECTED_EQUALIZED,
        "domain": EXPECTED_DOMAIN,
        "out_len": EXPECTED_OUT_LEN,
        "max_samples_per_combo": EXPECTED_MAX_PER_COMBO,
        "max_samples_per_tx": EXPECTED_MAX_PER_TX,
        "export_seed": EXPECTED_EXPORT_SEED,
        "batch_size": EXPECTED_BATCH_SIZE,
        "channel_view": EXPECTED_CHANNEL_VIEW,
        "runtime_view": EXPECTED_RUNTIME_VIEW,
        "satellite_scenarios": list(EXPECTED_SCENARIOS),
        "source_sat_seed": EXPECTED_SOURCE_SAT_SEED,
        "satellite_tta_policy": EXPECTED_TTA_POLICY,
        "star_ground_channel_impl": EXPECTED_STAR_GROUND_IMPL,
        "reconstructed_size": len(rebuilt),
        "generic_source_info": dict(rebuilt_info),
    }
    selection["selection_sha256"] = _canonical_json_sha256(selection)
    return {
        "schema": EXPECTED_BINDING_SCHEMA,
        "candidate_id": frozen["candidate"],
        "fold_index": frozen["fold"],
        "arm": frozen["arm"],
        "training_run_root": str(frozen["training_root"]),
        "postfreeze_output_root": str(frozen["postfreeze_root"]),
        "checkpoint_path": str(frozen["checkpoint"]),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_role": "training_final_only",
        "training_run_id": EXPECTED_TRAINING_RUN_LEAF,
        "classification_head_contract": EXPECTED_HEAD_CONTRACT,
        "leo_npz_path": str(frozen["out_npz"]),
        "leo_npz_sha256": _sha256_file(frozen["out_npz"]),
        "leo_manifest_sha256": _canonical_json_sha256(manifest),
        "dataset_path": str(frozen["dataset"]),
        "dataset_sha256": frozen["dataset_sha256"],
        "source_selection": selection,
        "physical_keys": physical_receipt,
        "scenario_assignment_sha256": _canonical_json_sha256(list(observed_scenarios)),
        "scenario_coverage": coverage_receipt,
        "all_source_rows_reconstructed": True,
        "all_scenarios_complete": True,
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ICMTLEOBindingError(f"refusing to overwrite LEO binding: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ICMTLEOBindingError(f"refusing to overwrite temporary LEO binding: {temporary}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
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
    parser.add_argument(
        "--max-samples-per-combo", type=int, default=EXPECTED_MAX_PER_COMBO
    )
    parser.add_argument("--seed", type=int, default=EXPECTED_EXPORT_SEED)
    parser.add_argument("--max-samples-per-tx", "--max_samples_per_tx", dest="max_samples_per_tx", type=int, default=EXPECTED_MAX_PER_TX)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--expected-wisig-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frozen = _validate_frozen_args(args)
    if frozen["out_npz"].exists():
        raise ICMTLEOBindingError(f"refusing to overwrite LEO NPZ: {frozen['out_npz']}")
    if frozen["binding_json"].exists():
        raise ICMTLEOBindingError(
            f"refusing to overwrite LEO binding: {frozen['binding_json']}"
        )
    frozen["candidate_dir"].mkdir(parents=True, exist_ok=True)
    subprocess.run(_generic_export_command(args, frozen), check=True)
    binding = build_binding_from_existing(args)
    _atomic_write_json(frozen["binding_json"], binding)
    print(
        json.dumps(
            {
                "out_npz": str(frozen["out_npz"]),
                "binding_json": str(frozen["binding_json"]),
                "binding_sha256": _sha256_file(frozen["binding_json"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
