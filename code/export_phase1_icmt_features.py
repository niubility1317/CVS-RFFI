#!/usr/bin/env python
"""Export the exact frozen Phase1 L/V split for P1-ICMT geometry.

The final checkpoint already seals the training split receipt.  This thin
entry reconstructs that split from the same local4 WiSig view, verifies every
L/U/V index hash against the checkpoint, and forwards only L and V.  U is
never wrapped in a loader and never reaches the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader


CODE_ROOT = Path(__file__).resolve().parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from SSDG.train_ssdg import (  # noqa: E402
    _build_source_split_receipt,
    _parse_wisig_axis_spec,
    _phase1_tx_partition_view,
    split_tx_rx_day_1_6_3,
)
from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint  # noqa: E402
from dataset_wisig import (  # noqa: E402
    WiSigCompactDataset,
    WiSigSubsetDataset,
    _resolve_days,
    _resolve_rxs,
    load_wisig_compact_pkl,
)
from export_spaceborne_features import (  # noqa: E402
    _build_wisig_dataset,
    _concat_payloads,
    _sha256_file,
    extract_features_with_metadata,
)


EXPECTED_SPLIT_MODE = "tx_rx_day_1_6_3"
EXPECTED_LABELED_RATIO = 0.07
EXPECTED_UNLABELED_RATIO = 0.63
EXPECTED_SOURCE_VAL_RATIO = 0.30
EXPECTED_SEED = 7281105
EXPECTED_FEATURE_NAME = "z_id"
EXPECTED_CLASSIFICATION_HEAD_CONTRACT = "dual_cvsincnet_tx_logits_v1"
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_MODEL_VARIANT = "lite_d"
FROZEN_WISIG_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
EXPECTED_TRAINING_RUN_ID = "phase1_icmt12_20260810_v1"
EXPECTED_RECEIPT_SCHEMA = "cvs.phase1.icmt_receipt.v1"
EXPECTED_RECEIPT_METHOD = "P1_ICMT"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_ICMT12$")
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400


class ICMTSplitExportError(RuntimeError):
    """Raised when the final checkpoint and reconstructed L/V split differ."""


def _require_frozen_dataset_sha256(
    *, actual: str, expected: str, checkpoint_declared: str
) -> dict[str, Any]:
    actual_value = str(actual).strip().lower()
    expected_value = str(expected).strip().lower()
    declared_value = str(checkpoint_declared).strip().lower()
    if expected_value != FROZEN_WISIG_SHA256:
        raise ICMTSplitExportError(
            f"expected WiSig SHA256 must equal frozen value {FROZEN_WISIG_SHA256}"
        )
    if actual_value != expected_value:
        raise ICMTSplitExportError("WiSig input bytes do not match expected SHA256")
    if declared_value and declared_value != expected_value:
        raise ICMTSplitExportError(
            "checkpoint-declared WiSig SHA256 does not match expected/actual bytes"
        )
    return {
        "actual": actual_value,
        "expected": expected_value,
        "checkpoint_declared": declared_value,
        "checkpoint_declared_empty_caveat": not bool(declared_value),
    }


def _require_close(name: str, observed: Any, expected: float) -> None:
    try:
        value = float(observed)
    except (TypeError, ValueError) as exc:
        raise ICMTSplitExportError(f"checkpoint arg {name} is not numeric") from exc
    if not math.isfinite(value) or abs(value - float(expected)) > 1e-12:
        raise ICMTSplitExportError(
            f"checkpoint arg {name} drifted: expected={expected} observed={observed}"
        )


def _parse_csv(value: Any, *, field: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in str(value or "").split(",") if item.strip())
    if not items or len(items) != len(set(items)):
        raise ICMTSplitExportError(f"{field} must be non-empty and duplicate-free")
    return items


def _require_frozen_proxy_selection(args: argparse.Namespace) -> dict[str, Any]:
    days = _parse_csv(args.proxy_days, field="proxy_days")
    rxs = _parse_csv(args.proxy_rxs, field="proxy_rxs")
    if days != FROZEN_PROXY_DAYS:
        raise ICMTSplitExportError(
            f"proxy_days must equal frozen value {FROZEN_PROXY_DAYS}"
        )
    if rxs != FROZEN_PROXY_RXS:
        raise ICMTSplitExportError(
            f"proxy_rxs must equal frozen value {FROZEN_PROXY_RXS}"
        )
    if int(args.max_proxy_samples_per_tx) != FROZEN_PROXY_MAX_SAMPLES_PER_TX:
        raise ICMTSplitExportError(
            "max_proxy_samples_per_tx must equal frozen value "
            f"{FROZEN_PROXY_MAX_SAMPLES_PER_TX}"
        )
    return {
        "days": list(FROZEN_PROXY_DAYS),
        "rxs": list(FROZEN_PROXY_RXS),
        "selection_seed": FROZEN_PROXY_SELECTION_SEED,
        "max_samples_per_tx": FROZEN_PROXY_MAX_SAMPLES_PER_TX,
        "expected_total_count": FROZEN_PROXY_TOTAL_COUNT,
    }


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _index_sha256(values: Sequence[Any]) -> str:
    return _canonical_json_sha256([int(value) for value in values])


def _physical_keys_for_indices(base: WiSigCompactDataset, values: Sequence[Any]) -> tuple[str, ...]:
    eq_list = getattr(base, "eq_list", None)
    if eq_list is None and hasattr(base, "base"):
        eq_list = getattr(base.base, "eq_list", None)
    if eq_list is None:
        raise ICMTSplitExportError("dataset lacks equalized-axis labels")
    keys: list[str] = []
    for value in values:
        item = base.index[int(value)]
        keys.append(
            "\x1f".join(
                (
                    str(base.tx_list[int(item.tx_i)]),
                    str(base.rx_list[int(item.rx_i)]),
                    str(base.day_list[int(item.day_i)]),
                    str(eq_list[int(item.eq_i)]),
                    str(int(item.sig_i)),
                )
            )
        )
    return tuple(keys)


def _physical_key_receipt(keys: Sequence[str]) -> dict[str, Any]:
    ordered = tuple(str(value) for value in keys)
    if len(ordered) != FROZEN_PROXY_TOTAL_COUNT:
        raise ICMTSplitExportError(
            "proxy physical row count must equal frozen value "
            f"{FROZEN_PROXY_TOTAL_COUNT}, got {len(ordered)}"
        )
    if len(set(ordered)) != len(ordered):
        raise ICMTSplitExportError("proxy physical keys must be unique")
    return {
        "row_count": len(ordered),
        "unique_count": len(set(ordered)),
        "ordered_sha256": _canonical_json_sha256(list(ordered)),
        "set_sha256": _canonical_json_sha256(sorted(ordered)),
    }


def _validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> Mapping[str, Any]:
    if str(checkpoint.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise ICMTSplitExportError(
            f"checkpoint_role must be {EXPECTED_CHECKPOINT_ROLE}"
        )
    if str(checkpoint.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise ICMTSplitExportError(
            f"checkpoint_selection must be {EXPECTED_CHECKPOINT_SELECTION}"
        )
    if not isinstance(checkpoint.get("model"), Mapping) or not isinstance(checkpoint.get("args"), Mapping):
        raise ICMTSplitExportError("checkpoint must contain model and args mappings")
    args = checkpoint["args"]
    expected_text = {
        "split_mode": EXPECTED_SPLIT_MODE,
        "model_variant": EXPECTED_MODEL_VARIANT,
        "id_feature_key": "feat_joint",
        "phase1_source_train_tx_ids": ",".join(source_tx_ids),
        "phase1_source_known_validation_tx_ids": ",".join(known_validation_tx_ids),
        "phase1_source_proxy_unknown_tx_ids": ",".join(proxy_unknown_tx_ids),
    }
    for field, expected in expected_text.items():
        if str(args.get(field, "")) != expected:
            raise ICMTSplitExportError(
                f"checkpoint arg {field} drifted: expected={expected} observed={args.get(field)}"
            )
    _require_close("labeled_ratio", args.get("labeled_ratio"), EXPECTED_LABELED_RATIO)
    _require_close("unlabeled_ratio", args.get("unlabeled_ratio"), EXPECTED_UNLABELED_RATIO)
    _require_close("source_val_ratio", args.get("source_val_ratio"), EXPECTED_SOURCE_VAL_RATIO)
    if int(args.get("seed", -1)) != EXPECTED_SEED:
        raise ICMTSplitExportError(
            f"checkpoint seed must be {EXPECTED_SEED}, got {args.get('seed')}"
        )
    if str(args.get("checkpoint_selection", "")) != EXPECTED_CHECKPOINT_SELECTION:
        raise ICMTSplitExportError("checkpoint args do not preserve final_only selection")
    if str(args.get("candidate_id", "")) != checkpoint_path.parent.name:
        raise ICMTSplitExportError("checkpoint candidate_id does not match its parent arm directory")
    if str(checkpoint.get("candidate_id", "")) != checkpoint_path.parent.name:
        raise ICMTSplitExportError("top-level checkpoint candidate_id does not match its parent arm directory")
    candidate_match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if candidate_match is None:
        raise ICMTSplitExportError("checkpoint candidate_id is not a frozen F1..F6 C/G ICMT12 arm")
    if str(checkpoint.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID:
        raise ICMTSplitExportError(
            f"checkpoint run_id must be {EXPECTED_TRAINING_RUN_ID}"
        )
    if str(args.get("run_id", "")) != EXPECTED_TRAINING_RUN_ID:
        raise ICMTSplitExportError(
            f"checkpoint args run_id must be {EXPECTED_TRAINING_RUN_ID}"
        )
    receipt = checkpoint.get("icmt_receipt")
    if not isinstance(receipt, Mapping):
        raise ICMTSplitExportError("checkpoint lacks P1-ICMT terminal receipt")
    if str(receipt.get("schema", "")) != EXPECTED_RECEIPT_SCHEMA:
        raise ICMTSplitExportError("P1-ICMT terminal receipt schema drifted")
    if str(receipt.get("method", "")) != EXPECTED_RECEIPT_METHOD:
        raise ICMTSplitExportError("P1-ICMT terminal receipt method drifted")
    if receipt.get("frozen_mode") is not True:
        raise ICMTSplitExportError("P1-ICMT terminal receipt is not frozen")
    expected_enabled = candidate_match.group(2) == "G"
    if receipt.get("enabled") is not expected_enabled:
        raise ICMTSplitExportError("P1-ICMT terminal receipt enabled flag does not match arm")
    _require_close("icmt receipt lambda", receipt.get("lambda"), 0.05 if expected_enabled else 0.0)
    if str(receipt.get("z_id_key", "")) != "feat_joint":
        raise ICMTSplitExportError("P1-ICMT terminal receipt z_id path drifted")
    if receipt.get("icmt_terminal_contract_passed") is not True:
        raise ICMTSplitExportError("P1-ICMT terminal receipt did not pass")
    if str(receipt.get("checkpoint_role", "")) != EXPECTED_CHECKPOINT_ROLE:
        raise ICMTSplitExportError("P1-ICMT receipt is not bound to training_final_only")
    if tuple(str(item) for item in receipt.get("source_train_tx", [])) != tuple(source_tx_ids):
        raise ICMTSplitExportError("P1-ICMT receipt source_train_tx order drifted")
    if tuple(str(item) for item in receipt.get("source_known_validation_tx", [])) != tuple(known_validation_tx_ids):
        raise ICMTSplitExportError("P1-ICMT receipt known-validation TX drifted")
    if tuple(str(item) for item in receipt.get("source_proxy_unknown_tx", [])) != tuple(proxy_unknown_tx_ids):
        raise ICMTSplitExportError("P1-ICMT receipt proxy-unknown TX drifted")
    if tuple(str(item) for item in receipt.get("local_tx_class_order", [])) != tuple(source_tx_ids):
        raise ICMTSplitExportError("P1-ICMT receipt local TX/head order drifted")
    if tuple(int(item) for item in receipt.get("local_to_head_class_ids", [])) != (0, 1, 2, 3):
        raise ICMTSplitExportError("P1-ICMT receipt local-to-head class order drifted")
    return args


def reconstruct_frozen_source_split(
    raw_dataset: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
    *,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
    wisig_pkl_sha256: str,
) -> dict[str, Any]:
    """Rebuild the exact local4 source base and its L/U/V indices."""

    filtered, tx_partition_receipt = _phase1_tx_partition_view(
        raw_dataset,
        train_spec=",".join(source_tx_ids),
        known_validation_spec=",".join(known_validation_tx_ids),
        proxy_unknown_spec=",".join(proxy_unknown_tx_ids),
    )
    day_list = list(filtered.get("capture_date_list", []))
    rx_list = list(filtered.get("rx_list", []))
    train_days = _resolve_days(
        day_list,
        _parse_wisig_axis_spec(checkpoint_args.get("wisig_train_days", "")),
        list(range(min(3, len(day_list)))),
    )
    test_days = _resolve_days(
        day_list,
        _parse_wisig_axis_spec(checkpoint_args.get("wisig_test_days", "")),
        [len(day_list) - 1],
    )
    train_rxs = _resolve_rxs(
        rx_list,
        _parse_wisig_axis_spec(checkpoint_args.get("wisig_train_rxs", "")),
        list(range(len(rx_list))),
    )
    test_rxs = _resolve_rxs(
        rx_list,
        _parse_wisig_axis_spec(checkpoint_args.get("wisig_test_rxs", "")),
        [],
    )
    train_days = [value for value in train_days if value not in test_days]
    train_rxs = [value for value in train_rxs if value not in test_rxs]
    equalized = checkpoint_args.get("wisig_equalized", "1")
    source_base = WiSigCompactDataset(
        filtered,
        out_len=int(checkpoint_args.get("wisig_out_len", 256)),
        crop_mode="center",
        normalize=True,
        equalized=("both" if str(equalized).lower() == "both" else int(equalized)),
        day_keep=train_days,
        rx_keep=train_rxs,
        domain=str(checkpoint_args.get("wisig_domain", "rx_day")),
        max_samples_per_combo=(
            None
            if int(checkpoint_args.get("wisig_max_day123_per_combo", 0)) <= 0
            else int(checkpoint_args["wisig_max_day123_per_combo"])
        ),
        seed=int(checkpoint_args.get("seed", EXPECTED_SEED)),
        build_index=True,
    )
    labeled, unlabeled, validation = split_tx_rx_day_1_6_3(
        source_base,
        labeled_ratio=EXPECTED_LABELED_RATIO,
        unlabeled_ratio=EXPECTED_UNLABELED_RATIO,
        source_val_ratio=EXPECTED_SOURCE_VAL_RATIO,
    )
    all_sets = [set(labeled), set(unlabeled), set(validation)]
    if any(all_sets[i] & all_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ICMTSplitExportError("reconstructed L/U/V indices overlap")
    if set().union(*all_sets) != set(range(len(source_base))):
        raise ICMTSplitExportError("reconstructed L/U/V indices do not cover source_base exactly")
    receipt = _build_source_split_receipt(
        seed=EXPECTED_SEED,
        split_mode=EXPECTED_SPLIT_MODE,
        source_days=train_days,
        target_days=test_days,
        source_receivers=train_rxs,
        target_receivers=test_rxs,
        labeled_indices=labeled,
        unlabeled_indices=unlabeled,
        source_validation_indices=validation,
        wisig_pkl_sha256=wisig_pkl_sha256,
        requested_labeled_ratio=EXPECTED_LABELED_RATIO,
        requested_unlabeled_ratio=EXPECTED_UNLABELED_RATIO,
        requested_source_val_ratio=EXPECTED_SOURCE_VAL_RATIO,
        realized_rho_tolerance=float(checkpoint_args.get("phase1_realized_rho_tolerance", 0.002)),
        realized_source_val_tolerance=float(
            checkpoint_args.get("phase1_realized_source_val_tolerance", 0.002)
        ),
    )
    return {
        "source_base": source_base,
        "labeled_indices": tuple(int(value) for value in labeled),
        "unlabeled_indices": tuple(int(value) for value in unlabeled),
        "source_validation_indices": tuple(int(value) for value in validation),
        "source_split_receipt": receipt,
        "tx_partition_receipt": tx_partition_receipt,
        "resolved_source_day_indices": tuple(int(value) for value in train_days),
        "resolved_target_day_indices": tuple(int(value) for value in test_days),
        "resolved_source_rx_indices": tuple(int(value) for value in train_rxs),
        "resolved_target_rx_indices": tuple(int(value) for value in test_rxs),
    }


def _require_split_receipts_match(checkpoint: Mapping[str, Any], reconstructed: Mapping[str, Any]) -> None:
    split_info = checkpoint.get("split_info")
    if not isinstance(split_info, Mapping):
        raise ICMTSplitExportError("checkpoint lacks split_info")
    expected_split = split_info.get("source_split_receipt")
    if not isinstance(expected_split, Mapping):
        raise ICMTSplitExportError("checkpoint lacks source_split_receipt")
    if dict(expected_split) != dict(reconstructed["source_split_receipt"]):
        raise ICMTSplitExportError("reconstructed source split receipt does not equal checkpoint receipt")
    expected_partition = split_info.get("tx_partition_receipt")
    if not isinstance(expected_partition, Mapping):
        raise ICMTSplitExportError("checkpoint lacks tx_partition_receipt")
    if dict(expected_partition) != dict(reconstructed["tx_partition_receipt"]):
        raise ICMTSplitExportError("reconstructed TX partition receipt does not equal checkpoint receipt")
    expected_sizes = {
        "labeled_size": len(reconstructed["labeled_indices"]),
        "unlabeled_size": len(reconstructed["unlabeled_indices"]),
        "source_val_size": len(reconstructed["source_validation_indices"]),
    }
    for field, expected in expected_sizes.items():
        if int(split_info.get(field, -1)) != int(expected):
            raise ICMTSplitExportError(
                f"checkpoint split_info {field} mismatch: expected={expected} observed={split_info.get(field)}"
            )


def _validate_local4_coverage(
    base: WiSigCompactDataset,
    indices: Sequence[int],
    source_tx_ids: Sequence[str],
    *,
    role: str,
    require_more_than_one: bool,
) -> dict[str, int]:
    counts = {str(tx): 0 for tx in source_tx_ids}
    for value in indices:
        item = base.index[int(value)]
        tx = str(base.tx_list[int(item.tx_i)])
        if tx not in counts:
            raise ICMTSplitExportError(f"{role} contains non-local4 TX {tx}")
        counts[tx] += 1
    minimum = 2 if require_more_than_one else 1
    if any(count < minimum for count in counts.values()):
        raise ICMTSplitExportError(
            f"{role} lacks required local4 coverage: minimum={minimum} counts={counts}"
        )
    return counts


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise ICMTSplitExportError(f"refusing to overwrite L/V export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise ICMTSplitExportError(f"refusing to overwrite temporary L/V export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **dict(payload))
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = Path(args.ckpt).resolve()
    dataset_path = Path(args.wisig_pkl).resolve()
    output_path = Path(args.out_npz).resolve()
    if not checkpoint_path.is_file():
        raise ICMTSplitExportError(f"missing final checkpoint: {checkpoint_path}")
    if not dataset_path.is_file():
        raise ICMTSplitExportError(f"missing WiSig dataset: {dataset_path}")
    proxy_selection = _require_frozen_proxy_selection(args)
    source_tx_ids = _parse_csv(args.source_tx_ids, field="source_tx_ids")
    known_validation_tx_ids = _parse_csv(
        args.known_validation_tx_ids, field="known_validation_tx_ids"
    )
    proxy_unknown_tx_ids = _parse_csv(args.proxy_unknown_tx_ids, field="proxy_unknown_tx_ids")
    if len(source_tx_ids) != 4 or len(known_validation_tx_ids) != 1 or len(proxy_unknown_tx_ids) != 1:
        raise ICMTSplitExportError("P1-ICMT L/V export requires local4 plus one held and one proxy TX")
    role_sets = [set(source_tx_ids), set(known_validation_tx_ids), set(proxy_unknown_tx_ids)]
    if any(role_sets[i] & role_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ICMTSplitExportError("source/known-validation/proxy-unknown TX roles overlap")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ICMTSplitExportError("checkpoint payload must be a mapping")
    checkpoint_args = _validate_checkpoint_contract(
        checkpoint,
        checkpoint_path=checkpoint_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    checkpoint_dataset_path = Path(str(checkpoint_args.get("wisig_pkl", ""))).resolve()
    if checkpoint_dataset_path != dataset_path:
        raise ICMTSplitExportError(
            "WiSig dataset path does not equal the path sealed in checkpoint args"
        )
    dataset_sha256 = _sha256_file(dataset_path)
    expected_dataset_sha256 = str(checkpoint_args.get("wisig_pkl_sha256", "")).lower()
    dataset_sha256_receipt = _require_frozen_dataset_sha256(
        actual=dataset_sha256,
        expected=args.expected_wisig_sha256,
        checkpoint_declared=expected_dataset_sha256,
    )
    raw_dataset = load_wisig_compact_pkl(str(dataset_path))
    reconstructed = reconstruct_frozen_source_split(
        raw_dataset,
        checkpoint_args,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
        # The frozen continuation checkpoint may omit a dataset digest. Preserve the
        # checkpoint's exact receipt value (possibly empty) so full receipt
        # equality still closes; the resolved path and actual bytes are
        # recorded separately in the export manifest below.
        wisig_pkl_sha256=expected_dataset_sha256,
    )
    _require_split_receipts_match(checkpoint, reconstructed)
    icmt_receipt = checkpoint["icmt_receipt"]
    reconstructed_labeled_sha = _index_sha256(reconstructed["labeled_indices"])
    if str(icmt_receipt.get("source_labeled_indices_sha256", "")) != reconstructed_labeled_sha:
        raise ICMTSplitExportError(
            "reconstructed labeled-index SHA256 does not equal P1-ICMT terminal receipt"
        )
    reconstructed_manifest_sha = str(
        reconstructed["source_split_receipt"].get("split_manifest_sha256", "")
    )
    if str(icmt_receipt.get("source_split_manifest_sha256", "")) != reconstructed_manifest_sha:
        raise ICMTSplitExportError(
            "reconstructed source-split manifest SHA256 does not equal P1-ICMT terminal receipt"
        )
    source_base = reconstructed["source_base"]
    labeled_indices = reconstructed["labeled_indices"]
    validation_indices = reconstructed["source_validation_indices"]
    unlabeled_indices = reconstructed["unlabeled_indices"]
    labeled_counts = _validate_local4_coverage(
        source_base,
        labeled_indices,
        source_tx_ids,
        role="labeled_fit",
        require_more_than_one=True,
    )
    validation_counts = _validate_local4_coverage(
        source_base,
        validation_indices,
        source_tx_ids,
        role="source_validation_known",
        require_more_than_one=False,
    )
    labeled_physical = _physical_keys_for_indices(source_base, labeled_indices)
    validation_physical = _physical_keys_for_indices(source_base, validation_indices)
    if set(labeled_physical) & set(validation_physical):
        raise ICMTSplitExportError("L/V physical keys overlap")

    dataset_cache = {str(dataset_path): raw_dataset}
    proxy_ds, proxy_info = _build_wisig_dataset(
        pkl_path=str(dataset_path),
        tx_spec=",".join(proxy_unknown_tx_ids),
        role="proxy_unknown",
        equalized=str(checkpoint_args.get("wisig_equalized", "1")),
        out_len=int(checkpoint_args.get("wisig_out_len", 256)),
        domain=str(checkpoint_args.get("wisig_domain", "rx_day")),
        days=",".join(FROZEN_PROXY_DAYS),
        rxs=",".join(FROZEN_PROXY_RXS),
        max_samples_per_combo=0,
        max_samples_per_tx=FROZEN_PROXY_MAX_SAMPLES_PER_TX,
        seed=FROZEN_PROXY_SELECTION_SEED,
        dataset_cache=dataset_cache,
    )
    proxy_physical = _physical_keys_for_indices(proxy_ds, range(len(proxy_ds)))
    proxy_physical_receipt = _physical_key_receipt(proxy_physical)
    if set(labeled_physical) & set(proxy_physical) or set(validation_physical) & set(proxy_physical):
        raise ICMTSplitExportError("L/V/proxy physical keys overlap")

    device = torch.device(str(args.device) if torch.cuda.is_available() else "cpu")
    model, checkpoint_load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint,
        input_len=int(checkpoint_args.get("wisig_out_len", 256)),
        device=device,
    )
    labeled_ds = WiSigSubsetDataset(source_base, labeled_indices, split_source="icmt_labeled_fit")
    validation_ds = WiSigSubsetDataset(
        source_base,
        validation_indices,
        split_source="icmt_source_validation_known",
    )
    labeled_loader = DataLoader(
        labeled_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    proxy_loader = DataLoader(
        proxy_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    labeled_payload = extract_features_with_metadata(
        model,
        labeled_loader,
        device=device,
        feature_name=EXPECTED_FEATURE_NAME,
        role="labeled_fit",
        channel_view="clean",
        satellite_tta_policy="none",
    )
    validation_payload = extract_features_with_metadata(
        model,
        validation_loader,
        device=device,
        feature_name=EXPECTED_FEATURE_NAME,
        role="source_validation_known",
        channel_view="clean",
        satellite_tta_policy="none",
    )
    proxy_payload = extract_features_with_metadata(
        model,
        proxy_loader,
        device=device,
        feature_name=EXPECTED_FEATURE_NAME,
        role="proxy_unknown",
        channel_view="clean",
        satellite_tta_policy="none",
    )
    labeled_payload["source_base_indices"] = np.asarray(labeled_indices, dtype=np.int64)
    validation_payload["source_base_indices"] = np.asarray(validation_indices, dtype=np.int64)
    # Proxy rows do not belong to the local4 source-base index namespace.
    proxy_payload["source_base_indices"] = -1 - np.arange(len(proxy_ds), dtype=np.int64)
    payload = _concat_payloads((labeled_payload, validation_payload, proxy_payload))
    manifest = {
        "schema": "cvs.phase1.icmt_lv_export.v1",
        "feature_name": EXPECTED_FEATURE_NAME,
        "feature_key": EXPECTED_FEATURE_NAME,
        "z_id_source_key": "feat_joint",
        "postfreeze_geometry_path": "checkpoint_model.feat_joint_as_z_id",
        "checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": _sha256_file(checkpoint_path),
        "checkpoint_role": str(checkpoint.get("checkpoint_role")),
        "checkpoint_selection": str(checkpoint.get("checkpoint_selection")),
        "candidate_id": str(checkpoint.get("candidate_id")),
        "run_id": str(checkpoint.get("run_id")),
        "training_run_contract": EXPECTED_TRAINING_RUN_ID,
        "icmt_receipt_schema": EXPECTED_RECEIPT_SCHEMA,
        "icmt_enabled": bool(icmt_receipt.get("enabled")),
        "icmt_source_labeled_indices_sha256": str(
            icmt_receipt.get("source_labeled_indices_sha256")
        ),
        "icmt_source_split_manifest_sha256": str(
            icmt_receipt.get("source_split_manifest_sha256")
        ),
        "classification_head_contract": EXPECTED_CLASSIFICATION_HEAD_CONTRACT,
        "class_id_to_tx": list(source_tx_ids),
        "logit_class_order": list(range(4)),
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": checkpoint_load_audit,
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_mismatch": 0,
        "source_only_export": True,
        "satellite_tta_policy": "none",
        "channel_profile": {
            "labeled_fit": {"view": "clean", "scenarios": []},
            "source_validation_known": {"view": "clean", "scenarios": []},
            "proxy_unknown": {"view": "clean", "scenarios": []},
        },
        "source_tx_ids": list(source_tx_ids),
        "known_validation_outer_tx_ids": list(known_validation_tx_ids),
        "proxy_unknown_tx_ids": list(proxy_unknown_tx_ids),
        "dataset_path": str(dataset_path),
        "wisig_pkl_sha256": dataset_sha256,
        "expected_wisig_pkl_sha256": dataset_sha256_receipt["expected"],
        "checkpoint_declared_wisig_pkl_sha256": expected_dataset_sha256,
        "checkpoint_declared_wisig_pkl_sha256_empty_caveat": dataset_sha256_receipt[
            "checkpoint_declared_empty_caveat"
        ],
        "dataset_path_checkpoint_equal": True,
        "split_mode": EXPECTED_SPLIT_MODE,
        "seed": EXPECTED_SEED,
        "labeled_ratio": EXPECTED_LABELED_RATIO,
        "unlabeled_ratio": EXPECTED_UNLABELED_RATIO,
        "source_val_ratio": EXPECTED_SOURCE_VAL_RATIO,
        "source_split_receipt": dict(reconstructed["source_split_receipt"]),
        "source_split_receipt_checkpoint_equal": True,
        "tx_partition_receipt": dict(reconstructed["tx_partition_receipt"]),
        "tx_partition_receipt_checkpoint_equal": True,
        "labeled_indices_sha256": _index_sha256(labeled_indices),
        "unlabeled_indices_sha256": _index_sha256(unlabeled_indices),
        "source_validation_indices_sha256": _index_sha256(validation_indices),
        "labeled_physical_keys_sha256": _canonical_json_sha256(list(labeled_physical)),
        "source_validation_physical_keys_sha256": _canonical_json_sha256(list(validation_physical)),
        "labeled_source_validation_physical_disjoint": True,
        "labeled_validation_proxy_physical_disjoint": True,
        "labeled_row_count": len(labeled_indices),
        "unlabeled_row_count": len(unlabeled_indices),
        "source_validation_row_count": len(validation_indices),
        "labeled_class_counts": labeled_counts,
        "source_validation_class_counts": validation_counts,
        "proxy_row_count": len(proxy_ds),
        "proxy_physical_keys_sha256": _canonical_json_sha256(list(proxy_physical)),
        "proxy_physical_key_receipt": proxy_physical_receipt,
        "proxy_export_info": proxy_info,
        "proxy_days": ",".join(FROZEN_PROXY_DAYS),
        "proxy_rxs": ",".join(FROZEN_PROXY_RXS),
        "proxy_seed": FROZEN_PROXY_SELECTION_SEED,
        "proxy_max_samples_per_tx": FROZEN_PROXY_MAX_SAMPLES_PER_TX,
        "proxy_expected_total_count": FROZEN_PROXY_TOTAL_COUNT,
        "proxy_selection": {
            **proxy_selection,
            "selection_sha256": _canonical_json_sha256(proxy_selection),
        },
        "forwarded_roles": ["labeled_fit", "source_validation_known", "proxy_unknown"],
        "unlabeled_loader_constructed": False,
        "unlabeled_loader_rows": 0,
        "unlabeled_forward_rows": 0,
        "unlabeled_features_persisted": False,
    }
    payload["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True))
    _atomic_save_npz(output_path, payload)
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
    parser.add_argument("--proxy_days", default=",".join(FROZEN_PROXY_DAYS))
    parser.add_argument("--proxy_rxs", default=",".join(FROZEN_PROXY_RXS))
    parser.add_argument(
        "--max_proxy_samples_per_tx",
        type=int,
        default=FROZEN_PROXY_MAX_SAMPLES_PER_TX,
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = export(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
