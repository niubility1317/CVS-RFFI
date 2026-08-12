"""CLIC clean source-L export sealing and terminal-envelope reopening.

The exporter validates the current final-only checkpoint together with its
versioned external terminal envelope before any source-L feature artifact can
be accepted.  It intentionally contains no source-V, proxy, target, receiver,
or query fitting path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from cvsrffi import phase1_clic as _clic


EXPECTED_TRAINING_RUN_ID = "phase1_clic12_20260811_v1"
EXPECTED_LV_EXPORT_SCHEMA = "cvs.phase1.clic_lv_export.v1"
EXPECTED_CHECKPOINT_ROLE = "training_final_only"
EXPECTED_CHECKPOINT_SELECTION = "final_only"
EXPECTED_CANDIDATE_PATTERN = re.compile(r"^F([1-6])([CG])_CLIC12$")
EXPECTED_METHOD = "P1_CLIC"
EXPECTED_HEAD_FEATURE_KEY = "z_id"
FROZEN_WISIG_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
FROZEN_PROXY_DAYS = ("2021_03_01", "2021_03_08")
FROZEN_PROXY_RXS = ("1-1", "1-19", "14-7", "18-2", "19-2", "2-1")
FROZEN_PROXY_SELECTION_SEED = 7281148
FROZEN_PROXY_MAX_SAMPLES_PER_TX = 400
FROZEN_PROXY_TOTAL_COUNT = 400


class CLICSplitExportError(RuntimeError):
    """Raised when a clean CLIC postfreeze export cannot be reopened safely."""


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_source_ids(values: Sequence[str], *, label: str, expected_count: int) -> tuple[str, ...]:
    parsed = tuple(str(value) for value in values)
    if len(parsed) != expected_count or len(set(parsed)) != expected_count or any(not value for value in parsed):
        raise CLICSplitExportError(f"{label} must contain exactly {expected_count} unique TX IDs")
    return parsed


def _require_bool(value: Any, *, label: str, expected: bool) -> None:
    if type(value) is not bool or value is not expected:
        raise CLICSplitExportError(f"{label} drifted")


def _require_close(value: Any, *, label: str, expected: float) -> None:
    if isinstance(value, bool):
        raise CLICSplitExportError(f"{label} must not be boolean")
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise CLICSplitExportError(f"{label} is invalid") from exc
    if observed != expected:
        raise CLICSplitExportError(f"{label} drifted")


def _reject_forbidden_terminal_state(value: Any, *, label: str) -> None:
    """Reject historical and row-level side channels before core validation."""

    legacy_prefixes = ("icmt_", "hnccd_", "hscf_", "rcmmc_", "rcat_", "recte_", "rcrmd_", "cagm_")
    forbidden_exact = {
        "source_receiver_ids", "receiver_ids", "target_rows", "target_ids", "raw_iq", "clean_iq",
        "sample_ids", "sample_features", "sample_logits", "proxy_rows", "proxy_ids",
    }
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower()
            if normalized.startswith(legacy_prefixes):
                raise CLICSplitExportError(f"legacy historical receipt field is forbidden in {label}: {key}")
            if normalized in forbidden_exact:
                raise CLICSplitExportError(f"forbidden raw/target/sample/receiver state in {label}: {key}")
            _reject_forbidden_terminal_state(item, label=label)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_terminal_state(item, label=label)


def _load_terminal_envelope(path: str | Path) -> dict[str, Any]:
    terminal = Path(path).resolve()
    if not terminal.is_file():
        raise CLICSplitExportError("CLIC versioned terminal receipt is missing")
    try:
        envelope = json.loads(terminal.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CLICSplitExportError("CLIC versioned terminal receipt is invalid") from exc
    if not isinstance(envelope, dict):
        raise CLICSplitExportError("CLIC versioned terminal receipt must be an object")
    _reject_forbidden_terminal_state(envelope, label="CLIC terminal envelope")
    try:
        return dict(_clic.validate_clic_terminal_envelope(envelope))
    except _clic.CLICTerminalError as exc:
        raise CLICSplitExportError(f"CLIC terminal receipt failed strict reopening: {exc}") from exc


def _validate_checkpoint_args(
    args: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    source_tx_ids: tuple[str, ...],
    known_validation_tx_ids: tuple[str, ...],
    proxy_unknown_tx_ids: tuple[str, ...],
) -> str:
    candidate_match = EXPECTED_CANDIDATE_PATTERN.fullmatch(checkpoint_path.parent.name)
    if candidate_match is None or checkpoint_path.name != "final_ssdg.pth":
        raise CLICSplitExportError("checkpoint must be a frozen CLIC F1..F6 final-only candidate")
    arm = candidate_match.group(2)
    expected_text = {
        "split_mode": "tx_rx_day_1_6_3",
        "model_variant": "lite_d",
        "id_feature_key": EXPECTED_HEAD_FEATURE_KEY,
        "phase1_source_train_tx_ids": ",".join(source_tx_ids),
        "phase1_source_known_validation_tx_ids": ",".join(known_validation_tx_ids),
        "phase1_source_proxy_unknown_tx_ids": ",".join(proxy_unknown_tx_ids),
        "checkpoint_selection": EXPECTED_CHECKPOINT_SELECTION,
        "candidate_id": checkpoint_path.parent.name,
        "run_id": EXPECTED_TRAINING_RUN_ID,
    }
    for field, expected in expected_text.items():
        if args.get(field) != expected:
            raise CLICSplitExportError(f"checkpoint arg {field} drifted")
    for field, expected in (("labeled_ratio", 0.07), ("unlabeled_ratio", 0.63), ("source_val_ratio", 0.30)):
        _require_close(args.get(field), label=f"checkpoint arg {field}", expected=expected)
    if type(args.get("seed")) is not int or int(args["seed"]) != int(_clic.CLIC_INIT_SEED):
        raise CLICSplitExportError("checkpoint seed drifted")
    _require_bool(args.get("phase1_clic_frozen_mode"), label="CLIC frozen mode", expected=True)
    _require_bool(args.get("phase1_clic_enabled"), label="CLIC enabled arm binding", expected=arm == "G")
    expected_operator = "complex_local_invariant_curvature" if arm == "G" else "raw_phase_control"
    if args.get("phase1_clic_operator_mode") != expected_operator:
        raise CLICSplitExportError("CLIC operator mode does not bind C/G arm")
    return arm


def validate_clic_training_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    terminal_receipt_path: str | Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Reopen both current checkpoint and external terminal digest, fail closed on drift."""

    if not isinstance(checkpoint, Mapping):
        raise CLICSplitExportError("CLIC checkpoint must be a mapping")
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise CLICSplitExportError("CLIC final checkpoint is missing")
    source = _parse_source_ids(source_tx_ids, label="source-L TX IDs", expected_count=4)
    known = _parse_source_ids(known_validation_tx_ids, label="known validation TX IDs", expected_count=1)
    proxy = _parse_source_ids(proxy_unknown_tx_ids, label="proxy TX IDs", expected_count=1)
    if set(source) & set(known) or set(source) & set(proxy) or set(known) & set(proxy):
        raise CLICSplitExportError("source-L, held validation, and proxy TX roles must be disjoint")
    if checkpoint.get("checkpoint_role") != EXPECTED_CHECKPOINT_ROLE or checkpoint.get("checkpoint_selection") != EXPECTED_CHECKPOINT_SELECTION:
        raise CLICSplitExportError("CLIC checkpoint must be final-only")
    args = checkpoint.get("args")
    if not isinstance(args, Mapping) or not isinstance(checkpoint.get("model"), Mapping):
        raise CLICSplitExportError("CLIC checkpoint lacks args/model mappings")
    arm = _validate_checkpoint_args(
        args, checkpoint_path=path, source_tx_ids=source, known_validation_tx_ids=known, proxy_unknown_tx_ids=proxy
    )
    if checkpoint.get("candidate_id") != path.parent.name or checkpoint.get("run_id") != EXPECTED_TRAINING_RUN_ID:
        raise CLICSplitExportError("CLIC checkpoint candidate/run binding drifted")
    if path.parent.parent.name != EXPECTED_TRAINING_RUN_ID:
        raise CLICSplitExportError("CLIC checkpoint is outside the frozen training run root")
    pre = checkpoint.get("clic_receipt_precheckpoint")
    if not isinstance(pre, Mapping) or pre.get("completed") is not False:
        raise CLICSplitExportError("CLIC checkpoint precheckpoint receipt is absent or not pre-terminal")
    _reject_forbidden_terminal_state(pre, label="CLIC checkpoint precheckpoint receipt")
    envelope = _load_terminal_envelope(terminal_receipt_path)
    actual_sha = _sha256_file(path)
    if Path(str(envelope["selected_checkpoint_path"])).resolve() != path:
        raise CLICSplitExportError("CLIC terminal envelope selected checkpoint path drifted")
    if envelope["selected_checkpoint_sha256"] != actual_sha:
        raise CLICSplitExportError("CLIC terminal envelope checkpoint SHA does not match current bytes")
    receipt = dict(envelope["strict_core"])
    if receipt.get("arm") != arm or receipt.get("source_l_only") is not True:
        raise CLICSplitExportError("CLIC terminal receipt arm/source-L-only contract drifted")
    if receipt.get("final_checkpoint_sha256") != actual_sha:
        raise CLICSplitExportError("CLIC terminal receipt final checkpoint SHA does not match current bytes")
    return dict(args), receipt, arm


def build_clean_export_manifest(
    *,
    checkpoint: Mapping[str, Any],
    checkpoint_path: str | Path,
    terminal_receipt_path: str | Path,
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
) -> dict[str, Any]:
    """Produce only a small source-L export manifest; no rows or features enter it."""

    args, receipt, arm = validate_clic_training_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        terminal_receipt_path=terminal_receipt_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_validation_tx_ids,
        proxy_unknown_tx_ids=proxy_unknown_tx_ids,
    )
    checkpoint_file = Path(checkpoint_path).resolve()
    terminal_file = Path(terminal_receipt_path).resolve()
    return {
        "schema": EXPECTED_LV_EXPORT_SCHEMA,
        "method": EXPECTED_METHOD,
        "source_only": True,
        "candidate_id": str(args["candidate_id"]),
        "run_id": EXPECTED_TRAINING_RUN_ID,
        "training_run_contract": EXPECTED_TRAINING_RUN_ID,
        "checkpoint": str(checkpoint_file),
        "source_checkpoint_sha256": _sha256_file(checkpoint_file),
        "terminal_receipt_sha256": _sha256_file(terminal_file),
        "clic_receipt_schema": "cvs.phase1.clic_receipt.v1",
        "clic_terminal_contract": str(receipt["terminal_contract"]),
        "clic_terminal_contract_passed": True,
        "clic_enabled": arm == "G",
        "z_id_source_key": EXPECTED_HEAD_FEATURE_KEY,
        "source_tx_ids": [str(item) for item in source_tx_ids],
        "known_validation_tx_ids": [str(item) for item in known_validation_tx_ids],
        "proxy_unknown_tx_ids": [str(item) for item in proxy_unknown_tx_ids],
        "proxy_selection_frozen_not_cli_tunable": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
    }


def _canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CLICSplitExportError("cannot canonicalize CLIC source split state") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _physical_keys_for_indices(base: Any, indices: Sequence[int]) -> tuple[str, ...]:
    eq_list = getattr(base, "eq_list", None)
    if eq_list is None and hasattr(base, "base"):
        eq_list = getattr(base.base, "eq_list", None)
    if eq_list is None:
        raise CLICSplitExportError("CLIC source dataset lacks equalized-axis labels")
    keys: list[str] = []
    for value in indices:
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
    if len(keys) != len(set(keys)):
        raise CLICSplitExportError("CLIC source physical selection contains duplicates")
    return tuple(keys)


def _reconstruct_source_l_v(
    *,
    raw_dataset: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
    source_tx_ids: Sequence[str],
    known_validation_tx_ids: Sequence[str],
    proxy_unknown_tx_ids: Sequence[str],
    wisig_sha256: str,
) -> dict[str, Any]:
    """Rebuild only source local4 indices; U is never materialized as a loader."""

    from SSDG.train_ssdg import (
        _build_source_split_receipt,
        _parse_wisig_axis_spec,
        _phase1_tx_partition_view,
        split_tx_rx_day_1_6_3,
    )
    from dataset_wisig import WiSigCompactDataset, _resolve_days, _resolve_rxs

    filtered, partition = _phase1_tx_partition_view(
        raw_dataset,
        train_spec=",".join(source_tx_ids),
        known_validation_spec=",".join(known_validation_tx_ids),
        proxy_unknown_spec=",".join(proxy_unknown_tx_ids),
    )
    days = list(filtered.get("capture_date_list", []))
    rxs = list(filtered.get("rx_list", []))
    train_days = _resolve_days(days, _parse_wisig_axis_spec(checkpoint_args.get("wisig_train_days", "")), list(range(min(3, len(days)))))
    test_days = _resolve_days(days, _parse_wisig_axis_spec(checkpoint_args.get("wisig_test_days", "")), [len(days) - 1])
    train_rxs = _resolve_rxs(rxs, _parse_wisig_axis_spec(checkpoint_args.get("wisig_train_rxs", "")), list(range(len(rxs))))
    test_rxs = _resolve_rxs(rxs, _parse_wisig_axis_spec(checkpoint_args.get("wisig_test_rxs", "")), [])
    train_days = [int(item) for item in train_days if item not in test_days]
    train_rxs = [int(item) for item in train_rxs if item not in test_rxs]
    if not train_days or not train_rxs:
        raise CLICSplitExportError("CLIC source-L reconstruction has no legal source day/RX rows")
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
        max_samples_per_combo=(None if int(checkpoint_args.get("wisig_max_day123_per_combo", 0)) <= 0 else int(checkpoint_args["wisig_max_day123_per_combo"])),
        seed=int(checkpoint_args["seed"]),
        build_index=True,
    )
    labeled, unlabeled, validation = split_tx_rx_day_1_6_3(
        source_base,
        labeled_ratio=float(checkpoint_args["labeled_ratio"]),
        unlabeled_ratio=float(checkpoint_args["unlabeled_ratio"]),
        source_val_ratio=float(checkpoint_args["source_val_ratio"]),
    )
    split_sets = (set(labeled), set(unlabeled), set(validation))
    if any(split_sets[i] & split_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise CLICSplitExportError("CLIC reconstructed L/U/V physical indices overlap")
    if set().union(*split_sets) != set(range(len(source_base))):
        raise CLICSplitExportError("CLIC reconstructed L/U/V indices do not cover source base")
    receipt = _build_source_split_receipt(
        seed=int(checkpoint_args["seed"]),
        split_mode=str(checkpoint_args["split_mode"]),
        source_days=train_days,
        target_days=test_days,
        source_receivers=train_rxs,
        target_receivers=test_rxs,
        labeled_indices=labeled,
        unlabeled_indices=unlabeled,
        source_validation_indices=validation,
        wisig_pkl_sha256=wisig_sha256,
        requested_labeled_ratio=float(checkpoint_args["labeled_ratio"]),
        requested_unlabeled_ratio=float(checkpoint_args["unlabeled_ratio"]),
        requested_source_val_ratio=float(checkpoint_args["source_val_ratio"]),
        realized_rho_tolerance=float(checkpoint_args.get("phase1_realized_rho_tolerance", 0.002)),
        realized_source_val_tolerance=float(checkpoint_args.get("phase1_realized_source_val_tolerance", 0.002)),
    )
    return {
        "source_base": source_base,
        "labeled_indices": tuple(int(item) for item in labeled),
        "unlabeled_indices": tuple(int(item) for item in unlabeled),
        "validation_indices": tuple(int(item) for item in validation),
        "source_split_receipt": receipt,
        "tx_partition_receipt": partition,
    }


def _assert_current_source_split(
    *, checkpoint: Mapping[str, Any], receipt: Mapping[str, Any],
    reconstructed: Mapping[str, Any], source_tx_ids: Sequence[str],
) -> None:
    split_info = checkpoint.get("split_info")
    if not isinstance(split_info, Mapping):
        raise CLICSplitExportError("CLIC checkpoint lacks source split receipt")
    if split_info.get("source_split_receipt") != reconstructed["source_split_receipt"]:
        raise CLICSplitExportError("CLIC reconstructed source split receipt does not equal checkpoint")
    if split_info.get("tx_partition_receipt") != reconstructed["tx_partition_receipt"]:
        raise CLICSplitExportError("CLIC reconstructed TX partition receipt does not equal checkpoint")
    labeled = tuple(reconstructed["labeled_indices"])
    if int(receipt.get("source_split_count", -1)) != len(labeled):
        raise CLICSplitExportError("CLIC terminal source-L count does not equal reconstructed split")
    if str(receipt.get("source_split_sha256", "")) != _canonical_json_sha256(list(labeled)):
        raise CLICSplitExportError("CLIC terminal source-L index SHA does not equal reconstructed split")
    class_order = [str(item) for item in getattr(reconstructed["source_base"], "tx_list", ())]
    if tuple(class_order) != tuple(source_tx_ids):
        raise CLICSplitExportError("CLIC reconstructed local class order drifted")
    if str(receipt.get("class_order_sha256", "")) != _canonical_json_sha256(class_order):
        raise CLICSplitExportError("CLIC terminal class-order SHA does not equal reconstructed split")
    if str(receipt.get("physical_order_sha256", "")) != _canonical_json_sha256(list(labeled)):
        raise CLICSplitExportError("CLIC terminal source-L physical-order SHA does not equal reconstructed split")


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLICSplitExportError(f"refusing to overwrite immutable CLIC clean export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICSplitExportError(f"refusing to overwrite temporary CLIC clean export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **dict(payload))
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Write a real immutable source L/V/fixed400 proxy feature NPZ.

    Geometry and tail fitting remain outside this exporter.  The only source
    rows that can later fit geometry are labelled ``labeled_fit`` rows; V and
    proxy rows are forwarded for score-only diagnostics and carry zero fit and
    threshold-fit counters in the manifest.
    """

    from torch.utils.data import DataLoader
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    from dataset_wisig import WiSigSubsetDataset, load_wisig_compact_pkl
    from export_spaceborne_features import _build_wisig_dataset, _concat_payloads, extract_features_with_metadata

    checkpoint_path = Path(args.ckpt).resolve()
    terminal_path = Path(args.terminal_receipt_json).resolve()
    dataset_path = Path(args.wisig_pkl).resolve()
    output_path = Path(args.out_npz).resolve()
    if not dataset_path.is_file():
        raise CLICSplitExportError("CLIC clean export WiSig dataset is missing")
    source = _parse_csv(args.source_tx_ids, label="source TX IDs")
    known = _parse_csv(args.known_validation_tx_ids, label="known validation TX IDs")
    proxy = _parse_csv(args.proxy_unknown_tx_ids, label="proxy TX IDs")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICSplitExportError("cannot load CLIC final checkpoint") from exc
    if not isinstance(checkpoint, Mapping):
        raise CLICSplitExportError("CLIC final checkpoint payload is malformed")
    checkpoint_args, terminal_receipt, arm = validate_clic_training_checkpoint(
        checkpoint,
        checkpoint_path=checkpoint_path,
        terminal_receipt_path=terminal_path,
        source_tx_ids=source,
        known_validation_tx_ids=known,
        proxy_unknown_tx_ids=proxy,
    )
    if Path(str(checkpoint_args.get("wisig_pkl", ""))).resolve() != dataset_path:
        raise CLICSplitExportError("CLIC clean export dataset path does not match checkpoint args")
    dataset_sha = _sha256_file(dataset_path)
    if str(args.expected_wisig_sha256).lower() != FROZEN_WISIG_SHA256 or dataset_sha != FROZEN_WISIG_SHA256:
        raise CLICSplitExportError("CLIC clean export WiSig bytes do not equal the frozen SHA256")
    declared_sha = str(checkpoint_args.get("wisig_pkl_sha256", "") or FROZEN_WISIG_SHA256).lower()
    if declared_sha != FROZEN_WISIG_SHA256:
        raise CLICSplitExportError("CLIC checkpoint WiSig SHA declaration drifted")
    raw_dataset = load_wisig_compact_pkl(str(dataset_path))
    reconstructed = _reconstruct_source_l_v(
        raw_dataset=raw_dataset,
        checkpoint_args=checkpoint_args,
        source_tx_ids=source,
        known_validation_tx_ids=known,
        proxy_unknown_tx_ids=proxy,
        wisig_sha256=str(checkpoint_args.get("wisig_pkl_sha256", "")),
    )
    _assert_current_source_split(
        checkpoint=checkpoint, receipt=terminal_receipt, reconstructed=reconstructed, source_tx_ids=source
    )
    proxy_days = _parse_csv(getattr(args, "proxy_days", ",".join(FROZEN_PROXY_DAYS)), label="proxy days")
    proxy_rxs = _parse_csv(getattr(args, "proxy_rxs", ",".join(FROZEN_PROXY_RXS)), label="proxy RXs")
    if (
        proxy_days != FROZEN_PROXY_DAYS
        or proxy_rxs != FROZEN_PROXY_RXS
        or int(getattr(args, "max_proxy_samples_per_tx", FROZEN_PROXY_MAX_SAMPLES_PER_TX)) != FROZEN_PROXY_MAX_SAMPLES_PER_TX
    ):
        raise CLICSplitExportError("fixed400 proxy selection arguments drifted")
    proxy_ds, proxy_info = _build_wisig_dataset(
        pkl_path=str(dataset_path),
        tx_spec=",".join(proxy),
        role="proxy_unknown",
        equalized=str(checkpoint_args.get("wisig_equalized", "1")),
        out_len=int(checkpoint_args.get("wisig_out_len", 256)),
        domain=str(checkpoint_args.get("wisig_domain", "rx_day")),
        days=",".join(FROZEN_PROXY_DAYS),
        rxs=",".join(FROZEN_PROXY_RXS),
        max_samples_per_combo=0,
        max_samples_per_tx=FROZEN_PROXY_MAX_SAMPLES_PER_TX,
        seed=FROZEN_PROXY_SELECTION_SEED,
        dataset_cache={str(dataset_path): raw_dataset},
    )
    if len(proxy_ds) != FROZEN_PROXY_TOTAL_COUNT:
        raise CLICSplitExportError("fixed400 proxy selection does not close")
    labeled = tuple(reconstructed["labeled_indices"])
    validation = tuple(reconstructed["validation_indices"])
    source_base = reconstructed["source_base"]
    labeled_keys = _physical_keys_for_indices(source_base, labeled)
    validation_keys = _physical_keys_for_indices(source_base, validation)
    proxy_keys = _physical_keys_for_indices(proxy_ds, range(len(proxy_ds)))
    if set(labeled_keys) & set(validation_keys) or set(labeled_keys) & set(proxy_keys) or set(validation_keys) & set(proxy_keys):
        raise CLICSplitExportError("CLIC clean L/V/proxy physical rows overlap")
    device = torch.device(str(getattr(args, "device", "cuda:0")) if torch.cuda.is_available() else "cpu")
    model, load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint, input_len=int(checkpoint_args.get("wisig_out_len", 256)), device=device
    )
    batch_size = int(getattr(args, "batch_size", 256))
    if batch_size <= 0:
        raise CLICSplitExportError("CLIC clean export batch size is invalid")
    loaders = (
        (WiSigSubsetDataset(source_base, labeled, split_source="clic_source_labeled_fit"), "labeled_fit"),
        (WiSigSubsetDataset(source_base, validation, split_source="clic_source_validation_known"), "source_validation_known"),
        (proxy_ds, "proxy_unknown"),
    )
    payloads: list[dict[str, np.ndarray]] = []
    for dataset, role in loaders:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
        payloads.append(
            extract_features_with_metadata(
                model,
                loader,
                device=device,
                feature_name=EXPECTED_HEAD_FEATURE_KEY,
                role=role,
                channel_view="clean",
                satellite_tta_policy="none",
            )
        )
    payload = _concat_payloads(payloads)
    payload["z_id"] = np.asarray(payload["features"], dtype=np.float32)
    manifest = build_clean_export_manifest(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        terminal_receipt_path=terminal_path,
        source_tx_ids=source,
        known_validation_tx_ids=known,
        proxy_unknown_tx_ids=proxy,
    )
    manifest.update(
        {
            "feature_name": EXPECTED_HEAD_FEATURE_KEY,
            "feature_key": EXPECTED_HEAD_FEATURE_KEY,
            "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
            "checkpoint_load_strict": True,
            "checkpoint_load_audit": load_audit,
            "wisig_pkl_sha256": dataset_sha,
            "source_labeled_indices_sha256": _canonical_json_sha256(list(labeled)),
            "source_validation_indices_sha256": _canonical_json_sha256(list(validation)),
            "source_labeled_physical_order_sha256": _canonical_json_sha256(list(labeled_keys)),
            "source_validation_physical_order_sha256": _canonical_json_sha256(list(validation_keys)),
            "labeled_validation_physical_disjoint": True,
            "labeled_validation_proxy_physical_disjoint": True,
            "labeled_row_count": len(labeled),
            "source_validation_row_count": len(validation),
            "proxy_row_count": len(proxy_ds),
            "proxy_physical_order_sha256": _canonical_json_sha256(list(proxy_keys)),
            "proxy_selection": {
                "days": list(FROZEN_PROXY_DAYS),
                "rxs": list(FROZEN_PROXY_RXS),
                "seed": FROZEN_PROXY_SELECTION_SEED,
                "max_samples_per_tx": FROZEN_PROXY_MAX_SAMPLES_PER_TX,
                "total_count": FROZEN_PROXY_TOTAL_COUNT,
            },
            "proxy_export_info": proxy_info,
            "forwarded_roles": ["labeled_fit", "source_validation_known", "proxy_unknown"],
            "unlabeled_loader_constructed": False,
            "unlabeled_forward_rows": 0,
            "geometry_fit_role": "labeled_fit_only",
            "validation_proxy_fit_rows": 0,
            "validation_proxy_threshold_rows": 0,
            "clic_enabled": arm == "G",
        }
    )
    payload["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    _atomic_save_npz(output_path, payload)
    return {"out_npz": str(output_path), "manifest": manifest}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--terminal-receipt-json", required=True)
    parser.add_argument("--wisig-pkl", "--wisig_pkl", dest="wisig_pkl", required=True)
    parser.add_argument("--out-npz", "--out_npz", dest="out_npz", required=True)
    parser.add_argument("--source-tx-ids", "--source_tx_ids", dest="source_tx_ids", required=True)
    parser.add_argument("--known-validation-tx-ids", "--known_validation_tx_ids", dest="known_validation_tx_ids", required=True)
    parser.add_argument("--proxy-unknown-tx-ids", "--proxy_unknown_tx_ids", dest="proxy_unknown_tx_ids", required=True)
    parser.add_argument("--expected-wisig-sha256", required=True)
    parser.add_argument("--proxy-days", default=",".join(FROZEN_PROXY_DAYS))
    parser.add_argument("--proxy-rxs", default=",".join(FROZEN_PROXY_RXS))
    parser.add_argument("--max-proxy-samples-per-tx", type=int, default=FROZEN_PROXY_MAX_SAMPLES_PER_TX)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def _parse_csv(value: str, *, label: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not parsed:
        raise CLICSplitExportError(f"{label} is empty")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
