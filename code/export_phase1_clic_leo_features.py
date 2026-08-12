"""Bind existing source-L received-IQ bytes for CLIC postfreeze LEO scoring.

This module does not build a channel view, resample rows, or fit a policy.  It
only proves that a C or G postfreeze reader is attached to one pre-existing
single-LEO observation table with the formal three-scene and physical-order
closure required by the frozen design.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import export_phase1_clic_features as _clean
from cvsrffi import phase1_clic as _clic


EXPECTED_BINDING_SCHEMA = "cvs.phase1.clic_leo_binding.v1"
EXPECTED_TRAINING_RUN_LEAF = "phase1_clic12_20260812_v5"
EXPECTED_SCENARIOS = tuple(_clic.FORMAL_LEO_WEAK_SCENARIOS)
EXPECTED_METHOD = "P1_CLIC"
EXPECTED_CACHE_RUN_LEAF = "phase1_clic_source_leo_20260812_v3"
EXPECTED_EXPORT_RUN_LEAF = "phase1_clic_source_leo_20260812_v4"
EXPECTED_CACHE_SCHEMA = "cvs.phase1.clic_source_leo_received_iq.v1"


class CLICLEOBindingError(RuntimeError):
    """Raised when existing received-IQ bytes cannot close the CLIC LEO contract."""


def _numpy_float32_to_tensor(value: np.ndarray) -> torch.Tensor:
    """Cross the NumPy/Torch boundary without the legacy ndarray C API."""

    source = np.ascontiguousarray(value, dtype=np.float32)
    if source.size <= 0 or not np.isfinite(source).all():
        raise CLICLEOBindingError("existing received-IQ tensor row is empty or non-finite")
    try:
        # N607 uses Torch 2.1 with NumPy 2.x.  The buffer protocol remains
        # compatible there; clone immediately detaches from the NPZ snapshot.
        tensor = torch.frombuffer(memoryview(source), dtype=torch.float32)
        return tensor.reshape(source.shape).clone()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise CLICLEOBindingError("existing received-IQ NumPy conversion failed") from exc


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise CLICLEOBindingError("cannot canonicalize CLIC LEO binding") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_csv(value: str, *, label: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if not parsed:
        raise CLICLEOBindingError(f"{label} is empty")
    return parsed


def _validate_sealed_source_leo_cache_asset(
    args: argparse.Namespace,
    *,
    received_path: Path,
    source_tx_ids: tuple[str, ...],
    fold: int,
    arm: str,
    candidate: str,
    checkpoint: Path,
    terminal: Path,
    terminal_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Reopen the exact immutable v3 cache receipt for the v4 export-only run."""

    if getattr(args, "require_sealed_source_leo_cache", False) is not True:
        return {}
    training_root = Path(args.training_run_root).resolve()
    cache_root = Path(args.cache_run_root).resolve()
    output_root = Path(args.postfreeze_output_root).resolve()
    if (
        training_root.name != EXPECTED_TRAINING_RUN_LEAF
        or cache_root.name != EXPECTED_CACHE_RUN_LEAF
        or output_root.name != EXPECTED_EXPORT_RUN_LEAF
        or not (training_root.parent == cache_root.parent == output_root.parent)
    ):
        raise CLICLEOBindingError("sealed source-LEO training/cache/output root drifted")
    expected_dir = cache_root / f"F{fold}_SHARED"
    expected_npz = expected_dir / "source_l_received_iq.npz"
    receipt_path = Path(args.existing_received_iq_receipt_json).resolve()
    expected_receipt = expected_dir / "source_l_received_iq.receipt.json"
    if received_path != expected_npz or receipt_path != expected_receipt or not receipt_path.is_file():
        raise CLICLEOBindingError("sealed source-LEO cache/receipt path drifted")
    if (
        Path(args.out_npz).resolve().parent != output_root / candidate
        or Path(args.binding_json).resolve().parent != output_root / candidate
    ):
        raise CLICLEOBindingError("sealed source-LEO output candidate root drifted")
    receipt_sha_before = _sha256_file(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CLICLEOBindingError("sealed source-LEO cache receipt is unreadable") from exc
    if not isinstance(receipt, Mapping):
        raise CLICLEOBindingError("sealed source-LEO cache receipt is malformed")
    expected_scalars = {
        "schema": EXPECTED_CACHE_SCHEMA,
        "method": EXPECTED_METHOD,
        "fold_index": fold,
        "source_only": True,
        "source_l_only": True,
        "same_received_iq_bytes_for_c_and_g": True,
        "source_row_count": 3920,
        "target_access": False,
        "query_access": False,
        "held_validation_forward_rows": 0,
        "proxy_forward_rows": 0,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
    }
    for field, expected in expected_scalars.items():
        if receipt.get(field) != expected or type(receipt.get(field)) is not type(expected):
            raise CLICLEOBindingError(f"sealed source-LEO cache receipt {field} drifted")
    if tuple(receipt.get("formal_scenarios", ())) != EXPECTED_SCENARIOS:
        raise CLICLEOBindingError("sealed source-LEO cache receipt scenario order drifted")
    if tuple(receipt.get("source_tx_ids", ())) != source_tx_ids:
        raise CLICLEOBindingError("sealed source-LEO cache receipt source TX order drifted")
    if receipt.get("source_split_sha256") != terminal_receipt.get("source_split_sha256"):
        raise CLICLEOBindingError("sealed source-LEO cache/terminal source split SHA drifted")
    checkpoint_hashes = receipt.get("checkpoint_sha256_by_arm")
    terminal_hashes = receipt.get("terminal_receipt_sha256_by_arm")
    if not isinstance(checkpoint_hashes, Mapping) or set(checkpoint_hashes) != {"C", "G"}:
        raise CLICLEOBindingError("sealed source-LEO cache checkpoint hash map drifted")
    if not isinstance(terminal_hashes, Mapping) or set(terminal_hashes) != {"C", "G"}:
        raise CLICLEOBindingError("sealed source-LEO cache terminal hash map drifted")
    if checkpoint_hashes.get(arm) != _sha256_file(checkpoint):
        raise CLICLEOBindingError("sealed source-LEO cache/current checkpoint SHA drifted")
    if terminal_hashes.get(arm) != _sha256_file(terminal):
        raise CLICLEOBindingError("sealed source-LEO cache/current terminal SHA drifted")
    if Path(str(receipt.get("received_iq_npz_path", ""))).resolve() != received_path:
        raise CLICLEOBindingError("sealed source-LEO cache receipt NPZ path drifted")
    observed_sha = _sha256_file(received_path)
    if receipt.get("received_iq_npz_sha256") != observed_sha:
        raise CLICLEOBindingError("sealed source-LEO cache receipt NPZ SHA drifted")
    if _sha256_file(receipt_path) != receipt_sha_before:
        raise CLICLEOBindingError("sealed source-LEO cache receipt changed during reopen")
    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha_before,
        "npz_sha256": observed_sha,
        "source_row_count": 3920,
    }


def _validate_args(args: argparse.Namespace) -> tuple[Path, Path, Path, tuple[str, ...], str, int, str]:
    fold = int(args.fold_index)
    arm = str(args.arm).upper()
    if fold not in range(1, 7) or arm not in {"C", "G"}:
        raise CLICLEOBindingError("CLIC LEO binding requires F1..F6 and C/G")
    candidate = f"F{fold}{arm}_CLIC12"
    if str(args.candidate_id) != candidate:
        raise CLICLEOBindingError("CLIC LEO candidate does not bind fold/arm")
    checkpoint = Path(args.ckpt).resolve()
    terminal = Path(args.terminal_receipt_json).resolve()
    received = Path(args.existing_received_iq_npz).resolve()
    training_root = Path(args.training_run_root).resolve()
    if not checkpoint.is_file() or not terminal.is_file() or not received.is_file():
        raise CLICLEOBindingError("CLIC LEO checkpoint, terminal, or existing received-IQ input is missing")
    if training_root.name != EXPECTED_TRAINING_RUN_LEAF or checkpoint.parent.parent != training_root:
        raise CLICLEOBindingError("CLIC LEO checkpoint does not bind the frozen training run root")
    expected_checkpoint = training_root / candidate / "final_ssdg.pth"
    if checkpoint != expected_checkpoint:
        raise CLICLEOBindingError("CLIC LEO checkpoint does not bind frozen candidate path")
    source_tx_ids = _parse_csv(args.source_tx_ids, label="source TX IDs")
    if len(source_tx_ids) != 4 or len(set(source_tx_ids)) != 4:
        raise CLICLEOBindingError("CLIC LEO binding requires exactly four source-L TX IDs")
    return checkpoint, terminal, received, source_tx_ids, arm, fold, candidate


def _load_existing_received_iq(path: Path, *, source_tx_ids: tuple[str, ...]) -> tuple[dict[str, np.ndarray], list[str], dict[str, dict[str, Any]]]:
    required = ("received_iq", "tx_ids", "rx_ids", "day_ids", "physical_sample_id", "sat_scenarios")
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(required):
                raise CLICLEOBindingError("existing received-IQ NPZ member set drifted")
            arrays = {name: np.array(archive[name], copy=True) for name in required}
    except (OSError, ValueError) as exc:
        if isinstance(exc, CLICLEOBindingError):
            raise
        raise CLICLEOBindingError("existing received-IQ NPZ is unreadable") from exc
    received = arrays["received_iq"]
    if received.ndim < 2 or received.shape[0] <= 0 or not np.isfinite(received).all():
        raise CLICLEOBindingError("existing received-IQ rows are non-finite or malformed")
    row_count = int(received.shape[0])
    text_arrays: dict[str, np.ndarray] = {}
    for name in required[1:]:
        values = np.asarray(arrays[name]).reshape(-1)
        if values.size != row_count:
            raise CLICLEOBindingError(f"existing received-IQ {name} does not align with rows")
        text_arrays[name] = np.asarray([str(value) for value in values], dtype=str)
    tx_ids = text_arrays["tx_ids"]
    if set(tx_ids).difference(source_tx_ids):
        raise CLICLEOBindingError("existing received-IQ contains non-source or target TX rows")
    scenes = text_arrays["sat_scenarios"]
    if set(scenes) != set(EXPECTED_SCENARIOS):
        raise CLICLEOBindingError("existing received-IQ formal three-scene coverage drifted")
    physical_ids = text_arrays["physical_sample_id"]
    if not all(str(value) for value in physical_ids):
        raise CLICLEOBindingError("existing received-IQ physical_sample_id is empty")
    # ``physical_sample_id`` is the stable physical identity.  Scene, receiver,
    # and day are observation metadata, not a namespace in which one physical
    # sample may be reused.  This enforces the formal three-scene disjointness
    # before any LEO forward or tail-calibration artifact is created.
    if len(set(physical_ids.tolist())) != row_count:
        raise CLICLEOBindingError("existing received-IQ physical_sample_id must be globally unique across scenes")
    physical_keys = [
        "|".join((tx_ids[index], text_arrays["rx_ids"][index], text_arrays["day_ids"][index], physical_ids[index]))
        for index in range(row_count)
    ]
    if len(physical_keys) != len(set(physical_keys)):
        raise CLICLEOBindingError("existing received-IQ physical key order contains duplicates")
    coverage: dict[str, dict[str, Any]] = {}
    for scene in EXPECTED_SCENARIOS:
        positions = np.flatnonzero(scenes == scene)
        if positions.size <= 0:
            raise CLICLEOBindingError("existing received-IQ scene coverage is empty")
        coverage[scene] = {
            "count": int(positions.size),
            "physical_order_sha256": _canonical_sha256([physical_keys[int(index)] for index in positions]),
        }
    return arrays, physical_keys, coverage


def _load_received_iq_snapshot(
    path: Path,
    *,
    source_tx_ids: tuple[str, ...],
) -> tuple[dict[str, np.ndarray], list[str], dict[str, dict[str, Any]], str]:
    """Read one immutable received-IQ snapshot and bind the source bytes.

    The in-memory arrays are the only IQ rows passed to the model.  Hashing
    immediately before and after the read prevents a path replacement between
    parser validation and snapshot acquisition.  The caller must additionally
    compare this digest after its forward, before writing any output.
    """

    before = _sha256_file(path)
    arrays, physical_keys, coverage = _load_existing_received_iq(path, source_tx_ids=source_tx_ids)
    after = _sha256_file(path)
    if before != after:
        raise CLICLEOBindingError("existing received-IQ changed while acquiring the immutable snapshot")
    return arrays, physical_keys, coverage, before


def _binding_payload(
    *,
    args: argparse.Namespace,
    checkpoint: Path,
    terminal: Path,
    source_tx_ids: tuple[str, ...],
    arm: str,
    fold: int,
    candidate: str,
    receipt: Mapping[str, Any],
    received_sha: str,
    physical_keys: list[str],
    coverage: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a binding only from the already validated IQ snapshot."""

    return {
        "schema": EXPECTED_BINDING_SCHEMA,
        "method": EXPECTED_METHOD,
        "fold_index": fold,
        "arm": arm,
        "candidate_id": candidate,
        "training_run_root": str(Path(args.training_run_root).resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "terminal_receipt_sha256": _sha256_file(terminal),
        "clic_terminal_contract": str(receipt["terminal_contract"]),
        "clic_terminal_contract_passed": True,
        "source_only": True,
        "single_leo_forward_bound": True,
        "single_leo_observation": True,
        "common_physical_order_bound": True,
        "existing_received_iq_sha256": received_sha,
        "received_iq_sha256": received_sha,
        "physical_order_sha256": _canonical_sha256(physical_keys),
        "physical_keys": list(physical_keys),
        "satellite_scenarios": list(EXPECTED_SCENARIOS),
        "scenario_coverage": dict(coverage),
        "source_tx_ids": list(source_tx_ids),
        "output_npz_path": str(Path(args.out_npz).resolve()),
        "binding_path": str(Path(args.binding_json).resolve()),
        "policy_fit_rows": 0,
        "threshold_fit_rows": 0,
    }


def build_binding_from_existing(args: argparse.Namespace) -> dict[str, Any]:
    """Return a small C/G-comparable binding for one existing source-L LEO table."""

    checkpoint, terminal, received, source_tx_ids, arm, fold, candidate = _validate_args(args)
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICLEOBindingError("cannot load CLIC final checkpoint for LEO binding") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("args"), Mapping):
        raise CLICLEOBindingError("CLIC final checkpoint is malformed")
    checkpoint_args = payload["args"]
    try:
        known = _parse_csv(checkpoint_args.get("phase1_source_known_validation_tx_ids", ""), label="checkpoint held validation TX IDs")
        proxy = _parse_csv(checkpoint_args.get("phase1_source_proxy_unknown_tx_ids", ""), label="checkpoint proxy TX IDs")
        _, receipt, observed_arm = _clean.validate_clic_training_checkpoint(
            payload,
            checkpoint_path=checkpoint,
            terminal_receipt_path=terminal,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known,
            proxy_unknown_tx_ids=proxy,
        )
    except _clean.CLICSplitExportError as exc:
        raise CLICLEOBindingError(f"CLIC LEO terminal/checkpoint reopening failed: {exc}") from exc
    if observed_arm != arm:
        raise CLICLEOBindingError("CLIC LEO terminal arm does not bind requested C/G arm")
    _, physical_keys, coverage, received_sha = _load_received_iq_snapshot(
        received, source_tx_ids=source_tx_ids
    )
    return _binding_payload(
        args=args,
        checkpoint=checkpoint,
        terminal=terminal,
        source_tx_ids=source_tx_ids,
        arm=arm,
        fold=fold,
        candidate=candidate,
        receipt=receipt,
        received_sha=received_sha,
        physical_keys=physical_keys,
        coverage=coverage,
    )


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLICLEOBindingError(f"refusing to overwrite immutable CLIC LEO export: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICLEOBindingError(f"refusing to overwrite temporary CLIC LEO export: {temporary}")
    with temporary.open("xb") as handle:
        np.savez(handle, **dict(payload))
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLICLEOBindingError(f"refusing to overwrite immutable CLIC LEO binding: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICLEOBindingError(f"refusing to overwrite temporary CLIC LEO binding: {temporary}")
    temporary.write_text(json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    """Forward every pre-existing received-IQ row once and write a sealed LEO NPZ.

    The input is already a received observation.  No channel is synthesized,
    no satellite TTA is applied, and no row is resampled.  This is therefore a
    strict one-forward-per-row C/G-comparable postfreeze operation.
    """

    from torch.utils.data import DataLoader, Dataset
    from cvsrffi.checkpoint_loading import build_exact_ssdg_model_from_checkpoint
    from export_spaceborne_features import extract_features_with_metadata

    checkpoint, terminal, received_path, source_tx_ids, arm, fold, candidate = _validate_args(args)
    if Path(args.out_npz).resolve().exists() or Path(args.binding_json).resolve().exists():
        raise CLICLEOBindingError("refusing to overwrite CLIC LEO output or binding")
    try:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CLICLEOBindingError("cannot load CLIC final checkpoint for LEO forward") from exc
    if not isinstance(checkpoint_payload, Mapping) or not isinstance(checkpoint_payload.get("args"), Mapping):
        raise CLICLEOBindingError("CLIC final checkpoint is malformed")
    checkpoint_args = checkpoint_payload["args"]
    known = _parse_csv(checkpoint_args.get("phase1_source_known_validation_tx_ids", ""), label="checkpoint held validation TX IDs")
    proxy = _parse_csv(checkpoint_args.get("phase1_source_proxy_unknown_tx_ids", ""), label="checkpoint proxy TX IDs")
    try:
        _, terminal_receipt, observed_arm = _clean.validate_clic_training_checkpoint(
            checkpoint_payload,
            checkpoint_path=checkpoint,
            terminal_receipt_path=terminal,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known,
            proxy_unknown_tx_ids=proxy,
        )
    except _clean.CLICSplitExportError as exc:
        raise CLICLEOBindingError(f"CLIC LEO terminal/checkpoint reopening failed: {exc}") from exc
    if observed_arm != arm:
        raise CLICLEOBindingError("CLIC LEO terminal arm does not bind requested C/G arm")
    raw_arrays, physical_keys, coverage, received_sha = _load_received_iq_snapshot(
        received_path, source_tx_ids=source_tx_ids
    )
    cache_asset = _validate_sealed_source_leo_cache_asset(
        args,
        received_path=received_path,
        source_tx_ids=source_tx_ids,
        fold=fold,
        arm=arm,
        candidate=candidate,
        checkpoint=checkpoint,
        terminal=terminal,
        terminal_receipt=terminal_receipt,
    )
    iq = np.asarray(raw_arrays["received_iq"], dtype=np.float32)
    if iq.ndim != 3 or iq.shape[1] != 2 or iq.shape[2] != int(checkpoint_args.get("wisig_out_len", 256)):
        raise CLICLEOBindingError("existing received-IQ must be finite [N,2,T] rows matching checkpoint input length")
    row_count = int(iq.shape[0])
    if cache_asset and row_count != cache_asset["source_row_count"]:
        raise CLICLEOBindingError("sealed source-LEO cache NPZ row count drifted")
    tx_ids = np.asarray(raw_arrays["tx_ids"]).reshape(-1).astype(str)
    rx_ids = np.asarray(raw_arrays["rx_ids"]).reshape(-1).astype(str)
    day_ids = np.asarray(raw_arrays["day_ids"]).reshape(-1).astype(str)
    physical_ids = np.asarray(raw_arrays["physical_sample_id"]).reshape(-1).astype(str)
    scenes = np.asarray(raw_arrays["sat_scenarios"]).reshape(-1).astype(str)
    rx_order = tuple(sorted(set(rx_ids.tolist())))
    if len(rx_order) != 7:
        raise CLICLEOBindingError("source-L LEO calibration requires exactly seven source RX slots")
    rx_to_slot = {name: index for index, name in enumerate(rx_order)}
    labels = np.asarray([source_tx_ids.index(value) for value in tx_ids], dtype=np.int64)

    class ExistingReceivedIQDataset(Dataset):
        def __len__(self) -> int:
            return row_count

        def __getitem__(self, index: int):
            return (
                _numpy_float32_to_tensor(iq[index]),
                torch.tensor(int(labels[index]), dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
                {
                    "tx": str(tx_ids[index]),
                    "rx": str(rx_ids[index]),
                    "day": str(day_ids[index]),
                    "equalized": "existing_received_iq",
                    "sig_i": str(physical_ids[index]),
                },
            )

    device = torch.device(str(getattr(args, "device", "cuda:0")) if torch.cuda.is_available() else "cpu")
    model, load_audit = build_exact_ssdg_model_from_checkpoint(
        checkpoint_payload, input_len=int(checkpoint_args.get("wisig_out_len", 256)), device=device
    )
    batch_size = int(getattr(args, "batch_size", 32))
    if batch_size <= 0:
        raise CLICLEOBindingError("CLIC LEO batch size is invalid")
    loader = DataLoader(ExistingReceivedIQDataset(), batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)
    payload = extract_features_with_metadata(
        model,
        loader,
        device=device,
        feature_name="z_id",
        role="source_L_leo_calibration",
        channel_view="received_existing",
        satellite_tta_policy="none",
        safe_numpy_bridge=True,
    )
    if int(np.asarray(payload["features"]).shape[0]) != row_count:
        raise CLICLEOBindingError("CLIC LEO forward did not preserve existing received-IQ row count")
    payload["z_id"] = np.asarray(payload["features"], dtype=np.float32)
    payload["physical_sample_id"] = physical_ids
    payload["source_rx_slot"] = np.asarray([rx_to_slot[item] for item in rx_ids], dtype=np.int64)
    payload["sat_scenarios"] = scenes
    if _sha256_file(received_path) != received_sha:
        raise CLICLEOBindingError("existing received-IQ changed after snapshot forward; refusing to write mixed-byte artifacts")
    physical_sha = _canonical_sha256(physical_keys)
    binding = _binding_payload(
        args=args,
        checkpoint=checkpoint,
        terminal=terminal,
        source_tx_ids=source_tx_ids,
        arm=arm,
        fold=fold,
        candidate=candidate,
        receipt=terminal_receipt,
        received_sha=received_sha,
        physical_keys=physical_keys,
        coverage=coverage,
    )
    binding.update(
        {
            "leo_npz_path": str(Path(args.out_npz).resolve()),
            "received_iq_sha256": received_sha,
            "existing_received_iq_sha256": received_sha,
            "physical_order_sha256": physical_sha,
            "physical_row_count": row_count,
            "source_rx_slot_order": list(rx_order),
            "scenario_coverage": coverage,
            "checkpoint_load_strict": True,
            "checkpoint_load_audit": load_audit,
            "clic_terminal_contract": str(terminal_receipt["terminal_contract"]),
            "clic_terminal_contract_passed": True,
            "single_leo_observation": True,
            "single_leo_forward_bound": True,
            "policy_fit_rows": 0,
            "threshold_fit_rows": 0,
        }
    )
    if cache_asset:
        binding["source_leo_cache_receipt_path"] = cache_asset["receipt_path"]
        binding["source_leo_cache_receipt_sha256"] = cache_asset["receipt_sha256"]
    manifest = {
        "schema": "cvs.phase1.clic_leo_export.v1",
        "method": EXPECTED_METHOD,
        "candidate_id": candidate,
        "fold_index": fold,
        "arm": arm,
        "checkpoint_sha256": _sha256_file(checkpoint),
        "terminal_receipt_sha256": _sha256_file(terminal),
        "source_only": True,
        "single_leo_observation_required": True,
        "single_leo_forward_count": row_count,
        "received_iq_sha256": received_sha,
        "physical_order_sha256": physical_sha,
        "satellite_scenarios": list(EXPECTED_SCENARIOS),
        "scenario_coverage": coverage,
        "source_rx_slot_order_sha256": _canonical_sha256(list(rx_order)),
        "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
        "checkpoint_load_strict": True,
        "checkpoint_load_audit": load_audit,
        "fit_rows": 0,
        "threshold_fit_rows": 0,
        "source_frozen_tail_calibration_only": True,
    }
    if cache_asset:
        manifest["source_leo_cache_receipt_sha256"] = cache_asset["receipt_sha256"]
    payload["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    output_path = Path(args.out_npz).resolve()
    binding_path = Path(args.binding_json).resolve()
    output_written = False
    binding_written = False
    try:
        _atomic_save_npz(output_path, payload)
        output_written = True
        binding["leo_npz_sha256"] = _sha256_file(output_path)
        binding["leo_manifest_sha256"] = _canonical_sha256(manifest)
        _atomic_write_json(binding_path, binding)
        binding_written = True
        if _sha256_file(received_path) != received_sha:
            raise CLICLEOBindingError("existing received-IQ changed while sealing the snapshot binding")
        if cache_asset and _sha256_file(cache_asset["receipt_path"]) != cache_asset["receipt_sha256"]:
            raise CLICLEOBindingError("sealed source-LEO cache receipt changed during forward")
    except Exception:
        # The targets were proven absent before work began.  Remove only files
        # this invocation finished writing so a detected snapshot race never
        # leaves a feature/binding pair that claims different received bytes.
        for target, written in ((binding_path, binding_written), (output_path, output_written)):
            if written and target.is_file():
                target.unlink()
        raise
    return {"out_npz": str(output_path), "binding_json": str(binding_path), "binding": binding}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--terminal-receipt-json", required=True)
    parser.add_argument("--existing-received-iq-npz", required=True)
    parser.add_argument("--existing-received-iq-receipt-json")
    parser.add_argument("--cache-run-root")
    parser.add_argument("--require-sealed-source-leo-cache", action="store_true")
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--binding-json", required=True)
    parser.add_argument("--training-run-root", required=True)
    parser.add_argument("--postfreeze-output-root", required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = export(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
