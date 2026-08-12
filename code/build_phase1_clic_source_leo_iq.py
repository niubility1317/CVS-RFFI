"""Build one immutable LEO-weak observation for every CLIC source-L row.

This is source-only, post-training preprocessing.  A physical source-L row is
assigned to exactly one formal weak satellite scene before the channel is
applied.  The assignment is deterministic and independent of input row order,
so the C/G arms of one fold can consume the same sealed received-IQ bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


FORMAL_LEO_WEAK_SCENARIOS = (
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
)
FROZEN_SOURCE_CLASS_COUNT = 4
FROZEN_SOURCE_RECEIVER_COUNT = 7
FROZEN_MIN_ROWS_PER_CELL_AND_SCENE = 20
FROZEN_ROWS_PER_TX_RX_CELL = 140
FROZEN_SOURCE_L_ROW_COUNT = (
    FROZEN_SOURCE_CLASS_COUNT
    * FROZEN_SOURCE_RECEIVER_COUNT
    * FROZEN_ROWS_PER_TX_RX_CELL
)
FROZEN_WISIG_SHA256 = "2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
SOURCE_LEO_SEED_OFFSET = 991
SOURCE_LEO_SCENE_SEED_STRIDE = 1_000_003
SOURCE_LEO_CACHE_SCHEMA = "cvs.phase1.clic_source_leo_received_iq.v1"


class CLICSourceLeoCacheError(RuntimeError):
    """Raised when a source-L LEO cache cannot be built fail-closed."""


def _strict_string_rows(values: Sequence[str], *, label: str) -> tuple[str, ...]:
    try:
        rows = tuple(values)
    except TypeError as exc:
        raise CLICSourceLeoCacheError(f"{label} rows are invalid") from exc
    if any(not isinstance(value, str) or not value for value in rows):
        raise CLICSourceLeoCacheError(f"{label} rows contain an empty or non-string ID")
    return rows


def assign_source_l_scenarios(
    tx_ids: Sequence[str],
    rx_ids: Sequence[str],
    physical_sample_ids: Sequence[str],
    *,
    min_per_cell: int = FROZEN_MIN_ROWS_PER_CELL_AND_SCENE,
) -> Mapping[str, str]:
    """Assign each source-L physical row to one formal LEO-weak scene.

    Rows are grouped by the exact local4 TX and seven source receivers.  Within
    each group, opaque physical IDs are sorted and assigned round-robin to the
    frozen scene tuple.  Consequently, permuting input rows cannot change an
    existing physical row's scene, all scene partitions are disjoint, and each
    TX/RX/scene calibration cell has at least twenty rows.
    """

    if type(min_per_cell) is not int or min_per_cell != FROZEN_MIN_ROWS_PER_CELL_AND_SCENE:
        raise CLICSourceLeoCacheError("source-L scene cell minimum must remain frozen at 20")
    tx_rows = _strict_string_rows(tx_ids, label="TX")
    rx_rows = _strict_string_rows(rx_ids, label="RX")
    physical_rows = _strict_string_rows(physical_sample_ids, label="physical")
    if not (len(tx_rows) == len(rx_rows) == len(physical_rows)):
        raise CLICSourceLeoCacheError("source-L TX/RX/physical row lengths drifted")
    if not physical_rows:
        raise CLICSourceLeoCacheError("source-L physical rows are empty")
    if len(physical_rows) != FROZEN_SOURCE_L_ROW_COUNT:
        raise CLICSourceLeoCacheError(
            "source-L must contain exactly 3920 frozen physical rows"
        )
    if len(set(physical_rows)) != len(physical_rows):
        raise CLICSourceLeoCacheError("source-L physical sample IDs must be globally unique")

    classes = tuple(sorted(set(tx_rows)))
    receivers = tuple(sorted(set(rx_rows)))
    if len(classes) != FROZEN_SOURCE_CLASS_COUNT:
        raise CLICSourceLeoCacheError("source-L must contain exactly four TX classes")
    if len(receivers) != FROZEN_SOURCE_RECEIVER_COUNT:
        raise CLICSourceLeoCacheError("source-L must contain exactly seven RX cells")

    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tx_id, rx_id, physical_id in zip(tx_rows, rx_rows, physical_rows, strict=True):
        grouped[(tx_id, rx_id)].append(physical_id)

    assignment: dict[str, str] = {}
    for tx_id in classes:
        for rx_id in receivers:
            cell = sorted(grouped.get((tx_id, rx_id), ()))
            if len(cell) != FROZEN_ROWS_PER_TX_RX_CELL:
                raise CLICSourceLeoCacheError(
                    "source-L TX/RX cell requires exactly 140 frozen physical rows: "
                    f"tx={tx_id!r} rx={rx_id!r} observed={len(cell)}"
                )
            for rank, physical_id in enumerate(cell):
                assignment[physical_id] = FORMAL_LEO_WEAK_SCENARIOS[
                    rank % len(FORMAL_LEO_WEAK_SCENARIOS)
                ]

    if set(assignment) != set(physical_rows):
        raise CLICSourceLeoCacheError("source-L scenario assignment did not cover every physical row")
    for tx_id in classes:
        for rx_id in receivers:
            cell_ids = grouped[(tx_id, rx_id)]
            for scene in FORMAL_LEO_WEAK_SCENARIOS:
                count = sum(assignment[physical_id] == scene for physical_id in cell_ids)
                if count < min_per_cell:
                    raise CLICSourceLeoCacheError(
                        "source-L scene calibration cell has fewer than 20 rows: "
                        f"tx={tx_id!r} rx={rx_id!r} scene={scene!r} count={count}"
                    )
    return assignment


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CLICSourceLeoCacheError("cannot canonicalize source-L cache state") from exc
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_csv(value: str, *, label: str, expected: int) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in str(value).split(",") if part.strip())
    if len(parsed) != expected or len(set(parsed)) != expected:
        raise CLICSourceLeoCacheError(
            f"{label} must contain exactly {expected} unique IDs"
        )
    return parsed


def _physical_sample_id(
    *,
    dataset_sha256: str,
    tx_id: str,
    rx_id: str,
    day_id: str,
    eq_id: str,
    sig_id: str,
) -> str:
    payload = "\x1f".join(
        (
            SOURCE_LEO_CACHE_SCHEMA,
            dataset_sha256,
            tx_id,
            rx_id,
            day_id,
            eq_id,
            sig_id,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_save_npz(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLICSourceLeoCacheError(
            f"refusing to overwrite immutable source-L received-IQ cache: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICSourceLeoCacheError(
            f"refusing to overwrite temporary source-L cache: {temporary}"
        )
    with temporary.open("xb") as handle:
        np.savez(handle, **dict(payload))
    temporary.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise CLICSourceLeoCacheError(
            f"refusing to overwrite immutable source-L cache receipt: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise CLICSourceLeoCacheError(
            f"refusing to overwrite temporary source-L cache receipt: {temporary}"
        )
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _data_config_projection(args: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "seed",
        "split_mode",
        "labeled_ratio",
        "unlabeled_ratio",
        "source_val_ratio",
        "wisig_pkl",
        "wisig_equalized",
        "wisig_out_len",
        "wisig_domain",
        "wisig_train_days",
        "wisig_test_days",
        "wisig_train_rxs",
        "wisig_test_rxs",
        "wisig_max_day123_per_combo",
        "phase1_source_train_tx_ids",
        "phase1_source_known_validation_tx_ids",
        "phase1_source_proxy_unknown_tx_ids",
        "sat_fs_hz",
        "sat_fc_hz",
    )
    return {field: args.get(field) for field in fields}


def _load_validated_arm(
    *,
    checkpoint_path: Path,
    terminal_path: Path,
    source_tx_ids: tuple[str, ...],
    known_validation_tx_ids: tuple[str, ...],
    proxy_unknown_tx_ids: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    import export_phase1_clic_features as clean_export

    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except Exception as exc:
        raise CLICSourceLeoCacheError("cannot load CLIC final checkpoint") from exc
    if not isinstance(checkpoint, Mapping):
        raise CLICSourceLeoCacheError("CLIC final checkpoint is malformed")
    try:
        checkpoint_args, terminal_receipt, arm = (
            clean_export.validate_clic_training_checkpoint(
                checkpoint,
                checkpoint_path=checkpoint_path,
                terminal_receipt_path=terminal_path,
                source_tx_ids=source_tx_ids,
                known_validation_tx_ids=known_validation_tx_ids,
                proxy_unknown_tx_ids=proxy_unknown_tx_ids,
            )
        )
    except clean_export.CLICSplitExportError as exc:
        raise CLICSourceLeoCacheError(
            f"CLIC checkpoint/terminal reopening failed: {exc}"
        ) from exc
    return dict(checkpoint), checkpoint_args, terminal_receipt, arm


def _collect_source_l_rows(
    dataset: Any,
    *,
    source_tx_ids: tuple[str, ...],
    dataset_sha256: str,
    batch_size: int,
) -> dict[str, Any]:
    from torch.utils.data import DataLoader
    from export_spaceborne_features import _meta_to_list

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    clean_rows: list[np.ndarray] = []
    tx_ids: list[str] = []
    rx_ids: list[str] = []
    day_ids: list[str] = []
    physical_ids: list[str] = []
    for batch in loader:
        if not isinstance(batch, (list, tuple)) or len(batch) != 4:
            raise CLICSourceLeoCacheError(
                "source-L cache builder expects (x,y,domain,metadata) batches"
            )
        x, y, _domain, meta = batch
        if not torch.is_tensor(x) or not torch.is_tensor(y):
            raise CLICSourceLeoCacheError("source-L cache batch tensors are malformed")
        clean = x.detach().cpu().float().numpy().astype(np.float32)
        if clean.ndim != 3 or clean.shape[1] != 2 or not np.isfinite(clean).all():
            raise CLICSourceLeoCacheError("source-L clean IQ rows are non-finite or malformed")
        count = int(clean.shape[0])
        meta_tx = [str(value) for value in _meta_to_list(meta, "tx", count)]
        meta_rx = [str(value) for value in _meta_to_list(meta, "rx", count)]
        meta_day = [str(value) for value in _meta_to_list(meta, "day", count)]
        meta_eq = [str(value) for value in _meta_to_list(meta, "equalized", count)]
        meta_sig = [str(value) for value in _meta_to_list(meta, "sig_i", count)]
        labels = [int(value) for value in y.detach().cpu().reshape(-1).tolist()]
        if len(labels) != count:
            raise CLICSourceLeoCacheError("source-L label row count drifted")
        for index in range(count):
            label = labels[index]
            if label not in range(len(source_tx_ids)) or source_tx_ids[label] != meta_tx[index]:
                raise CLICSourceLeoCacheError("source-L local label/TX binding drifted")
            physical_ids.append(
                _physical_sample_id(
                    dataset_sha256=dataset_sha256,
                    tx_id=meta_tx[index],
                    rx_id=meta_rx[index],
                    day_id=meta_day[index],
                    eq_id=meta_eq[index],
                    sig_id=meta_sig[index],
                )
            )
        clean_rows.append(clean)
        tx_ids.extend(meta_tx)
        rx_ids.extend(meta_rx)
        day_ids.extend(meta_day)
    if not clean_rows:
        raise CLICSourceLeoCacheError("source-L cache builder observed no labeled rows")
    return {
        "clean_iq": np.concatenate(clean_rows, axis=0).astype(np.float32),
        "tx_ids": tx_ids,
        "rx_ids": rx_ids,
        "day_ids": day_ids,
        "physical_sample_ids": physical_ids,
    }


def build_source_l_received_iq(args: argparse.Namespace) -> dict[str, Any]:
    """Materialize one source-only received-IQ cache shared by a fold's C/G arms."""

    import export_phase1_clic_features as clean_export
    import export_phase1_clic_leo_features as leo_export
    from cvsrffi.eval import apply_sat_channel_for_scenario
    from cvsrffi.tensors import make_torch_generator
    from dataset_wisig import WiSigSubsetDataset, load_wisig_compact_pkl
    from training_controls import sat_channel_config_for_scenario

    fold_index = int(args.fold_index)
    if fold_index not in range(1, 7):
        raise CLICSourceLeoCacheError("source-L cache fold_index must be F1..F6")
    source_tx_ids = _parse_csv(args.source_tx_ids, label="source TX", expected=4)
    known_ids = _parse_csv(
        args.known_validation_tx_ids, label="known validation TX", expected=1
    )
    proxy_ids = _parse_csv(
        args.proxy_unknown_tx_ids, label="proxy unknown TX", expected=1
    )
    if set(source_tx_ids) & (set(known_ids) | set(proxy_ids)) or set(known_ids) & set(proxy_ids):
        raise CLICSourceLeoCacheError("source-L, held, and proxy TX roles overlap")

    c_checkpoint_path = Path(args.c_ckpt).resolve()
    c_terminal_path = Path(args.c_terminal_receipt_json).resolve()
    g_checkpoint_path = Path(args.g_ckpt).resolve()
    g_terminal_path = Path(args.g_terminal_receipt_json).resolve()
    dataset_path = Path(args.wisig_pkl).resolve()
    output_path = Path(args.out_npz).resolve()
    receipt_path = Path(args.receipt_json).resolve()
    for path, label in (
        (c_checkpoint_path, "C checkpoint"),
        (c_terminal_path, "C terminal"),
        (g_checkpoint_path, "G checkpoint"),
        (g_terminal_path, "G terminal"),
        (dataset_path, "WiSig dataset"),
    ):
        if not path.is_file():
            raise CLICSourceLeoCacheError(f"source-L cache {label} is missing")
    if output_path.exists() or receipt_path.exists():
        raise CLICSourceLeoCacheError("refusing to overwrite source-L cache output")
    input_hashes_before = {
        "c_checkpoint": _sha256_file(c_checkpoint_path),
        "c_terminal": _sha256_file(c_terminal_path),
        "g_checkpoint": _sha256_file(g_checkpoint_path),
        "g_terminal": _sha256_file(g_terminal_path),
        "wisig": _sha256_file(dataset_path),
    }

    expected_candidates = {
        c_checkpoint_path.parent.name: f"F{fold_index}C_CLIC12",
        g_checkpoint_path.parent.name: f"F{fold_index}G_CLIC12",
    }
    if any(observed != expected for observed, expected in expected_candidates.items()):
        raise CLICSourceLeoCacheError("source-L cache C/G checkpoint paths do not bind the fold")
    c_payload, c_args, c_receipt, c_arm = _load_validated_arm(
        checkpoint_path=c_checkpoint_path,
        terminal_path=c_terminal_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_ids,
        proxy_unknown_tx_ids=proxy_ids,
    )
    g_payload, g_args, g_receipt, g_arm = _load_validated_arm(
        checkpoint_path=g_checkpoint_path,
        terminal_path=g_terminal_path,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_ids,
        proxy_unknown_tx_ids=proxy_ids,
    )
    if (c_arm, g_arm) != ("C", "G"):
        raise CLICSourceLeoCacheError("source-L cache requires one C and one G arm")
    if _data_config_projection(c_args) != _data_config_projection(g_args):
        raise CLICSourceLeoCacheError("source-L cache C/G data or channel configuration drifted")
    if Path(str(c_args.get("wisig_pkl", ""))).resolve() != dataset_path:
        raise CLICSourceLeoCacheError("source-L cache dataset path differs from checkpoint")
    if Path(str(g_args.get("wisig_pkl", ""))).resolve() != dataset_path:
        raise CLICSourceLeoCacheError("source-L cache G dataset path differs from checkpoint")
    expected_wisig_sha = str(args.expected_wisig_sha256).lower()
    dataset_sha = input_hashes_before["wisig"]
    if expected_wisig_sha != FROZEN_WISIG_SHA256 or dataset_sha != FROZEN_WISIG_SHA256:
        raise CLICSourceLeoCacheError("source-L cache WiSig bytes drifted")

    raw_dataset = load_wisig_compact_pkl(str(dataset_path))
    if _sha256_file(dataset_path) != dataset_sha:
        raise CLICSourceLeoCacheError("WiSig bytes changed while loading source-L rows")
    reconstructed = clean_export._reconstruct_source_l_v(
        raw_dataset=raw_dataset,
        checkpoint_args=c_args,
        source_tx_ids=source_tx_ids,
        known_validation_tx_ids=known_ids,
        proxy_unknown_tx_ids=proxy_ids,
        wisig_sha256=dataset_sha,
    )
    for checkpoint, receipt in ((c_payload, c_receipt), (g_payload, g_receipt)):
        clean_export._assert_current_source_split(
            checkpoint=checkpoint,
            receipt=receipt,
            reconstructed=reconstructed,
            source_tx_ids=source_tx_ids,
            known_validation_tx_ids=known_ids,
            proxy_unknown_tx_ids=proxy_ids,
        )
    labeled = tuple(int(value) for value in reconstructed["labeled_indices"])
    labeled_dataset = WiSigSubsetDataset(
        reconstructed["source_base"],
        labeled,
        split_source="clic_source_l_received_iq_cache",
    )
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise CLICSourceLeoCacheError("source-L cache batch size must be positive")
    rows = _collect_source_l_rows(
        labeled_dataset,
        source_tx_ids=source_tx_ids,
        dataset_sha256=dataset_sha,
        batch_size=batch_size,
    )
    if len(rows["physical_sample_ids"]) != len(labeled):
        raise CLICSourceLeoCacheError("source-L cache row count differs from frozen split")
    assignment = assign_source_l_scenarios(
        rows["tx_ids"], rows["rx_ids"], rows["physical_sample_ids"]
    )
    scenes = [assignment[physical_id] for physical_id in rows["physical_sample_ids"]]

    device = torch.device(
        str(args.device) if torch.cuda.is_available() else "cpu"
    )
    clean_iq = np.asarray(rows["clean_iq"], dtype=np.float32)
    received_iq = np.empty_like(clean_iq)
    base_seed = int(c_args["seed"]) + SOURCE_LEO_SEED_OFFSET
    scene_seeds = {
        scene: base_seed + index * SOURCE_LEO_SCENE_SEED_STRIDE
        for index, scene in enumerate(FORMAL_LEO_WEAK_SCENARIOS)
    }
    channel_config_sha256: dict[str, str] = {}
    channel_args = argparse.Namespace(
        sat_fs_hz=float(c_args.get("sat_fs_hz", 25e6)),
        sat_fc_hz=float(c_args.get("sat_fc_hz", 2.462e9)),
    )
    scene_array = np.asarray(scenes, dtype=str)
    for scene in FORMAL_LEO_WEAK_SCENARIOS:
        channel_config = dict(sat_channel_config_for_scenario(scene))
        channel_config.update(
            {
                "fs_hz": channel_args.sat_fs_hz,
                "fc_hz": channel_args.sat_fc_hz,
            }
        )
        channel_config_sha256[scene] = _canonical_sha256(channel_config)
        generator = make_torch_generator(device, scene_seeds[scene])
        positions = np.flatnonzero(scene_array == scene)
        if positions.size <= 0:
            raise CLICSourceLeoCacheError("source-L cache scene assignment is empty")
        for start in range(0, int(positions.size), batch_size):
            current = positions[start : start + batch_size]
            source = torch.from_numpy(clean_iq[current]).to(device)
            with torch.no_grad():
                received, metadata = apply_sat_channel_for_scenario(
                    source,
                    scene,
                    channel_args,
                    gen=generator,
                    return_meta=True,
                )
            if not isinstance(metadata, Mapping) or metadata.get("channel_model") != "leo_residual":
                raise CLICSourceLeoCacheError("source-L cache channel metadata drifted")
            observed = received.detach().cpu().float().numpy().astype(np.float32)
            if observed.shape != clean_iq[current].shape or not np.isfinite(observed).all():
                raise CLICSourceLeoCacheError("source-L received-IQ rows are malformed or non-finite")
            received_iq[current] = observed

    if _sha256_file(dataset_path) != dataset_sha:
        raise CLICSourceLeoCacheError("WiSig bytes changed before source-L cache sealing")
    input_hashes_after = {
        "c_checkpoint": _sha256_file(c_checkpoint_path),
        "c_terminal": _sha256_file(c_terminal_path),
        "g_checkpoint": _sha256_file(g_checkpoint_path),
        "g_terminal": _sha256_file(g_terminal_path),
        "wisig": _sha256_file(dataset_path),
    }
    if input_hashes_after != input_hashes_before:
        raise CLICSourceLeoCacheError(
            "source-L cache input bytes changed during immutable generation"
        )
    checkpoint_hashes = {
        "C": input_hashes_before["c_checkpoint"],
        "G": input_hashes_before["g_checkpoint"],
    }
    terminal_hashes = {
        "C": input_hashes_before["c_terminal"],
        "G": input_hashes_before["g_terminal"],
    }
    payload = {
        "received_iq": received_iq,
        "tx_ids": np.asarray(rows["tx_ids"], dtype=str),
        "rx_ids": np.asarray(rows["rx_ids"], dtype=str),
        "day_ids": np.asarray(rows["day_ids"], dtype=str),
        "physical_sample_id": np.asarray(rows["physical_sample_ids"], dtype=str),
        "sat_scenarios": scene_array,
    }
    output_written = False
    receipt_written = False
    try:
        _atomic_save_npz(output_path, payload)
        output_written = True
        output_sha = _sha256_file(output_path)
        # Reopen through the production consumer before declaring the cache usable.
        _, physical_keys, coverage = leo_export._load_existing_received_iq(
            output_path, source_tx_ids=source_tx_ids
        )
        cell_counts: dict[str, int] = {}
        for tx_id, rx_id, scene in zip(
            rows["tx_ids"], rows["rx_ids"], scenes, strict=True
        ):
            key = "|".join((tx_id, rx_id, scene))
            cell_counts[key] = cell_counts.get(key, 0) + 1
        if len(cell_counts) != 4 * 7 * 3 or min(cell_counts.values()) < 20:
            raise CLICSourceLeoCacheError("source-L cache calibration cell coverage drifted")
        receipt = {
            "schema": SOURCE_LEO_CACHE_SCHEMA,
            "method": "P1_CLIC",
            "fold_index": fold_index,
            "training_run_id": clean_export.EXPECTED_TRAINING_RUN_ID,
            "source_only": True,
            "source_l_only": True,
            "same_received_iq_bytes_for_c_and_g": True,
            "single_leo_observation_per_physical_sample": True,
            "cross_scene_physical_sample_reuse": False,
            "formal_scenarios": list(FORMAL_LEO_WEAK_SCENARIOS),
            "source_tx_ids": list(source_tx_ids),
            "source_rx_ids": sorted(set(rows["rx_ids"])),
            "source_row_count": len(labeled),
            "source_split_sha256": str(c_receipt["source_split_sha256"]),
            "physical_order_sha256": _canonical_sha256(physical_keys),
            "scenario_assignment_sha256": _canonical_sha256(
                list(zip(rows["physical_sample_ids"], scenes, strict=True))
            ),
            "scenario_coverage": coverage,
            "cell_counts": cell_counts,
            "minimum_cell_count": min(cell_counts.values()),
            "scene_seeds": scene_seeds,
            "channel_config_sha256": channel_config_sha256,
            "wisig_pkl_sha256": dataset_sha,
            "checkpoint_sha256_by_arm": checkpoint_hashes,
            "terminal_receipt_sha256_by_arm": terminal_hashes,
            "received_iq_npz_path": str(output_path),
            "received_iq_npz_sha256": output_sha,
            "target_access": False,
            "query_access": False,
            "held_validation_forward_rows": 0,
            "proxy_forward_rows": 0,
            "fit_rows": 0,
            "threshold_fit_rows": 0,
        }
        _atomic_write_json(receipt_path, receipt)
        receipt_written = True
    except Exception:
        for path, written in (
            (receipt_path, receipt_written),
            (output_path, output_written),
        ):
            if written and path.is_file():
                path.unlink()
        raise
    return {
        "out_npz": str(output_path),
        "receipt_json": str(receipt_path),
        "received_iq_npz_sha256": receipt["received_iq_npz_sha256"],
        "source_row_count": receipt["source_row_count"],
        "minimum_cell_count": receipt["minimum_cell_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, required=True)
    parser.add_argument("--c-ckpt", required=True)
    parser.add_argument("--c-terminal-receipt-json", required=True)
    parser.add_argument("--g-ckpt", required=True)
    parser.add_argument("--g-terminal-receipt-json", required=True)
    parser.add_argument("--wisig-pkl", required=True)
    parser.add_argument("--expected-wisig-sha256", required=True)
    parser.add_argument("--source-tx-ids", required=True)
    parser.add_argument("--known-validation-tx-ids", required=True)
    parser.add_argument("--proxy-unknown-tx-ids", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--receipt-json", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = build_source_l_received_iq(build_parser().parse_args(argv))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


__all__ = [
    "CLICSourceLeoCacheError",
    "FORMAL_LEO_WEAK_SCENARIOS",
    "FROZEN_MIN_ROWS_PER_CELL_AND_SCENE",
    "FROZEN_ROWS_PER_TX_RX_CELL",
    "FROZEN_SOURCE_L_ROW_COUNT",
    "SOURCE_LEO_CACHE_SCHEMA",
    "assign_source_l_scenarios",
    "build_source_l_received_iq",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
