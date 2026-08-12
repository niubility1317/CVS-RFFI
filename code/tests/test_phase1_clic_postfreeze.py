from __future__ import annotations

"""RED contract tests for the frozen P1-CLIC postfreeze stage.

The four postfreeze modules are intentionally imported directly.  Their absence
must produce a real collection failure until Task 6 production code lands;
``importorskip`` and test-only stand-ins are deliberately not used.

These tests are data-free mechanical contracts.  They do not launch N607,
read target performance, choose a threshold, or claim Phase1/Phase3 progress.
"""

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import zipfile
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluate_phase1_clic_postfreeze_pair as PAIR
import evaluate_phase1_clic_target_leo as TARGET_EVAL
import export_phase1_clic_deployment_bundle as BUNDLE
import export_phase1_clic_features as CLEAN
import export_phase1_clic_leo_features as LEO
from cvsrffi import phase1_clic as CLIC
import cvsrffi.phase1_clic_target_leo as TARGET


CODE_ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = tuple(CLIC.FORMAL_LEO_WEAK_SCENARIOS)
SOURCE_TX = ("tx-a", "tx-b", "tx-c", "tx-d")
HELD_TX = ("tx-held",)
PROXY_TX = ("tx-proxy",)
SOURCE_RX = ("rx-0", "rx-1")
SOURCE_DAYS = ("day-0", "day-1")
TRAINING_RUN = "phase1_clic12_20260811_v1"
POSTFREEZE_MATRIX = "phase1_clic_postfreeze_20260811_v1"


def _load_clic_core_fixture():
    path = CODE_ROOT / "tests" / "test_phase1_clic.py"
    spec = importlib.util.spec_from_file_location("_clic_core_postfreeze_fixture", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_CORE = _load_clic_core_fixture()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _receipt(arm: str) -> dict[str, object]:
    """Use the already verified Task 5 CLIC receipt builder for raw fixtures."""

    return dict(_CORE._complete_receipt(arm))


_REAL_G_MODEL_STATE: dict[str, torch.Tensor] | None = None


def _real_g_model_state() -> dict[str, torch.Tensor]:
    """Build one exact production G model state for the F6 bundle chain."""

    global _REAL_G_MODEL_STATE
    if _REAL_G_MODEL_STATE is None:
        from model_dual_cvsincnet import build_dual_model

        runtime_defaults = dict(BUNDLE.RUNTIME_MODEL_DEFAULTS)
        runtime_defaults["id_feature_key"] = "z_id"
        model = build_dual_model(**runtime_defaults)
        _REAL_G_MODEL_STATE = {
            key: value.detach().cpu().contiguous().clone()
            for key, value in model.state_dict().items()
        }
    return {
        key: value.detach().cpu().contiguous().clone()
        for key, value in _REAL_G_MODEL_STATE.items()
    }


def _checkpoint_fixture(
    tmp_path: Path,
    *,
    arm: str = "G",
    fold: int = 1,
    real_model: bool = False,
) -> dict[str, Path | str | dict[str, object]]:
    """Create one final-only checkpoint plus the external versioned envelope."""

    candidate = f"F{fold}{arm}_CLIC12"
    training_root = tmp_path / TRAINING_RUN
    checkpoint = training_root / candidate / "final_ssdg.pth"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pre = _receipt(arm)
    pre.update(
        {
            "completed": False,
            "terminal_contract": "AWAITING_EXTERNAL_CHECKPOINT_SHA",
            "terminal_contract_passed": False,
            "final_checkpoint_sha256": "",
        }
    )
    args = {
        "split_mode": "tx_rx_day_1_6_3",
        "model_variant": "lite_d",
        "id_feature_key": "z_id",
        "phase1_source_train_tx_ids": ",".join(SOURCE_TX),
        "phase1_source_known_validation_tx_ids": ",".join(HELD_TX),
        "phase1_source_proxy_unknown_tx_ids": ",".join(PROXY_TX),
        "checkpoint_selection": "final_only",
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_val_ratio": 0.30,
        "seed": 7281164,
        "candidate_id": candidate,
        "run_id": TRAINING_RUN,
        "phase1_clic_frozen_mode": True,
        "phase1_clic_enabled": arm == "G",
        "phase1_clic_operator_mode": "complex_local_invariant_curvature" if arm == "G" else "raw_phase_control",
    }
    model_state: dict[str, torch.Tensor] = {}
    if real_model:
        if arm != "G":
            raise AssertionError("only the frozen G arm may carry a real deployment model fixture")
        args.update({key: value for key, value in BUNDLE.RUNTIME_MODEL_DEFAULTS.items()})
        args["wisig_out_len"] = int(BUNDLE.RUNTIME_MODEL_DEFAULTS["input_len"])
        args["id_feature_key"] = "z_id"
        args["phase1_clic_enabled"] = True
        args["phase1_clic_frozen_mode"] = True
        args["phase1_clic_operator_mode"] = "complex_local_invariant_curvature"
        model_state = _real_g_model_state()
    payload = {
        "checkpoint_schema": "ssdg_phase1_training_state_v2",
        "checkpoint_role": "training_final_only",
        "checkpoint_selection": "final_only",
        "candidate_id": candidate,
        "run_id": TRAINING_RUN,
        "args": args,
        "model": model_state,
        "optimizer": {},
        "scaler": {},
        "epoch": 40,
        "final_epoch": 40,
        "clic_receipt_precheckpoint": pre,
        "split_info": {
            "source_split_receipt": {
                "schema": "cvs.phase1.source_split_receipt.v1",
                "source_receivers": list(SOURCE_RX),
                "source_days": list(SOURCE_DAYS),
                "labeled_indices_sha256": _sha_text("labeled"),
                "split_manifest_sha256": _sha_text("split"),
            },
            "tx_partition_receipt": {"partition_sha256": _sha_text("partition")},
        },
    }
    torch.save(payload, checkpoint)
    checkpoint_sha = _sha_file(checkpoint)
    strict = _receipt(arm)
    strict.update(
        {
            "completed": True,
            "terminal_contract": "VERSIONED_EXTERNAL_CHECKPOINT_SHA",
            "terminal_contract_passed": True,
            "final_checkpoint_sha256": checkpoint_sha,
        }
    )
    envelope = {
        "schema": "cvs.phase1.clic_terminal_envelope.v1",
        "method": "P1_CLIC",
        "strict_core": strict,
        "selected_checkpoint_path": str(checkpoint),
        "selected_checkpoint_sha256": checkpoint_sha,
    }
    terminal = checkpoint.parent / "phase1_clic_terminal_receipt.json"
    terminal.write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "candidate": candidate,
        "training_root": training_root,
        "checkpoint": checkpoint,
        "terminal": terminal,
        "checkpoint_sha": checkpoint_sha,
        "pre": pre,
        "strict": strict,
    }


def _clean_manifest(paths: dict[str, Path | str | dict[str, object]], *, arm: str) -> dict[str, object]:
    return {
        "schema": "cvs.phase1.clic_lv_export.v1",
        "method": "P1_CLIC",
        "source_only": True,
        "candidate_id": paths["candidate"],
        "run_id": TRAINING_RUN,
        "training_run_contract": TRAINING_RUN,
        "checkpoint": str(paths["checkpoint"]),
        "source_checkpoint_sha256": paths["checkpoint_sha"],
        "terminal_receipt_sha256": _sha_file(Path(paths["terminal"])),
        "clic_receipt_schema": "cvs.phase1.clic_receipt.v1",
        "clic_terminal_contract": "STRICT_CLIC_SOURCE_L_COMMON_C_G_RAW_UNSCALED_VJP_AMP_RESOURCE_GRAPH_RELEASE",
        "clic_terminal_contract_passed": True,
        "clic_enabled": arm == "G",
        "z_id_source_key": "z_id",
        "feature_name": "z_id",
        "feature_key": "z_id",
        "classification_head_contract": "dual_cvsincnet_tx_logits_v1",
        "source_tx_ids": list(SOURCE_TX),
        "known_validation_tx_ids": list(HELD_TX),
        "proxy_unknown_tx_ids": list(PROXY_TX),
        "proxy_selection_frozen_not_cli_tunable": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
        "labeled_row_count": 8,
        "source_validation_row_count": 4,
        "proxy_row_count": 400,
        "forwarded_roles": ["labeled_fit", "source_validation_known", "proxy_unknown"],
        "geometry_fit_role": "labeled_fit_only",
        "validation_proxy_fit_rows": 0,
        "validation_proxy_threshold_rows": 0,
        "unlabeled_loader_constructed": False,
        "unlabeled_forward_rows": 0,
    }


def _write_feature_npz(
    path: Path,
    paths: dict[str, Path | str | dict[str, object]],
    *,
    arm: str,
    feature_dim: int = 2,
) -> None:
    if type(feature_dim) is not int or feature_dim < 2:
        raise AssertionError("feature fixture dimension must be at least two")
    path.parent.mkdir(parents=True, exist_ok=True)
    fit_labels = np.asarray([tx for tx in SOURCE_TX for _ in range(2)], dtype=str)
    validation_labels = np.asarray([HELD_TX[0]] * 4, dtype=str)
    proxy_labels = np.asarray([PROXY_TX[0]] * 400, dtype=str)
    labels = np.concatenate([fit_labels, validation_labels, proxy_labels])
    roles = np.asarray(
        ["labeled_fit"] * fit_labels.size + ["source_validation_known"] * validation_labels.size + ["proxy_unknown"] * proxy_labels.size,
        dtype=str,
    )
    base = np.zeros((len(SOURCE_TX), feature_dim), dtype=np.float32)
    base[:, :2] = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
    if feature_dim > len(SOURCE_TX):
        base[:, len(SOURCE_TX)] = np.arange(len(SOURCE_TX), dtype=np.float32) * 0.05
    fit_rows: list[np.ndarray] = []
    for index in range(len(SOURCE_TX)):
        for offset in (0.0, 1.0):
            row = base[index].copy()
            row[0] += np.float32(offset * 0.1)
            fit_rows.append(row)
    fit_z = np.vstack(fit_rows)
    validation_z = np.zeros((validation_labels.size, feature_dim), dtype=np.float32)
    validation_z[:, :2] = np.asarray([[0.25, 0.25], [0.30, 0.20], [0.20, 0.30], [0.25, 0.20]], dtype=np.float32)
    proxy_z = np.zeros((400, feature_dim), dtype=np.float32)
    proxy_z[:, :2] = np.asarray([2.0, 2.0], dtype=np.float32)
    z_id = np.vstack([fit_z, validation_z, proxy_z])
    fit_logits = np.asarray([[3.0 if column == index else 0.0 for column in range(4)] for index in range(4) for _ in (0, 1)], dtype=np.float32)
    validation_logits = np.zeros((validation_labels.size, 4), dtype=np.float32)
    logits = np.vstack([fit_logits, validation_logits, np.zeros((proxy_labels.size, 4), dtype=np.float32)])
    physical = np.asarray([f"p-{i}" for i in range(labels.size)], dtype=str)
    receiver = np.asarray([SOURCE_RX[index % len(SOURCE_RX)] for index in range(labels.size)], dtype=str)
    days = np.asarray([SOURCE_DAYS[index % len(SOURCE_DAYS)] for index in range(labels.size)], dtype=str)
    raw_labels = np.asarray([SOURCE_TX.index(value) if value in SOURCE_TX else -1 for value in labels], dtype=np.int64)
    domain_labels = np.zeros(labels.size, dtype=np.int64)
    manifest = _clean_manifest(paths, arm=arm)
    np.savez(
        path,
        z_id=z_id,
        features=z_id.copy(),
        tx_logits=logits,
        raw_labels=raw_labels,
        domain_labels=domain_labels,
        tx_ids=labels,
        rx_ids=receiver,
        day_ids=days,
        eq_ids=np.asarray([f"eq-{index}" for index in range(labels.size)], dtype=str),
        sig_ids=physical,
        dataset_role=roles,
        channel_views=np.asarray(["clean"] * labels.size, dtype=str),
        sat_scenarios=np.asarray([""] * labels.size, dtype=str),
        manifest_json=np.asarray(json.dumps(manifest, ensure_ascii=True, sort_keys=True)),
    )


def _write_received_iq_fixture(path: Path) -> None:
    """Write one sealed received-IQ table reused by all three scenes."""

    rows = []
    iq = []
    for scene_index, scene in enumerate(SCENARIOS):
        for row_index, tx in enumerate(SOURCE_TX):
            physical = f"physical-{scene_index}-{row_index}"
            rows.append((tx, SOURCE_RX[row_index % len(SOURCE_RX)], SOURCE_DAYS[row_index % len(SOURCE_DAYS)], physical, scene))
            iq.append(np.asarray([scene_index + 1, row_index + 1, scene_index + row_index + 2], dtype=np.float32))
    arr = np.asarray(rows, dtype=str)
    np.savez(
        path,
        received_iq=np.asarray(iq, dtype=np.float32),
        tx_ids=arr[:, 0],
        rx_ids=arr[:, 1],
        day_ids=arr[:, 2],
        physical_sample_id=arr[:, 3],
        sat_scenarios=arr[:, 4],
    )


def _write_leo_export_received_iq_fixture(path: Path) -> None:
    """Write the runtime-shaped 7-RX×4-TX×3-scene existing IQ input."""

    rows = []
    iq = []
    for scene_index, scene in enumerate(SCENARIOS):
        for rx_index in range(7):
            for tx_index, tx in enumerate(SOURCE_TX):
                physical = f"runtime-{scene_index}-{rx_index}-{tx_index}"
                rows.append((tx, f"rx-{rx_index}", f"day-{rx_index % 2}", physical, scene))
                iq.append(np.full((2, 256), scene_index + rx_index + tx_index + 1, dtype=np.float32))
    arr = np.asarray(rows, dtype=str)
    np.savez(
        path,
        received_iq=np.asarray(iq, dtype=np.float32),
        tx_ids=arr[:, 0],
        rx_ids=arr[:, 1],
        day_ids=arr[:, 2],
        physical_sample_id=arr[:, 3],
        sat_scenarios=arr[:, 4],
    )


def _common_receipt(arm: str) -> dict[str, object]:
    receipt = _receipt(arm)
    return {
        "arm": arm,
        "fold_index": 1,
        "training_run_root": TRAINING_RUN,
        "physical_order_sha256": receipt["physical_order_sha256"],
        "class_order_sha256": receipt["class_order_sha256"],
        "source_split_sha256": receipt["source_split_sha256"],
        "common_batch_sequence_sha256": receipt["common_batch_sequence_sha256"],
        "scene_order": list(SCENARIOS),
        "physical_row_count": 12,
    }


def _clic_proxy_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return source-L fit, source-validation score-only, and fixed400 proxy rows."""

    source = np.asarray(
        [[1.0, 0.0], [1.0, 0.5], [0.0, 1.0], [0.0, 1.5], [-1.0, 0.0], [-1.0, 0.5], [0.0, -1.0], [0.0, -1.5]],
        dtype=np.float64,
    )
    source_labels = np.asarray([SOURCE_TX[0], SOURCE_TX[0], SOURCE_TX[1], SOURCE_TX[1], SOURCE_TX[2], SOURCE_TX[2], SOURCE_TX[3], SOURCE_TX[3]], dtype=str)
    validation = source + 0.05
    proxy = np.tile(np.asarray([[2.0, 2.0]], dtype=np.float64), (400, 1))
    return source, source_labels, validation, proxy


def _write_pair_received_iq(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create one existing source-L observation: 3 scenes × 7 RX × 4 TX × 20 rows."""

    rows: list[tuple[str, str, str, str, str, int, int]] = []
    iq: list[np.ndarray] = []
    for scene_index, scene in enumerate(SCENARIOS):
        for rx_slot in range(7):
            for tx_index, tx in enumerate(SOURCE_TX):
                for repeat in range(20):
                    rows.append((tx, f"rx-{rx_slot}", f"day-{repeat % 2}", f"pair-{scene_index}-{rx_slot}-{tx_index}-{repeat}", scene, rx_slot, tx_index))
                    iq.append(np.full((2, 256), scene_index + rx_slot + tx_index + repeat + 1, dtype=np.float32))
    table = np.asarray(rows, dtype=object)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        received_iq=np.asarray(iq, dtype=np.float32),
        tx_ids=np.asarray(table[:, 0], dtype=str),
        rx_ids=np.asarray(table[:, 1], dtype=str),
        day_ids=np.asarray(table[:, 2], dtype=str),
        physical_sample_id=np.asarray(table[:, 3], dtype=str),
        sat_scenarios=np.asarray(table[:, 4], dtype=str),
    )
    return (
        np.asarray(table[:, 0], dtype=str),
        np.asarray(table[:, 1], dtype=str),
        np.asarray(table[:, 2], dtype=str),
        np.asarray(table[:, 3], dtype=str),
        np.asarray(table[:, 4], dtype=str),
    )


def _duplicate_physical_id_across_scenes(source: Path, destination: Path) -> Path:
    """Copy a received-IQ table while reusing one physical ID in another scene."""

    with np.load(source, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    scenes = np.asarray(arrays["sat_scenarios"], dtype=str).reshape(-1)
    physical = np.asarray(arrays["physical_sample_id"], dtype=str).reshape(-1)
    first = int(np.flatnonzero(scenes == SCENARIOS[0])[0])
    second_candidates = np.flatnonzero(scenes == SCENARIOS[1])
    second = next(
        int(index)
        for index in second_candidates
        if str(arrays["tx_ids"].reshape(-1)[index]) == str(arrays["tx_ids"].reshape(-1)[first])
        and str(arrays["rx_ids"].reshape(-1)[index]) == str(arrays["rx_ids"].reshape(-1)[first])
        and str(arrays["day_ids"].reshape(-1)[index]) == str(arrays["day_ids"].reshape(-1)[first])
    )
    physical[second] = physical[first]
    arrays["physical_sample_id"] = physical
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)
    return destination


def _resign_leo_npz_with_cross_scene_duplicate(npz_path: Path, binding_path: Path) -> None:
    """Tamper a valid LEO artifact, then re-seal its manifest and binding hashes."""

    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    scenes = np.asarray(arrays["sat_scenarios"], dtype=str).reshape(-1)
    tx_ids = np.asarray(arrays["tx_ids"], dtype=str).reshape(-1)
    rx_ids = np.asarray(arrays["rx_ids"], dtype=str).reshape(-1)
    day_ids = np.asarray(arrays["day_ids"], dtype=str).reshape(-1)
    physical_ids = np.asarray(arrays["physical_sample_id"], dtype=str).reshape(-1)
    first = int(np.flatnonzero(scenes == SCENARIOS[0])[0])
    second = next(
        int(index)
        for index in np.flatnonzero(scenes == SCENARIOS[1])
        if tx_ids[index] == tx_ids[first] and rx_ids[index] == rx_ids[first] and day_ids[index] == day_ids[first]
    )
    physical_ids[second] = physical_ids[first]
    arrays["physical_sample_id"] = physical_ids
    physical_keys = [
        "|".join((tx_ids[index], rx_ids[index], day_ids[index], physical_ids[index]))
        for index in range(physical_ids.size)
    ]
    manifest = json.loads(str(np.asarray(arrays["manifest_json"]).item()))
    scenario_coverage = dict(manifest["scenario_coverage"])
    for scene in SCENARIOS:
        positions = np.flatnonzero(scenes == scene)
        scene_keys = [physical_keys[int(index)] for index in positions]
        scenario_coverage[scene] = dict(scenario_coverage[scene])
        scenario_coverage[scene]["physical_order_sha256"] = _canonical(scene_keys)
    manifest["scenario_coverage"] = scenario_coverage
    manifest["physical_order_sha256"] = _canonical(physical_keys)
    arrays["manifest_json"] = np.asarray(json.dumps(manifest, ensure_ascii=True, sort_keys=True))
    np.savez(npz_path, **arrays)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    binding["physical_keys"] = physical_keys
    binding["physical_order_sha256"] = _canonical(physical_keys)
    binding["scenario_coverage"] = scenario_coverage
    binding["leo_npz_sha256"] = _sha_file(npz_path)
    binding["leo_manifest_sha256"] = _canonical(manifest)
    binding_path.write_text(json.dumps(binding, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")


def _write_pair_leo_npz(
    path: Path,
    *,
    arm: str,
    paths: dict[str, Path | str | dict[str, object]],
    existing: Path,
    rows: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fold: int = 1,
    feature_dim: int = 2,
) -> dict[str, object]:
    """Use the production LEO exporter to seal manifest/member/binding hashes."""

    import cvsrffi.checkpoint_loading as checkpoint_loading
    import export_spaceborne_features

    tx_ids, rx_ids, day_ids, physical_ids, scenes = rows

    class DummyModel:
        def eval(self):
            return self

    def fake_extract(_model, loader, *, role: str, **_kwargs):
        count = len(loader.dataset)
        if count != len(tx_ids):
            raise AssertionError("pair LEO fixture row count drifted")
        metadata = [loader.dataset[index][3] for index in range(count)]
        actual_physical_ids = np.asarray([str(item["sig_i"]) for item in metadata], dtype=str)
        tx_index = np.asarray([SOURCE_TX.index(value) for value in tx_ids], dtype=np.int64)
        rx_slot = np.asarray([int(value.split("-")[1]) for value in rx_ids], dtype=np.int64)
        repeat = np.asarray([int(value.rsplit("-", 1)[1]) for value in actual_physical_ids], dtype=np.float64)
        base = np.zeros((len(SOURCE_TX), feature_dim), dtype=np.float32)
        base[:, :2] = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]], dtype=np.float32)
        features = base[tx_index]
        features[:, 0] += (rx_slot * 1e-3).astype(np.float32)
        features[:, 1] += (repeat * 1e-4).astype(np.float32)
        if feature_dim > len(SOURCE_TX):
            features[:, len(SOURCE_TX)] = (rx_slot * 1e-3 + repeat * 1e-4).astype(np.float32)
        logits = np.zeros((count, len(SOURCE_TX)), dtype=np.float32)
        logits[np.arange(count), tx_index] = 3.0
        return {
            "features": features,
            "tx_logits": logits,
            "raw_labels": tx_index,
            "domain_labels": np.zeros(count, dtype=np.int64),
            "tx_ids": tx_ids,
            "rx_ids": rx_ids,
            "day_ids": day_ids,
            "eq_ids": np.asarray(["existing_received_iq"] * count, dtype=str),
            "sig_ids": actual_physical_ids,
            "dataset_role": np.asarray([role] * count, dtype=str),
            "channel_views": np.asarray(["received_existing"] * count, dtype=str),
            "sat_scenarios": scenes,
        }

    args = LEO.build_parser().parse_args(
        [
            "--ckpt", str(paths["checkpoint"]),
            "--terminal-receipt-json", str(paths["terminal"]),
            "--existing-received-iq-npz", str(existing),
            "--out-npz", str(path),
            "--binding-json", str(path.with_name(path.stem + "_binding.json")),
            "--training-run-root", str(paths["training_root"]),
            "--postfreeze-output-root", str(path.parent),
            "--candidate-id", str(paths["candidate"]),
            "--fold-index", str(fold),
            "--arm", arm,
            "--source-tx-ids", ",".join(SOURCE_TX),
            "--device", "cpu",
            "--batch-size", "64",
        ]
    )
    patch = pytest.MonkeyPatch()
    try:
        patch.setattr(checkpoint_loading, "build_exact_ssdg_model_from_checkpoint", lambda *_args, **_kwargs: (DummyModel(), {"fixture": True}))
        patch.setattr(export_spaceborne_features, "extract_features_with_metadata", fake_extract)
        LEO.export(args)
    finally:
        patch.undo()
    binding_path = Path(args.binding_json)
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as exported:
        assert {"z_id", "features", "tx_logits", "dataset_role", "manifest_json", "source_rx_slot", "sat_scenarios"} <= set(exported.files)
        assert np.asarray(exported["z_id"]).shape == (len(tx_ids), feature_dim)
        assert np.isfinite(np.asarray(exported["z_id"], dtype=np.float64)).all()
        manifest = json.loads(str(np.asarray(exported["manifest_json"]).item()))
    assert manifest["schema"] == "cvs.phase1.clic_leo_export.v1"
    assert manifest["single_leo_observation_required"] is True
    assert manifest["single_leo_forward_count"] == len(tx_ids)
    assert binding["schema"] == "cvs.phase1.clic_leo_binding.v1"
    assert binding["leo_npz_sha256"] == _sha_file(path)
    assert binding["leo_manifest_sha256"] == _canonical(manifest)
    assert binding["received_iq_sha256"] == _sha_file(existing)
    return binding


def _pair_artifact_fixture(tmp_path: Path) -> dict[str, object]:
    """Create C/G raw feature, binding, receipt, and proxy-summary artifacts."""

    artifacts: dict[str, object] = {}
    existing = tmp_path / "existing_received_iq.npz"
    pair_rows = _write_pair_received_iq(existing)
    for arm in ("C", "G"):
        paths = _checkpoint_fixture(tmp_path / arm, arm=arm)
        clean = tmp_path / arm / "clean.npz"
        _write_feature_npz(clean, paths, arm=arm)
        leo = tmp_path / arm / "leo.npz"
        binding = _write_pair_leo_npz(leo, arm=arm, paths=paths, existing=existing, rows=pair_rows)
        binding_path = tmp_path / arm / "leo_binding.json"
        binding_path.write_text(json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8")
        receipt_path = tmp_path / arm / "common_receipt.json"
        receipt_path.write_text(json.dumps(_common_receipt(arm), sort_keys=True) + "\n", encoding="utf-8")
        proxy_path = tmp_path / arm / "proxy_diagnostic.json"
        PAIR.export_clic_proxy_diagnostic(
            clean_npz_path=clean,
            output_json_path=proxy_path,
        )
        artifacts[f"{arm.lower()}_paths"] = paths
        artifacts[f"{arm.lower()}_clean"] = clean
        artifacts[f"{arm.lower()}_leo"] = leo
        artifacts[f"{arm.lower()}_binding"] = binding_path
        artifacts[f"{arm.lower()}_receipt"] = receipt_path
        artifacts[f"{arm.lower()}_proxy"] = proxy_path
    artifacts["existing_received_iq"] = existing
    pair_json = tmp_path / "pair.json"
    artifacts["pair_json"] = pair_json
    return artifacts


def _pair_fold_artifact_fixture(
    tmp_path: Path,
    *,
    fold: int,
    real_g_bundle: bool = False,
) -> dict[str, object]:
    """Build one complete C/G pair chain with the frozen shared LEO bytes."""

    if fold not in range(1, 7):
        raise AssertionError("pair fixture fold must be one-based and bounded")
    artifacts: dict[str, object] = {}
    existing = tmp_path / "existing_received_iq.npz"
    pair_rows = _write_pair_received_iq(existing)
    feature_dim = 160 if real_g_bundle else 2
    for arm in ("C", "G"):
        paths = _checkpoint_fixture(
            tmp_path / arm,
            arm=arm,
            fold=fold,
            real_model=bool(real_g_bundle and arm == "G"),
        )
        clean = tmp_path / arm / "clean.npz"
        _write_feature_npz(clean, paths, arm=arm, feature_dim=feature_dim)
        leo = tmp_path / arm / "leo.npz"
        binding = _write_pair_leo_npz(
            leo,
            arm=arm,
            paths=paths,
            existing=existing,
            rows=pair_rows,
            fold=fold,
            feature_dim=feature_dim,
        )
        binding_path = tmp_path / arm / "leo_binding.json"
        binding_path.write_text(json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8")
        receipt = _common_receipt(arm)
        receipt["fold_index"] = fold
        receipt_path = tmp_path / arm / "common_receipt.json"
        receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
        proxy_path = tmp_path / arm / "proxy_diagnostic.json"
        PAIR.export_clic_proxy_diagnostic(
            clean_npz_path=clean,
            output_json_path=proxy_path,
        )
        artifacts[f"{arm.lower()}_paths"] = paths
        artifacts[f"{arm.lower()}_clean"] = clean
        artifacts[f"{arm.lower()}_leo"] = leo
        artifacts[f"{arm.lower()}_binding"] = binding_path
        artifacts[f"{arm.lower()}_receipt"] = receipt_path
        artifacts[f"{arm.lower()}_proxy"] = proxy_path
    if real_g_bundle:
        bundle_path = tmp_path / "G" / "deployment.bundle.zip"
        BUNDLE.export_bundle(
            checkpoint_path=artifacts["g_paths"]["checkpoint"],
            terminal_receipt_path=artifacts["g_paths"]["terminal"],
            output_path=bundle_path,
            clean_npz_path=artifacts["g_clean"],
            leo_npz_path=artifacts["g_leo"],
            leo_binding_path=artifacts["g_binding"],
        )
        verified = BUNDLE.verify_clic_bundle(bundle_path)
        assert verified["state_origin"] == "checkpoint_model_exact"
        assert verified["real_checkpoint_state_rebuild_verified"] is True
        assert verified["real_checkpoint_reload_verified"] is False
        artifacts["g_bundle"] = bundle_path
    artifacts["existing_received_iq"] = existing
    artifacts["pair_json"] = tmp_path / "pair.json"
    return artifacts


def _write_proxy_without_field(source: Path, destination: Path, field: str) -> None:
    """Copy a production-written proxy record while removing one sealed field."""

    payload = json.loads(source.read_text(encoding="utf-8"))
    assert field in payload, (field, sorted(payload))
    payload.pop(field)
    destination.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _pair_cli_argv(artifacts: dict[str, object], *, fold: int = 1) -> list[str]:
    return [
        "--c-checkpoint", str(artifacts["c_paths"]["checkpoint"]),
        "--g-checkpoint", str(artifacts["g_paths"]["checkpoint"]),
        "--c-terminal-receipt-json", str(artifacts["c_paths"]["terminal"]),
        "--g-terminal-receipt-json", str(artifacts["g_paths"]["terminal"]),
        "--c-clean-npz", str(artifacts["c_clean"]),
        "--g-clean-npz", str(artifacts["g_clean"]),
        "--c-leo-npz", str(artifacts["c_leo"]),
        "--g-leo-npz", str(artifacts["g_leo"]),
        "--c-leo-binding-json", str(artifacts["c_binding"]),
        "--g-leo-binding-json", str(artifacts["g_binding"]),
        "--c-common-receipt-json", str(artifacts["c_receipt"]),
        "--g-common-receipt-json", str(artifacts["g_receipt"]),
        "--c-proxy-diagnostic-json", str(artifacts["c_proxy"]),
        "--g-proxy-diagnostic-json", str(artifacts["g_proxy"]),
        "--fold-index", str(fold),
        "--training-run-root", TRAINING_RUN,
        "--source-tx-ids", ",".join(SOURCE_TX),
        "--output-pair-json", str(artifacts["pair_json"]),
    ]


def test_clic_clean_exporter_reopens_versioned_terminal_and_checkpoint_contract(tmp_path: Path) -> None:
    paths = _checkpoint_fixture(tmp_path, arm="G")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    args, receipt, arm = CLEAN.validate_clic_training_checkpoint(
        checkpoint,
        checkpoint_path=paths["checkpoint"],
        terminal_receipt_path=paths["terminal"],
        source_tx_ids=SOURCE_TX,
        known_validation_tx_ids=HELD_TX,
        proxy_unknown_tx_ids=PROXY_TX,
    )
    assert arm == "G"
    assert args["checkpoint_selection"] == "final_only"
    assert receipt["completed"] is True
    assert receipt["final_checkpoint_sha256"] == paths["checkpoint_sha"]
    assert receipt["terminal_contract"] != "AWAITING_EXTERNAL_CHECKPOINT_SHA"
    assert receipt["source_l_only"] is True
    assert CLEAN.EXPECTED_LV_EXPORT_SCHEMA == "cvs.phase1.clic_lv_export.v1"


def test_clic_clean_export_writes_feature_npz_not_manifest_only(tmp_path: Path) -> None:
    """The public export path must materialize finite feature rows and manifest."""

    import cvsrffi.checkpoint_loading as checkpoint_loading
    import dataset_wisig
    import export_spaceborne_features

    paths = _checkpoint_fixture(tmp_path, arm="G")
    dataset_path = tmp_path / "synthetic_wisig.pkl"
    dataset_path.write_bytes(b"mechanical-clic-export-fixture")
    dataset_sha = _sha_file(dataset_path)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    checkpoint["args"].update({"wisig_pkl": str(dataset_path), "wisig_pkl_sha256": dataset_sha})
    torch.save(checkpoint, paths["checkpoint"])
    checkpoint_sha = _sha_file(paths["checkpoint"])
    terminal = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    terminal["selected_checkpoint_sha256"] = checkpoint_sha
    terminal["strict_core"]["final_checkpoint_sha256"] = checkpoint_sha
    paths["terminal"].write_text(json.dumps(terminal, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

    class DummyRows:
        tx_list = list(SOURCE_TX)

        def __init__(self, count: int):
            self.count = int(count)

        def __len__(self) -> int:
            return self.count

        def __getitem__(self, index: int):
            return (
                torch.zeros(3, dtype=torch.float32),
                torch.tensor(0, dtype=torch.long),
                torch.tensor(0, dtype=torch.long),
                {"tx": SOURCE_TX[index % len(SOURCE_TX)], "rx": SOURCE_RX[0], "day": SOURCE_DAYS[0], "equalized": "eq", "sig_i": str(index)},
            )

    class DummyModel:
        def eval(self):
            return self

    def fake_extract(_model, _loader, *, role: str, **_kwargs):
        count = {"labeled_fit": 4, "source_validation_known": 2, "proxy_unknown": 400}[role]
        tx = np.asarray([SOURCE_TX[index % len(SOURCE_TX)] for index in range(count)])
        return {
            "features": np.ones((count, 2), dtype=np.float32),
            "tx_logits": np.zeros((count, 4), dtype=np.float32),
            "raw_labels": np.zeros(count, dtype=np.int64),
            "domain_labels": np.zeros(count, dtype=np.int64),
            "tx_ids": tx,
            "rx_ids": np.asarray([SOURCE_RX[0]] * count),
            "day_ids": np.asarray([SOURCE_DAYS[0]] * count),
            "eq_ids": np.asarray(["eq"] * count),
            "sig_ids": np.asarray([str(index) for index in range(count)]),
            "dataset_role": np.asarray([role] * count),
            "channel_views": np.asarray(["clean"] * count),
            "sat_scenarios": np.asarray([""] * count),
        }

    fake_source = DummyRows(6)
    fake_reconstructed = {
        "source_base": fake_source,
        "labeled_indices": (0, 1, 2, 3),
        "validation_indices": (4, 5),
        "unlabeled_indices": (),
        "source_split_receipt": {},
        "tx_partition_receipt": {},
    }
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(CLEAN, "FROZEN_WISIG_SHA256", dataset_sha)
        monkeypatch.setattr(CLEAN, "_reconstruct_source_l_v", lambda **_kwargs: fake_reconstructed)
        monkeypatch.setattr(CLEAN, "_assert_current_source_split", lambda **_kwargs: None)
        monkeypatch.setattr(CLEAN, "_physical_keys_for_indices", lambda base, indices: tuple(f"{id(base)}-{int(index)}" for index in indices))
        monkeypatch.setattr(dataset_wisig, "load_wisig_compact_pkl", lambda _path: {})
        monkeypatch.setattr(dataset_wisig, "WiSigSubsetDataset", lambda _base, indices, split_source: DummyRows(len(tuple(indices))))
        monkeypatch.setattr(export_spaceborne_features, "_build_wisig_dataset", lambda **_kwargs: (DummyRows(400), {}))
        monkeypatch.setattr(export_spaceborne_features, "extract_features_with_metadata", fake_extract)
        monkeypatch.setattr(checkpoint_loading, "build_exact_ssdg_model_from_checkpoint", lambda *_args, **_kwargs: (DummyModel(), {}))

        output = tmp_path / "source_l_export.npz"
        args = argparse.Namespace(
            ckpt=str(paths["checkpoint"]),
            terminal_receipt_json=str(paths["terminal"]),
            wisig_pkl=str(dataset_path),
            expected_wisig_sha256=dataset_sha,
            source_tx_ids=",".join(SOURCE_TX),
            known_validation_tx_ids=",".join(HELD_TX),
            proxy_unknown_tx_ids=",".join(PROXY_TX),
            source_feature_npz="",
            source_l_npz="",
            output_npz=str(output),
            out_npz=str(output),
            batch_size=32,
            device="cpu",
        )
        result = CLEAN.export(args)
    finally:
        monkeypatch.undo()
    assert output.is_file(), "CLEAN.export must write args.output_npz"
    if isinstance(result, dict) and "out_npz" in result:
        assert Path(str(result["out_npz"])).resolve() == output.resolve()
    with np.load(output, allow_pickle=False) as data:
        members = set(data.files)
        assert {"z_id", "features", "tx_logits", "tx_ids", "dataset_role", "manifest_json"} <= members
        z_id = np.asarray(data["z_id"])
        assert z_id.ndim == 2 and z_id.shape[0] == 406 and z_id.shape[0] > 0
        assert np.issubdtype(z_id.dtype, np.floating)
        assert np.isfinite(z_id).all()
        manifest = json.loads(str(np.asarray(data["manifest_json"]).item()))
    assert manifest["schema"] == "cvs.phase1.clic_lv_export.v1"
    assert manifest["source_only"] is True


def test_clic_clean_rejects_synchronized_checkpoint_seed_drift(tmp_path: Path) -> None:
    """A valid fixture reopens, but a rehashed seed mutation fails closed."""

    paths = _checkpoint_fixture(tmp_path, arm="G")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    CLEAN.validate_clic_training_checkpoint(
        checkpoint,
        checkpoint_path=paths["checkpoint"],
        terminal_receipt_path=paths["terminal"],
        source_tx_ids=SOURCE_TX,
        known_validation_tx_ids=HELD_TX,
        proxy_unknown_tx_ids=PROXY_TX,
    )

    checkpoint["args"]["seed"] = 7281105
    torch.save(checkpoint, paths["checkpoint"])
    drift_sha = _sha_file(paths["checkpoint"])
    envelope = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    envelope["selected_checkpoint_sha256"] = drift_sha
    envelope["strict_core"]["final_checkpoint_sha256"] = drift_sha
    paths["terminal"].write_text(json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    drifted_checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    with pytest.raises(Exception, match="seed|drift"):
        CLEAN.validate_clic_training_checkpoint(
            drifted_checkpoint,
            checkpoint_path=paths["checkpoint"],
            terminal_receipt_path=paths["terminal"],
            source_tx_ids=SOURCE_TX,
            known_validation_tx_ids=HELD_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )


@pytest.mark.parametrize("forbidden", ("icmt_receipt_schema", "source_receiver_ids", "target_rows", "raw_iq", "sample_ids"))
def test_clic_clean_terminal_rejects_nested_legacy_receiver_target_and_sample_state(tmp_path: Path, forbidden: str) -> None:
    paths = _checkpoint_fixture(tmp_path, arm="G")
    envelope = json.loads(paths["terminal"].read_text(encoding="utf-8"))
    envelope["strict_core"]["nested"] = {"deep": {forbidden: ["forbidden"]}}
    paths["terminal"].write_text(json.dumps(envelope, sort_keys=True), encoding="utf-8")
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu")
    with pytest.raises(Exception, match="legacy|historical|raw|target|sample|receiver|forbidden|receipt"):
        CLEAN.validate_clic_training_checkpoint(
            checkpoint,
            checkpoint_path=paths["checkpoint"],
            terminal_receipt_path=paths["terminal"],
            source_tx_ids=SOURCE_TX,
            known_validation_tx_ids=HELD_TX,
            proxy_unknown_tx_ids=PROXY_TX,
        )


def test_clic_leo_exporter_reuses_one_existing_received_iq_with_three_scene_and_common_order_binding(tmp_path: Path) -> None:
    existing = tmp_path / "existing_received_iq.npz"
    _write_received_iq_fixture(existing)
    bindings: dict[str, dict[str, object]] = {}
    for arm in ("C", "G"):
        paths = _checkpoint_fixture(tmp_path / arm, arm=arm)
        args = LEO.build_parser().parse_args(
            [
                "--ckpt", str(paths["checkpoint"]),
                "--terminal-receipt-json", str(paths["terminal"]),
                "--existing-received-iq-npz", str(existing),
                "--out-npz", str(tmp_path / f"{arm.lower()}_leo.npz"),
                "--binding-json", str(tmp_path / f"{arm.lower()}_leo_binding.json"),
                "--training-run-root", str(paths["training_root"]),
                "--postfreeze-output-root", str(tmp_path / POSTFREEZE_MATRIX),
                "--candidate-id", str(paths["candidate"]),
                "--fold-index", "1",
                "--arm", arm,
                "--source-tx-ids", ",".join(SOURCE_TX),
            ]
        )
        binding = LEO.build_binding_from_existing(args)
        bindings[arm] = binding
        assert binding["schema"] == LEO.EXPECTED_BINDING_SCHEMA
        assert tuple(binding["satellite_scenarios"]) == SCENARIOS
        assert binding["single_leo_forward_bound"] is True
        assert binding["common_physical_order_bound"] is True
        assert binding["existing_received_iq_sha256"] == _sha_file(existing)
        physical = binding["physical_keys"]
        assert len(physical) == len(set(physical))
        for scene in SCENARIOS:
            assert int(binding["scenario_coverage"][scene]["count"]) > 0
    assert bindings["C"]["existing_received_iq_sha256"] == bindings["G"]["existing_received_iq_sha256"]
    assert bindings["C"]["physical_keys"] == bindings["G"]["physical_keys"]


def test_clic_leo_main_writes_existing_iq_features_before_binding_without_regeneration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LEO main must materialize out_npz from one existing received-IQ table."""

    import cvsrffi.checkpoint_loading as checkpoint_loading
    import export_spaceborne_features

    paths = _checkpoint_fixture(tmp_path, arm="G")
    existing = tmp_path / "existing_received_iq.npz"
    _write_leo_export_received_iq_fixture(existing)
    output = tmp_path / "leo_features.npz"
    binding_path = tmp_path / "leo_binding.json"
    calls: list[Path] = []
    original_loader = LEO._load_existing_received_iq

    def counted_loader(path: Path, *, source_tx_ids: tuple[str, ...]):
        calls.append(Path(path).resolve())
        return original_loader(path, source_tx_ids=source_tx_ids)

    class DummyModel:
        def eval(self):
            return self

    def fake_extract(_model, loader, *, role: str, **_kwargs):
        count = len(loader.dataset)
        return {
            "features": np.ones((count, 2), dtype=np.float32),
            "tx_logits": np.zeros((count, 4), dtype=np.float32),
            "raw_labels": np.zeros(count, dtype=np.int64),
            "domain_labels": np.zeros(count, dtype=np.int64),
            "tx_ids": np.asarray([SOURCE_TX[index % len(SOURCE_TX)] for index in range(count)]),
            "rx_ids": np.asarray([f"rx-{index % 7}" for index in range(count)]),
            "day_ids": np.asarray([f"day-{index % 2}" for index in range(count)]),
            "eq_ids": np.asarray(["existing_received_iq"] * count),
            "sig_ids": np.asarray([str(index) for index in range(count)]),
            "dataset_role": np.asarray([role] * count),
            "channel_views": np.asarray(["received_existing"] * count),
            "sat_scenarios": np.asarray([SCENARIOS[index // 28] for index in range(count)]),
        }

    monkeypatch.setattr(LEO, "_load_existing_received_iq", counted_loader)
    monkeypatch.setattr(checkpoint_loading, "build_exact_ssdg_model_from_checkpoint", lambda *_args, **_kwargs: (DummyModel(), {}))
    monkeypatch.setattr(export_spaceborne_features, "extract_features_with_metadata", fake_extract)
    before_sha = _sha_file(existing)
    rc = LEO.main(
        [
            "--ckpt", str(paths["checkpoint"]),
            "--terminal-receipt-json", str(paths["terminal"]),
            "--existing-received-iq-npz", str(existing),
            "--out-npz", str(output),
            "--binding-json", str(binding_path),
            "--training-run-root", str(paths["training_root"]),
            "--postfreeze-output-root", str(tmp_path),
            "--candidate-id", str(paths["candidate"]),
            "--fold-index", "1",
            "--arm", "G",
            "--source-tx-ids", ",".join(SOURCE_TX),
        ]
    )
    assert rc == 0
    assert output.is_file(), "LEO export must write out_npz before binding"
    assert binding_path.is_file()
    assert calls and set(calls) == {existing.resolve()}, "LEO must reuse one existing received-IQ table"
    assert _sha_file(existing) == before_sha
    assert len(list(tmp_path.glob("*received_iq*.npz"))) == 1
    with np.load(existing, allow_pickle=False) as source, np.load(output, allow_pickle=False) as exported:
        assert {"z_id", "features", "tx_logits", "manifest_json"} <= set(exported.files)
        assert exported["z_id"].shape[0] == source["received_iq"].shape[0]
        assert np.isfinite(np.asarray(exported["z_id"], dtype=np.float64)).all()
        leo_manifest = json.loads(str(np.asarray(exported["manifest_json"]).item()))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["existing_received_iq_sha256"] == before_sha
    assert binding["single_leo_observation"] is True
    assert leo_manifest["received_iq_sha256"] == before_sha


def test_clic_leo_loader_and_pair_reject_physical_id_reuse_across_scenes(tmp_path: Path) -> None:
    """A physical sample identity is global; scene is never part of its uniqueness key."""

    base = tmp_path / "existing_received_iq.npz"
    _write_pair_received_iq(base)
    duplicate = _duplicate_physical_id_across_scenes(base, tmp_path / "duplicate_received_iq.npz")
    with pytest.raises(Exception, match="duplicate|physical|unique|scene"):
        LEO._load_existing_received_iq(duplicate, source_tx_ids=SOURCE_TX)

    artifacts = _pair_artifact_fixture(tmp_path / "pair")
    for arm in ("C", "G"):
        _resign_leo_npz_with_cross_scene_duplicate(
            Path(artifacts[f"{arm.lower()}_leo"]),
            Path(artifacts[f"{arm.lower()}_binding"]),
        )
    with pytest.raises(Exception, match="duplicate|physical|unique|scene"):
        PAIR.evaluate(PAIR.build_parser().parse_args(_pair_cli_argv(artifacts)))


def test_clic_leo_export_fails_closed_if_existing_iq_changes_after_first_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cvsrffi.checkpoint_loading as checkpoint_loading
    import export_spaceborne_features

    paths = _checkpoint_fixture(tmp_path / "G", arm="G")
    existing = tmp_path / "existing_received_iq.npz"
    _write_leo_export_received_iq_fixture(existing)
    output = tmp_path / "leo_features.npz"
    binding_path = tmp_path / "leo_binding.json"
    args = LEO.build_parser().parse_args(
        [
            "--ckpt", str(paths["checkpoint"]),
            "--terminal-receipt-json", str(paths["terminal"]),
            "--existing-received-iq-npz", str(existing),
            "--out-npz", str(output),
            "--binding-json", str(binding_path),
            "--training-run-root", str(paths["training_root"]),
            "--postfreeze-output-root", str(tmp_path),
            "--candidate-id", str(paths["candidate"]),
            "--fold-index", "1",
            "--arm", "G",
            "--source-tx-ids", ",".join(SOURCE_TX),
        ]
    )
    original_loader = LEO._load_existing_received_iq

    def replacing_loader(path: Path, *, source_tx_ids: tuple[str, ...]):
        loaded = original_loader(path, source_tx_ids=source_tx_ids)
        replacement = path.with_name(path.stem + ".replacement.npz")
        _write_leo_export_received_iq_fixture(replacement)
        with np.load(replacement, allow_pickle=False) as archive:
            replacement_arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
        replacement_arrays["received_iq"][0, 0, 0] += np.float32(1.0)
        np.savez(replacement, **replacement_arrays)
        replacement.replace(path)
        return loaded

    class DummyModel:
        def eval(self):
            return self

    def fake_extract(_model, loader, *, role: str, **_kwargs):
        count = len(loader.dataset)
        metadata = [loader.dataset[index][3] for index in range(count)]
        return {
            "features": np.ones((count, 2), dtype=np.float32),
            "tx_logits": np.zeros((count, 4), dtype=np.float32),
            "raw_labels": np.zeros(count, dtype=np.int64),
            "domain_labels": np.zeros(count, dtype=np.int64),
            "tx_ids": np.asarray([str(item["tx"]) for item in metadata], dtype=str),
            "rx_ids": np.asarray([str(item["rx"]) for item in metadata], dtype=str),
            "day_ids": np.asarray([str(item["day"]) for item in metadata], dtype=str),
            "eq_ids": np.asarray(["existing_received_iq"] * count, dtype=str),
            "sig_ids": np.asarray([str(item["sig_i"]) for item in metadata], dtype=str),
            "dataset_role": np.asarray([role] * count, dtype=str),
            "channel_views": np.asarray(["received_existing"] * count, dtype=str),
            "sat_scenarios": np.asarray([SCENARIOS[index // 28] for index in range(count)], dtype=str),
        }

    monkeypatch.setattr(LEO, "_load_existing_received_iq", replacing_loader)
    monkeypatch.setattr(checkpoint_loading, "build_exact_ssdg_model_from_checkpoint", lambda *_args, **_kwargs: (DummyModel(), {}))
    monkeypatch.setattr(export_spaceborne_features, "extract_features_with_metadata", fake_extract)
    with pytest.raises(Exception) as exc_info:
        LEO.export(args)
    assert re.search("changed|snapshot|hash|SHA|received|immutable", str(exc_info.value), re.IGNORECASE), str(exc_info.value)
    assert not output.exists()
    assert not binding_path.exists()


def test_clic_float64_totalized_l2_preserves_exact_zero_and_rejects_nonfinite() -> None:
    features = np.asarray([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)
    normalized = PAIR.safe_totalized_l2_float64(features, label="CLIC fixture")
    assert normalized.dtype == np.float64
    np.testing.assert_array_equal(normalized[1], np.zeros(2, dtype=np.float64))
    np.testing.assert_allclose(normalized[0], np.asarray([0.6, 0.8], dtype=np.float64))
    with pytest.raises(Exception, match="non-finite"):
        PAIR.safe_totalized_l2_float64(np.asarray([[np.nan, 1.0]], dtype=np.float64), label="bad")


def test_clic_source_geometry_fits_source_l_only_and_never_accepts_v_or_proxy_rows() -> None:
    source = np.asarray(
        [[1.0, 0.0], [1.0, 0.5], [0.0, 1.0], [0.0, 1.5], [-1.0, 0.0], [-1.0, 0.5], [0.0, -1.0], [0.0, -1.5]],
        dtype=np.float64,
    )
    labels = np.asarray([SOURCE_TX[0], SOURCE_TX[0], SOURCE_TX[1], SOURCE_TX[1], SOURCE_TX[2], SOURCE_TX[2], SOURCE_TX[3], SOURCE_TX[3]], dtype=str)
    geometry = PAIR.fit_clic_source_geometry(source, labels, SOURCE_TX)
    assert geometry["class_counts"] == {tx: 2 for tx in SOURCE_TX}
    assert np.asarray(geometry["means"]).dtype == np.float64
    assert np.asarray(geometry["variances"]).dtype == np.float64
    validation = source + 0.05
    proxy = np.tile(np.asarray([[2.0, 2.0]], dtype=np.float64), (400, 1))
    with pytest.raises(Exception, match="source|fit|role|validation|proxy"):
        PAIR.fit_clic_source_geometry(
            np.vstack([source, validation, proxy]),
            np.asarray(list(labels) + ["source_validation_known"] * validation.shape[0] + [PROXY_TX[0]] * proxy.shape[0], dtype=str),
            SOURCE_TX,
        )


def _leo_calibration_fixture(scene: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    rows_per_cell = 20
    cell_count = 7 * len(SOURCE_TX)
    cell = np.arange(cell_count, dtype=np.int64).repeat(rows_per_cell)
    rx_slot = np.repeat(np.arange(7, dtype=np.int64), len(SOURCE_TX) * rows_per_cell)
    y = np.tile(np.repeat(np.asarray(SOURCE_TX, dtype=str), rows_per_cell), 7)
    base = np.asarray([[1.0, 0.0], [1.0, 0.5], [0.0, 1.0], [0.0, 1.5]], dtype=np.float64)
    z = np.vstack([base[(index % len(SOURCE_TX))] + (index % rows_per_cell) * 1e-4 for index in range(cell.size)])
    scene_rows = np.asarray([scene] * cell.size, dtype=str)
    physical_binding = {
        "received_iq_sha256": _sha_text("same-received-iq-bytes"),
        "physical_order_sha256": _canonical([f"physical-{index}" for index in range(cell.size)]),
        "single_leo_observation": True,
        "source_only": True,
    }
    return z, scene_rows, rx_slot, {"y": y, "cell": cell, "physical_binding": physical_binding}


def test_clic_source_leo_tail_policy_uses_28_cells_and_same_c_g_physical_binding_without_rows() -> None:
    clean = np.asarray(
        [[1.0, 0.0], [1.0, 0.5], [0.0, 1.0], [0.0, 1.5], [-1.0, 0.0], [-1.0, 0.5], [0.0, -1.0], [0.0, -1.5]],
        dtype=np.float64,
    )
    labels = np.asarray([SOURCE_TX[0], SOURCE_TX[0], SOURCE_TX[1], SOURCE_TX[1], SOURCE_TX[2], SOURCE_TX[2], SOURCE_TX[3], SOURCE_TX[3]], dtype=str)
    geometry = PAIR.fit_clic_source_geometry(clean, labels, SOURCE_TX)
    policies = {}
    for scene in SCENARIOS:
        leo_z, scene_rows, rx_slot, extra = _leo_calibration_fixture(scene)
        policy = PAIR.freeze_clic_tail_policy(
            geometry,
            leo_z,
            scene_rows,
            rx_slot,
            extra["y"],
            extra["physical_binding"],
        )
        policies[scene] = policy
        assert policy["scene"] == scene
        assert policy["cell_count"] == 28
        assert policy["min_cell_total"] >= 20
        assert policy["min_cell_positive"] >= 20
        assert float(policy["a_s"]) <= float(policy["b_s"])
        assert policy["received_iq_sha256"] == extra["physical_binding"]["received_iq_sha256"]
        assert policy["physical_order_sha256"] == extra["physical_binding"]["physical_order_sha256"]
        assert "rows" not in policy and "physical_ids" not in policy
        tx_logits = np.zeros((leo_z.shape[0], len(SOURCE_TX)), dtype=np.float64)
        tx_logits[:, 0] = 1.0
        scored = PAIR.score_clic_open_set(geometry, policy, leo_z, tx_logits, scene)
        energy = np.asarray(scored["e_unknown"], dtype=np.float64)
        for cell in range(28):
            mask = extra["cell"] == cell
            assert float(np.mean(energy[mask] > float(policy["a_s"]))) <= 0.10 + 1e-12
            assert float(np.mean(energy[mask] > float(policy["b_s"]))) <= 0.05 + 1e-12
    leo_z, scene_rows, rx_slot, extra = _leo_calibration_fixture(SCENARIOS[0])
    c_policy = PAIR.freeze_clic_tail_policy(geometry, leo_z, scene_rows, rx_slot, extra["y"], extra["physical_binding"])
    g_policy = PAIR.freeze_clic_tail_policy(geometry, leo_z.copy(), scene_rows.copy(), rx_slot.copy(), extra["y"].copy(), dict(extra["physical_binding"]))
    assert c_policy["state_sha256"] == g_policy["state_sha256"]


def test_clic_open_set_score_is_zero_tie_safe_and_fail_closed_on_nonfinite() -> None:
    clean = np.asarray(
        [[1.0, 0.0], [1.0, 0.5], [0.0, 1.0], [0.0, 1.5], [-1.0, 0.0], [-1.0, 0.5], [0.0, -1.0], [0.0, -1.5]],
        dtype=np.float64,
    )
    labels = np.asarray([SOURCE_TX[0], SOURCE_TX[0], SOURCE_TX[1], SOURCE_TX[1], SOURCE_TX[2], SOURCE_TX[2], SOURCE_TX[3], SOURCE_TX[3]], dtype=str)
    geometry = PAIR.fit_clic_source_geometry(clean, labels, SOURCE_TX)
    leo_z, scene_rows, rx_slot, extra = _leo_calibration_fixture(SCENARIOS[0])
    policy = PAIR.freeze_clic_tail_policy(geometry, leo_z, scene_rows, rx_slot, extra["y"], extra["physical_binding"])
    z_id = np.asarray([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    tx_logits = np.asarray([[3.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [2.0, 2.0, 0.0, 0.0]], dtype=np.float64)
    scored = PAIR.score_clic_open_set(geometry, policy, z_id, tx_logits, SCENARIOS[0])
    decisions = np.asarray(scored["decision"], dtype=str).reshape(-1)
    assert decisions[1] == "defer"  # exact zero row has no registered identity
    assert decisions[2] == "defer"  # exact head tie is never assigned to a class
    assert np.isfinite(np.asarray(scored["e_unknown"], dtype=np.float64)).all()
    with pytest.raises(Exception, match="non-finite|finite|fail"):
        PAIR.score_clic_open_set(geometry, policy, np.asarray([[np.nan, 0.0]]), np.asarray([[1.0, 0.0, 0.0, 0.0]]), SCENARIOS[0])


def test_clic_decision_boundaries_are_exact_zero_tie_and_nonfinite_safe() -> None:
    unique = np.asarray([3.0, 0.0, 0.0, 0.0], dtype=np.float64)
    tie = np.asarray([3.0, 3.0, 0.0, 0.0], dtype=np.float64)
    registered = PAIR.decide_clic_open_set(1.0, unique, a_s=1.0, b_s=2.0, zero_flag=False)
    assert registered["decision"] == "registered"
    assert registered["predicted_index"] == 0
    assert PAIR.decide_clic_open_set(1.5, unique, a_s=1.0, b_s=2.0, zero_flag=False)["decision"] == "defer"
    assert PAIR.decide_clic_open_set(2.0, unique, a_s=1.0, b_s=2.0, zero_flag=False)["decision"] == "defer"
    assert PAIR.decide_clic_open_set(2.0000001, unique, a_s=1.0, b_s=2.0, zero_flag=False)["decision"] == "unknown"
    zero = PAIR.decide_clic_open_set(0.0, unique, a_s=1.0, b_s=2.0, zero_flag=True)
    assert zero["decision"] == "defer" and zero["predicted_index"] in (None, -1)
    tied = PAIR.decide_clic_open_set(0.5, tie, a_s=1.0, b_s=2.0, zero_flag=False)
    assert tied["decision"] == "defer" and tied["predicted_index"] in (None, -1)
    with pytest.raises(Exception, match="non-finite|finite"):
        PAIR.decide_clic_open_set(float("nan"), unique, a_s=1.0, b_s=2.0, zero_flag=False)


def test_clic_fixed400_proxy_scores_continuous_unknown_energy_without_fit_or_threshold() -> None:
    source, source_labels, validation, proxy = _clic_proxy_arrays()
    geometry = PAIR.fit_clic_source_geometry(source, source_labels, SOURCE_TX)
    leo_z, scene_rows, rx_slot, extra = _leo_calibration_fixture(SCENARIOS[0])
    policy = PAIR.freeze_clic_tail_policy(geometry, leo_z, scene_rows, rx_slot, extra["y"], extra["physical_binding"])
    tx_logits = np.tile(np.asarray([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64), (proxy.shape[0], 1))
    scored = PAIR.score_clic_open_set(geometry, policy, proxy, tx_logits, SCENARIOS[0])
    assert proxy.shape[0] == 400
    assert np.asarray(scored["e_unknown"]).shape == (400,)
    assert np.isfinite(np.asarray(scored["e_unknown"], dtype=np.float64)).all()
    assert scored.get("fit_rows", 0) == 0
    assert scored.get("threshold_fit_rows", 0) == 0
    with pytest.raises(Exception, match="tie|defer|400|proxy|row"):
        PAIR.score_clic_open_set(geometry, policy, proxy[:399], tx_logits[:399], SCENARIOS[0], expected_proxy_count=400)


def test_clic_proxy_diagnostic_is_source_l_fit_only_and_fixed400_score_only() -> None:
    source, source_labels, validation, proxy = _clic_proxy_arrays()
    result = PAIR.compute_clic_proxy_diagnostic(
        source_l_features=source,
        source_l_tx_ids=source_labels,
        source_validation_features=validation,
        proxy_features=proxy,
        proxy_tx_ids=np.asarray([PROXY_TX[0]] * proxy.shape[0], dtype=str),
        source_tx_ids=SOURCE_TX,
    )
    assert result["schema"] == "cvs.phase1.clic_proxy_diagnostic.v1"
    assert result["fit"]["role"] == "source_L_only"
    assert int(result["fit"]["fit_rows"]) == source.shape[0]
    assert int(result["fit"]["threshold_fit_rows"]) == 0
    assert int(result["source_validation_known"]["count"]) == validation.shape[0]
    assert int(result["source_validation_known"]["fit_rows"]) == 0
    assert int(result["source_validation_known"]["threshold_fit_rows"]) == 0
    assert int(result["proxy_unknown"]["count"]) == 400
    assert int(result["proxy_unknown"]["fit_rows"]) == 0
    assert int(result["proxy_unknown"]["threshold_fit_rows"]) == 0
    assert result["threshold_used"] is False
    assert result["tail_policy_used"] is False
    assert np.isfinite(float(result["AUROC_unknown"]))
    assert 0.0 <= float(result["AUROC_unknown"]) <= 1.0
    assert np.isfinite(float(result["u_gap"]))


@pytest.mark.parametrize("case", ("short_proxy", "validation_in_fit", "proxy_overlap"))
def test_clic_proxy_diagnostic_rejects_short_vfit_or_nonmutual_proxy_tx(case: str) -> None:
    source, source_labels, validation, proxy = _clic_proxy_arrays()
    source_fit = source
    fit_labels = source_labels
    proxy_rows = proxy
    if case == "short_proxy":
        proxy_rows = proxy[:399]
        proxy_labels = np.asarray([PROXY_TX[0]] * proxy_rows.shape[0], dtype=str)
    elif case == "validation_in_fit":
        source_fit = np.vstack([source, validation[:1]])
        fit_labels = np.concatenate([source_labels, np.asarray([HELD_TX[0]], dtype=str)])
        proxy_labels = np.asarray([PROXY_TX[0]] * proxy_rows.shape[0], dtype=str)
    else:
        proxy_labels = np.asarray([SOURCE_TX[0]] * proxy_rows.shape[0], dtype=str)
    with pytest.raises(Exception, match="400|proxy|source|validation|disjoint|overlap|TX|label"):
        PAIR.compute_clic_proxy_diagnostic(
            source_l_features=source_fit,
            source_l_tx_ids=fit_labels,
            source_validation_features=validation,
            proxy_features=proxy_rows,
            proxy_tx_ids=proxy_labels,
            source_tx_ids=SOURCE_TX,
        )


def test_clic_proxy_writer_cli_recomputes_clean_raw_and_pair_never_trusts_hand_json(
    tmp_path: Path,
) -> None:
    artifacts = _pair_artifact_fixture(tmp_path / "artifacts")
    clean = Path(artifacts["g_clean"])
    loaded = PAIR._load_feature_npz(clean, CLEAN.EXPECTED_LV_EXPORT_SCHEMA, "G")
    roles = np.asarray(loaded["roles"], dtype=str)
    tx_ids = np.asarray(loaded["tx_ids"], dtype=str)
    manifest = loaded["manifest"]
    source_order = tuple(str(item) for item in manifest["source_tx_ids"])
    labeled = roles == "labeled_fit"
    validation = roles == "source_validation_known"
    proxy = roles == "proxy_unknown"
    expected = PAIR.compute_clic_proxy_diagnostic(
        loaded["z_id"][labeled],
        tx_ids[labeled],
        loaded["z_id"][validation],
        loaded["z_id"][proxy],
        tx_ids[proxy],
        source_order,
    )

    output = tmp_path / "proxy_writer.json"
    writer = PAIR.export_clic_proxy_diagnostic
    result = writer(clean_npz_path=clean, output_json_path=output)
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert set(payload) == set(expected) | {"clean_npz_sha256"}
    assert payload["clean_npz_sha256"] == _sha_file(clean)
    assert payload["geometry_state_sha256"] == expected["geometry_state_sha256"]
    assert payload["fit"] == expected["fit"]
    assert payload["source_validation_known"] == expected["source_validation_known"]
    assert payload["proxy_unknown"] == expected["proxy_unknown"]
    assert payload["AUROC_unknown"] == pytest.approx(expected["AUROC_unknown"])
    assert payload["u_gap"] == pytest.approx(expected["u_gap"])
    assert payload["threshold_used"] is False
    assert payload["tail_policy_used"] is False
    if isinstance(result, Mapping):
        assert Path(str(result.get("output_json", result.get("output_path", output)))).resolve() == output.resolve()
    written_bytes = output.read_bytes()
    with pytest.raises(Exception, match="overwrite|immutable|exists"):
        writer(clean_npz_path=clean, output_json_path=output)
    assert output.read_bytes() == written_bytes

    cli_output = tmp_path / "proxy_writer_cli.json"
    cli = subprocess.run(
        [
            sys.executable,
            str(CODE_ROOT / "evaluate_phase1_clic_postfreeze_pair.py"),
            "--export-proxy-diagnostic",
            "--clean-npz",
            str(clean),
            "--output-proxy-diagnostic-json",
            str(cli_output),
        ],
        cwd=str(CODE_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr or cli.stdout
    assert json.loads(cli_output.read_text(encoding="utf-8")) == payload

    pair_artifacts = _pair_artifact_fixture(tmp_path / "pair")
    for arm in ("C", "G"):
        generated = tmp_path / f"{arm.lower()}_generated_proxy.json"
        writer(clean_npz_path=pair_artifacts[f"{arm.lower()}_clean"], output_json_path=generated)
        pair_artifacts[f"{arm.lower()}_proxy"] = generated
    args = PAIR.build_parser().parse_args(_pair_cli_argv(pair_artifacts))
    PAIR.evaluate(args)
    tampered = json.loads(Path(pair_artifacts["g_proxy"]).read_text(encoding="utf-8"))
    tampered["AUROC_unknown"] = 0.0 if float(tampered["AUROC_unknown"]) != 0.0 else 1.0
    Path(pair_artifacts["g_proxy"]).write_text(json.dumps(tampered, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="proxy|diagnostic|recompute|AUROC|u_gap|hash|SHA|drift"):
        PAIR.evaluate(args)


def test_clic_pair_script_help_is_a_real_executable_cli() -> None:
    completed = subprocess.run(
        [sys.executable, str(CODE_ROOT / "evaluate_phase1_clic_postfreeze_pair.py"), "--help"],
        cwd=str(CODE_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert "usage:" in completed.stdout.lower()
    assert "--c-checkpoint" in completed.stdout
    assert "--export-proxy-diagnostic" in completed.stdout


def test_clic_pair_parser_evaluate_writes_cg_artifact_summary_and_common_binding(tmp_path: Path) -> None:
    artifacts = _pair_artifact_fixture(tmp_path)
    parser = PAIR.build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    required_options = {
        "--c-checkpoint", "--g-checkpoint", "--c-terminal-receipt-json", "--g-terminal-receipt-json",
        "--c-clean-npz", "--g-clean-npz", "--c-leo-npz", "--g-leo-npz",
        "--c-leo-binding-json", "--g-leo-binding-json", "--c-common-receipt-json", "--g-common-receipt-json",
        "--c-proxy-diagnostic-json", "--g-proxy-diagnostic-json", "--output-pair-json",
    }
    assert required_options.issubset(option_strings)
    args = parser.parse_args(_pair_cli_argv(artifacts))
    result = PAIR.evaluate(args)
    output = Path(args.output_pair_json)
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema"] == payload["schema"] == "cvs.phase1.clic_postfreeze_pair.v1"
    assert payload["same_fold"] is True
    assert payload["common_binding"]["passed"] is True
    assert set(payload["geometry"]) == {"C", "G"}
    assert set(payload["policies"]) == {"C", "G"}
    assert set(payload["clic_source_policy_state"]) == {"C", "G"}
    for arm in ("C", "G"):
        state = payload["clic_source_policy_state"][arm]
        assert state["checkpoint_sha256"] == _sha_file(artifacts[f"{arm.lower()}_paths"]["checkpoint"])
        assert state["terminal_receipt_sha256"] == _sha_file(artifacts[f"{arm.lower()}_paths"]["terminal"])
        assert len(state["state_sha256"]) == 64
    assert set(payload["proxy_diagnostic"]) == {"C", "G"}
    assert payload["raw_artifacts"]["C"]["clean"] == str(Path(artifacts["c_clean"]).resolve())
    assert payload["raw_artifacts"]["G"]["leo_binding"] == str(Path(artifacts["g_binding"]).resolve())


@pytest.mark.parametrize(
    "missing_field",
    ("clean_npz_sha256", "geometry", "geometry_state_sha256", "score_rule"),
)
def test_clic_pair_evaluate_rejects_incomplete_production_proxy_diagnostic(
    tmp_path: Path, missing_field: str
) -> None:
    artifacts = _pair_artifact_fixture(tmp_path / "artifacts")
    complete = Path(artifacts["c_proxy"])
    complete_payload = json.loads(complete.read_text(encoding="utf-8"))
    assert {"clean_npz_sha256", "geometry", "geometry_state_sha256", "score_rule"} <= set(complete_payload)
    missing = tmp_path / f"proxy_missing_{missing_field}.json"
    _write_proxy_without_field(complete, missing, missing_field)
    artifacts["c_proxy"] = missing
    args = PAIR.build_parser().parse_args(_pair_cli_argv(artifacts))
    with pytest.raises(Exception, match="proxy|diagnostic|field|schema|missing|geometry|drift"):
        PAIR.evaluate(args)


def test_clic_pair_evaluate_rejects_missing_raw_npz_or_tampered_binding(tmp_path: Path) -> None:
    artifacts = _pair_artifact_fixture(tmp_path / "missing")
    parser = PAIR.build_parser()
    args = parser.parse_args(_pair_cli_argv(artifacts))
    Path(artifacts["g_clean"]).unlink()
    with pytest.raises(Exception, match="missing|raw|artifact|NPZ|clean"):
        PAIR.evaluate(args)

    tampered = _pair_artifact_fixture(tmp_path / "tampered")
    tampered_binding = Path(tampered["g_binding"])
    binding = json.loads(tampered_binding.read_text(encoding="utf-8"))
    binding["physical_order_sha256"] = "0" * 64
    tampered_binding.write_text(json.dumps(binding, sort_keys=True) + "\n", encoding="utf-8")
    tampered_args = parser.parse_args(_pair_cli_argv(tampered))
    with pytest.raises(Exception, match="binding|physical|order|SHA|tamper|drift"):
        PAIR.evaluate(tampered_args)


def test_clic_same_fold_binding_and_noncompensating_gate_are_strict() -> None:
    c = _common_receipt("C")
    g = _common_receipt("G")
    binding = PAIR.validate_clic_common_training_binding(c, g)
    assert binding["passed"] is True
    assert binding["same_fold"] is True
    assert binding["training_run_root"] == TRAINING_RUN
    g["physical_order_sha256"] = _sha_text("tampered-order")
    with pytest.raises(Exception, match="common|physical|order|binding"):
        PAIR.validate_clic_common_training_binding(c, g)
    gates = PAIR.clic_noncompensating_gates(
        clean_delta_pp={name: 0.0 for name in ("overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy")},
        leo_delta_pp={scene: {name: 0.0 for name in ("overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy")} for scene in SCENARIOS},
        proxy_guard={"strict_AUROC_improvement": True, "strict_proxy_known_gap_improvement": True},
    )
    assert gates["non_compensating"] is True
    assert set(gates["required_floors"]) == {"overall_accuracy", "min_class_accuracy", "min_rx_accuracy", "min_day_accuracy"}


def test_clic_f6_reopens_f1_f5_raw_checkpoint_terminal_clean_leo_binding_proxy_bundle_and_rejects_tamper(tmp_path: Path) -> None:
    priors: list[Path] = []
    raw_by_fold: dict[int, dict[str, Path | str | dict[str, object]]] = {}
    for fold in range(1, 6):
        root = tmp_path / POSTFREEZE_MATRIX
        paths = _checkpoint_fixture(tmp_path / f"F{fold}", arm="G", fold=fold)
        clean = tmp_path / f"F{fold}" / "clean.npz"
        _write_feature_npz(clean, paths, arm="G")
        leo_binding = tmp_path / f"F{fold}" / "leo_binding.json"
        leo_binding.write_text(
            json.dumps(
                {
                    "schema": "cvs.phase1.clic_leo_binding.v1",
                    "fold_index": fold,
                    "checkpoint_sha256": paths["checkpoint_sha"],
                    "terminal_receipt_sha256": _sha_file(paths["terminal"]),
                    "clean_npz_sha256": _sha_file(clean),
                    "common_physical_order_bound": True,
                    "satellite_scenarios": list(SCENARIOS),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        proxy = tmp_path / f"F{fold}" / "proxy.json"
        proxy.write_text(json.dumps({"schema": "cvs.phase1.clic_proxy.v1", "proxy_row_count": 400}, sort_keys=True), encoding="utf-8")
        bundle = tmp_path / f"F{fold}" / "bundle.tar"
        bundle.write_bytes(f"bundle-{fold}".encode("ascii"))
        record = tmp_path / f"F{fold}" / f"F{fold}_pair.json"
        record.write_text(
            json.dumps(
                {
                    "schema": "cvs.phase1.clic_postfreeze_pair.v1",
                    "fold_index": fold,
                    "postfreeze_matrix_id": POSTFREEZE_MATRIX,
                    "training_run_root": TRAINING_RUN,
                    "raw_artifacts": {
                        "checkpoint": str(paths["checkpoint"]),
                        "terminal": str(paths["terminal"]),
                        "clean": str(clean),
                        "leo_binding": str(leo_binding),
                        "proxy": str(proxy),
                        "bundle": str(bundle),
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        raw_by_fold[fold] = paths
        priors.append(record)
    f6 = _checkpoint_fixture(tmp_path / "F6", arm="G", fold=6)
    with pytest.raises(Exception, match="C/G|pair|bundle|raw|checkpoint|terminal"):
        PAIR.reopen_f6_raw_artifacts(
            prior_pair_metrics=priors,
            current_fold=6,
            expected_matrix_id=POSTFREEZE_MATRIX,
            expected_training_run=TRAINING_RUN,
            current_checkpoint=f6["checkpoint"],
            raw_artifacts_by_fold=raw_by_fold,
        )
    tampered = json.loads(priors[0].read_text(encoding="utf-8"))
    tampered["raw_artifacts"]["clean"] = str(tmp_path / "missing-or-tampered.npz")
    priors[0].write_text(json.dumps(tampered, sort_keys=True), encoding="utf-8")
    with pytest.raises(Exception, match="raw|artifact|SHA|tamper|missing"):
        PAIR.reopen_f6_raw_artifacts(
            prior_pair_metrics=priors,
            current_fold=6,
            expected_matrix_id=POSTFREEZE_MATRIX,
            expected_training_run=TRAINING_RUN,
            current_checkpoint=f6["checkpoint"],
            raw_artifacts_by_fold=raw_by_fold,
        )


def test_clic_f6_positive_reopens_all_cg_raw_artifacts_and_rejects_binding_byte_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    priors: list[Path] = []
    raw_by_fold: dict[int, dict[str, object]] = {}
    binding_paths: list[Path] = []
    for fold in range(1, 6):
        artifacts = _pair_fold_artifact_fixture(
            tmp_path / f"F{fold}", fold=fold, real_g_bundle=True
        )
        args = PAIR.build_parser().parse_args(_pair_cli_argv(artifacts, fold=fold))
        pair_payload = PAIR.evaluate(args)
        pair_path = Path(args.output_pair_json)
        assert pair_path.is_file()
        persisted = json.loads(pair_path.read_text(encoding="utf-8"))
        assert persisted == pair_payload
        assert set(pair_payload["clic_source_policy_state"]) == {"C", "G"}
        for arm in ("C", "G"):
            state = pair_payload["clic_source_policy_state"][arm]
            paths = artifacts[f"{arm.lower()}_paths"]
            assert state["checkpoint_sha256"] == _sha_file(paths["checkpoint"])
            assert state["terminal_receipt_sha256"] == _sha_file(paths["terminal"])
            assert state["state_sha256"] == _canonical(
                {key: value for key, value in state.items() if key != "state_sha256"}
            )
            assert PAIR._validated_clic_source_policy_state(
                state,
                fold_index=fold,
                arm=arm,
                checkpoint_sha256=_sha_file(paths["checkpoint"]),
                terminal_receipt_sha256=_sha_file(paths["terminal"]),
            ) == state
        assert pair_payload["clic_source_policy_state"]["C"]["state_sha256"] != pair_payload["clic_source_policy_state"]["G"]["state_sha256"]
        raw_by_fold[fold] = {
            arm: dict(pair_payload["raw_artifacts"][arm]) for arm in ("C", "G")
        }
        raw_by_fold[fold]["G"]["bundle"] = artifacts["g_bundle"]
        binding_paths.append(Path(artifacts["g_binding"]))
        priors.append(pair_path)
    f6 = _checkpoint_fixture(tmp_path / "F6G", arm="G", fold=6)
    calls: list[tuple[str, str]] = []
    original_load_json = PAIR._load_json
    original_require_existing = PAIR._require_regular_existing

    def recording_load_json(path: str | Path, *, label: str) -> dict[str, object]:
        calls.append(("json", label))
        return original_load_json(path, label=label)

    def recording_require_existing(value: object, *, label: str) -> Path:
        calls.append(("file", label))
        return original_require_existing(value, label=label)

    monkeypatch.setattr(PAIR, "_load_json", recording_load_json)
    monkeypatch.setattr(PAIR, "_require_regular_existing", recording_require_existing)
    reopened = PAIR.reopen_f6_raw_artifacts(
        prior_pair_metrics=priors,
        current_fold=6,
        expected_matrix_id=POSTFREEZE_MATRIX,
        expected_training_run=TRAINING_RUN,
        current_checkpoint=f6["checkpoint"],
        raw_artifacts_by_fold=raw_by_fold,
    )
    assert reopened["passed"] is True
    assert reopened["raw_reopen_only"] is True
    assert sum(kind == "json" for kind, _ in calls) >= 5
    assert sum(kind == "file" for kind, _ in calls) >= 5 * 6 + 1 + 2 * 5 * 2

    binding_paths[0].write_text("{\"physical_order_sha256\":\"tampered\"}\n", encoding="utf-8")
    with pytest.raises(Exception, match="binding|physical|order|SHA|tamper|drift|invalid"):
        PAIR.reopen_f6_raw_artifacts(
            prior_pair_metrics=priors,
            current_fold=6,
            expected_matrix_id=POSTFREEZE_MATRIX,
            expected_training_run=TRAINING_RUN,
            current_checkpoint=f6["checkpoint"],
            raw_artifacts_by_fold=raw_by_fold,
        )


def test_clic_f6_reopen_rejects_each_incomplete_production_proxy_diagnostic(tmp_path: Path) -> None:
    priors: list[Path] = []
    raw_by_fold: dict[int, dict[str, object]] = {}
    for fold in range(1, 6):
        artifacts = _pair_fold_artifact_fixture(
            tmp_path / f"F{fold}", fold=fold, real_g_bundle=True
        )
        args = PAIR.build_parser().parse_args(_pair_cli_argv(artifacts, fold=fold))
        pair_payload = PAIR.evaluate(args)
        pair_path = Path(args.output_pair_json)
        assert pair_path.is_file()
        raw_by_fold[fold] = {
            arm: dict(pair_payload["raw_artifacts"][arm]) for arm in ("C", "G")
        }
        raw_by_fold[fold]["G"]["bundle"] = artifacts["g_bundle"]
        priors.append(pair_path)
    f6 = _checkpoint_fixture(tmp_path / "F6G", arm="G", fold=6)
    fields = ("clean_npz_sha256", "geometry", "geometry_state_sha256", "score_rule")
    for missing_field in fields:
        mutated_proxy = tmp_path / f"F1_missing_{missing_field}.json"
        _write_proxy_without_field(Path(raw_by_fold[1]["C"]["proxy_diagnostic"]), mutated_proxy, missing_field)
        mutated_raw = {
            fold: {arm: dict(values) for arm, values in fold_raw.items()}
            for fold, fold_raw in raw_by_fold.items()
        }
        mutated_raw[1]["C"]["proxy_diagnostic"] = str(mutated_proxy)
        mutated_raw[1]["C"]["proxy_diagnostic_sha256"] = _sha_file(mutated_proxy)
        mutated_record = tmp_path / f"F1_pair_missing_{missing_field}.json"
        record = json.loads(priors[0].read_text(encoding="utf-8"))
        record["raw_artifacts"]["C"]["proxy_diagnostic"] = str(mutated_proxy)
        record["raw_artifacts"]["C"]["proxy_diagnostic_sha256"] = _sha_file(mutated_proxy)
        mutated_record.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
        mutated_priors = [mutated_record, *priors[1:]]
        with pytest.raises(Exception, match="proxy|diagnostic|field|schema|missing|geometry|drift"):
            PAIR.reopen_f6_raw_artifacts(
                prior_pair_metrics=mutated_priors,
                current_fold=6,
                expected_matrix_id=POSTFREEZE_MATRIX,
                expected_training_run=TRAINING_RUN,
                current_checkpoint=f6["checkpoint"],
                raw_artifacts_by_fold=mutated_raw,
            )


def _bundle_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    paths = _checkpoint_fixture(tmp_path, arm="G")
    output = tmp_path / "clic_bundle"
    model_state = {"id_backbone.clic.weight": np.arange(6, dtype=np.float32).reshape(2, 3).tobytes()}
    source_geometry = {
        "source_fit_roles": ["source_L"],
        "class_order": list(SOURCE_TX),
        "radius": {tx: 1.0 for tx in SOURCE_TX},
        "energy": {tx: 0.5 for tx in SOURCE_TX},
        "tail": {tx: {"q": 0.95} for tx in SOURCE_TX},
    }
    source_rule = {
        "direction": "higher_is_unknown",
        "threshold": {"source_frozen": True},
        "defer": {"source_frozen": True},
        "rule_sha256": _canonical({"direction": "higher_is_unknown", "threshold": {"source_frozen": True}, "defer": {"source_frozen": True}}),
    }
    bundle = BUNDLE.export_bundle(
        checkpoint_path=paths["checkpoint"],
        terminal_receipt_path=paths["terminal"],
        output_path=output,
        model_state=model_state,
        clic_state=model_state,
        source_geometry=source_geometry,
        source_frozen_unknown_rule=source_rule,
        operator_mode="complex_local_invariant_curvature",
        config={"z_id_dim": 2, "z_dom_dim": 2, "q_clic_dim": 1},
    )
    return Path(bundle), source_rule, paths["checkpoint"]


def _copy_bundle(source: Path, destination: Path) -> Path:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    return destination


@pytest.mark.parametrize("state_slot", ("model_state", "clic_state"))
@pytest.mark.parametrize(
    "bad_state",
    (
        np.asarray([np.nan], dtype=np.float32),
        np.asarray([np.inf], dtype=np.float64),
        torch.tensor([float("nan")], dtype=torch.float32),
        {"nested_tensor": torch.tensor([float("inf")], dtype=torch.float32)},
        {"nested_array": np.asarray([np.nan], dtype=np.float64)},
    ),
)
def test_clic_bundle_pack_and_export_reject_nonfinite_model_or_clic_state(
    tmp_path: Path,
    state_slot: str,
    bad_state: object,
) -> None:
    paths = _checkpoint_fixture(tmp_path, arm="G")
    source_geometry = {
        "source_fit_roles": ["source_L"],
        "class_order": list(SOURCE_TX),
        "radius": {tx: 1.0 for tx in SOURCE_TX},
        "energy": {tx: 0.5 for tx in SOURCE_TX},
        "tail": {tx: {"q": 0.95} for tx in SOURCE_TX},
    }
    source_rule = {
        "direction": "higher_is_unknown",
        "threshold": {"source_frozen": True},
        "defer": {"source_frozen": True},
        "rule_sha256": _canonical(
            {
                "direction": "higher_is_unknown",
                "threshold": {"source_frozen": True},
                "defer": {"source_frozen": True},
            }
        ),
    }
    model_state = {"weight": np.asarray([1.0], dtype=np.float32)}
    clic_state = {"weight": np.asarray([1.0], dtype=np.float32)}
    if state_slot == "model_state":
        model_state = {"weight": bad_state}
    else:
        clic_state = {"weight": bad_state}
    with pytest.raises(Exception, match="finite|non-finite|state|tensor|array|unsupported"):
        BUNDLE._pack_state({"bad": bad_state}, label=state_slot)
    output = tmp_path / f"bad_{state_slot}.bundle.zip"
    with pytest.raises(Exception, match="finite|non-finite|state|tensor|array|unsupported"):
        BUNDLE.export_bundle(
            checkpoint_path=paths["checkpoint"],
            terminal_receipt_path=paths["terminal"],
            output_path=output,
            model_state=model_state,
            clic_state=clic_state,
            source_geometry=source_geometry,
            source_frozen_unknown_rule=source_rule,
            operator_mode="complex_local_invariant_curvature",
            config={"z_id_dim": 2, "z_dom_dim": 2, "q_clic_dim": 1},
        )
    assert not output.exists()


def _rewrite_bundle_archive(path: Path, mutate) -> None:
    """Rewrite a copied ZIP bundle only for a negative test; never production."""

    if path.is_dir():
        raise AssertionError("archive mutation helper received a directory")
    temporary = path.with_name(path.name + ".tamper.tmp")
    with zipfile.ZipFile(path, "r") as source, zipfile.ZipFile(temporary, "w") as target:
        for info in source.infolist():
            data = source.read(info.filename)
            data = mutate(info.filename, data)
            if data is None:
                data = source.read(info.filename)
            target.writestr(info, data)
    temporary.replace(path)


def _mutate_bundle_manifest(path: Path, mutation: str) -> None:
    """Change one sealed manifest field in a copied bundle for a negative test."""

    def mutate(payload: bytes) -> bytes:
        manifest = json.loads(payload.decode("utf-8"))
        if mutation == "shape":
            manifest["z_id_shape"] = [99, 99]
        elif mutation == "dtype":
            manifest["z_id_dtype"] = "float32"
        elif mutation == "operator":
            manifest["operator_mode"] = "raw_phase_control"
        elif mutation == "real":
            manifest["real_checkpoint_reload_verified"] = True
        elif mutation == "hash":
            manifest["state_sha256"] = "0" * 64
        return (json.dumps(manifest, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")

    if path.is_dir():
        manifests = sorted(path.rglob("*manifest*.json"))
        assert manifests, "bundle fixture must contain a manifest member"
        manifests[0].write_bytes(mutate(manifests[0].read_bytes()))
        return
    _rewrite_bundle_archive(path, lambda name, data: mutate(data) if name.endswith("manifest.json") else data)


def test_clic_bundle_schema_exact_members_and_reload_forward_are_strict(tmp_path: Path) -> None:
    bundle_path, source_rule, checkpoint = _bundle_fixture(tmp_path)
    verified = BUNDLE.verify_clic_bundle(bundle_path)
    assert verified["schema"] == "cvs.phase1.clic_deployment_bundle.v1"
    assert verified["clean_source_runtime_access"] is False
    assert verified["query_fit_access"] is False
    assert verified["single_leo_observation_required"] is True
    assert verified["source_frozen_unknown_rule"]["rule_sha256"] == source_rule["rule_sha256"]
    assert verified["checkpoint_sha256"] == _sha_file(checkpoint)
    forbidden = {"raw_iq", "sample_features", "sample_logits", "target_rows", "proxy_rows", "receiver_ids", "sample_ids"}
    assert not forbidden.intersection(BUNDLE.bundle_member_names(bundle_path))

    fixture = np.asarray([0.25, -0.50, 0.75], dtype=np.float32)
    first = BUNDLE.reload_forward(bundle_path, fixture)
    second = BUNDLE.reload_forward(bundle_path, fixture.copy())
    for field in ("z_id", "z_dom", "q_clic", "tx_logits", "e_unknown", "decision"):
        assert field in first and field in second
        if isinstance(first[field], np.ndarray):
            np.testing.assert_array_equal(first[field], second[field])
        else:
            assert first[field] == second[field]
    assert first["state_sha256"] == second["state_sha256"] == verified["state_sha256"]


def test_clic_bundle_synthetic_fixture_cannot_claim_real_checkpoint_reload(tmp_path: Path) -> None:
    """The real-state marker requires checkpoint model/hash/forward evidence."""

    bundle_path, _, _ = _bundle_fixture(tmp_path)
    verified = BUNDLE.verify_clic_bundle(bundle_path)
    assert verified["real_checkpoint_reload_verified"] is False
    synthetic_claim = _copy_bundle(bundle_path, tmp_path / "synthetic_claim")
    _mutate_bundle_manifest(synthetic_claim, "real")
    with pytest.raises(BUNDLE.CLICBundleError, match="real|checkpoint|model|forward|synthetic|receipt|state|hash"):
        BUNDLE.verify_clic_bundle(synthetic_claim)


def test_clic_bundle_main_reads_g_artifacts_and_has_no_cli_state_geometry_or_rule_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _pair_artifact_fixture(tmp_path / "inputs")
    g_paths = artifacts["g_paths"]
    baseline, _, _ = _bundle_fixture(tmp_path / "baseline")
    output = tmp_path / "deployment.bundle.zip"
    parser = BUNDLE.build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    required_options = {
        "--checkpoint", "--terminal-receipt-json", "--clean-npz", "--leo-npz", "--leo-binding-json", "--output-bundle",
    }
    assert required_options.issubset(option_strings)
    forbidden_options = {"--model-state", "--model-state-json", "--source-geometry", "--source-geometry-json", "--source-frozen-unknown-rule", "--rule-json"}
    assert not forbidden_options.intersection(option_strings)

    argv = [
        "--checkpoint", str(g_paths["checkpoint"]),
        "--terminal-receipt-json", str(g_paths["terminal"]),
        "--clean-npz", str(artifacts["g_clean"]),
        "--leo-npz", str(artifacts["g_leo"]),
        "--leo-binding-json", str(artifacts["g_binding"]),
        "--output-bundle", str(output),
    ]
    calls: dict[str, object] = {}

    def fake_export_bundle(**kwargs: object) -> str:
        calls.update(kwargs)
        expected = {
            "checkpoint_path": g_paths["checkpoint"],
            "terminal_receipt_path": g_paths["terminal"],
            "clean_npz_path": artifacts["g_clean"],
            "leo_npz_path": artifacts["g_leo"],
            "leo_binding_path": artifacts["g_binding"],
        }
        for key, value in expected.items():
            assert Path(str(kwargs[key])).resolve() == Path(value).resolve()
            assert Path(value).is_file()
        destination = Path(str(kwargs["output_path"]))
        shutil.copy2(baseline, destination)
        return str(destination)

    monkeypatch.setattr(BUNDLE, "export_bundle", fake_export_bundle)
    BUNDLE.main(argv)
    assert output.is_file()
    verified = BUNDLE.verify_clic_bundle(output)
    assert verified["real_checkpoint_reload_verified"] is False
    assert verified["state_origin"] == "synthetic_fixture"


def test_clic_real_g_bundle_seals_candidate_train_data_config_member_and_rejects_member_or_manifest_drift(
    tmp_path: Path,
) -> None:
    """A real raw-derived G bundle must carry the immutable candidate data config.

    The config is a bundle member rather than a caller-side sidecar so the
    target predictor can reopen the exact training-data contract later.  This
    is intentionally RED until the deployment exporter adds the member and
    verifier bindings.
    """

    artifacts = _pair_fold_artifact_fixture(
        tmp_path / "real_g_inputs", fold=1, real_g_bundle=True
    )
    bundle_path = Path(artifacts["g_bundle"])
    member_name = "candidate_train_data_config.json"
    members = BUNDLE.bundle_member_names(bundle_path)
    assert member_name in members
    assert members == set(BUNDLE.MEMBER_NAMES)

    verified = BUNDLE.verify_clic_bundle(bundle_path)
    assert verified["train_config_manifest_container_path"] == str(bundle_path.resolve())
    assert verified["train_config_member_name"] == member_name
    assert verified["train_config_raw_sha256"] == verified["members"][member_name]["sha256"]
    assert verified["train_config_normalized_sha256"] == _sha_text(
        json.dumps(
            json.loads(
                zipfile.ZipFile(bundle_path, "r").read(member_name).decode("utf-8")
            )["normalized"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )

    with zipfile.ZipFile(bundle_path, "r") as archive:
        raw_config = json.loads(archive.read(member_name).decode("utf-8"))
        clean_manifest = json.loads(
            str(np.asarray(np.load(artifacts["g_clean"], allow_pickle=False)["manifest_json"]).item())
        )
    assert raw_config["schema"] == "cvs.phase1.clic_train_data_config.v1"
    assert raw_config["real_checkpoint_config"] is True
    assert raw_config["checkpoint_sha256"] == _sha_file(artifacts["g_paths"]["checkpoint"])
    assert raw_config["terminal_receipt_sha256"] == _sha_file(artifacts["g_paths"]["terminal"])
    assert raw_config["clean_manifest_sha256"] == _canonical(clean_manifest)
    normalized = raw_config["normalized"]
    assert normalized["split_mode"] == "tx_rx_day_1_6_3"
    assert normalized["source_train_tx_ids"] == list(SOURCE_TX)
    assert normalized["source_validation_tx_ids"] == list(HELD_TX)
    assert normalized["source_proxy_tx_ids"] == list(PROXY_TX)
    assert normalized["source_receiver_ids"] == list(SOURCE_RX)
    assert normalized["source_day_ids"] == list(SOURCE_DAYS)
    assert normalized["role_construction"] == {
        "split_mode": "tx_rx_day_1_6_3",
        "labeled_ratio": 0.07,
        "unlabeled_ratio": 0.63,
        "source_val_ratio": 0.30,
    }
    assert normalized["input_len"] == int(BUNDLE.RUNTIME_MODEL_DEFAULTS["input_len"])
    assert normalized["single_leo_training_scenes"] == list(SCENARIOS)
    for forbidden in ("epoch", "optimizer", "loss", "model", "model_architecture", "model_state"):
        assert forbidden not in normalized

    def _rewrite_without_candidate_config(source: Path, destination: Path) -> None:
        with zipfile.ZipFile(source, "r") as src, zipfile.ZipFile(destination, "w") as dst:
            for info in src.infolist():
                if info.filename != member_name:
                    dst.writestr(info, src.read(info.filename))

    tampered_member = _copy_bundle(bundle_path, tmp_path / "tampered_train_config.zip")
    _rewrite_bundle_archive(
        tampered_member,
        lambda name, data: b"{\"schema\":\"tampered\"}\n" if name == member_name else data,
    )
    with pytest.raises(BUNDLE.CLICBundleError, match="member|config|hash|drift"):
        BUNDLE.verify_clic_bundle(tampered_member)

    tampered_manifest = _copy_bundle(bundle_path, tmp_path / "tampered_train_manifest.zip")

    def _tamper_train_descriptor(name: str, data: bytes) -> bytes:
        if name != "manifest.json":
            return data
        manifest = json.loads(data.decode("utf-8"))
        manifest["members"][member_name]["sha256"] = "0" * 64
        return (json.dumps(manifest, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")

    _rewrite_bundle_archive(tampered_manifest, _tamper_train_descriptor)
    with pytest.raises(BUNDLE.CLICBundleError, match="member|config|hash|drift"):
        BUNDLE.verify_clic_bundle(tampered_manifest)

    missing_member = tmp_path / "missing_train_config.zip"
    _rewrite_without_candidate_config(bundle_path, missing_member)
    with pytest.raises(BUNDLE.CLICBundleError, match="member|allowlist|config|missing"):
        BUNDLE.verify_clic_bundle(missing_member)


@pytest.mark.parametrize("mutation", ("member", "byte", "shape", "dtype", "operator", "hash"))
def test_clic_bundle_reload_rejects_any_member_byte_shape_dtype_operator_or_hash_drift(tmp_path: Path, mutation: str) -> None:
    bundle_path, _, _ = _bundle_fixture(tmp_path)
    mutated = _copy_bundle(bundle_path, tmp_path / f"tampered_{mutation}")
    if mutation == "member":
        if mutated.is_dir():
            (mutated / "sample_logits.npy").write_bytes(b"forbidden")
        else:
            with zipfile.ZipFile(mutated, "a") as archive:
                archive.writestr("sample_logits.npy", b"forbidden")
    elif mutation == "byte":
        if mutated.is_dir():
            (mutated / "model_state.bin").write_bytes(b"tampered")
        else:
            _rewrite_bundle_archive(mutated, lambda name, data: b"tampered" if name == "model_state.bin" else data)
    elif mutation in {"shape", "dtype", "operator"}:
        _mutate_bundle_manifest(mutated, mutation)
    else:
        _mutate_bundle_manifest(mutated, "hash")
    with pytest.raises(BUNDLE.CLICBundleError, match="member|byte|shape|dtype|operator|hash|state|forbidden"):
        BUNDLE.verify_clic_bundle(mutated)


# ---------------------------------------------------------------------------
# Task 7 RED contracts.  These fixtures intentionally contain only tiny,
# byte-sealed LEO-IQ rows; they never train, tune, or inspect target metrics.
# ---------------------------------------------------------------------------


def _write_target_config_manifest(path: Path, *, schema: str, normalized: dict[str, object]) -> Path:
    payload = {
        "schema": schema,
        "normalized": dict(normalized),
        "normalized_sha256": _canonical(normalized),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _target_train_config() -> dict[str, object]:
    return {
        "dataset_provenance": "wisig_phase1_source",
        "source_train_tx_ids": list(SOURCE_TX),
        "source_validation_tx_ids": list(HELD_TX),
        "source_proxy_tx_ids": list(PROXY_TX),
        "source_receiver_ids": list(SOURCE_RX),
        "source_day_ids": list(SOURCE_DAYS),
        "split_mode": "tx_rx_day_1_6_3",
        "role_construction": {
            "split_mode": "tx_rx_day_1_6_3",
            "labeled_ratio": 0.07,
            "unlabeled_ratio": 0.63,
            "source_val_ratio": 0.30,
        },
        "physical_row_selection": "fixed_pre_registered_rows",
        "preprocessing": {"input_len": 256, "iq_dtype": "float32"},
        "single_leo_training_scenes": list(SCENARIOS),
        # Deliberately not part of normalized data-config equality.
        "epoch": 40,
        "optimizer": "adam",
        "loss": "ssdg",
        "model_architecture": "clic12_lite_d",
    }


def _target_known_test_config(*, capsule_id: str = "candidate-capsule-v1") -> dict[str, object]:
    return {
        "target_receiver_ids": ["target-rx-0", "target-rx-1"],
        "target_known_tx_ids": ["known-tx-a", "known-tx-b"],
        "class_order": ["known-tx-a", "known-tx-b"],
        "scenes": list(SCENARIOS),
        "leo_weak_channel": {
            "model": "leo_residual",
            "clear": {"elevation_deg": 45.0},
            "low_elev": {"elevation_deg": 15.0},
            "rain": {"attenuation_db": 8.0},
        },
        "preprocessing": {"input_len": 256, "iq_dtype": "float32"},
        "zero_adaptation": True,
        "metric_definitions": {
            "known_accuracy": "accepted_true_class_fraction",
            "unknown_rejection": "decision_unknown_over_true_unknown",
        },
        # These fields must be ignored by ADV3B02 normalized equivalence.
        "capsule_id": capsule_id,
        "physical_sample_ids": [f"physical-{capsule_id}-0"],
        "received_iq_sha256": _sha_text(capsule_id),
        "scene_seed": 17,
    }


def _write_target_cache_set_fixture(
    tmp_path: Path,
    *,
    duplicate_cross_scene: bool = False,
    non_leo_view: bool = False,
) -> dict[str, Path | str | dict[str, object]]:
    """Create a valid tiny three-scene p2_min_v1 cache-set and receipt."""

    from cvsrffi.leo_weak_cache import (
        LEO_WEAK_CACHE_SCHEMA,
        LEO_WEAK_CACHE_SET_SCHEMA,
        LEO_WEAK_CACHE_STAGE,
        PHASE2_PHYSICAL_SAMPLE_OBSERVATION_POLICY,
        PHASE2_PHYSICAL_SAMPLE_ROOT_ID_POLICY,
        PHASE2_SAMPLE_VIEW_POLICY,
        canonical_json_sha256,
        ids_sha256,
        overlay_id,
        physical_sample_id_from_values,
        post_channel_iq_sha256,
    )

    root = tmp_path / "target_cache"
    root.mkdir(parents=True, exist_ok=True)
    dataset_sha = _sha_text("target-dataset")
    known_test = _target_known_test_config()
    known_class_order = [str(tx_id) for tx_id in known_test["class_order"]]
    known_config = _write_target_config_manifest(
        root / "known_test_config.json",
        schema="cvs.phase1.clic_known_test_config.v1",
        normalized=known_test,
    )
    mapping: dict[str, str] = {}
    hashes: dict[str, str] = {}
    ids_by_scene: dict[str, list[str]] = {}
    roles = ("registered_known", "unknown")

    for scene_index, scene in enumerate(SCENARIOS):
        rows: list[dict[str, object]] = []
        for role_index, role in enumerate(roles):
            for repeat in range(2):
                identity_scene = 0 if duplicate_cross_scene and scene_index == 1 and role_index == 0 and repeat == 0 else scene_index
                identity_index = role_index * 2 + repeat
                source_index = identity_scene * 100 + identity_index
                if role == "registered_known":
                    tx_id = known_class_order[repeat % len(known_class_order)]
                else:
                    tx_id = f"unknown-tx-{repeat}"
                rx_id = f"target-rx-{repeat % 2}"
                day_id = f"day-{repeat % 2}"
                eq_id = "eq-0"
                sig_id = f"sig-{identity_scene}-{role_index}-{repeat}"
                physical_id = physical_sample_id_from_values(
                    dataset_sha256=dataset_sha,
                    source_record_index=source_index,
                    role=role,
                    tx_id=tx_id,
                    rx_id=rx_id,
                    day_id=day_id,
                    eq_id=eq_id,
                    sig_id=sig_id,
                )
                iq_level = (10.0 if role == "registered_known" else 100.0) + scene_index + repeat
                iq = np.full((2, 16), iq_level, dtype=np.float32)
                rows.append(
                    {
                        "role": role,
                        "tx": tx_id,
                        "rx": rx_id,
                        "day": day_id,
                        "eq": eq_id,
                        "sig": sig_id,
                        "source_index": source_index,
                        "physical": physical_id,
                        "iq": iq,
                    }
                )
        iq_rows = np.asarray([row["iq"] for row in rows], dtype=np.float32)
        tx_ids = np.asarray([row["tx"] for row in rows], dtype=str)
        rx_ids = np.asarray([row["rx"] for row in rows], dtype=str)
        day_ids = np.asarray([row["day"] for row in rows], dtype=str)
        eq_ids = np.asarray([row["eq"] for row in rows], dtype=str)
        sig_ids = np.asarray([row["sig"] for row in rows], dtype=str)
        role_ids = np.asarray([row["role"] for row in rows], dtype=str)
        source_indices = np.asarray([row["source_index"] for row in rows], dtype=np.int64)
        sample_ids = np.asarray([row["physical"] for row in rows], dtype=str)
        dataset_hashes = np.asarray([dataset_sha] * len(rows), dtype=str)
        seeds = np.asarray([17 + scene_index] * len(rows), dtype=np.int64)
        channel_hash = canonical_json_sha256({"scenario": scene, "model": "leo_residual"})
        iq_hashes = np.asarray([post_channel_iq_sha256(row) for row in iq_rows], dtype=str)
        overlay_ids = np.asarray(
            [
                overlay_id(
                    sample_id=str(sample_ids[index]),
                    scenario=scene,
                    satellite_seed=int(seeds[index]),
                    channel_config_sha256=channel_hash,
                    iq_sha256=str(iq_hashes[index]),
                )
                for index in range(len(rows))
            ],
            dtype=str,
        )
        manifest = {
            "schema": LEO_WEAK_CACHE_SCHEMA,
            "artifact_stage": LEO_WEAK_CACHE_STAGE,
            "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "contains_post_channel_iq_only": True,
            "contains_clean_rows": False,
            "target_channel_view": "leo_weak_only",
            "target_channel_scenarios": [scene],
            "scenario": scene,
            "iq_array_key": "leo_weak_iq",
            "raw_or_clean_iq_key_present": False,
            "overlay_applied_before_phase2": True,
            "star_ground_channel_impl": "simplified_leo_residual",
            "channel_model": "leo_residual",
            "phase2_physical_sample_observation_policy": PHASE2_PHYSICAL_SAMPLE_OBSERVATION_POLICY,
            "phase2_cross_scenario_physical_sample_reuse": False,
            "phase2_additional_leo_channel_state_generation": False,
            "phase2_post_reception_equalization_augmentation_transform_allowed": True,
            "phase2_post_reception_view_from_fixed_received_iq_only": True,
            "phase2_post_reception_view_counts_as_additional_physical_sample": False,
            "phase2_physical_sample_root_id_policy": PHASE2_PHYSICAL_SAMPLE_ROOT_ID_POLICY,
            "phase2_query_post_reception_view_fit_access": False,
            "builder_sha256": _sha_text("builder"),
            "output_roles": list(roles),
            "sample_overlay_provenance_fields": [
                "sample_ids", "source_dataset_sha256", "source_record_indices",
                "sat_scenarios", "satellite_seeds", "post_channel_iq_sha256", "overlay_ids",
            ],
            "channel_config_sha256": channel_hash,
            "physical_sample_ids_sha256": ids_sha256(sample_ids.tolist()),
            "row_count": len(rows),
        }
        cache_path = root / f"{scene}.npz"
        np.savez(
            cache_path,
            leo_weak_iq=iq_rows,
            raw_labels=np.asarray([0, 1, 0, 1], dtype=np.int64),
            domain_labels=np.zeros(len(rows), dtype=np.int64),
            tx_ids=tx_ids,
            rx_ids=rx_ids,
            day_ids=day_ids,
            eq_ids=eq_ids,
            sig_ids=sig_ids,
            source_dataset_sha256=dataset_hashes,
            source_record_indices=source_indices,
            dataset_role=role_ids,
            channel_views=np.asarray(["clean" if non_leo_view else "rx_base"] * len(rows), dtype=str),
            sat_scenarios=np.asarray([scene] * len(rows), dtype=str),
            satellite_seeds=seeds,
            overlay_applied=np.asarray([not non_leo_view] * len(rows), dtype=bool),
            sample_ids=sample_ids,
            post_channel_iq_sha256=iq_hashes,
            overlay_ids=overlay_ids,
            manifest_json=np.asarray(json.dumps(manifest, ensure_ascii=True, sort_keys=True)),
        )
        mapping[scene] = cache_path.name
        hashes[scene] = _sha_file(cache_path)
        ids_by_scene[scene] = sample_ids.astype(str).tolist()

    cache_manifest = {
        "schema": LEO_WEAK_CACHE_SET_SCHEMA,
        "artifact_stage": LEO_WEAK_CACHE_STAGE,
        "phase2_sample_view_policy": PHASE2_SAMPLE_VIEW_POLICY,
        "clean_sample_access": False,
        "clean_derived_signal_access": False,
        "target_channel_view": "leo_weak_only",
        "cache_scope": "stage2_registered",
        "output_roles": list(roles),
        "cache_npz_by_scenario": mapping,
        "cache_sha256_by_scenario": hashes,
        "physical_sample_ids_sha256_by_scenario": {
            scene: ids_sha256(ids_by_scene[scene]) for scene in SCENARIOS
        },
        "physical_sample_scenario_assignment_sha256": canonical_json_sha256(ids_by_scene),
        "phase2_physical_sample_observation_policy": PHASE2_PHYSICAL_SAMPLE_OBSERVATION_POLICY,
        "phase2_cross_scenario_physical_sample_reuse": False,
        "phase2_additional_leo_channel_state_generation": False,
        "phase2_post_reception_equalization_augmentation_transform_allowed": True,
        "phase2_post_reception_view_from_fixed_received_iq_only": True,
        "phase2_post_reception_view_counts_as_additional_physical_sample": False,
        "phase2_physical_sample_root_id_policy": PHASE2_PHYSICAL_SAMPLE_ROOT_ID_POLICY,
        "phase2_query_post_reception_view_fit_access": False,
    }
    manifest_path = root / "cache_set.json"
    manifest_path.write_text(json.dumps(cache_manifest, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = root / "validator_receipt.json"
    receipt = {
        "schema": "cvs.phase2.data_validation_receipt.v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "protocol_schema": "p2_min_v1",
        "capsule_id": "target-capsule-v1",
        "split_id": "target-split-v1",
        "cache_set_manifest_path": str(manifest_path.resolve()),
        "cache_set_manifest_sha256": _sha_file(manifest_path),
        "cache_scope": "stage2_registered",
        "truth_role_blind_scene_assignment": True,
        "known_test_config_manifest_path": str(known_config.resolve()),
        "known_test_config_raw_sha256": _sha_file(known_config),
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "root": root,
        "manifest": manifest_path,
        "receipt": receipt_path,
        "known_test_config": known_config,
        "capsule_id": str(receipt["capsule_id"]),
        "split_id": str(receipt["split_id"]),
    }


def _target_manifest(package: Path | str) -> dict[str, object]:
    path = Path(package)
    if path.is_dir():
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(path, "r") as archive:
        return json.loads(archive.read("manifest.json").decode("utf-8"))


def _target_prediction_path(result: object, fallback: Path) -> Path:
    if isinstance(result, (str, Path)):
        return Path(result)
    if isinstance(result, Mapping):
        for key in ("prediction_path", "output_path", "path"):
            value = result.get(key)
            if value:
                return Path(str(value))
    return fallback


def _write_fake_predictor_artifacts(tmp_path: Path) -> dict[str, Path]:
    train_config = _write_target_config_manifest(
        tmp_path / "candidate_train_config.json",
        schema="cvs.phase1.clic_train_data_config.v1",
        normalized=_target_train_config(),
    )
    checkpoint = tmp_path / "final_checkpoint.pth"
    checkpoint.write_bytes(b"immutable-checkpoint")
    terminal = tmp_path / "terminal.json"
    terminal.write_text(json.dumps({"schema": "cvs.phase1.clic_terminal_envelope.v1"}) + "\n", encoding="utf-8")
    pair_policy = tmp_path / "pair_policy.json"
    pair_policy.write_text(json.dumps({"schema": "cvs.phase1.clic_source_policy_state.v1", "arm": "C"}) + "\n", encoding="utf-8")
    c_state = tmp_path / "c_predictor_state.v1.json"
    c_state.write_text(
        json.dumps(
            {
                "schema": "cvs.phase1.clic_predictor_state.v1",
                "arm": "C",
                "operator": "raw_phase_control",
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": _sha_file(checkpoint),
                "terminal_receipt_path": str(terminal),
                "terminal_receipt_sha256": _sha_file(terminal),
                "pair_policy_state_path": str(pair_policy),
                "pair_policy_state_sha256": _sha_file(pair_policy),
                "train_config_manifest_path": str(train_config),
                "train_config_raw_sha256": _sha_file(train_config),
                "train_config_normalized_sha256": _canonical(_target_train_config()),
                "immutable": True,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    g_bundle = tmp_path / "g_verified.bundle.zip"
    g_bundle.write_bytes(b"verified-g-bundle")
    return {"train_config": train_config, "c_state": c_state, "g_bundle": g_bundle}


def _fake_runtime_factory(
    paths: dict[str, Path],
    *,
    calls: list[Path],
    forward_calls: list[np.ndarray],
    scene_calls: list[str] | None = None,
) -> object:
    class FakeRuntime(TARGET._CLICTargetPredictorRuntime):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.arm = "G" if path.suffix == ".zip" else "C"
            self.operator = "complex_local_invariant_curvature" if self.arm == "G" else "raw_phase_control"
            super().__init__(
                arm=self.arm,
                operator=self.operator,
                state_sha256=_sha_file(path),
                source_frozen_rule_sha256=_sha_text("source-frozen-rule"),
                train_config_manifest_path=str(paths["train_config"]),
                train_config_raw_sha256=_sha_file(paths["train_config"]),
                train_config_normalized_sha256=_canonical(_target_train_config()),
                train_config_member_name=None,
                forward_impl=self._forward_impl,
            )

        def _forward_impl(self, received_i: object, *, scene: str) -> dict[str, object]:
            if scene not in SCENARIOS:
                raise AssertionError(f"fixture received non-formal scene: {scene}")
            if scene_calls is not None:
                scene_calls.append(scene)
            values = np.asarray(received_i, dtype=np.float32)
            forward_calls.append(np.array(values, copy=True))
            decision = "unknown" if float(values.mean()) >= 50.0 else "registered"
            return {
                "z_id": np.asarray([1.0, 0.0], dtype=np.float32),
                "z_dom": np.asarray([0.0, 1.0], dtype=np.float32),
                "q_clic": np.asarray([0.1], dtype=np.float32),
                "tx_logits": np.asarray([2.0, 1.0], dtype=np.float32),
                "e_unknown": 0.9 if decision == "unknown" else 0.1,
                "decision": decision,
            }

    def load(path: str | Path) -> FakeRuntime:
        resolved = Path(path)
        calls.append(resolved)
        return FakeRuntime(resolved)

    return load


def _write_adv3b02_reference_fixture(tmp_path: Path, *, different_capsule: bool = True) -> dict[str, Path | dict[str, object]]:
    train = _target_train_config()
    known = _target_known_test_config(capsule_id="baseline-capsule-v2" if different_capsule else "candidate-capsule-v1")
    train_path = _write_target_config_manifest(
        tmp_path / "adv_train_config.json",
        schema="cvs.phase1.adv3b02_train_data_config.v1",
        normalized=train,
    )
    known_path = _write_target_config_manifest(
        tmp_path / "adv_known_test_config.json",
        schema="cvs.phase1.adv3b02_known_test_config.v1",
        normalized=known,
    )
    checkpoint = tmp_path / "adv3b02_checkpoint.pth"
    checkpoint.write_bytes(b"adv3b02-checkpoint")
    metrics_path = tmp_path / "adv3b02_stratified_metrics.json"
    cells = []
    receiver_sha = _canonical(["target-rx-0", "target-rx-1"])
    tx_sha = _canonical(["known-tx-a", "known-tx-b"])
    class_sha = _canonical(["known-tx-a", "known-tx-b"])
    known_sha = _canonical(known)
    for fold in range(1, 2):
        for scene in SCENARIOS:
            cells.append(
                {
                    "fold_config_key": f"adv-f{fold}",
                    "scene": scene,
                    "target_receiver_set_sha256": receiver_sha,
                    "target_known_tx_set_sha256": tx_sha,
                    "class_order_sha256": class_sha,
                    "known_test_config_sha256": known_sha,
                    "numerator": 1,
                    "denominator": 1,
                    "accuracy": 1.0,
                }
            )
    metrics_path.write_text(json.dumps({"schema": "cvs.phase1.adv3b02_target_known_metrics.v1", "cells": cells}, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "checkpoint": checkpoint,
        "train_config": train_path,
        "known_test_config": known_path,
        "metrics": metrics_path,
        "train": train,
        "known": known,
    }


def _true_unknown_rows(*, unknown: int, defer: int, per_scene: bool = True) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    scenes = SCENARIOS if per_scene else (SCENARIOS[0],)
    for scene in scenes:
        for index in range(unknown):
            rows.append({"scene": scene, "role": "unknown", "truth": "unknown", "decision": "unknown"})
        for index in range(defer):
            rows.append({"scene": scene, "role": "unknown", "truth": "unknown", "decision": "defer"})
    return rows


def test_target_sealer_reuses_validated_receipt_and_emits_iq_only_role_blind_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = _write_target_cache_set_fixture(tmp_path)
    called: list[str] = []

    def forbidden_builder(*_args: object, **_kwargs: object) -> None:
        called.append("builder")
        raise AssertionError("target sealer must not rebuild or revalidate an existing cache")

    monkeypatch.setattr(TARGET_EVAL, "build_phase2_cache", forbidden_builder, raising=False)
    monkeypatch.setattr(TARGET_EVAL, "revalidate_phase2_cache", forbidden_builder, raising=False)
    package, truth = TARGET_EVAL.seal_clic_target_package(
        artifacts["manifest"],
        tmp_path / "sealed_target",
        validator_receipt_path=artifacts["receipt"],
        expected_capsule_id=artifacts["capsule_id"],
        expected_split_id=artifacts["split_id"],
        expected_protocol_schema="p2_min_v1",
    )
    manifest = _target_manifest(package)
    assert manifest["query_truth_included"] is False
    assert manifest["query_role_included"] is False
    assert manifest["single_leo_observation"] is True
    assert set(manifest["scenes"]) == set(SCENARIOS)
    assert manifest["scene_physical_id_pairwise_disjoint"] is True
    package_names = {path.name.lower() for path in Path(package).rglob("*")}
    assert not package_names.intersection({"label", "role", "tx_id", "rx_id", "day_id", "truth.json"})
    truth_payload = json.loads(Path(truth).read_text(encoding="utf-8"))
    assert truth_payload["schema"] == "cvs.phase1.clic_target_truth_sidecar.v1"
    assert called == []


@pytest.mark.parametrize(
    "receipt_patch, expected",
    [
        ({"phase2_data_status": "PENDING"}, "VALIDATED_ONCE"),
        ({"protocol_schema": "wrong_schema"}, "p2_min_v1"),
        ({"capsule_id": "other-capsule"}, "capsule"),
        ({"split_id": "other-split"}, "split"),
    ],
)
def test_target_sealer_rejects_receipt_drift_before_opening_iq(tmp_path: Path, receipt_patch: dict[str, object], expected: str) -> None:
    artifacts = _write_target_cache_set_fixture(tmp_path)
    receipt = json.loads(Path(artifacts["receipt"]).read_text(encoding="utf-8"))
    receipt.update(receipt_patch)
    bad_receipt = tmp_path / "bad_receipt.json"
    bad_receipt.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match=expected):
        TARGET_EVAL.seal_clic_target_package(
            artifacts["manifest"],
            tmp_path / "sealed_target",
            validator_receipt_path=bad_receipt,
            expected_capsule_id=artifacts["capsule_id"],
            expected_split_id=artifacts["split_id"],
        )


def test_target_sealer_rejects_cross_scene_physical_reuse_and_non_leo_view(tmp_path: Path) -> None:
    duplicate = _write_target_cache_set_fixture(tmp_path / "duplicate", duplicate_cross_scene=True)
    with pytest.raises(Exception, match="physical|scene|duplicate|reuse|disjoint"):
        TARGET_EVAL.seal_clic_target_package(
            duplicate["manifest"],
            tmp_path / "duplicate_out",
            validator_receipt_path=duplicate["receipt"],
            expected_capsule_id=duplicate["capsule_id"],
            expected_split_id=duplicate["split_id"],
        )
    non_leo = _write_target_cache_set_fixture(tmp_path / "non_leo", non_leo_view=True)
    with pytest.raises(Exception, match="LEO|leo|received|view|overlay|clean"):
        TARGET_EVAL.seal_clic_target_package(
            non_leo["manifest"],
            tmp_path / "non_leo_out",
            validator_receipt_path=non_leo["receipt"],
            expected_capsule_id=non_leo["capsule_id"],
            expected_split_id=non_leo["split_id"],
        )


def test_publish_predictor_state_is_path_only_and_loader_returns_verified_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = _write_target_cache_set_fixture(tmp_path)
    package, _truth = TARGET_EVAL.seal_clic_target_package(
        artifacts["manifest"],
        tmp_path / "sealed_target",
        validator_receipt_path=artifacts["receipt"],
        expected_capsule_id=artifacts["capsule_id"],
        expected_split_id=artifacts["split_id"],
    )
    predictor_artifacts = _write_fake_predictor_artifacts(tmp_path / "predictor")
    calls: list[Path] = []
    forwards: list[np.ndarray] = []
    scene_calls: list[str] = []
    loader = _fake_runtime_factory(
        predictor_artifacts,
        calls=calls,
        forward_calls=forwards,
        scene_calls=scene_calls,
    )
    monkeypatch.setattr(TARGET_EVAL, "load_verified_clic_predictor_state", loader, raising=False)
    monkeypatch.setattr(TARGET, "load_verified_clic_predictor_state", loader, raising=False)
    output = tmp_path / "c_prediction.json"
    result = TARGET_EVAL.publish_clic_target_prediction(predictor_artifacts["c_state"], package, output)
    assert output.is_file()
    assert calls == [predictor_artifacts["c_state"]]
    assert len(forwards) > 0
    assert set(scene_calls) == set(SCENARIOS)
    assert isinstance(result, (str, Path, Mapping))
    with pytest.raises(Exception, match="path|artifact|state|predictor|inject"):
        TARGET_EVAL.publish_clic_target_prediction({"model_state": {}}, package, tmp_path / "injected.json")


def test_c_and_g_prediction_bind_identical_iq_package_and_forward_once_per_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = _write_target_cache_set_fixture(tmp_path)
    package, _truth = TARGET_EVAL.seal_clic_target_package(
        artifacts["manifest"], tmp_path / "sealed_target", validator_receipt_path=artifacts["receipt"],
        expected_capsule_id=artifacts["capsule_id"], expected_split_id=artifacts["split_id"],
    )
    predictor_artifacts = _write_fake_predictor_artifacts(tmp_path / "predictor")
    calls: list[Path] = []
    forwards: list[np.ndarray] = []
    scene_calls: list[str] = []
    loader = _fake_runtime_factory(
        predictor_artifacts,
        calls=calls,
        forward_calls=forwards,
        scene_calls=scene_calls,
    )
    monkeypatch.setattr(TARGET_EVAL, "load_verified_clic_predictor_state", loader, raising=False)
    monkeypatch.setattr(TARGET, "load_verified_clic_predictor_state", loader, raising=False)
    c_out = tmp_path / "c_prediction.json"
    g_out = tmp_path / "g_prediction.json"
    c_result = TARGET_EVAL.publish_clic_target_prediction(predictor_artifacts["c_state"], package, c_out)
    c_forward_count = len(forwards)
    g_result = TARGET_EVAL.publish_clic_target_prediction(predictor_artifacts["g_bundle"], package, g_out)
    manifest = _target_manifest(package)
    c_payload = json.loads(c_out.read_text(encoding="utf-8"))
    g_payload = json.loads(g_out.read_text(encoding="utf-8"))
    assert c_payload["predictor_package_sha256"] == g_payload["predictor_package_sha256"] == manifest["package_sha256"]
    assert c_payload["forward_count"] == g_payload["forward_count"] == c_payload["row_count"]
    assert c_payload["predictor_state_sha256"] != g_payload["predictor_state_sha256"]
    assert len(forwards) == 2 * int(c_payload["row_count"])
    assert len(scene_calls) == 2 * int(c_payload["row_count"])
    assert set(scene_calls) == set(SCENARIOS)
    assert _target_prediction_path(c_result, c_out).is_file()
    assert _target_prediction_path(g_result, g_out).is_file()


@pytest.mark.parametrize("tampered_config", ("train", "known_test"))
def test_prediction_seals_candidate_config_paths_raw_and_normalized_sha_and_rejects_postseal_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tampered_config: str,
) -> None:
    artifacts = _write_target_cache_set_fixture(tmp_path)
    package, truth = TARGET_EVAL.seal_clic_target_package(
        artifacts["manifest"], tmp_path / "sealed_target", validator_receipt_path=artifacts["receipt"],
        expected_capsule_id=artifacts["capsule_id"], expected_split_id=artifacts["split_id"],
    )
    predictor_artifacts = _write_fake_predictor_artifacts(tmp_path / "predictor")
    loader = _fake_runtime_factory(predictor_artifacts, calls=[], forward_calls=[])
    monkeypatch.setattr(TARGET_EVAL, "load_verified_clic_predictor_state", loader, raising=False)
    monkeypatch.setattr(TARGET, "load_verified_clic_predictor_state", loader, raising=False)
    prediction = tmp_path / "prediction.json"
    TARGET_EVAL.publish_clic_target_prediction(predictor_artifacts["c_state"], package, prediction)
    payload = json.loads(prediction.read_text(encoding="utf-8"))
    for field in ("train_config_manifest_path", "train_config_raw_sha256", "train_config_normalized_sha256", "known_test_config_manifest_path", "known_test_config_raw_sha256", "known_test_config_normalized_sha256"):
        assert field in payload
    baseline = _write_adv3b02_reference_fixture(tmp_path / "baseline")
    adv_reference = tmp_path / "adv_reference.json"
    TARGET_EVAL.ingest_adv3b02_target_known_reference(
        baseline["checkpoint"],
        baseline["train_config"],
        baseline["known_test_config"],
        baseline["metrics"],
        adv_reference,
    )
    config_path = (
        Path(predictor_artifacts["train_config"])
        if tampered_config == "train"
        else Path(artifacts["known_test_config"])
    )
    config_path.write_text(
        config_path.read_text(encoding="utf-8") + "postseal-byte-tamper\n",
        encoding="utf-8",
    )
    expected = "train.*config|known.*test.*config|sha|tamper|drift"
    with pytest.raises(Exception, match=expected):
        TARGET_EVAL.score_clic_target_prediction(
            prediction, truth, adv_reference, tmp_path / "score.json"
        )


def test_adv3b02_reference_ingest_binds_own_artifacts_without_unknown_claim(tmp_path: Path) -> None:
    baseline = _write_adv3b02_reference_fixture(tmp_path / "baseline", different_capsule=True)
    output = tmp_path / "adv3b02_reference.json"
    result = TARGET_EVAL.ingest_adv3b02_target_known_reference(
        baseline["checkpoint"], baseline["train_config"], baseline["known_test_config"], baseline["metrics"], output,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "cvs.phase1.adv3b02_target_known_reference.v1"
    assert payload["checkpoint_sha256"] == _sha_file(baseline["checkpoint"])
    assert payload["train_config_raw_sha256"] == _sha_file(baseline["train_config"])
    assert payload["known_test_config_raw_sha256"] == _sha_file(baseline["known_test_config"])
    assert payload["stratified_metric_artifact_sha256"] == _sha_file(baseline["metrics"])
    assert all(int(cell["denominator"]) > 0 for cell in payload["semantic_cells"])
    assert "unknown" not in json.dumps(payload, ensure_ascii=True).lower()
    assert _target_prediction_path(result, output).is_file()


def test_adv3b02_config_equivalence_ignores_capsule_physical_seed_but_rejects_train_or_test_drift(tmp_path: Path) -> None:
    candidate_train = _target_train_config()
    baseline_train = dict(candidate_train, epoch=200, optimizer="sgd", model_architecture="adv3b02")
    candidate_known = _target_known_test_config(capsule_id="candidate-capsule-v1")
    baseline_known = _target_known_test_config(capsule_id="baseline-capsule-v2")
    baseline_known["physical_sample_ids"] = ["baseline-physical-row"]
    baseline_known["scene_seed"] = 90210
    passed = TARGET_EVAL.validate_adv3b02_config_equivalence(
        candidate_train_config=candidate_train,
        candidate_known_test_config=candidate_known,
        baseline_train_config=baseline_train,
        baseline_known_test_config=baseline_known,
    )
    assert passed["passed"] is True
    drifted_train = dict(baseline_train, split_mode="tx_rx_day_drift")
    with pytest.raises(Exception, match="config|equivalence|split|drift"):
        TARGET_EVAL.validate_adv3b02_config_equivalence(
            candidate_train_config=candidate_train,
            candidate_known_test_config=candidate_known,
            baseline_train_config=drifted_train,
            baseline_known_test_config=baseline_known,
        )
    drifted_known = dict(baseline_known, zero_adaptation=False)
    with pytest.raises(Exception, match="config|equivalence|zero|adapt"):
        TARGET_EVAL.validate_adv3b02_config_equivalence(
            candidate_train_config=candidate_train,
            candidate_known_test_config=candidate_known,
            baseline_train_config=baseline_train,
            baseline_known_test_config=drifted_known,
        )


@pytest.mark.parametrize(
    "ratio_field, drifted_value",
    [
        ("labeled_ratio", 0.08),
        ("unlabeled_ratio", 0.62),
        ("source_val_ratio", 0.31),
    ],
)
def test_adv3b02_config_equivalence_rejects_any_training_role_ratio_drift(
    ratio_field: str,
    drifted_value: float,
) -> None:
    candidate_train = _target_train_config()
    baseline_train = _target_train_config()
    baseline_role = dict(baseline_train["role_construction"])
    baseline_role[ratio_field] = drifted_value
    baseline_train["role_construction"] = baseline_role
    known = _target_known_test_config()
    with pytest.raises(Exception, match="config|equivalence|role|ratio|drift"):
        TARGET_EVAL.validate_adv3b02_config_equivalence(
            candidate_train_config=candidate_train,
            candidate_known_test_config=known,
            baseline_train_config=baseline_train,
            baseline_known_test_config=known,
        )


def test_adv3b02_reference_rejects_missing_or_zero_denominator_semantic_cell(tmp_path: Path) -> None:
    baseline = _write_adv3b02_reference_fixture(tmp_path / "baseline")
    metrics = json.loads(Path(baseline["metrics"]).read_text(encoding="utf-8"))
    metrics["cells"][0]["denominator"] = 0
    bad_metrics = tmp_path / "bad_metrics.json"
    bad_metrics.write_text(json.dumps(metrics, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="denominator|cell|positive|missing"):
        TARGET_EVAL.ingest_adv3b02_target_known_reference(
            baseline["checkpoint"], baseline["train_config"], baseline["known_test_config"], bad_metrics, tmp_path / "bad_reference.json",
        )


def test_true_unknown_defer_never_counts_toward_explicit_rejection(tmp_path: Path) -> None:
    rows = _true_unknown_rows(unknown=69, defer=31, per_scene=True)
    audit = TARGET_EVAL.recompute_unknown_counts(rows)
    assert audit["unknown_denominator_global"] == 300
    assert audit["unknown_numerator_global"] == 207
    assert all(audit["unknown_denominator_by_scene"][scene] == 100 for scene in SCENARIOS)
    assert all(audit["unknown_numerator_by_scene"][scene] == 69 for scene in SCENARIOS)
    with pytest.raises(Exception, match="70|explicit|unknown"):
        TARGET_EVAL.score_target_rows(rows)


def test_truth_sidecar_is_unreadable_before_sealed_prediction_and_target_path_has_no_fit_update_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    truth = tmp_path / "truth_directory"
    truth.mkdir()
    unsealed = tmp_path / "unsealed_prediction.json"
    unsealed.write_text(json.dumps({"schema": "cvs.phase1.clic_target_prediction.v1", "sealed": False}) + "\n", encoding="utf-8")
    with pytest.raises(Exception, match="seal|immutable|verified"):
        TARGET_EVAL.score_clic_target_prediction(unsealed, truth, tmp_path / "adv.json", tmp_path / "score.json")

    artifacts = _write_target_cache_set_fixture(tmp_path / "sealed")
    package, _truth = TARGET_EVAL.seal_clic_target_package(
        artifacts["manifest"], tmp_path / "sealed" / "package", validator_receipt_path=artifacts["receipt"],
        expected_capsule_id=artifacts["capsule_id"], expected_split_id=artifacts["split_id"],
    )
    predictor_artifacts = _write_fake_predictor_artifacts(tmp_path / "sealed" / "predictor")
    calls: list[Path] = []
    forwards: list[np.ndarray] = []
    loader = _fake_runtime_factory(predictor_artifacts, calls=calls, forward_calls=forwards)
    monkeypatch.setattr(TARGET_EVAL, "load_verified_clic_predictor_state", loader, raising=False)
    monkeypatch.setattr(TARGET, "load_verified_clic_predictor_state", loader, raising=False)
    prediction = tmp_path / "sealed" / "prediction.json"
    TARGET_EVAL.publish_clic_target_prediction(predictor_artifacts["c_state"], package, prediction)
    assert len(forwards) == 12


def test_opaque_prediction_truth_join_is_exact_and_rejects_duplicate_missing_or_scene_mismatch() -> None:
    token_a = _sha_text("target-opaque-a")
    token_b = _sha_text("target-opaque-b")
    prediction_rows = [
        {"opaque_token": token_a, "scene": SCENARIOS[0], "decision": "registered"},
        {"opaque_token": token_b, "scene": SCENARIOS[1], "decision": "unknown"},
    ]
    truth_rows = [
        {"opaque_token": token_a, "scene": SCENARIOS[0], "role": "registered_known", "truth": "known-a"},
        {"opaque_token": token_b, "scene": SCENARIOS[1], "role": "unknown", "truth": "unknown"},
    ]
    joined = TARGET.join_prediction_and_truth_by_opaque_token(prediction_rows, truth_rows)
    assert len(joined) == 2
    assert [row["opaque_token"] for row in joined] == [token_a, token_b]
    assert all(row["prediction"]["opaque_token"] == row["truth"]["opaque_token"] for row in joined)

    with pytest.raises(Exception, match="duplicate|opaque|token|join"):
        TARGET.join_prediction_and_truth_by_opaque_token(
            [*prediction_rows, dict(prediction_rows[0])], truth_rows
        )
    with pytest.raises(Exception, match="missing|opaque|token|join"):
        TARGET.join_prediction_and_truth_by_opaque_token(prediction_rows[:1], truth_rows)
    mismatched_truth = [dict(row) for row in truth_rows]
    mismatched_truth[0]["scene"] = SCENARIOS[1]
    with pytest.raises(Exception, match="scene|opaque|token|join|mismatch"):
        TARGET.join_prediction_and_truth_by_opaque_token(prediction_rows, mismatched_truth)


def test_target_scorer_receipt_has_zero_feedback_counters_and_never_reopens_predictor_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _write_target_cache_set_fixture(tmp_path / "inputs")
    package, truth = TARGET_EVAL.seal_clic_target_package(
        artifacts["manifest"],
        tmp_path / "sealed_target",
        validator_receipt_path=artifacts["receipt"],
        expected_capsule_id=artifacts["capsule_id"],
        expected_split_id=artifacts["split_id"],
    )
    predictor_artifacts = _write_fake_predictor_artifacts(tmp_path / "predictor")
    loader = _fake_runtime_factory(predictor_artifacts, calls=[], forward_calls=[])
    monkeypatch.setattr(TARGET_EVAL, "load_verified_clic_predictor_state", loader, raising=False)
    monkeypatch.setattr(TARGET, "load_verified_clic_predictor_state", loader, raising=False)
    prediction = tmp_path / "prediction.json"
    TARGET_EVAL.publish_clic_target_prediction(
        predictor_artifacts["c_state"], package, prediction
    )
    prediction_payload = json.loads(prediction.read_text(encoding="utf-8"))
    assert prediction_payload["truth_sidecar_opened"] is False
    for field in (
        "target_fit_rows",
        "target_update_rows",
        "target_retry_count",
        "target_selection_count",
    ):
        assert prediction_payload[field] == 0
    assert prediction_payload["target_selection_feedback"] is False
    baseline = _write_adv3b02_reference_fixture(tmp_path / "baseline")
    reference = tmp_path / "adv_reference.json"
    TARGET_EVAL.ingest_adv3b02_target_known_reference(
        baseline["checkpoint"],
        baseline["train_config"],
        baseline["known_test_config"],
        baseline["metrics"],
        reference,
    )
    score_path = tmp_path / "score.json"
    result = TARGET_EVAL.score_clic_target_prediction(
        prediction, truth, reference, score_path
    )
    assert score_path.is_file()
    payload = json.loads(score_path.read_text(encoding="utf-8"))
    for field in (
        "target_fit_rows",
        "target_update_rows",
        "target_retry_count",
        "target_selection_count",
    ):
        assert payload[field] == 0
    assert payload["target_selection_feedback"] is False
    assert payload["truth_sidecar_opened"] is True
    assert isinstance(result, (str, Path, Mapping))


def test_true_unknown_gate_accepts_exact_70_percent_and_ignores_registered_rows() -> None:
    rows = _true_unknown_rows(unknown=70, defer=30, per_scene=True)
    rows.extend(
        {
            "scene": scene,
            "role": "registered_known",
            "truth": "known-tx-a",
            "decision": "unknown",
        }
        for scene in SCENARIOS
    )
    audit = TARGET_EVAL.recompute_unknown_counts(rows)
    assert audit["unknown_denominator_global"] == 300
    assert audit["unknown_numerator_global"] == 210
    assert audit["unknown_defer_global"] == 90
    assert audit["unknown_rejection_rate_global"] == pytest.approx(0.70)
    assert all(
        audit["unknown_rejection_rate_by_scene"][scene] == pytest.approx(0.70)
        for scene in SCENARIOS
    )
    result = TARGET_EVAL.score_target_rows(rows)
    assert result["explicit_unknown_gate_passed"] is True


@pytest.mark.parametrize(
    "surface",
    (
        "train_receivers",
        "train_days",
        "train_preprocess",
        "known_channel",
        "known_scenes",
    ),
)
def test_adv3b02_equivalence_rejects_training_or_known_test_data_surface_drift(surface: str) -> None:
    candidate_train = _target_train_config()
    baseline_train = json.loads(json.dumps(candidate_train))
    candidate_known = _target_known_test_config(capsule_id="candidate-capsule-v1")
    baseline_known = json.loads(json.dumps(candidate_known))
    if surface == "train_receivers":
        baseline_train["source_receiver_ids"] = ["drifted-rx"]
    elif surface == "train_days":
        baseline_train["source_day_ids"] = ["drifted-day"]
    elif surface == "train_preprocess":
        baseline_train["preprocessing"]["input_len"] = 512
    elif surface == "known_channel":
        baseline_known["leo_weak_channel"]["rain"]["attenuation_db"] = 9.0
    elif surface == "known_scenes":
        baseline_known["scenes"] = [SCENARIOS[1], SCENARIOS[0], SCENARIOS[2]]
    with pytest.raises(Exception, match="config|equivalence|receiver|day|split|preprocess|channel|scene|drift"):
        TARGET_EVAL.validate_adv3b02_config_equivalence(
            candidate_train_config=candidate_train,
            candidate_known_test_config=candidate_known,
            baseline_train_config=baseline_train,
            baseline_known_test_config=baseline_known,
        )
