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
import shutil
import zipfile
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import evaluate_phase1_clic_postfreeze_pair as PAIR
import export_phase1_clic_deployment_bundle as BUNDLE
import export_phase1_clic_features as CLEAN
import export_phase1_clic_leo_features as LEO
from cvsrffi import phase1_clic as CLIC


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


def _checkpoint_fixture(tmp_path: Path, *, arm: str = "G", fold: int = 1) -> dict[str, Path | str | dict[str, object]]:
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
        "seed": 7281105,
        "candidate_id": candidate,
        "run_id": TRAINING_RUN,
        "phase1_clic_frozen_mode": True,
        "phase1_clic_enabled": arm == "G",
        "phase1_clic_operator_mode": "complex_local_invariant_curvature" if arm == "G" else "raw_phase_control",
    }
    payload = {
        "checkpoint_schema": "ssdg_phase1_training_state_v2",
        "checkpoint_role": "training_final_only",
        "checkpoint_selection": "final_only",
        "candidate_id": candidate,
        "run_id": TRAINING_RUN,
        "args": args,
        "model": {},
        "optimizer": {},
        "scaler": {},
        "epoch": 40,
        "final_epoch": 40,
        "clic_receipt_precheckpoint": pre,
        "split_info": {
            "source_split_receipt": {
                "schema": "cvs.phase1.source_split_receipt.v1",
                "source_receivers": list(SOURCE_RX),
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
        "clic_receipt_schema": "cvs.phase1.clic_receipt.v1",
        "clic_terminal_contract": "STRICT_CLIC_SOURCE_L_COMMON_C_G_RAW_UNSCALED_VJP_AMP_RESOURCE_GRAPH_RELEASE",
        "clic_terminal_contract_passed": True,
        "clic_enabled": arm == "G",
        "z_id_source_key": "z_id",
        "source_tx_ids": list(SOURCE_TX),
        "known_validation_tx_ids": list(HELD_TX),
        "proxy_unknown_tx_ids": list(PROXY_TX),
        "proxy_selection_frozen_not_cli_tunable": True,
        "clean_source_runtime_access": False,
        "query_fit_access": False,
    }


def _write_feature_npz(path: Path, paths: dict[str, Path | str | dict[str, object]], *, arm: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = np.asarray([SOURCE_TX[0], SOURCE_TX[0], SOURCE_TX[1], SOURCE_TX[1], PROXY_TX[0]], dtype=str)
    roles = np.asarray(
        ["labeled_fit", "labeled_fit", "source_validation_known", "source_validation_known", "proxy_unknown"],
        dtype=str,
    )
    z_id = np.asarray(
        [[1.0, 0.0], [1.0, 0.5], [0.0, 1.0], [0.0, 1.5], [-1.0, -1.0]],
        dtype=np.float32,
    )
    logits = np.asarray(
        [[3.0, 0.0, 0.0, 0.0], [2.5, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0], [0.0, 2.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    physical = np.asarray([f"p-{i}" for i in range(labels.size)], dtype=str)
    manifest = _clean_manifest(paths, arm=arm)
    np.savez(
        path,
        z_id=z_id,
        features=z_id.copy(),
        tx_logits=logits,
        tx_ids=labels,
        dataset_role=roles,
        receiver_ids=np.asarray([SOURCE_RX[0], SOURCE_RX[1], SOURCE_RX[0], SOURCE_RX[1], SOURCE_RX[0]], dtype=str),
        rx_ids=np.asarray([SOURCE_RX[0], SOURCE_RX[1], SOURCE_RX[0], SOURCE_RX[1], SOURCE_RX[0]], dtype=str),
        day_ids=np.asarray([SOURCE_DAYS[0], SOURCE_DAYS[0], SOURCE_DAYS[1], SOURCE_DAYS[1], SOURCE_DAYS[0]], dtype=str),
        physical_sample_id=physical,
        sig_ids=physical,
        sat_scenarios=np.asarray(["", "", "", "", ""], dtype=str),
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
