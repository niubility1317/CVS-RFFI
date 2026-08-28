from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from cvsrffi.stage2_sf_erbt_four_state import (
    fit_erbt_registration_pair,
    fit_registered_erbt,
    run_four_state_prediction,
    score_four_state_predictions,
)
from cvsrffi.stage2_sf_d3_erbt_plan import build_data_handle, build_d3_config


def test_registered_erbt_uses_old_metric_and_task_balanced_covariance() -> None:
    rng = np.random.default_rng(713101)
    labels = np.repeat(np.arange(11, dtype=np.int64), 10)
    identity = rng.normal(0.0, 0.01, (110, 160)).astype(np.float32)
    fft = rng.normal(0.0, 0.01, (110, 96)).astype(np.float32)
    for class_id in range(11):
        mask = labels == class_id
        identity[mask, class_id] += 5.0
        fft[mask, class_id] += 2.0

    state = fit_registered_erbt(
        identity,
        fft,
        labels,
        class_ids=tuple(range(11)),
        old_class_count=6,
        seed=713101,
        device="cpu",
    )

    assert np.array_equal(state.predict(identity, fft), labels)
    assert state.audit["arm"] == "M29-FFT96-A4"
    assert state.audit["method_lock"] == "D92-E0-NORF32"
    assert state.audit["rf32_used"] is False
    assert state.audit["registration_state"] == "REG1"
    assert state.audit["metric_support_rows"] == 60
    assert state.audit["metric_new_support_rows"] == 0
    assert state.audit["d92_registration_balanced_active"] is True
    assert state.audit["d92_old_class_count"] == 6
    assert state.audit["d92_new_class_count"] == 5


def test_registration_pair_reuses_one_old_domain_metric() -> None:
    rng = np.random.default_rng(31)
    labels = np.repeat(np.arange(11, dtype=np.int64), 10)
    identity = rng.normal(0.0, 0.02, (110, 160)).astype(np.float32)
    fft = rng.normal(0.0, 0.02, (110, 96)).astype(np.float32)
    for class_id in range(11):
        mask = labels == class_id
        identity[mask, class_id] += 4.0
        fft[mask, class_id] += 2.0

    reg0, reg1, audit = fit_erbt_registration_pair(
        identity[:60],
        fft[:60],
        labels[:60],
        identity,
        fft,
        labels,
        old_class_ids=tuple(range(6)),
        registered_class_ids=tuple(range(11)),
        seed=713101,
        device="cpu",
    )

    assert audit["metric_fit_count"] == 1
    assert audit["metric_support_rows"] == 60
    assert np.array_equal(reg0.log_diag, reg1.log_diag)
    assert reg0.audit["registration_state"] == "REG0"
    assert reg1.audit["registration_state"] == "REG1"
    assert reg1.audit["d92_registration_balanced_active"] is True


class _TinyIdentityModel(torch.nn.Module):
    def forward(self, rows, return_aux=False):
        return {"z_id": rows.flatten(1)[:, :160]}


def test_four_state_prediction_freezes_support_states_before_query_open(tmp_path: Path) -> None:
    old_labels = np.repeat(np.arange(6, dtype=np.int64), 10)
    registered_labels = np.repeat(np.arange(11, dtype=np.int64), 10)
    rng = np.random.default_rng(20260828)
    registered_iq = rng.normal(0.0, 0.01, (110, 2, 256)).astype(np.float32)
    for class_id in range(11):
        registered_iq[registered_labels == class_id, 0, class_id] = 5.0
    old_iq = registered_iq[:60].copy()
    old_ids = np.asarray([f"s{index:03d}" for index in range(60)])
    registered_ids = np.asarray([f"s{index:03d}" for index in range(110)])
    np.savez(
        tmp_path / "old_support.npz",
        received_iq=old_iq,
        support_labels=old_labels,
        support_physical_ids=old_ids,
    )
    np.savez(
        tmp_path / "registered_support.npz",
        received_iq=registered_iq,
        support_labels=registered_labels,
        support_physical_ids=registered_ids,
    )
    query_iq = np.zeros((11, 2, 256), dtype=np.float32)
    for class_id in range(11):
        query_iq[class_id, 0, class_id] = 5.0
    query_ids = np.asarray([f"q{index:03d}" for index in range(11)])
    np.savez(tmp_path / "query.npz", received_iq=query_iq, query_ids=query_ids)
    handle = {
        "schema": "cvs.sf_erbt_four_state.handle.v1",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "capsule_id": "capsule-new5",
        "split_id": "split-new5-k10",
        "da_split_id": "split-new5-k10",
        "scenario": "leo_clear_weak",
        "k_shot": 10,
        "old_class_count": 6,
        "new_class_count": 5,
        "old_support_rows": 60,
        "registered_support_rows": 110,
        "query_rows": 11,
        "base_checkpoint_path": str(tmp_path / "base.pth"),
    }
    (tmp_path / "handle.json").write_text(json.dumps(handle), encoding="utf-8")

    def base_loader(path, *, device):
        return _TinyIdentityModel().to(device).eval()

    def delta_loader(path, *, device, expected_target_binding):
        assert expected_target_binding["capsule_id"] == "capsule-new5"
        model = _TinyIdentityModel().to(device).eval()
        return model, object(), {
            "capsule_id": "capsule-new5",
            "split_id": "split-new5-k10",
            "schema": "cvs.sf_tapft.delta.v2",
        }

    output = tmp_path / "prediction"

    def guarded_query_loader(path):
        assert (output / "support_state_receipt.json").is_file()
        with np.load(path, allow_pickle=False) as payload:
            return {name: np.asarray(payload[name]) for name in payload.files}

    receipt = run_four_state_prediction(
        base_checkpoint_path=tmp_path / "base.pth",
        d3_delta_path=tmp_path / "d3.pt",
        old_support_path=tmp_path / "old_support.npz",
        registered_support_path=tmp_path / "registered_support.npz",
        query_path=tmp_path / "query.npz",
        data_handle_path=tmp_path / "handle.json",
        output_root=output,
        seed=713101,
        device="cpu",
        base_model_loader=base_loader,
        delta_bundle_loader=delta_loader,
        query_loader=guarded_query_loader,
    )

    with np.load(output / "predictions.npz", allow_pickle=False) as payload:
        assert payload["query_ids"].shape == (11,)
        for state in ("da0_reg0", "da1_reg0", "da0_reg1", "da1_reg1"):
            assert payload[f"{state}_predictions"].shape == (11,)
            expected_columns = 6 if state.endswith("reg0") else 11
            assert payload[f"{state}_logits"].shape == (11, expected_columns)
            assert payload[f"{state}_class_ids"].shape == (expected_columns,)
    assert receipt["support_states_frozen_before_query_open"] is True
    assert receipt["query_truth_opened"] is False
    assert receipt["query_role_opened"] is False
    assert receipt["four_states"] == ["DA0_REG0", "DA1_REG0", "DA0_REG1", "DA1_REG1"]


def test_truth_last_scorer_reports_four_effects_and_reg0_new_is_na(tmp_path: Path) -> None:
    query_ids = np.asarray([f"q{index:02d}" for index in range(22)])
    truth = np.repeat(np.arange(11, dtype=np.int64), 2)
    payload: dict[str, np.ndarray] = {"query_ids": query_ids}
    predictions_by_state = {
        "da0_reg0": np.asarray([0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 0] + [0] * 10),
        "da1_reg0": np.asarray(list(np.repeat(np.arange(6), 2)) + [0] * 10),
        "da0_reg1": truth.copy(),
        "da1_reg1": truth.copy(),
    }
    predictions_by_state["da0_reg1"][20] = 0
    for state, predictions in predictions_by_state.items():
        class_count = 6 if state.endswith("reg0") else 11
        class_ids = np.arange(class_count, dtype=np.int64)
        logits = np.full((22, class_count), -2.0, dtype=np.float32)
        logits[np.arange(22), predictions] = 3.0
        payload[f"{state}_class_ids"] = class_ids
        payload[f"{state}_logits"] = logits
        payload[f"{state}_predictions"] = predictions
    np.savez(tmp_path / "predictions.npz", **payload)
    np.savez(tmp_path / "truth.npz", query_ids=query_ids, query_labels=truth)
    binding = {
        "capsule_id": "capsule-new5",
        "split_id": "split-new5-k10",
        "scenario": "leo_clear_weak",
        "k_shot": 10,
        "old_class_count": 6,
        "new_class_count": 5,
        "query_rows": 22,
    }
    (tmp_path / "prediction_receipt.json").write_text(
        json.dumps(
            {
                **binding,
                "schema": "cvs.sf_erbt_four_state.prediction.v1",
                "status": "PREDICTIONS_COMPLETE",
                "query_truth_opened": False,
                "query_role_opened": False,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "handle.json").write_text(
        json.dumps(
            {
                **binding,
                "schema": "cvs.sf_erbt_four_state.handle.v1",
                "protocol_schema": "p2_min_v1",
                "phase2_data_status": "VALIDATED_ONCE",
            }
        ),
        encoding="utf-8",
    )

    result = score_four_state_predictions(
        tmp_path / "predictions.npz",
        tmp_path / "truth.npz",
        tmp_path / "prediction_receipt.json",
        tmp_path / "handle.json",
        tmp_path / "score.json",
    )

    assert result["states"]["DA0_REG0"]["seen_new_acc"] is None
    assert result["states"]["DA0_REG0"]["H_old_new"] is None
    assert result["states"]["DA1_REG0"]["old_acc"] == 1.0
    assert result["states"]["DA0_REG1"]["seen_new_acc"] == 0.9
    assert result["states"]["DA1_REG1"]["seen_new_acc"] == 1.0
    assert result["effects"]["da_before_registration"]["old_acc"] > 0.0
    assert result["effects"]["registration_without_da"]["old_acc"] > 0.0
    assert result["effects"]["registration_with_da"]["old_acc"] == 0.0
    assert "old_acc" in result["effects"]["interaction"]
    assert result["truth_join_after_prediction_only"] is True


def test_four_state_cli_help_runs_from_repo_root_without_pythonpath() -> None:
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            str(root / "code" / "scripts" / "run_sf_erbt_four_state.py"),
            "--help",
        ],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "predict" in result.stdout
    assert "score" in result.stdout


def test_d3_erbt_plan_builds_scene_bound_config_and_handle() -> None:
    plan = {
        "schema": "cvs.sf_d3_erbt.plan.v1",
        "run_id": "run-r1",
        "capsule_id": "capsule-new5",
        "base_checkpoint_path": "/model/base.pth",
        "phase1_bundle": {"package_root": "/phase1/package"},
        "d3_config": {"phase_steps": [327, 0, 0], "seed": 392002},
        "scenes": {
            "leo_clear_weak": {
                "gpu": 0,
                "split_id": "split-clear",
                "old_support_input": "/package/before/support_clear.npz",
                "registered_support_input": "/package/new5/support_clear.npz",
                "query_input": "/package/new5/query_clear.npz",
                "old_support_output": "/run/input/clear/old.npz",
                "registered_support_output": "/run/input/clear/registered.npz",
                "query_output": "/run/input/clear/query.npz",
            }
        },
    }

    config = build_d3_config(plan, "leo_clear_weak")
    handle = build_data_handle(
        plan,
        "leo_clear_weak",
        new_class_count=20,
        query_rows=520,
        split_id="split-clear-new20",
    )

    assert config["candidate_id"] == "D3_R1_T_ERBT_LEO_CLEAR_WEAK"
    assert config["capsule_id"] == "capsule-new5"
    assert config["split_id"] == "split-clear"
    assert config["support_path"] == "/run/input/clear/old.npz"
    assert config["sf_tapft"]["oof_temperature_calibration"] is True
    assert config["sf_tapft"]["phase_steps"] == (327, 0, 0)
    assert handle["scenario"] == "leo_clear_weak"
    assert handle["query_rows"] == 520
    assert handle["split_id"] == "split-clear-new20"
    assert handle["da_split_id"] == "split-clear"
    assert handle["old_support_rows"] == 60
    assert handle["registered_support_rows"] == 260
