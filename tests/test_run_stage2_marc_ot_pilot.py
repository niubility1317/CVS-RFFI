from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvsrffi.stage2_marc_ot_pilot import FORMAL_ARMS, SCENARIOS, validate_pilot_config


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "code" / "scripts" / "run_stage2_marc_ot_pilot.py"
CONFIG = ROOT / "configs" / "marc_ot_k10_pilot_20260901.json"


def _module():
    spec = importlib.util.spec_from_file_location("run_stage2_marc_ot_pilot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_has_exact_smoke_pilot_score_commands_and_smoke_has_no_query_path() -> None:
    module = _module()
    parser = module.parser()
    subparsers = next(
        action for action in parser._actions if hasattr(action, "choices") and action.choices
    )
    assert tuple(subparsers.choices) == ("smoke", "pilot", "score")
    smoke = subparsers.choices["smoke"]
    smoke_options = {
        option
        for action in smoke._actions
        for option in action.option_strings
    }
    assert all("query" not in option for option in smoke_options)
    score = subparsers.choices["score"]
    score_options = {
        option
        for action in score._actions
        for option in action.option_strings
    }
    assert {"--prediction-root", "--truth-sidecar", "--output-root"} <= score_options
    assert "--manifest" not in score_options
    assert "--config" not in score_options


def test_output_root_is_immutable(tmp_path) -> None:
    module = _module()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="immutable"):
        module.create_immutable_output_root(existing)
    created = module.create_immutable_output_root(tmp_path / "new")
    assert created.is_dir()


def test_tensor_uses_bounded_python_copy_when_numpy_bridge_is_incompatible(monkeypatch) -> None:
    module = _module()
    values = np.asarray([[1.25, -2.5], [3.75, 4.5]], dtype=np.float32)

    def incompatible_from_numpy(_values):
        raise TypeError("NumPy ABI is unavailable")

    monkeypatch.setattr(module.torch, "from_numpy", incompatible_from_numpy)
    result = module._tensor(values, "cpu")

    assert result.shape == (2, 2)
    assert result.dtype == torch.float32
    assert result.tolist() == [[1.25, -2.5], [3.75, 4.5]]


def test_checkpoint_loader_rejects_same_shape_state_with_different_identity(tmp_path) -> None:
    module = _module()
    expected = "ADV3B02_CORE90_SOFT_E200"
    state = {"id_backbone.t1.weight": torch.ones(2, 2)}
    valid = tmp_path / "valid.pt"
    wrong = tmp_path / "wrong.pt"
    missing = tmp_path / "missing.pt"
    torch.save({"candidate_id": expected, "model_state_dict": state}, valid)
    torch.save({"candidate_id": "SAME_SHAPE_OTHER_METHOD", "model_state_dict": state}, wrong)
    torch.save({"model_state_dict": state}, missing)

    assert module._validate_checkpoint_identity(valid, expected) == expected
    with pytest.raises(ValueError, match="identity"):
        module._validate_checkpoint_identity(wrong, expected)
    with pytest.raises(ValueError, match="identity"):
        module._validate_checkpoint_identity(missing, expected)


def test_resource_receipt_uses_measured_values_or_explicit_na_status() -> None:
    module = _module()
    measured_rss = module._peak_rss_bytes()
    assert measured_rss is None or measured_rss > 0

    resources = module._resource_receipt(
        training_seconds=1.25,
        inference_seconds=0.5,
        peak_rss_bytes=8192,
        peak_cuda_bytes=None,
        peak_cuda_status="NOT_APPLICABLE",
        trainable_parameter_count=8,
    )

    assert resources["peak_rss_bytes"] == 8192
    assert resources["peak_rss_status"] == "MEASURED"
    assert resources["peak_cuda_bytes"] == "N/A"
    assert resources["peak_cuda_status"] == "NOT_APPLICABLE"


def test_frozen_k10_config_is_complete_and_has_no_mrior_history_fields() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    validated = validate_pilot_config(payload)
    assert tuple(validated["arms"]) == FORMAL_ARMS
    assert tuple(validated["scenarios"]) == SCENARIOS
    assert validated["k_shot"] == 10
    controls = json.dumps(validated["mrior_controls"], sort_keys=True).lower()
    assert "historical" not in controls
    assert "mrior_sda_result" not in controls


def _write_prediction_unit(root, scenario, arm, *, corrupt=False) -> None:
    root.mkdir(parents=True)
    tokens = np.asarray(["q0", "q1"])
    logits = np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)
    predictions = np.asarray([0, 1], dtype=np.int64)
    members = {"query_tokens": tokens}
    for prefix in ("p1", "p2", "p3"):
        members[f"{prefix}_logits"] = logits.copy()
        members[f"{prefix}_predictions"] = predictions.copy()
    if corrupt:
        members["p3_logits"][1, 1] = np.nan
    np.savez_compressed(root / "predictions.npz", **members)
    receipt = {
        "schema": "cvs.phase2.marc_ot.prediction_receipt.v1",
        "status": "PREDICTIONS_COMPLETE",
        "protocol_schema": "p2_min_v1",
        "phase2_data_status": "VALIDATED_ONCE",
        "outer_key": "outer",
        "capsule_id": "capsule",
        "split_id": "split",
        "receiver": "3-19",
        "scenario": scenario,
        "arm": arm,
        "query_rows": 2,
        "expected_query_tokens": tokens.tolist(),
        "class_registry": ["old0", "old1"],
        "query_truth_opened": False,
        "query_role_opened": False,
        "support_state_frozen_before_query": True,
        "resources": {
            "training_seconds": 1.0,
            "inference_seconds": 0.1,
            "peak_rss_bytes": 4096,
            "peak_cuda_bytes": 1024,
            "trainable_parameter_count": 8,
        },
    }
    (root / "prediction_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


class _CountingTruthPath(os.PathLike):
    def __init__(self, path) -> None:
        self.path = path
        self.open_attempts = 0

    def __fspath__(self):
        self.open_attempts += 1
        return os.fspath(self.path)


def test_score_preflights_all_18_predictions_before_first_truth_open(tmp_path) -> None:
    module = _module()
    prediction_root = tmp_path / "pilot"
    for index, (scenario, arm) in enumerate(
        (scenario, arm) for scenario in SCENARIOS for arm in FORMAL_ARMS
    ):
        _write_prediction_unit(
            prediction_root / scenario / arm / "prediction",
            scenario,
            arm,
            corrupt=index == len(SCENARIOS) * len(FORMAL_ARMS) - 1,
        )
    pilot = {
        "status": "ARTIFACTS_COMPLETE",
        "support_frozen_unit_count": 18,
        "prediction_unit_count": 18,
        "arms": list(FORMAL_ARMS),
        "scenarios": list(SCENARIOS),
        "truth_opened": False,
        "promotion_gates": {
            "median_p3_ba_delta_pp": 3.0,
            "worst_scene_p3_ba_delta_pp": -0.5,
            "median_p3_floor_delta_pp": 0.0,
            "low_elev_p3_floor_delta_pp": 0.0,
            "max_p1_p2_scene_drop_pp": 2.0,
            "minimum_help_gt_harm_scenes": 2,
        },
    }
    (prediction_root / "pilot_result.json").write_text(json.dumps(pilot), encoding="utf-8")
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "receiver": "3-19",
                "capsule_id": "capsule",
                "split_id": "split",
                "rows": [
                    {"query_token": "q0", "true_class_index": 0},
                    {"query_token": "q1", "true_class_index": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    observed_truth = _CountingTruthPath(truth_path)
    args = SimpleNamespace(
        prediction_root=prediction_root,
        truth_sidecar=observed_truth,
        output_root=tmp_path / "score",
    )

    with pytest.raises(ValueError, match="finiteness"):
        module._score(args)

    assert observed_truth.open_attempts == 0
