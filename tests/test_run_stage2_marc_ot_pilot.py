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
from cvsrffi.stage2_marc_ot_runner import MARCOTTrainingAudit


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


def test_real_adapt_unit_r8_builds_fold_scoped_plans_and_learning_rates(
    monkeypatch,
) -> None:
    module = _module()
    model = torch.nn.Linear(2, 2, bias=False)
    base_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    plan_tokens: list[tuple[str, ...]] = []

    class Encoder:
        def __call__(self, _features, _labels, tokens):
            return SimpleNamespace(tokens=tuple(tokens))

    entry = SimpleNamespace(spec=SimpleNamespace(name="t1"))
    bundle = SimpleNamespace(
        support_encoder=Encoder(),
        bank=SimpleNamespace(entries=(entry,)),
    )

    def fake_plan(_base, _checkpoint_id, _bank, support_state, **_kwargs):
        tokens = tuple(support_state.tokens)
        plan_tokens.append(tokens)
        learning_rate = 1.0e-5 * len(tokens)
        return SimpleNamespace(
            state_dict={
                name: value.detach().clone() for name, value in base_state.items()
            },
            block_lrs=(learning_rate,),
            applied=True,
            reason="APPLIED",
            uncertainty=0.25,
            block_gates=(1.0,),
        )

    def fake_train(_model, values, labels, tokens, **kwargs):
        initial_factory = kwargs["initial_state_factory"]
        learning_rate_factory = kwargs["block_learning_rate_factory"]
        folds = ((0, 2), (1, 3))
        observed_rates = []
        for fold in folds:
            indices = torch.tensor(fold, dtype=torch.long)
            fit_tokens = tuple(tokens[index] for index in fold)
            initial_factory(values[indices], labels[indices], fit_tokens, "crossfit")
            observed_rates.append(
                learning_rate_factory(
                    values[indices], labels[indices], fit_tokens, "crossfit"
                )["t1"]
            )
        initial_factory(values, labels, tuple(tokens), "full_support")
        observed_rates.append(
            learning_rate_factory(values, labels, tuple(tokens), "full_support")["t1"]
        )
        assert observed_rates == [2.0e-5, 2.0e-5, 4.0e-5]
        return MARCOTTrainingAudit(
            arm="R8",
            selected_alpha=1.0,
            initial_selected_alpha=1.0,
            stage_selected_alphas=(1.0, 1.0, 1.0, 1.0),
            optimizer_steps=12,
            query_rows_used=0,
            stage_audits=(),
            final_duals={},
            config={},
            training_seconds=0.1,
            peak_cuda_bytes=None,
            reached_parameter_names=(),
        )

    monkeypatch.setattr(
        module,
        "_load_model_and_bundle",
        lambda _args, _config: (model, bundle, base_state),
    )
    monkeypatch.setattr(module, "_identity_features", lambda _model, values: values.float())
    monkeypatch.setattr(module, "calibrate_weight_plan", fake_plan)
    monkeypatch.setattr(module, "train_marc_ot_arm", fake_train)
    monkeypatch.setattr(module, "_bank_task_features", lambda _bundle, _device: torch.ones(1, 1))
    monkeypatch.setattr(module, "_calibration_transform", lambda _bundle: None)
    args = SimpleNamespace(device="cpu")
    config = {
        "checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
        "fold_count": 2,
        "stage_steps": [1, 1, 1, 1],
        "learning_rate_bounds": {"min": 1.0e-5, "max": 3.0e-4},
        "ot": {"epsilon": 0.1, "iterations": 2},
        "ratio_cap": 0.5,
        "interpolation_grid": [1.0, 0.0],
        "seed": 713102,
    }
    support = SimpleNamespace(
        iq=np.asarray([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        tokens=("a0", "a1", "b0", "b1"),
    )

    result = module._adapt_unit(args, config, support, "R8", smoke=False)

    assert plan_tokens == [("a0", "b0"), ("a1", "b1"), support.tokens]
    assert result["bank_initialization"]["block_lrs"] == [4.0e-5]


def test_real_adapt_unit_restores_train_mode_when_late_stage_first_builds_full_plan(
    monkeypatch,
) -> None:
    module = _module()
    plan_forward_modes: list[bool] = []
    update_modes: list[tuple[str, bool]] = []
    plan_tokens: list[tuple[str, ...]] = []

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for name in (
                "t1",
                "t2",
                "t3",
                "f1",
                "f2",
                "f3",
                "time_projection",
                "frequency_projection",
                "fusion",
                "identity_mapping",
            ):
                setattr(self, name, torch.nn.Linear(2, 2, bias=False))
            self.norm = torch.nn.LayerNorm(2)

    class ModeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.id_backbone = Backbone()

        def forward(self, values, return_aux=True):
            del return_aux
            plan_forward_modes.append(self.training)
            features = values.float()
            return {"tx_logits": features, "z_id": features}

    model = ModeModel()
    base_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    class Encoder:
        def __call__(self, _features, _labels, tokens):
            return SimpleNamespace(tokens=tuple(tokens))

    entry = SimpleNamespace(spec=SimpleNamespace(name="t1"))
    bundle = SimpleNamespace(
        support_encoder=Encoder(),
        bank=SimpleNamespace(entries=(entry,)),
    )

    def fake_plan(_base, _checkpoint_id, _bank, support_state, **_kwargs):
        tokens = tuple(support_state.tokens)
        plan_tokens.append(tokens)
        return SimpleNamespace(
            state_dict={
                name: value.detach().clone() for name, value in base_state.items()
            },
            block_lrs=(1.0e-5 * len(tokens),),
            applied=True,
            reason="APPLIED",
            uncertainty=0.25,
            block_gates=(1.0,),
        )

    original_train = module.train_marc_ot_arm

    def stage_update(
        current,
        _stage,
        trainable_names,
        _steps,
        duals,
        _fit_iq,
        _fit_labels,
        _fit_tokens,
        fit_scope,
    ):
        update_modes.append((fit_scope, current.training))
        candidate = {
            name: value.detach().clone() for name, value in current.state_dict().items()
        }
        for name in trainable_names:
            candidate[name] = candidate[name] + 0.01
        return candidate, {
            name: value.detach().clone() for name, value in duals.items()
        }

    def support_evaluator(state, _duals, *_fold):
        changed = any(
            value.is_floating_point() and not torch.equal(value, base_state[name])
            for name, value in state.items()
        )
        return {"safe": changed, "oof_ba": 1.0, "oof_floor": 1.0}

    def train_with_mode_spy(*args, **kwargs):
        return original_train(
            *args,
            **kwargs,
            stage_update=stage_update,
            support_evaluator=support_evaluator,
        )

    monkeypatch.setattr(
        module,
        "_load_model_and_bundle",
        lambda _args, _config: (model, bundle, base_state),
    )
    monkeypatch.setattr(module, "calibrate_weight_plan", fake_plan)
    monkeypatch.setattr(module, "train_marc_ot_arm", train_with_mode_spy)
    monkeypatch.setattr(
        module, "_bank_task_features", lambda _bundle, _device: torch.ones(1, 1)
    )
    monkeypatch.setattr(module, "_calibration_transform", lambda _bundle: None)
    args = SimpleNamespace(device="cpu")
    config = {
        "checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
        "fold_count": 2,
        "stage_steps": [1, 1, 1, 1],
        "learning_rate_bounds": {"min": 1.0e-5, "max": 3.0e-4},
        "ot": {"epsilon": 0.1, "iterations": 2},
        "ratio_cap": 0.5,
        "interpolation_grid": [1.0, 0.0],
        "seed": 713102,
    }
    support = SimpleNamespace(
        iq=np.asarray([[2.0, 0.0], [1.5, 0.0], [0.0, 2.0], [0.0, 1.5]]),
        labels=np.asarray([0, 0, 1, 1], dtype=np.int64),
        tokens=("a0", "a1", "b0", "b1"),
    )

    result = module._adapt_unit(args, config, support, "R8", smoke=False)

    assert result["audit"]["initial_selected_alpha"] == 0.0
    assert any(alpha > 0.0 for alpha in result["audit"]["stage_selected_alphas"])
    assert len(plan_tokens) == 3
    assert set(plan_tokens[0]).isdisjoint(plan_tokens[1])
    assert set(plan_tokens[0]) | set(plan_tokens[1]) == set(support.tokens)
    assert plan_tokens[2] == support.tokens
    assert plan_forward_modes and all(mode is False for mode in plan_forward_modes)
    assert update_modes and all(mode is True for _scope, mode in update_modes)
    assert any(scope == "crossfit" for scope, _mode in update_modes)
    assert any(scope == "full_support" for scope, _mode in update_modes)
    assert model.training is False
    assert all(parameter.requires_grad is False for parameter in model.parameters())


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
            "peak_rss_status": "MEASURED",
            "peak_cuda_status": "MEASURED",
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
