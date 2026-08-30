from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cvsrffi.phase1_bicad_xr.metrics import (
    FORMAL_EVAL_SCENARIOS,
    BiCADXRMetricStore,
    evaluate_final_checkpoint,
    validate_artifact_closure,
    validate_checkpoint_runtime,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _runtime(**overrides: object) -> dict[str, object]:
    runtime: dict[str, object] = {
        "phase1_method": "bicad_xr",
        "candidate_id": "D5",
        "fold": 1,
        "seed": 392001,
        "optimizer_update": 5000,
        "total_updates": 5000,
        "source_receivers": ["rx1", "rx2"],
        "train_days": [1, 2, 3],
        "source_only": True,
        "target_access": False,
        "phase2_access": False,
        "support_access": False,
        "query_access": False,
        "truth_access": False,
    }
    runtime.update(overrides)
    return runtime


def _expectation() -> dict[str, object]:
    return {
        "candidate_id": "D5",
        "fold": 1,
        "seed": 392001,
        "optimizer_updates": 5000,
        "source_receivers": ("rx1", "rx2"),
        "train_days": (1, 2, 3),
    }


def _write_complete_base(row_root: Path, reconstruction: dict[str, list[str]] | None = None) -> None:
    checkpoint = row_root / "final_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    _write_json(
        row_root / "checkpoint_runtime.json",
        {
            "checkpoint_path": checkpoint.name,
            "runtime": _runtime(),
            "reconstruction": reconstruction
            or {"missing": [], "unexpected": [], "shape_mismatch": []},
            "strict_reconstruction": True,
            "trainer_runtime_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
        },
    )
    _write_json(row_root / "diagnostics.json", BiCADXRMetricStore().snapshot())


def _write_eval(row_root: Path, scene: str) -> None:
    log_path = row_root / "evaluations" / f"{scene}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(f"{scene} complete\n", encoding="utf-8")
    _write_json(
        row_root / "evaluations" / f"{scene}.json",
        {
            "scene": scene,
            "checkpoint": "final_checkpoint.pt",
            "checkpoint_load_strict": True,
            "missing_keys": [],
            "unexpected_keys": [],
            "shape_mismatches": [],
            "per_class_accuracy": {"0": 1.0, "1": 0.5},
            "floor_accuracy": 0.5,
            "accuracy": 0.75,
            "log_path": log_path.name,
        },
    )


def test_metric_store_exposes_the_complete_diagnostic_schema() -> None:
    snapshot = BiCADXRMetricStore().snapshot()

    assert {
        "conditional_receiver_probe",
        "zdom_tx_probe",
        "domain_classifier_accuracy",
        "xdc_donor_query_matrix",
        "paired_satellite",
        "margin_q0_1",
        "worst_tx_receiver",
        "worst_tx_receiver_day",
        "worst_tx_receiver_channel",
        "gradient_ratios",
        "projection_trigger_rate",
        "effective_xdc_donors",
        "ridge_condition_numbers",
        "throughput_samples_per_second",
        "peak_gpu_memory_bytes",
        "gpu_hours",
        "extra_forward_ratio",
        "inference_parameter_count",
    } <= set(snapshot)
    assert snapshot["paired_satellite"] == "N/A"


def test_checkpoint_runtime_requires_exact_row_identity() -> None:
    valid = validate_checkpoint_runtime(_runtime(), _expectation())
    wrong = validate_checkpoint_runtime(_runtime(seed=392002), _expectation())

    assert valid == {"valid": True, "missing": [], "mismatches": []}
    assert wrong["valid"] is False
    assert wrong["mismatches"] == ["seed"]


def test_closure_requires_clean_and_each_leo_scenario(tmp_path: Path) -> None:
    _write_complete_base(tmp_path)
    for scene in ("clean", "leo_clear_weak", "leo_low_elev_weak"):
        _write_eval(tmp_path, scene)

    result = validate_artifact_closure(tmp_path)

    assert result["complete"] is False
    assert result["status"] != "ARTIFACTS_COMPLETE"
    assert result["missing"] == ["leo_rain_weak"]


@pytest.mark.parametrize("failure_key", ["missing", "unexpected", "shape_mismatch"])
def test_reconstruction_failure_prevents_artifacts_complete(
    tmp_path: Path, failure_key: str
) -> None:
    reconstruction = {"missing": [], "unexpected": [], "shape_mismatch": []}
    reconstruction[failure_key] = ["identity_head.weight"]
    _write_complete_base(tmp_path, reconstruction)
    for scene in FORMAL_EVAL_SCENARIOS:
        _write_eval(tmp_path, scene)

    result = validate_artifact_closure(tmp_path)

    assert result["complete"] is False
    assert result["status"] != "ARTIFACTS_COMPLETE"
    assert result["reconstruction"][failure_key] == ["identity_head.weight"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("strict_reconstruction", False),
        ("missing_keys", ["identity_head.weight"]),
        ("unexpected_keys", ["legacy_head.weight"]),
        ("shape_mismatches", ["identity_head.weight"]),
    ],
)
def test_closure_rejects_non_strict_checkpoint_artifact_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_complete_base(tmp_path)
    runtime_path = tmp_path / "checkpoint_runtime.json"
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_payload[field] = value
    _write_json(runtime_path, runtime_payload)
    for scene in FORMAL_EVAL_SCENARIOS:
        _write_eval(tmp_path, scene)

    result = validate_artifact_closure(tmp_path)

    assert result["complete"] is False
    assert result["status"] != "ARTIFACTS_COMPLETE"


@pytest.mark.parametrize(
    "field",
    [
        "strict_reconstruction",
        "trainer_runtime_strict",
        "missing_keys",
        "unexpected_keys",
        "shape_mismatches",
    ],
)
def test_closure_rejects_missing_checkpoint_strict_field(
    tmp_path: Path, field: str
) -> None:
    _write_complete_base(tmp_path)
    runtime_path = tmp_path / "checkpoint_runtime.json"
    runtime_payload = json.loads(runtime_path.read_text(encoding="utf-8"))
    del runtime_payload[field]
    _write_json(runtime_path, runtime_payload)
    for scene in FORMAL_EVAL_SCENARIOS:
        _write_eval(tmp_path, scene)

    result = validate_artifact_closure(tmp_path)

    assert result["complete"] is False
    assert result["status"] != "ARTIFACTS_COMPLETE"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("checkpoint_load_strict", False),
        ("missing_keys", ["identity_head.weight"]),
        ("unexpected_keys", ["legacy_head.weight"]),
        ("shape_mismatches", ["identity_head.weight"]),
    ],
)
def test_closure_rejects_non_strict_evaluation_artifact_fields(
    tmp_path: Path, field: str, value: object
) -> None:
    _write_complete_base(tmp_path)
    for scene in FORMAL_EVAL_SCENARIOS:
        _write_eval(tmp_path, scene)
    eval_path = tmp_path / "evaluations" / "leo_rain_weak.json"
    eval_payload = json.loads(eval_path.read_text(encoding="utf-8"))
    eval_payload[field] = value
    _write_json(eval_path, eval_payload)

    result = validate_artifact_closure(tmp_path)

    assert result["complete"] is False
    assert result["status"] != "ARTIFACTS_COMPLETE"
    assert "leo_rain_weak" in result["missing"]


def test_final_checkpoint_is_strictly_loaded_then_evaluated_per_scene(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "final_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    calls: list[str] = []
    trainer_runtime_calls: list[object] = []

    class FakeModel:
        def load_state_dict(self, state: object, strict: bool = False) -> SimpleNamespace:
            assert state == {"weight": 1}
            assert strict is True
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def evaluator(model: object, scene: str) -> dict[str, object]:
        assert isinstance(model, FakeModel)
        calls.append(scene)
        return {
            "accuracy": 1.0,
            "per_class_accuracy": {"0": 1.0},
            "floor_accuracy": 1.0,
            "log": f"{scene} complete\n",
        }

    result = evaluate_final_checkpoint(
        checkpoint_path,
        expected_runtime=_expectation(),
        output_dir=tmp_path,
        checkpoint_loader=lambda _: {
            "model": {"weight": 1},
            "bicad_xr_runtime": {
                key: value
                for key, value in _runtime().items()
                if key not in {"fold", "seed", "source_receivers", "train_days"}
            },
            "args": {
                "row_key": json.dumps(
                    {
                        "row_id": "D5-F1-S392001",
                        "fold": 1,
                        "optimizer_updates": 5000,
                    }
                ),
                "seed": 392001,
                "wisig_train_rxs": "rx1,rx2",
                "wisig_train_days": "1,2,3",
                "run_id": "formal-D5-F1-S392001",
                "output_dir": str(tmp_path),
            },
        },
        model_builder=lambda _: FakeModel(),
        trainer_runtime_restorer=lambda model, payload: trainer_runtime_calls.append(
            (model, payload["bicad_xr_runtime"])
        ),
        evaluator=evaluator,
    )

    assert calls == list(FORMAL_EVAL_SCENARIOS)
    assert len(trainer_runtime_calls) == 1
    assert result["complete"] is True
    assert result["status"] == "ARTIFACTS_COMPLETE"
    assert result["reconstruction"] == {
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [],
    }


def test_final_checkpoint_serializes_multielement_training_head_state(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "final_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")

    class FakeModel:
        def load_state_dict(self, state: object, strict: bool = False) -> SimpleNamespace:
            assert strict is True
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    runtime = _runtime(
        training_state={
            "factorized_heads": {
                "dom_tx.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)
            }
        }
    )
    result = evaluate_final_checkpoint(
        checkpoint_path,
        expected_runtime=_expectation(),
        output_dir=tmp_path,
        checkpoint_loader=lambda _: {"model": {}, "bicad_xr_runtime": runtime},
        model_builder=lambda _: FakeModel(),
        trainer_runtime_restorer=lambda _model, _payload: None,
        evaluator=lambda _model, scene: {
            "accuracy": 1.0,
            "per_class_accuracy": {"0": 1.0},
            "floor_accuracy": 1.0,
            "log": f"{scene} complete\n",
        },
    )

    saved = json.loads((tmp_path / "checkpoint_runtime.json").read_text(encoding="utf-8"))
    assert result["complete"] is True
    assert saved["runtime"]["training_state"]["factorized_heads"][
        "dom_tx.weight"
    ] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
