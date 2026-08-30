from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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


def test_final_checkpoint_is_strictly_loaded_then_evaluated_per_scene(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "final_checkpoint.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    calls: list[str] = []

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
            "bicad_xr_runtime": _runtime(),
        },
        model_builder=lambda _: FakeModel(),
        evaluator=evaluator,
    )

    assert calls == list(FORMAL_EVAL_SCENARIOS)
    assert result["complete"] is True
    assert result["status"] == "ARTIFACTS_COMPLETE"
    assert result["reconstruction"] == {
        "missing": [],
        "unexpected": [],
        "shape_mismatch": [],
    }
