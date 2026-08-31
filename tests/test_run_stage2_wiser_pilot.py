from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cvsrffi.stage2_wiser_pilot import WISERQueryPackage, WISERSupportPackage
from cvsrffi.stage2_wiser_runner import WISERP3TrainingAudit, WISERTrainingAudit


def _script_module():
    path = Path(__file__).parents[1] / "code" / "scripts" / "run_stage2_wiser_pilot.py"
    spec = importlib.util.spec_from_file_location("run_stage2_wiser_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wiser_pilot_cli_exposes_smoke_pilot_and_score_commands() -> None:
    script = Path(__file__).parents[1] / "code" / "scripts" / "run_stage2_wiser_pilot.py"

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "smoke" in result.stdout
    assert "pilot" in result.stdout
    assert "score-pilot" in result.stdout
    assert "p3-smoke" in result.stdout
    assert "p3-pilot" in result.stdout
    assert "p3-score-pilot" in result.stdout


def test_p3_cli_normalizes_n_series_subset_and_rejects_legacy_names() -> None:
    module = _script_module()

    assert module.normalize_p3_arms(("N6",)) == ("N0", "N6")
    with pytest.raises(ValueError, match="mixed"):
        module.normalize_p3_arms(("N1", "A"))


def test_p3_config_is_strict_before_output_root_creation(tmp_path: Path) -> None:
    module = _script_module()
    config = module._default_p3_config_payload()
    config["unexpected"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown"):
        module._load_p3_config(path)


def test_p3_unit_training_audit_is_bound_to_its_receipt(tmp_path: Path) -> None:
    module = _script_module()
    unit = tmp_path / "leo_clear_weak" / "N6"
    prediction = unit / "prediction"
    prediction.mkdir(parents=True)
    audit = {
        "outer_arm": "N6", "trainer_arm": "N6", "query_rows_used": 0,
        "support_state_frozen": True,
        "baseline_joint_condition_number": 2.0,
        "final_joint_condition_number": 3.0,
        "final_zero_identity_count": 0,
    }
    (unit / "training_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    receipt = {
        "schema": "cvs.phase2.wiser_rf.p3_primary.prediction_receipt.v1",
        "training_audit": audit,
    }

    assert module._p3_unit_training_audit(unit, receipt, arm="N6") == audit
    receipt["training_audit"] = {**audit, "outer_arm": "N5"}
    with pytest.raises(ValueError, match="audit"):
        module._p3_unit_training_audit(unit, receipt, arm="N6")


def test_p3_config_binds_real_manifest_receiver_separately_from_outer_key() -> None:
    module = _script_module()
    config = module._default_p3_config_payload()
    job = {
        "outer_key": "rx_3_19__seed_713102__k_10__new_5",
        "capsule_id": config["capsule_id"], "split_id": config["split_id"],
        "receiver": "3-19", "seed": 713102, "k_shot": 10, "new_class_count": 5,
    }
    binding = {
        "checkpoint_id": config["checkpoint_id"],
        "feature_schema": config["source_binding"]["feature_schema"],
        "feature_dim": 160,
        "class_registry": config["source_binding"]["class_registry"],
    }

    module._validate_p3_job_binding(config, job, binding)


def test_p3_pilot_freezes_all_three_by_seven_support_units_before_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    config = module._default_p3_config_payload()
    config_path = tmp_path / "p3_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {"protocol_schema": "p2_min_v1", "jobs": [{
                "outer_key": config["pilot_outer_key"], "protocol_schema": "p2_min_v1",
                "phase2_data_status": "VALIDATED_ONCE", "capsule_id": config["capsule_id"],
                "split_id": config["split_id"], "receiver": "3-19", "seed": 713102,
                "k_shot": 10, "new_class_count": 5, "truth_sidecar": str(tmp_path / "truth.json"),
                "packages": {"before_enrollment": {"package_root": str(tmp_path / "support")}, "before_apply": {"package_root": str(tmp_path / "query")}},
            }]}),
        encoding="utf-8",
    )
    support = WISERSupportPackage(
        iq=np.zeros((12, 2, 256), np.float32), labels=np.repeat(np.arange(6), 2),
        tokens=tuple(f"s{index}" for index in range(12)),
    )
    query = WISERQueryPackage(
        iq=np.zeros((6, 2, 256), np.float32), tokens=tuple(f"q{index}" for index in range(6)),
    )
    registry = tuple(config["source_binding"]["class_registry"])
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(module, "_validate_phase1_binding", lambda *_args: {
        "checkpoint_id": config["checkpoint_id"], "class_registry": list(registry),
        "feature_schema": config["source_binding"]["feature_schema"], "feature_dim": 160,
    })
    monkeypatch.setattr(module, "load_quantized_source_summary", lambda *_args: SimpleNamespace(
        class_registry=registry, feature_schema=config["source_binding"]["feature_schema"], centers=torch.zeros((6, 160)),
    ))
    monkeypatch.setattr(module, "frozen_checkpoint", lambda *_args: torch.nn.Linear(1, 1))
    monkeypatch.setattr(module, "load_support_package", lambda *_args: support)
    def fake_query(_path):
        assert (tmp_path / "run" / "support_audit.json").is_file()
        calls.append(("query", "opened"))
        return query
    monkeypatch.setattr(module, "load_query_package", fake_query)
    monkeypatch.setattr(module, "train_wiser_arm", lambda *_args, **kwargs: WISERTrainingAudit(
        arm="A", optimizer_steps=1, query_rows_used=0, vsw_enabled=False,
        model_inversion_enabled=False, stage_audits=(), config=asdict(kwargs["config"]),
    ))
    def fake_p3(*_args, **kwargs):
        calls.append(("p3", kwargs["arm"]))
        assert tuple(kwargs["expected_source_class_registry"]) == registry
        assert kwargs["expected_source_feature_schema"] == config["source_binding"]["feature_schema"]
        return WISERP3TrainingAudit(
            arm=kwargs["arm"], optimizer_steps=1, query_rows_used=0, stage_audits=(),
            reached_parameter_names=(), final_oof_p3_ba=0.5, final_oof_p3_floor=0.5,
            baseline_joint_condition_number=2.0, final_joint_condition_number=2.0,
            final_zero_identity_count=0, final_duals=(), config=asdict(kwargs["config"]),
        )
    monkeypatch.setattr(module, "train_wiser_p3_arm", fake_p3)
    monkeypatch.setattr(module, "_save_adapted_state_new", lambda *_args: None)
    monkeypatch.setattr(module, "_load_adapted_state", lambda *_args: None)
    monkeypatch.setattr(module, "predict_wiser_representation_probes", lambda *_args, query_tokens, **_kwargs: {
        "query_tokens": np.asarray(query_tokens), "p1_predictions": np.zeros(len(query_tokens), np.int64),
        "p2_predictions": np.zeros(len(query_tokens), np.int64), "p3_predictions": np.zeros(len(query_tokens), np.int64),
        "p1_logits": np.zeros((len(query_tokens), 6), np.float32), "p2_logits": np.zeros((len(query_tokens), 6), np.float32),
        "p3_logits": np.zeros((len(query_tokens), 6), np.float32), "query_z_id": np.zeros((len(query_tokens), 160), np.float32),
    })

    result = module._p3_pilot(Namespace(
        p3_config=config_path, manifest=manifest_path, pilot_outer_key=config["pilot_outer_key"],
        checkpoint=tmp_path / "checkpoint.pth", source_summary=tmp_path / "summary.npz",
        source_binding=tmp_path / "binding.json", output_root=tmp_path / "run", device="cpu", arms=module.P3_ARMS,
    ))

    assert result["schema"] == "cvs.phase2.wiser_rf.p3_primary.pilot.v1"
    assert result["scene_arm_unit_count"] == 21
    assert len([call for call in calls if call[0] == "p3"]) == 15
    assert len([call for call in calls if call[0] == "query"]) == 3


def test_cli_normalizes_bounded_arm_subset_with_required_baseline() -> None:
    module = _script_module()
    args = module.parser().parse_args(
        [
            "pilot",
            "--manifest", "manifest.json",
            "--checkpoint", "checkpoint.pth",
            "--source-summary", "summary.npz",
            "--source-binding", "binding.json",
            "--output-root", "run",
            "--device", "cpu",
            "--arms", "B0", "A",
        ]
    )

    assert module._normalize_arms(args.arms) == ("B0", "A")
    with pytest.raises(ValueError, match="B0"):
        module._normalize_arms(("B",))


def test_pilot_rejects_invalid_arm_subset_before_claiming_output_root(
    tmp_path: Path,
) -> None:
    module = _script_module()
    output_root = tmp_path / "pilot"

    with pytest.raises(ValueError, match="one candidate"):
        module._pilot(Namespace(output_root=output_root, arms=("B0",)))

    assert not output_root.exists()


def test_pilot_rejects_missing_validated_once_binding() -> None:
    module = _script_module()
    manifest = {
        "protocol_schema": "p2_min_v1",
        "jobs": [
            {
                "outer_key": "pilot",
                "capsule_id": "capsule",
                "split_id": "split",
            }
        ],
    }

    with pytest.raises(ValueError, match="VALIDATED_ONCE"):
        module._pilot_job(manifest, "pilot")


def test_phase1_binding_rejects_mismatched_summary(tmp_path: Path) -> None:
    module = _script_module()
    checkpoint = tmp_path / "checkpoint.pth"
    summary = tmp_path / "summary.npz"
    checkpoint.write_bytes(b"checkpoint")
    summary.write_bytes(b"summary")
    binding = tmp_path / "binding.json"
    binding.write_text(
        json.dumps(
            {
                "schema": "cvs.phase1.wiser_rf.source_binding.v1",
                "checkpoint_id": "ADV3B02_CORE90_SOFT_E200",
                "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
                "source_summary_sha256": "0" * 64,
                "feature_schema": "ADV3B02:z_id:unit_l2:160:v1",
                "feature_dim": 160,
                "class_registry": [
                    "14-10", "14-7", "20-15", "20-19", "6-15", "8-20"
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact binding"):
        module._validate_phase1_binding(checkpoint, summary, binding)


def test_tensor_falls_back_when_n607_numpy_bridge_rejects_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _script_module()
    monkeypatch.setattr(
        module.torch,
        "from_numpy",
        lambda _value: (_ for _ in ()).throw(TypeError("N607 bridge")),
    )

    value = module._tensor(np.asarray([[1.0, 2.0]], np.float32), "cpu")

    assert torch.equal(value, torch.tensor([[1.0, 2.0]]))


def test_wiser_uses_enrollment_support_and_apply_query_package_roots(
    tmp_path: Path,
) -> None:
    module = _script_module()
    job = {
        "packages": {
            "before_enrollment": {"package_root": str(tmp_path / "enrollment")},
            "before_apply": {"package_root": str(tmp_path / "apply")},
        }
    }

    assert module._support_path(job, "leo_clear_weak") == (
        tmp_path / "enrollment" / "support_leo_clear_weak.npz"
    )
    assert module._query_path(job, "leo_clear_weak") == (
        tmp_path / "apply" / "query_leo_clear_weak.npz"
    )


def test_pilot_freezes_every_support_state_before_first_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _script_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_schema": "p2_min_v1",
                "jobs": [
                    {
                        "outer_key": "pilot",
                        "protocol_schema": "p2_min_v1",
                        "phase2_data_status": "VALIDATED_ONCE",
                        "capsule_id": "capsule",
                        "split_id": "split",
                        "receiver": "3-19",
                        "seed": 713102,
                        "packages": {
                            "before_enrollment": {
                                "package_root": str(tmp_path / "enrollment")
                            },
                            "before_apply": {
                                "package_root": str(tmp_path / "apply")
                            },
                        },
                        "truth_sidecar": str(tmp_path / "truth.json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    events: list[str] = []
    support = WISERSupportPackage(
        iq=np.zeros((12, 2, 256), dtype=np.float32),
        labels=np.repeat(np.arange(6), 2),
        tokens=tuple(f"s{index}" for index in range(12)),
    )
    query = WISERQueryPackage(
        iq=np.zeros((3, 2, 256), dtype=np.float32),
        tokens=("q0", "q1", "q2"),
    )

    def fake_model(*_args, **_kwargs):
        model = torch.nn.Linear(1, 1)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def fake_train(*_args, arm, config, **_kwargs):
        events.append(f"train:{arm}")
        return WISERTrainingAudit(
            arm=arm,
            optimizer_steps=sum(config.stage_steps),
            query_rows_used=0,
            vsw_enabled=arm in {"B", "ABC"},
            model_inversion_enabled=arm in {"C", "ABC"},
            stage_audits=(),
            config=asdict(config),
        )

    def fake_query(_path):
        events.append("query")
        return query

    def fake_predict(*_args, query_tokens, **_kwargs):
        events.append("predict")
        rows = len(query_tokens)
        return {
            "query_tokens": np.asarray(query_tokens),
            "p1_predictions": np.zeros(rows, dtype=np.int64),
            "p2_predictions": np.zeros(rows, dtype=np.int64),
            "p3_predictions": np.zeros(rows, dtype=np.int64),
            "query_z_id": np.zeros((rows, 160), dtype=np.float32),
        }

    monkeypatch.setattr(module, "frozen_checkpoint", fake_model)
    monkeypatch.setattr(module, "load_support_package", lambda _path: support)
    monkeypatch.setattr(module, "load_query_package", fake_query)
    registry = ("14-10", "14-7", "20-15", "20-19", "6-15", "8-20")
    monkeypatch.setattr(
        module,
        "_validate_phase1_binding",
        lambda *_args: {
            "class_registry": list(registry),
            "feature_schema": "ADV3B02:z_id:unit_l2:160:v1",
            "feature_dim": 160,
        },
    )
    monkeypatch.setattr(module, "load_quantized_source_summary", lambda _path: SimpleNamespace(
        class_registry=registry,
        feature_schema="ADV3B02:z_id:unit_l2:160:v1",
        centers=torch.zeros((6, 160)),
    ))
    monkeypatch.setattr(module, "train_wiser_arm", fake_train)
    monkeypatch.setattr(module, "predict_wiser_representation_probes", fake_predict)
    monkeypatch.setattr(module, "_save_adapted_state_new", lambda *_args: events.append("freeze"))
    monkeypatch.setattr(module, "_load_adapted_state", lambda *_args: None)
    args = Namespace(
        output_root=tmp_path / "run",
        manifest=manifest_path,
        pilot_outer_key="pilot",
        source_summary=tmp_path / "summary.npz",
        source_binding=tmp_path / "binding.json",
        checkpoint=tmp_path / "checkpoint.pth",
        device="cpu",
        stage_steps=(1, 1, 1),
        lambda_proto=0.5,
        lambda_sp=1.0,
        lambda_vsw=0.5,
        lambda_inversion=0.25,
        num_vsw_projections=4,
        inversion_steps=1,
        inversion_samples_per_class=1,
        seed=713102,
        arms=("B0", "A"),
    )

    result = module._pilot(args)

    first_query = events.index("query")
    assert all(not event.startswith("train:") for event in events[first_query:])
    assert events[:first_query].count("freeze") == 6
    assert result["status"] == "ARTIFACTS_COMPLETE"
    assert result["scene_arm_unit_count"] == 6


def test_score_pilot_honors_same_bounded_arm_subset(tmp_path: Path) -> None:
    module = _script_module()
    tokens = [f"q{index}" for index in range(6)]
    truth_path = tmp_path / "truth.json"
    truth_path.write_text(
        json.dumps(
            {
                "receiver": "3-19",
                "rows": [
                    {"query_token": token, "true_class_index": index}
                    for index, token in enumerate(tokens)
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "protocol_schema": "p2_min_v1",
                "jobs": [
                    {
                        "outer_key": "pilot",
                        "protocol_schema": "p2_min_v1",
                        "phase2_data_status": "VALIDATED_ONCE",
                        "capsule_id": "capsule",
                        "split_id": "split",
                        "receiver": "3-19",
                        "truth_sidecar": str(truth_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    prediction_root = tmp_path / "predictions"
    (prediction_root).mkdir(parents=True)
    (prediction_root / "pilot_result.json").write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.wiser_rf.pilot.v1",
                "status": "ARTIFACTS_COMPLETE",
                "arms": ["B0", "A"],
            }
        ),
        encoding="utf-8",
    )
    for scenario in module.SCENARIOS:
        for arm in ("B0", "A"):
            unit = prediction_root / scenario / arm / "prediction"
            unit.mkdir(parents=True)
            exact = np.arange(6, dtype=np.int64)
            np.savez_compressed(
                unit / "predictions.npz",
                query_tokens=np.asarray(tokens),
                p1_predictions=exact,
                p2_predictions=exact,
                p3_predictions=exact,
                query_z_id=np.eye(6, dtype=np.float32),
            )
            (unit / "prediction_receipt.json").write_text(
                json.dumps(
                    {
                        "status": "PREDICTIONS_COMPLETE",
                        "arm": arm,
                        "receiver": "3-19",
                        "scenario": scenario,
                        "query_rows": 6,
                        "expected_query_tokens": tokens,
                        "query_truth_opened": False,
                        "query_role_opened": False,
                        "support_state_frozen_before_query": True,
                    }
                ),
                encoding="utf-8",
            )

    result = module._score_pilot(
        Namespace(
            manifest=manifest_path,
            pilot_outer_key="pilot",
            prediction_root=prediction_root,
            output_root=tmp_path / "scores",
            arms=None,
        )
    )

    assert result["scene_arm_unit_count"] == 6
    assert set(result["formal_decisions"]) == {"A"}


def test_score_pilot_rejects_arm_registry_mismatch_before_claiming_output_root(
    tmp_path: Path,
) -> None:
    module = _script_module()
    prediction_root = tmp_path / "predictions"
    prediction_root.mkdir()
    (prediction_root / "pilot_result.json").write_text(
        json.dumps(
            {
                "schema": "cvs.phase2.wiser_rf.pilot.v1",
                "status": "ARTIFACTS_COMPLETE",
                "arms": ["B0", "A"],
            }
        ),
        encoding="utf-8",
    )
    output_root = tmp_path / "scores"

    with pytest.raises(ValueError, match="arm registry mismatch"):
        module._score_pilot(
            Namespace(
                prediction_root=prediction_root,
                output_root=output_root,
                arms=("B0", "B"),
            )
        )

    assert not output_root.exists()
