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
from cvsrffi.stage2_wiser_runner import WISERTrainingAudit


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
                            "before_enrollment": {"package_root": str(tmp_path / "sealed")}
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
    )

    result = module._pilot(args)

    first_query = events.index("query")
    assert all(not event.startswith("train:") for event in events[first_query:])
    assert events[:first_query].count("freeze") == 15
    assert result["status"] == "ARTIFACTS_COMPLETE"
