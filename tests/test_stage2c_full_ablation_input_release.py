from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = (
    REPO_ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "cvs_full_ablation_phase2c_t1_20260730_v1_14552df1"
    / "release_evidence"
)
SOURCE_SUMMARY = (
    REPO_ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "cvs_full_ablation_phase2_t1_20260729_v1"
    / "release_evidence"
    / "n607_v4_25c725c4"
    / "package_build_summary.json"
)


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EVIDENCE_ROOT / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_package_templates_have_exact_identity_coverage() -> None:
    module = _load(
        "stage2c_package_completion_controller_test",
        "stage2c_package_completion_controller.py",
    )
    source = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    templates = module._templates(source)
    assert set(templates) == {
        (receiver, stage)
        for receiver in module.RECEIVERS
        for stage in ("before", "new20")
    }


def test_source_package_templates_reject_identity_duplication() -> None:
    module = _load(
        "stage2c_package_completion_controller_duplicate_test",
        "stage2c_package_completion_controller.py",
    )
    source = json.loads(SOURCE_SUMMARY.read_text(encoding="utf-8"))
    source["results"][1] = dict(source["results"][0])
    with pytest.raises(ValueError, match="duplicated"):
        module._templates(source)


def test_feature_slots_count_existing_compute_processes() -> None:
    module = _load(
        "stage2c_feature_completion_controller_test",
        "stage2c_feature_completion_controller.py",
    )
    occupancy = {0: 0, 1: 1, 2: 2, 3: 3, 4: 0, 5: 1, 6: 2, 7: 0}
    assert module._available_gpu_slots(occupancy) == (
        0,
        0,
        1,
        4,
        4,
        5,
        7,
        7,
    )


def _write_summary(path: Path, artifact: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "test.completion.v1",
                "expected": 2,
                "completed": 2,
                "results": [
                    {
                        "receiver": "20-1",
                        "seed": 1,
                        "returncode": 0,
                        "artifact_validated": True,
                        "output": str(artifact),
                    },
                    {
                        "receiver": "3-19",
                        "seed": 2,
                        "returncode": 0,
                        "artifact_validated": True,
                        "output": str(artifact),
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_completion_verifier_binds_digest_identity_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(
        "verify_input_completion_summary_test",
        "verify_input_completion_summary.py",
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    summary = tmp_path / "summary.json"
    _write_summary(summary, artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_input_completion_summary.py",
            "--summary",
            str(summary),
            "--expected-sha256",
            module._sha256(summary),
            "--schema",
            "test.completion.v1",
            "--equal",
            "expected=2",
            "--equal",
            "completed=2",
            "--result-identity-fields",
            "receiver,seed",
            "--result-equal",
            "returncode=0",
            "--result-equal",
            "artifact_validated=true",
            "--result-path-field",
            "output",
        ],
    )
    assert module.main() == 0


def test_completion_verifier_rejects_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(
        "verify_input_completion_summary_digest_test",
        "verify_input_completion_summary.py",
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    summary = tmp_path / "summary.json"
    _write_summary(summary, artifact)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_input_completion_summary.py",
            "--summary",
            str(summary),
            "--expected-sha256",
            "0" * 64,
            "--schema",
            "test.completion.v1",
        ],
    )
    with pytest.raises(ValueError, match="SHA-256 drift"):
        module.main()
