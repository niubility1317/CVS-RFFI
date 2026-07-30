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
SOURCE_PLAN = (
    REPO_ROOT
    / "automation_reports"
    / "CV-SincNet"
    / "cvs_full_ablation_phase2_t1_20260729_v1"
    / "stage2c_screening_plan_14552df1.json"
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


@pytest.mark.parametrize(
    "filename",
    [
        "stage2c_package_completion_controller.py",
        "stage2c_feature_completion_controller.py",
        "stage2c_sidecar_completion_controller.py",
    ],
)
def test_completion_summary_fields_are_json_serializable(
    filename: str,
) -> None:
    module = _load(f"{filename}_json_test", filename)
    result = module._json_ready_fields(
        {"receiver": "20-1", "output": Path("/tmp/fresh-output")},
        ("receiver", "output"),
    )
    assert result["output"] == str(Path("/tmp/fresh-output"))
    json.dumps({"results": [result]})


def test_package_gate_uses_source_sidecar_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(
        "verify_input_completion_summary_source_loader_test",
        "verify_input_completion_summary.py",
    )
    from cvsrffi import stage2_predictor_bundle
    from cvsrffi import stage2_scoring_sidecar

    output = tmp_path / "package"
    (output / "predictor").mkdir(parents=True)
    (output / "scorer").mkdir()
    (output / "predictor.seal.json").write_text("{}", encoding="utf-8")
    (output / "scorer" / "scoring_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    called = {"source_loader": 0}

    def fake_preflight(*_args, **_kwargs):
        return (
            {"stage": "stage2c", "receiver": "20-1", "seed": 7282101},
            {},
            {},
        )

    def fake_source_loader(_path):
        called["source_loader"] += 1
        return (
            {
                "schema": "cvs.phase2.query_truth_sidecar.v2",
                "stage": "stage2c",
                "receiver": "20-1",
                "seed": 7282101,
                "rows": [{"opaque_query_token": "q"}],
            },
            {},
            {},
        )

    monkeypatch.setattr(
        stage2_predictor_bundle,
        "preflight_stage2_predictor_package",
        fake_preflight,
    )
    monkeypatch.setattr(
        stage2_scoring_sidecar,
        "load_verified_scoring_sidecar",
        fake_source_loader,
    )
    module._verify_package_artifacts(
        {
            "output": str(output),
            "stage": "new20",
            "receiver": "20-1",
            "method_seed": 7282101,
        }
    )
    assert called["source_loader"] == 1


def test_stage2c_source_plan_has_exact_1425_row_identity() -> None:
    module = _load(
        "verify_stage2c_plan_identity_test",
        "verify_stage2c_plan_identity.py",
    )
    plan = json.loads(SOURCE_PLAN.read_text(encoding="utf-8"))
    module._validate_source(
        plan,
        expected_git_commit="14552df1ca50f8fe100621f5fd4f099942b08322",
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


def test_completion_verifier_rejects_incomplete_identity_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load(
        "verify_input_completion_summary_identity_test",
        "verify_input_completion_summary.py",
    )
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    summary = tmp_path / "summary.json"
    _write_summary(summary, artifact)
    payload = json.loads(summary.read_text(encoding="utf-8"))
    payload["results"][0].update({"method_seed": 7282101, "stage": "before"})
    payload["results"][1].update({"method_seed": 7282101, "stage": "before"})
    summary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
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
            "--result-identity-fields",
            "receiver,method_seed,stage",
            "--identity-profile",
            "stage2c-package",
        ],
    )
    with pytest.raises(ValueError, match="identity coverage drift"):
        module.main()
