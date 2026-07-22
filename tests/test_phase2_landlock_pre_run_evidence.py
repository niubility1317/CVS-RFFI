from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cvsrffi.phase2_landlock_pre_run_evidence as evidence
from cvsrffi.phase2_runtime_contract import (
    PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    closure = tmp_path / "closure"
    runtime = closure / "runtime"
    package = tmp_path / "package"
    scorer = tmp_path / "scorer"
    system = tmp_path / "system"
    controller = tmp_path / "controller"
    for path in (
        runtime,
        package,
        scorer,
        system,
        controller / "scripts",
        controller / "cvsrffi",
    ):
        path.mkdir(parents=True)
    seal = tmp_path / "package.seal.json"
    seal.write_text("{}\n", encoding="utf-8")
    launcher = controller / "scripts" / "launcher.py"
    policy = controller / "cvsrffi" / "policy.py"
    strace = system / "strace"
    python = system / "python"
    for path in (launcher, policy, strace, python):
        path.write_text(path.name, encoding="utf-8")
    monkeypatch.setattr(evidence.platform, "system", lambda: "Linux")
    monkeypatch.setattr(evidence, "query_landlock_abi", lambda: 4)
    monkeypatch.setattr(
        evidence,
        "_validated_system_roots",
        lambda values: (
            [Path(value).resolve() for value in values],
            [system.resolve()],
        ),
    )
    monkeypatch.setattr(
        evidence,
        "_executable_descriptor",
        lambda path, **_kwargs: {
            "requested_path": str(path),
            "resolved_path": str(Path(path).resolve()),
            "sha256": "a" * 64,
            "size_bytes": 1,
        },
    )
    monkeypatch.setattr(
        evidence,
        "verify_phase2_runtime_closure",
        lambda *_args, **_kwargs: {
            "schema": "test.closure",
            "runtime_root": str(runtime.resolve()),
            "root_sha256": "1" * 64,
            "verified": True,
        },
    )
    monkeypatch.setattr(
        evidence,
        "preflight_stage2_predictor_package",
        lambda *_args, **_kwargs: (
            {"package_root_sha256": "2" * 64},
            {"artifact_member_allowlist_sha256": "3" * 64},
            {"status": "PASS"},
        ),
    )
    return {
        "closure": closure,
        "package": package,
        "scorer": scorer,
        "system": system,
        "seal": seal,
        "launcher": launcher,
        "policy": policy,
        "strace": strace,
        "python": python,
    }


def _kwargs(paths: dict[str, Path], output: Path) -> dict[str, object]:
    return {
        "runtime_closure_root": paths["closure"],
        "package_root": paths["package"],
        "detached_seal": paths["seal"],
        "expected_package_seal_sha256": _sha(paths["seal"]),
        "output_root": output,
        "landlock_launcher": paths["launcher"],
        "landlock_policy_module": paths["policy"],
        "strace_executable": paths["strace"],
        "python_executable": paths["python"],
        "system_read_roots": [paths["system"]],
        "forbidden_scorer_truth_roots": [paths["scorer"]],
    }


def test_landlock_evidence_uses_exact_contract_and_diagnostic_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    result = evidence.build_phase2_landlock_pre_run_evidence(
        **_kwargs(paths, output)
    )
    payload = json.loads(
        Path(result["runtime_isolation_evidence"]).read_text(encoding="utf-8")
    )
    assert set(payload) == set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS)
    assert payload["os_isolation_mode"] == "equivalent_verified_isolation"
    assert result["formal_launch_authority"] is False
    assert result["formal_launch_blockers"]
    attestation = json.loads(
        Path(result["os_isolation_attestation"]).read_text(encoding="utf-8")
    )
    assert attestation["landlock_abi"] == 4
    assert attestation["filesystem_namespace_unshared"] is False
    assert attestation["full_lifecycle_strace_required"] is True


def test_landlock_evidence_verifier_rejects_controller_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    result = evidence.build_phase2_landlock_pre_run_evidence(
        **_kwargs(paths, output)
    )
    paths["policy"].write_text("changed", encoding="utf-8")
    with pytest.raises(evidence.Phase2LandlockEvidenceError):
        evidence.verify_phase2_landlock_pre_run_evidence(
            evidence_root=output,
            expected_evidence=result["evidence"],
            **{
                key: value
                for key, value in _kwargs(paths, output).items()
                if key != "output_root"
            },
        )
