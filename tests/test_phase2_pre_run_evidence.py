from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import cvsrffi.phase2_pre_run_evidence as pre_run
import cvsrffi.somph_predictor_bundle as somph_bundle
from cvsrffi.phase2_runtime_contract import PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, monkeypatch) -> dict[str, object]:
    closure = tmp_path / "closure"
    runtime = closure / "runtime"
    package = tmp_path / "package"
    scorer = tmp_path / "scorer"
    system = tmp_path / "system"
    for path in (runtime, package, scorer, system):
        path.mkdir(parents=True)
    seal_path = tmp_path / "package.seal.json"
    seal_path.write_text("{}\n", encoding="utf-8")
    executables = []
    for name in ("bwrap", "strace", "python"):
        path = system / name
        path.write_bytes(name.encode("ascii"))
        executables.append(path)
    monkeypatch.setattr(pre_run.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        pre_run, "_trusted_system_root_allowlist", lambda: [system.resolve()]
    )
    monkeypatch.setattr(
        pre_run,
        "verify_phase2_runtime_closure",
        lambda _path, **_kwargs: {
            "schema": "test.closure",
            "runtime_root": str(runtime),
            "runtime_mount_path": "/runtime/code",
            "entrypoint": "/runtime/code/scripts/run_cvs_stage2_predictor.py",
            "member_count": 7,
            "root_sha256": "1" * 64,
            "verified": True,
        },
    )
    monkeypatch.setattr(
        pre_run,
        "preflight_stage2_predictor_package",
        lambda *_args, **_kwargs: (
            {"package_root_sha256": "2" * 64},
            {"artifact_member_allowlist_sha256": "3" * 64},
            {"schema": "test.preopen", "status": "PASS"},
        ),
    )
    return {
        "closure": closure,
        "runtime": runtime,
        "package": package,
        "scorer": scorer,
        "system": system,
        "seal": seal_path,
        "executables": executables,
    }


def _build(paths: dict[str, object], output: Path):
    executables: list[Path] = paths["executables"]  # type: ignore[assignment]
    return pre_run.build_phase2_pre_run_evidence(
        runtime_closure_root=paths["closure"],
        package_root=paths["package"],
        detached_seal=paths["seal"],
        expected_package_seal_sha256=_sha(paths["seal"]),  # type: ignore[arg-type]
        output_root=output,
        bwrap_executable=executables[0],
        strace_executable=executables[1],
        python_executable=executables[2],
        system_read_roots=[paths["system"]],
        forbidden_scorer_truth_roots=[paths["scorer"]],
    )


def _verify(paths: dict[str, object], result: dict, **overrides):
    executables: list[Path] = paths["executables"]  # type: ignore[assignment]
    evidence_path = Path(result["runtime_isolation_evidence"])
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    values = {
        "evidence_root": evidence_path.parent,
        "runtime_closure_root": paths["closure"],
        "package_root": paths["package"],
        "detached_seal": paths["seal"],
        "expected_package_seal_sha256": _sha(paths["seal"]),  # type: ignore[arg-type]
        "bwrap_executable": executables[0],
        "strace_executable": executables[1],
        "python_executable": executables[2],
        "system_read_roots": [paths["system"]],
        "forbidden_scorer_truth_roots": [paths["scorer"]],
        "expected_evidence": evidence,
    }
    values.update(overrides)
    return pre_run.verify_phase2_pre_run_evidence(**values)


def test_builds_exact_pre_run_evidence_without_claiming_post_run_pass(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    result = _build(paths, output)
    evidence = json.loads(
        Path(result["runtime_isolation_evidence"]).read_text(encoding="utf-8")
    )
    assert set(evidence) == set(PRE_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS)
    assert evidence["sealed_inference_package_sha256"] == _sha(paths["seal"])  # type: ignore[arg-type]
    assert evidence["runtime_code_sha256"] == "1" * 64
    assert result["post_run_filesystem_access_audit_pending"] is True
    attestation = json.loads(
        Path(result["os_isolation_attestation"]).read_text(encoding="utf-8")
    )
    assert attestation["runtime_mount"]["target"] == "/runtime/code"
    assert attestation["actual_open_ledger_required_post_run"] is True
    assert attestation["post_run_pass_not_claimed"] is True
    assert attestation["claim_scope"] == "pre_run_prerequisites_only"
    verified = _verify(paths, result)
    assert verified["status"] == "PASS"
    assert verified["runtime_root"] == str(paths["runtime"].resolve())  # type: ignore[union-attr]


def test_rejects_non_linux_before_creating_evidence(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(pre_run.platform, "system", lambda: "Windows")
    output = tmp_path / "evidence"
    with pytest.raises(pre_run.Phase2PreRunEvidenceError, match="Linux"):
        _build(paths, output)
    assert not output.exists()


def test_rejects_scorer_root_overlapping_predictor_package(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    truth = paths["package"] / "truth"  # type: ignore[operator]
    truth.mkdir()
    paths["scorer"] = truth
    with pytest.raises(pre_run.Phase2PreRunEvidenceError, match="overlaps"):
        _build(paths, tmp_path / "evidence")


def test_refuses_to_overwrite_existing_evidence_root(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    output = tmp_path / "evidence"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        _build(paths, output)


def test_verifier_rejects_attestation_tamper(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = _build(paths, tmp_path / "evidence")
    attestation_path = Path(result["os_isolation_attestation"])
    attestation_path.chmod(0o644)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["network_namespace_unshared"] = False
    attestation_path.write_text(
        json.dumps(attestation, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(pre_run.Phase2PreRunEvidenceError):
        _verify(paths, result)


def test_verifier_rejects_actual_executable_drift(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = _build(paths, tmp_path / "evidence")
    replacement = paths["system"] / "bwrap-replacement"  # type: ignore[operator]
    replacement.write_bytes(b"different")
    with pytest.raises(pre_run.Phase2PreRunEvidenceError, match="actual runner inputs"):
        _verify(paths, result, bwrap_executable=replacement)


def test_verifier_rejects_runtime_closure_digest_drift(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = _build(paths, tmp_path / "evidence")
    runtime = paths["runtime"]
    monkeypatch.setattr(
        pre_run,
        "verify_phase2_runtime_closure",
        lambda _path, **_kwargs: {
            "schema": "test.closure",
            "runtime_root": str(runtime),
            "runtime_mount_path": "/runtime/code",
            "entrypoint": "/runtime/code/scripts/run_cvs_stage2_predictor.py",
            "member_count": 7,
            "root_sha256": "9" * 64,
            "verified": True,
        },
    )
    with pytest.raises(pre_run.Phase2PreRunEvidenceError):
        _verify(paths, result)


def test_verifier_rejects_missing_forbidden_scorer_root(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    result = _build(paths, tmp_path / "evidence")
    with pytest.raises(pre_run.Phase2PreRunEvidenceError, match="at least one"):
        _verify(paths, result, forbidden_scorer_truth_roots=[])


def test_builder_rejects_untrusted_current_seal_digest(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    executables: list[Path] = paths["executables"]  # type: ignore[assignment]
    with pytest.raises(pre_run.Phase2PreRunEvidenceError, match="external trusted"):
        pre_run.build_phase2_pre_run_evidence(
            runtime_closure_root=paths["closure"],
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256="f" * 64,
            output_root=tmp_path / "evidence",
            bwrap_executable=executables[0],
            strace_executable=executables[1],
            python_executable=executables[2],
            system_read_roots=[paths["system"]],
            forbidden_scorer_truth_roots=[paths["scorer"]],
        )


def test_builder_rejects_caller_nominated_system_data_root(tmp_path: Path, monkeypatch) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    other = tmp_path / "untrusted_data"
    other.mkdir()
    executables: list[Path] = paths["executables"]  # type: ignore[assignment]
    with pytest.raises(pre_run.Phase2PreRunEvidenceError, match="fixed trusted allowlist"):
        pre_run.build_phase2_pre_run_evidence(
            runtime_closure_root=paths["closure"],
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=_sha(paths["seal"]),  # type: ignore[arg-type]
            output_root=tmp_path / "evidence",
            bwrap_executable=executables[0],
            strace_executable=executables[1],
            python_executable=executables[2],
            system_read_roots=[other],
            forbidden_scorer_truth_roots=[paths["scorer"]],
        )


def test_builds_profile_bound_somph_pre_run_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        pre_run,
        "verify_phase2_runtime_closure",
        lambda _path, **kwargs: {
            "schema": "test.closure",
            "profile": kwargs["expected_profile"],
            "runtime_root": str(paths["runtime"]),
            "runtime_mount_path": "/runtime/code",
            "entrypoint": "/runtime/code/scripts/run_cvs_somph_apply.py",
            "member_count": 12,
            "root_sha256": "1" * 64,
            "verified": True,
        },
    )
    monkeypatch.setattr(
        somph_bundle,
        "preflight_somph_predictor_bundle",
        lambda *_args, **_kwargs: (
            {
                "profile": somph_bundle.APPLY_ONLY,
                "package_root_sha256": "2" * 64,
            },
            {"artifact_member_allowlist_sha256": "3" * 64},
            {
                "schema": "test.somph_preopen",
                "status": "STRUCTURAL_SELF_CONSISTENCY_PASS",
            },
        ),
    )
    executables: list[Path] = paths["executables"]  # type: ignore[assignment]
    result = pre_run.build_phase2_pre_run_evidence(
        runtime_closure_root=paths["closure"],
        package_root=paths["package"],
        detached_seal=paths["seal"],
        expected_package_seal_sha256=_sha(paths["seal"]),  # type: ignore[arg-type]
        output_root=tmp_path / "somph_evidence",
        bwrap_executable=executables[0],
        strace_executable=executables[1],
        python_executable=executables[2],
        system_read_roots=[paths["system"]],
        forbidden_scorer_truth_roots=[paths["scorer"]],
        isolation_profile=pre_run.SOMPH_APPLY_PROFILE,
    )
    assert result["isolation_profile"] == pre_run.SOMPH_APPLY_PROFILE
    attestation = json.loads(
        Path(result["os_isolation_attestation"]).read_text(encoding="utf-8")
    )
    assert attestation["isolation_profile"] == pre_run.SOMPH_APPLY_PROFILE
    assert attestation["policy_contract"]["entrypoint"].endswith(
        "run_cvs_somph_apply.py"
    )
