from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import cvsrffi.phase2_isolated_runner as isolated_runner
from cvsrffi.phase2_isolated_runner import (
    AUDIT_NAME,
    POST_EVIDENCE_NAME,
    STDOUT_RECEIPT_NAME,
    TRACE_NAME,
    Phase2IsolatedRunnerError,
    audit_open_ledger,
    build_production_predictor_argv,
    execute_phase2_isolated,
    parse_successful_execve_trace,
    parse_successful_open_trace,
)
from cvsrffi.phase2_runtime_contract import (
    PHASE2_FULL_CONTRACT,
    POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS,
)
from cvsrffi.stage2_prediction_artifact import publish_prediction_artifact


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fixture_tree(tmp_path: Path) -> dict[str, object]:
    paths: dict[str, object] = {}
    for name in ("runtime", "closure", "evidence", "package", "output", "system", "scorer"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    python = paths["system"] / "python"  # type: ignore[index,operator]
    strace = paths["system"] / "strace"  # type: ignore[index,operator]
    bwrap = paths["system"] / "bwrap"  # type: ignore[index,operator]
    python.write_bytes(b"python")
    strace.write_bytes(b"strace")
    bwrap.write_bytes(b"bwrap")
    paths["python"] = python
    paths["strace"] = strace
    paths["bwrap"] = bwrap

    seal = tmp_path / "package.seal.json"
    seal_bytes = b'{"sealed":true}\n'
    seal.write_bytes(seal_bytes)
    paths["seal"] = seal
    class_handle = "cls_" + "1" * 64
    pre = {
        "sealed_inference_package_sha256": _sha(seal_bytes),
        "package_root_sha256": "2" * 64,
        "runtime_code_sha256": "3" * 64,
        "artifact_member_allowlist_sha256": "4" * 64,
        "os_isolation_mode": "bwrap_readonly_mounts",
        "os_isolation_attestation_sha256": "5" * 64,
        "preopen_audit_status": "PASS",
        "preopen_audit_receipt_sha256": "6" * 64,
        "predict_score_process_isolation": True,
    }
    descriptor = {
        "relative_path": "artifact.bin",
        "sha256": "7" * 64,
        "size_bytes": 1,
        "artifact_role": "placeholder",
        "schema": "test.v1",
    }
    scenarios = ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
    request = {
        "schema_version": "cvs.phase2.predict_request.v2",
        "request_id": "request-test",
        "row_id": "row-test",
        "stage": "stage2c",
        "receiver": "rx-test",
        "scenarios": scenarios,
        "k_shot": 1,
        "satellite_seed": 11,
        "candidate_lock_sha256": "8" * 64,
        "package_root_sha256": pre["package_root_sha256"],
        "runtime_code_sha256": pre["runtime_code_sha256"],
        "registered_class_count": 1,
        "registered_classes": [{"class_index": 0, "class_handle": class_handle}],
        "support_artifacts": [
            {**descriptor, "relative_path": f"support-{index}.npz", "artifact_role": f"support:{scenario}"}
            for index, scenario in enumerate(scenarios)
        ],
        "query_artifacts": [
            {**descriptor, "relative_path": f"query-{index}.npz", "artifact_role": f"query:{scenario}"}
            for index, scenario in enumerate(scenarios)
        ],
        "checkpoint_artifact": {**descriptor, "artifact_role": "checkpoint"},
        "adapter_artifact": {**descriptor, "artifact_role": "adapter"},
        "head_artifact": {**descriptor, "artifact_role": "head"},
        "tta_policy": {"mode": "adaptive_1_3_5"},
        "tta_policy_sha256": "9" * 64,
        "output_contract": {
            "schema": "cvs.phase2.prediction.v2",
            "relative_path": "prediction_artifact.cvspred",
            "sealed_immutable_required": True,
        },
        "phase2_runtime_isolation_evidence": pre,
        **PHASE2_FULL_CONTRACT,
    }
    request_path = tmp_path / "request.json"
    _write_json(request_path, request)
    paths["request"] = request_path
    paths["request_payload"] = request
    paths["class_handle"] = class_handle
    return paths


def test_production_argv_is_fixed_and_contains_no_truth_or_scorer_route() -> None:
    argv = build_production_predictor_argv(expected_seal_sha256="a" * 64)
    assert argv[0] == "/runtime/code/scripts/run_cvs_stage2_predictor.py"
    assert argv[argv.index("--request-json") + 1] == "/sealed/request.json"
    assert argv[argv.index("--output-root") + 1] == "/output"
    lowered = " ".join(argv).lower()
    assert "truth" not in lowered
    assert "scorer" not in lowered
    assert "dataset" not in lowered


def test_trace_parser_keeps_only_successful_opens_and_resolves_dirfd() -> None:
    trace = "\n".join(
        [
            '101 openat(AT_FDCWD, "/sealed/request.json", O_RDONLY) = 3</sealed/request.json>',
            '[pid 102] openat(3</sealed/package>, "query.npz", O_RDONLY <unfinished ...>',
            '[pid 102] <... openat resumed>) = 4</sealed/package/query.npz>',
            '101 open("relative.py", O_RDONLY) = 5',
            '101 openat(AT_FDCWD, "/missing", O_RDONLY) = -1 ENOENT (No such file or directory)',
        ]
    )
    ledger = parse_successful_open_trace(trace)
    assert [row["path"] for row in ledger] == [
        "/runtime/code/relative.py",
        "/sealed/package/query.npz",
        "/sealed/request.json",
    ]
    assert sum(row["successful_open_count"] for row in ledger) == 3


def test_execve_parser_reports_only_successful_additional_processes() -> None:
    trace = "\n".join(
        [
            '201 execve("/usr/bin/helper", ["helper"], 0x0) = 0',
            '201 execve("/usr/bin/missing", ["missing"], 0x0) = -1 ENOENT (No such file or directory)',
            '[pid 202] execve("/bin/sh", ["sh", "-c", "true"], 0x0 <unfinished ...>',
            '[pid 202] <... execve resumed>) = 0',
        ]
    )
    assert parse_successful_execve_trace(trace) == [
        {"line_number": 1, "executable": "/usr/bin/helper"},
        {"line_number": 3, "executable": "/bin/sh"},
    ]


def test_execve_parser_fails_closed_on_incomplete_resumed_trace() -> None:
    with pytest.raises(Phase2IsolatedRunnerError, match="unfinished execve"):
        parse_successful_execve_trace(
            '201 execve("/bin/sh", ["sh"], 0x0 <unfinished ...>\n'
        )
    with pytest.raises(Phase2IsolatedRunnerError, match="orphan resumed execve"):
        parse_successful_execve_trace("201 <... execve resumed>) = 0\n")


def test_open_ledger_rejects_truth_and_unmounted_project_paths() -> None:
    ledger = [
        {"path": "/runtime/code/predict.py", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/request.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/package.seal.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/package/query.npz", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/output/.prediction.tmp", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/srv/cvs/scorer/truth_sidecar.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/srv/cvs/raw.pkl", "successful_open_count": 1, "syscalls": ["open"]},
    ]
    audit = audit_open_ledger(
        ledger, system_read_roots=["/usr"], forbidden_project_roots=["/srv/cvs"]
    )
    assert audit["status"] == "FAIL"
    assert len(audit["violations"]) == 2


def test_open_ledger_enforces_exact_and_required_sealed_package_members() -> None:
    ledger = [
        {"path": "/runtime/code/predict.py", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/request.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/package.seal.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/package/package_manifest.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/package/extra.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/output/prediction.cvspred", "successful_open_count": 1, "syscalls": ["openat"]},
    ]
    audit = audit_open_ledger(
        ledger,
        system_read_roots=["/usr"],
        sealed_package_members=["package_manifest.json", "query.npz"],
        required_package_members=["package_manifest.json", "query.npz"],
    )
    assert audit["status"] == "FAIL"
    assert audit["sealed_package_exact_members"] == [
        "/sealed/package/package_manifest.json",
        "/sealed/package/query.npz",
    ]
    assert audit["required_package_members"] == [
        "/sealed/package/package_manifest.json",
        "/sealed/package/query.npz",
    ]
    assert {"path": "/sealed/package/extra.json", "reason": "sealed_package_member_not_allowlisted"} in audit[
        "violations"
    ]
    assert {"path": "/sealed/package/query.npz", "reason": "required_package_member_not_opened"} in audit[
        "violations"
    ]


def test_open_ledger_default_keeps_directory_level_package_compatibility() -> None:
    ledger = [
        {"path": "/runtime/code/predict.py", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/request.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/package.seal.json", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/sealed/package/legacy_member.bin", "successful_open_count": 1, "syscalls": ["openat"]},
        {"path": "/output/prediction.cvspred", "successful_open_count": 1, "syscalls": ["openat"]},
    ]
    audit = audit_open_ledger(ledger, system_read_roots=["/usr"])
    assert audit["status"] == "PASS"
    assert audit["sealed_package_exact_members"] is None
    assert audit["required_package_members"] == []


def test_fake_isolated_run_emits_immutable_bound_post_run_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture_tree(tmp_path)
    output: Path = paths["output"]  # type: ignore[assignment]
    request_path: Path = paths["request"]  # type: ignore[assignment]
    request: dict = paths["request_payload"]  # type: ignore[assignment]
    class_handle = str(paths["class_handle"])
    monkeypatch.setattr(
        isolated_runner,
        "verify_phase2_pre_run_evidence",
        lambda **_kwargs: {
            "status": "PASS",
            "runtime_root": str(paths["runtime"]),
            "binding_sha256": "a" * 64,
            "trusted_system_read_roots": [str(paths["system"].resolve())],
        },
    )

    def fake_run(command, **kwargs):
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert "pass_fds" not in kwargs
        assert command[0] == str(Path(paths["strace"]).resolve())
        trace_target = Path(command[command.index("-o") + 1])
        assert trace_target.parent == output.parent
        assert output not in trace_target.parents
        assert "/proc/self/fd/" not in " ".join(command)
        assert command.index(str(Path(paths["bwrap"]).resolve())) > command.index("-o")
        trace_target.write_text(
            "\n".join(
                [
                    '199 openat(AT_FDCWD, "/host/bwrap/setup-source", O_RDONLY) = 2</host/bwrap/setup-source>',
                    f'201 execve({json.dumps(str(Path(paths["python"]).resolve()))}, ["python"], 0x0) = 0',
                    '201 openat(AT_FDCWD, "/sealed/request.json", O_RDONLY) = 3</sealed/request.json>',
                    '201 openat(AT_FDCWD, "/sealed/package.seal.json", O_RDONLY) = 3</sealed/package.seal.json>',
                    '201 openat(AT_FDCWD, "/sealed/package/manifest.json", O_RDONLY) = 4</sealed/package/manifest.json>',
                    '201 openat(AT_FDCWD, "/runtime/code/scripts/run_cvs_stage2_predictor.py", O_RDONLY) = 5</runtime/code/scripts/run_cvs_stage2_predictor.py>',
                    '201 openat(AT_FDCWD, "/output/prediction_artifact.cvspred", O_WRONLY|O_CREAT|O_EXCL) = 6</output/prediction_artifact.cvspred>',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        scenarios = np.asarray(request["scenarios"])
        published = publish_prediction_artifact(
            output / request["output_contract"]["relative_path"],
            stage="Stage2-C",
            row_id=request["row_id"],
            receiver=request["receiver"],
            k_shot=request["k_shot"],
            candidate_lock_sha256=request["candidate_lock_sha256"],
            package_root_sha256=request["package_root_sha256"],
            package_seal_sha256=request["phase2_runtime_isolation_evidence"]["sealed_inference_package_sha256"],
            query_tokens=np.asarray(["qry_" + str(index) + "a" * 64 for index in range(3)]),
            scenarios=scenarios,
            candidate_after=np.asarray([class_handle] * 3),
            candidate_before=np.asarray([class_handle] * 3),
            identity_after=np.asarray([class_handle] * 3),
            identity_before=np.asarray([class_handle] * 3),
            direct=np.asarray([class_handle] * 3),
            shared_view_counts=np.asarray([1, 3, 5]),
        )
        result = {
            "artifact_sha256": published["artifact_sha256"],
            "seal_sha256": published["seal_sha256"],
            "request_sha256": _sha(request_path.read_bytes()),
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(result), stderr="")

    result = execute_phase2_isolated(
        bwrap=paths["bwrap"],
        strace_executable=paths["strace"],
        runtime_closure_root=paths["closure"],
        pre_run_evidence_root=paths["evidence"],
        package_root=paths["package"],
        detached_seal=paths["seal"],
        expected_package_seal_sha256=request["phase2_runtime_isolation_evidence"]["sealed_inference_package_sha256"],
        request_json=request_path,
        output_root=output,
        python_executable=paths["python"],
        system_read_roots=[paths["system"]],
        forbidden_roots=[paths["scorer"]],
        forbidden_project_roots=["/srv/cvs"],
        command_runner=fake_run,
    )
    assert result["status"] == "LOCAL_DIAGNOSTIC_PASS"
    assert result["formal_launch_authority"] is False
    assert result["formal_launch_blockers"]
    command = result["command"]
    assert "--unshare-all" in command and "--share-net" not in command
    assert command.count("--bind") == 1
    assert str(paths["scorer"]) not in command
    runtime_bind = command.index(str(paths["runtime"].resolve()))
    assert command[runtime_bind - 1] == "--ro-bind"
    assert command[runtime_bind + 1] == "/runtime/code"
    assert command[command.index("--chdir") + 1] == "/runtime/code"
    for name in (TRACE_NAME, AUDIT_NAME, STDOUT_RECEIPT_NAME, POST_EVIDENCE_NAME):
        path = output / name
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) & 0o222 == 0
    diagnostic_post = json.loads((output / POST_EVIDENCE_NAME).read_text(encoding="utf-8"))
    assert diagnostic_post["status"] == "LOCAL_DIAGNOSTIC_PASS"
    assert diagnostic_post["formal_launch_authority"] is False
    assert diagnostic_post["protocol_valid_claim_allowed"] is False
    post = diagnostic_post["formal_post_run_contract_evidence"]
    assert set(post) == set(POST_RUN_RUNTIME_EVIDENCE_REQUIRED_FIELDS)
    assert post["filesystem_access_audit_status"] == "PASS"
    assert post["prediction_artifact_sha256"] == result["prediction_artifact_sha256"]
    audit = json.loads((output / AUDIT_NAME).read_text(encoding="utf-8"))
    assert audit["trace_sha256"] == result["trace_sha256"]
    assert audit["request_sha256"] == result["request_sha256"]
    assert audit["unique_opened_path_count"] == 5
    assert audit["trace_scope"] == "after_bound_predictor_python_execve"
    assert "/host/bwrap/setup-source" not in {
        row["path"] for row in audit["opened_file_ledger"]
    }


def test_production_runner_rejects_additional_successful_execve(
    tmp_path: Path, monkeypatch
) -> None:
    paths = _fixture_tree(tmp_path)
    monkeypatch.setattr(
        isolated_runner,
        "verify_phase2_pre_run_evidence",
        lambda **_kwargs: {
            "status": "PASS",
            "runtime_root": str(paths["runtime"]),
            "binding_sha256": "a" * 64,
            "trusted_system_read_roots": [str(paths["system"].resolve())],
        },
    )

    def fake_run(command, **_kwargs):
        trace_target = Path(command[command.index("-o") + 1])
        trace_target.write_text(
            "\n".join(
                [
                    f'201 execve({json.dumps(str(Path(paths["python"]).resolve()))}, ["python"], 0x0) = 0',
                    '201 openat(AT_FDCWD, "/sealed/request.json", O_RDONLY) = 3</sealed/request.json>',
                    '201 execve("/bin/sh", ["sh", "-c", "true"], 0x0) = 0',
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=1, stdout="", stderr="")

    request = paths["request_payload"]
    with pytest.raises(
        Phase2IsolatedRunnerError, match="additional successful execve.*?/bin/sh"
    ):
        execute_phase2_isolated(
            bwrap=paths["bwrap"],
            strace_executable=paths["strace"],
            runtime_closure_root=paths["closure"],
            pre_run_evidence_root=paths["evidence"],
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=request["phase2_runtime_isolation_evidence"][
                "sealed_inference_package_sha256"
            ],
            request_json=paths["request"],
            output_root=paths["output"],
            python_executable=paths["python"],
            system_read_roots=[paths["system"]],
            forbidden_roots=[paths["scorer"]],
            command_runner=fake_run,
        )


def test_production_runner_rejects_nonempty_output_before_subprocess(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)
    (paths["output"] / "existing.json").write_text("{}", encoding="utf-8")  # type: ignore[index,operator]
    with pytest.raises(Phase2IsolatedRunnerError, match="must be empty"):
        execute_phase2_isolated(
            bwrap=paths["bwrap"],
            strace_executable=paths["strace"],
            runtime_closure_root=paths["closure"],
            pre_run_evidence_root=paths["evidence"],
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=paths["request_payload"]["phase2_runtime_isolation_evidence"]["sealed_inference_package_sha256"],
            request_json=paths["request"],
            output_root=paths["output"],
            python_executable=paths["python"],
            system_read_roots=[paths["system"]],
            forbidden_roots=[paths["scorer"]],
        )


def test_production_runner_rejects_empty_scorer_truth_root_set(tmp_path: Path) -> None:
    paths = _fixture_tree(tmp_path)
    with pytest.raises(Phase2IsolatedRunnerError, match="at least one"):
        execute_phase2_isolated(
            bwrap=paths["bwrap"],
            strace_executable=paths["strace"],
            runtime_closure_root=paths["closure"],
            pre_run_evidence_root=paths["evidence"],
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=paths["request_payload"]["phase2_runtime_isolation_evidence"]["sealed_inference_package_sha256"],
            request_json=paths["request"],
            output_root=paths["output"],
            python_executable=paths["python"],
            system_read_roots=[paths["system"]],
            forbidden_roots=[],
        )
