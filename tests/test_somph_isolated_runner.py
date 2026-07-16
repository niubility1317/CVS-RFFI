from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cvsrffi.somph_isolated_runner as runner
from cvsrffi.phase2_runtime_contract import PHASE2_FULL_CONTRACT
from cvsrffi.somph_predictor_bundle import APPLY_ONLY, ENROLLMENT_ONLY
from cvsrffi.somph_runtime_request import SOMPH_APPLY_REQUEST_SCHEMA


def _fixture(tmp_path: Path) -> dict:
    paths = {}
    for name in ("package", "output", "system", "scorer", "runtime"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    (paths["runtime"] / "scripts").mkdir()
    (paths["runtime"] / "scripts/run_cvs_somph_apply.py").write_text(
        "# fixed\n", encoding="utf-8"
    )
    for name in ("python", "strace", "bwrap"):
        path = paths["system"] / name
        path.write_bytes(name.encode("ascii"))
        paths[name] = path
    seal = tmp_path / "package.seal.json"
    seal.write_text("{}\n", encoding="utf-8")
    paths["seal"] = seal
    seal_sha = hashlib.sha256(seal.read_bytes()).hexdigest()
    request = {
        "schema": SOMPH_APPLY_REQUEST_SCHEMA,
        "package_seal_sha256": seal_sha,
        "head_capsule_sha256": "2" * 64,
        "head_enrollment_binding_sha256": "3" * 64,
        "row_handle": "row_" + "4" * 64,
        "row_manifest_sha256": "5" * 64,
        "prediction_output_leaf": "prediction.cvspred",
        "device": "cpu",
        **PHASE2_FULL_CONTRACT,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request, sort_keys=True), encoding="utf-8")
    paths["request"] = request_path
    paths["request_sha"] = hashlib.sha256(request_path.read_bytes()).hexdigest()
    paths["seal_sha"] = seal_sha
    members = [
        ("checkpoint", "checkpoint.pt"),
        ("method_lock", "method_lock.json"),
        ("head_capsule", "head_capsule.npz"),
        ("overlay_provenance", "overlay_provenance.json"),
        ("query:leo_clear_weak", "query_leo_clear_weak.npz"),
        ("query:leo_low_elev_weak", "query_leo_low_elev_weak.npz"),
        ("query:leo_rain_weak", "query_leo_rain_weak.npz"),
    ]
    paths["manifest"] = {
        "profile": APPLY_ONLY,
        "package_root_sha256": "6" * 64,
        "checkpoint_sha256": "9" * 64,
        "method_lock_sha256": "c" * 64,
        "overlay_provenance_sha256": "d" * 64,
        "head_capsule_sha256": "2" * 64,
        "head_enrollment_binding_sha256": "3" * 64,
        "members": [
            {"kind": kind, "relative_path": relative}
            for kind, relative in members
        ],
    }
    return paths


def _trace(paths: dict, *, extra_execve: bool = False, support_open: bool = False) -> str:
    lines = [
        f'101 execve({json.dumps(str(Path(paths["python"]).resolve()))}, ["python"], 0x0) = 0',
        '101 openat(AT_FDCWD, "/sealed/request.json", O_RDONLY) = 3</sealed/request.json>',
        '101 openat(AT_FDCWD, "/sealed/package.seal.json", O_RDONLY) = 3</sealed/package.seal.json>',
        '101 openat(AT_FDCWD, "/runtime/code/scripts/run_cvs_somph_apply.py", O_RDONLY) = 4</runtime/code/scripts/run_cvs_somph_apply.py>',
    ]
    for item in paths["manifest"]["members"]:
        relative = item["relative_path"]
        lines.append(
            f'101 openat(AT_FDCWD, "/sealed/package/{relative}", O_RDONLY) = '
            f'5</sealed/package/{relative}>'
        )
    lines.append(
        '101 openat(AT_FDCWD, "/sealed/package/package_manifest.json", O_RDONLY) = '
        '5</sealed/package/package_manifest.json>'
    )
    if support_open:
        lines.append(
            '101 openat(AT_FDCWD, "/sealed/package/support_leo_clear_weak.npz", '
            'O_RDONLY) = 6</sealed/package/support_leo_clear_weak.npz>'
        )
    if extra_execve:
        lines.append('101 execve("/usr/bin/helper", ["helper"], 0x0) = 0')
    lines.append(
        '101 openat(AT_FDCWD, "/output/prediction.cvspred", '
        'O_WRONLY|O_CREAT|O_EXCL) = 7</output/prediction.cvspred>'
    )
    return "\n".join(lines) + "\n"


def _patch_preflight(monkeypatch: pytest.MonkeyPatch, paths: dict) -> None:
    monkeypatch.setattr(
        runner,
        "verify_phase2_pre_run_evidence",
        lambda *args, **kwargs: {
            "runtime_root": str(paths["runtime"]),
            "runtime_code_sha256": "7" * 64,
            "trusted_system_read_roots": [str(paths["system"].resolve())],
            "binding_sha256": "e" * 64,
        },
    )
    monkeypatch.setattr(
        runner,
        "preflight_somph_predictor_bundle",
        lambda *args, **kwargs: (
            paths["manifest"],
            {},
            {"manifest_sha256": "8" * 64},
        ),
    )
    monkeypatch.setattr(
        runner,
        "verify_somph_prediction_artifact",
        lambda *args, **kwargs: {
            "artifact_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
        },
    )


def _write_apply_outputs(paths: dict, *, receipt_sha_drift: bool = False) -> str:
    (paths["output"] / "prediction.cvspred").write_bytes(b"prediction")
    receipt = {
        "schema": "cvs.phase2.somph_apply_execution_receipt.v1",
        "status": "LOCAL_PROTOCOL_REPAIR_REQUIRED",
        "formal_launch_authority": False,
        "formal_metric_claim_allowed": False,
        "request_sha256": paths["request_sha"],
        "package_root_sha256": paths["manifest"]["package_root_sha256"],
        "package_seal_sha256": paths["seal_sha"],
        "checkpoint_sha256": paths["manifest"]["checkpoint_sha256"],
        "method_lock_sha256": paths["manifest"]["method_lock_sha256"],
        "overlay_provenance_sha256": paths["manifest"][
            "overlay_provenance_sha256"
        ],
        "head_capsule_sha256": paths["manifest"]["head_capsule_sha256"],
        "enrollment_binding_sha256": paths["manifest"][
            "head_enrollment_binding_sha256"
        ],
        "prediction_artifact_sha256": "a" * 64,
        "prediction_seal_sha256": "b" * 64,
        "preopen_audit": {"status": "STRUCTURAL_SELF_CONSISTENCY_PASS"},
        "resource": {
            "schema": "cvs.phase2.somph_apply_resource_receipt.v1"
        },
        "peak_cuda_memory_bytes": 0,
    }
    raw = (
        json.dumps(receipt, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    (paths["output"] / runner.APPLY_RECEIPT_NAME).write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    return ("f" * 64) if receipt_sha_drift else digest


def test_fixed_argv_exposes_no_entrypoint_or_truth_control() -> None:
    enrollment = runner._fixed_argv(
        profile=ENROLLMENT_ONLY, expected_seal_sha256="1" * 64
    )
    apply = runner._fixed_argv(
        profile=APPLY_ONLY, expected_seal_sha256="1" * 64
    )
    assert enrollment[0].endswith("run_cvs_somph_enrollment.py")
    assert apply[0].endswith("run_cvs_somph_apply.py")
    for argv in (enrollment, apply):
        text = " ".join(argv).lower()
        assert "truth" not in text
        assert "scorer" not in text
        assert "dataset" not in text
    assert "command_runner" not in inspect.signature(
        runner.execute_somph_isolated
    ).parameters


def test_apply_runner_uses_exact_members_and_real_trace_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_preflight(monkeypatch, paths)

    def fake_run(command, **kwargs):
        trace_path = Path(command[command.index("-o") + 1])
        trace_path.write_text(_trace(paths), encoding="utf-8")
        receipt_sha = _write_apply_outputs(paths)
        stdout = {
            "schema": "cvs.phase2.somph_apply_stdout.v1",
            "profile": APPLY_ONLY,
            "request_sha256": paths["request_sha"],
            "prediction_output_leaf": "prediction.cvspred",
            "artifact_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
            "execution_receipt_sha256": receipt_sha,
            "formal_launch_authority": False,
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(stdout), stderr="")

    result = runner._execute_somph_isolated_impl(
        profile=APPLY_ONLY,
        bwrap=paths["bwrap"],
        strace_executable=paths["strace"],
        runtime_closure_root=tmp_path / "closure",
        pre_run_evidence_root=tmp_path / "pre_run",
        expected_pre_run_binding_sha256="e" * 64,
        expected_runtime_closure_sha256="7" * 64,
        package_root=paths["package"],
        detached_seal=paths["seal"],
        expected_package_seal_sha256=paths["seal_sha"],
        request_json=paths["request"],
        output_root=paths["output"],
        python_executable=paths["python"],
        system_read_roots=[paths["system"]],
        forbidden_roots=[paths["scorer"]],
        forbidden_project_roots=["/srv/cvs"],
        command_runner=fake_run,
    )
    assert result["status"] == "LOCAL_PROTOCOL_REPAIR_REQUIRED"
    assert result["formal_launch_authority"] is False
    assert result["output_sha256"] == "a" * 64
    audit = json.loads(
        (paths["output"] / runner.AUDIT_NAME).read_text(encoding="utf-8")
    )
    assert audit["status"] == "PASS"
    assert audit["sealed_package_exact_members"] == sorted(
        [
            f"/sealed/package/{item['relative_path']}"
            for item in paths["manifest"]["members"]
        ]
        + ["/sealed/package/package_manifest.json"]
    )


@pytest.mark.parametrize(
    ("extra_execve", "support_open", "message"),
    [
        (True, False, "additional successful execve"),
        (False, True, "opened-file ledger"),
    ],
)
def test_runner_rejects_extra_process_or_forbidden_support_member(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extra_execve: bool,
    support_open: bool,
    message: str,
) -> None:
    paths = _fixture(tmp_path)
    _patch_preflight(monkeypatch, paths)

    def fake_run(command, **kwargs):
        trace_path = Path(command[command.index("-o") + 1])
        trace_path.write_text(
            _trace(paths, extra_execve=extra_execve, support_open=support_open),
            encoding="utf-8",
        )
        receipt_sha = _write_apply_outputs(paths)
        stdout = {
            "schema": "cvs.phase2.somph_apply_stdout.v1",
            "profile": APPLY_ONLY,
            "request_sha256": paths["request_sha"],
            "prediction_output_leaf": "prediction.cvspred",
            "artifact_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
            "execution_receipt_sha256": receipt_sha,
            "formal_launch_authority": False,
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(stdout), stderr="")

    with pytest.raises(runner.SomphIsolatedRunnerError, match=message):
        runner._execute_somph_isolated_impl(
            profile=APPLY_ONLY,
            bwrap=paths["bwrap"],
            strace_executable=paths["strace"],
            runtime_closure_root=tmp_path / "closure",
            pre_run_evidence_root=tmp_path / "pre_run",
            expected_pre_run_binding_sha256="e" * 64,
            expected_runtime_closure_sha256="7" * 64,
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=paths["seal_sha"],
            request_json=paths["request"],
            output_root=paths["output"],
            python_executable=paths["python"],
            system_read_roots=[paths["system"]],
            forbidden_roots=[paths["scorer"]],
            command_runner=fake_run,
        )


def test_runner_rejects_execution_receipt_digest_before_stdout_pass_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_preflight(monkeypatch, paths)

    def fake_run(command, **kwargs):
        trace_path = Path(command[command.index("-o") + 1])
        trace_path.write_text(_trace(paths), encoding="utf-8")
        receipt_sha = _write_apply_outputs(paths, receipt_sha_drift=True)
        stdout = {
            "schema": "cvs.phase2.somph_apply_stdout.v1",
            "profile": APPLY_ONLY,
            "request_sha256": paths["request_sha"],
            "prediction_output_leaf": "prediction.cvspred",
            "artifact_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
            "execution_receipt_sha256": receipt_sha,
            "formal_launch_authority": False,
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(stdout), stderr="")

    with pytest.raises(runner.SomphIsolatedRunnerError, match="receipt SHA256"):
        runner._execute_somph_isolated_impl(
            profile=APPLY_ONLY,
            bwrap=paths["bwrap"],
            strace_executable=paths["strace"],
            runtime_closure_root=tmp_path / "closure",
            pre_run_evidence_root=tmp_path / "pre_run",
            expected_pre_run_binding_sha256="e" * 64,
            expected_runtime_closure_sha256="7" * 64,
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=paths["seal_sha"],
            request_json=paths["request"],
            output_root=paths["output"],
            python_executable=paths["python"],
            system_read_roots=[paths["system"]],
            forbidden_roots=[paths["scorer"]],
            command_runner=fake_run,
        )
    assert not (paths["output"] / runner.STDOUT_RECEIPT_NAME).exists()


def test_runner_rejects_extra_output_member(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_preflight(monkeypatch, paths)

    def fake_run(command, **kwargs):
        trace_path = Path(command[command.index("-o") + 1])
        trace_path.write_text(_trace(paths), encoding="utf-8")
        receipt_sha = _write_apply_outputs(paths)
        (paths["output"] / "side_channel.json").write_text(
            "{}", encoding="utf-8"
        )
        stdout = {
            "schema": "cvs.phase2.somph_apply_stdout.v1",
            "profile": APPLY_ONLY,
            "request_sha256": paths["request_sha"],
            "prediction_output_leaf": "prediction.cvspred",
            "artifact_sha256": "a" * 64,
            "seal_sha256": "b" * 64,
            "execution_receipt_sha256": receipt_sha,
            "formal_launch_authority": False,
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(stdout), stderr="")

    with pytest.raises(runner.SomphIsolatedRunnerError, match="output exact"):
        runner._execute_somph_isolated_impl(
            profile=APPLY_ONLY,
            bwrap=paths["bwrap"],
            strace_executable=paths["strace"],
            runtime_closure_root=tmp_path / "closure",
            pre_run_evidence_root=tmp_path / "pre_run",
            expected_pre_run_binding_sha256="e" * 64,
            expected_runtime_closure_sha256="7" * 64,
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=paths["seal_sha"],
            request_json=paths["request"],
            output_root=paths["output"],
            python_executable=paths["python"],
            system_read_roots=[paths["system"]],
            forbidden_roots=[paths["scorer"]],
            command_runner=fake_run,
        )


def test_runner_rejects_unapproved_pre_run_or_runtime_trust_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _fixture(tmp_path)
    _patch_preflight(monkeypatch, paths)
    called = False

    def fake_run(command, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess must not start")

    with pytest.raises(runner.SomphIsolatedRunnerError, match="trust root mismatch"):
        runner._execute_somph_isolated_impl(
            profile=APPLY_ONLY,
            bwrap=paths["bwrap"],
            strace_executable=paths["strace"],
            runtime_closure_root=tmp_path / "closure",
            pre_run_evidence_root=tmp_path / "pre_run",
            expected_pre_run_binding_sha256="0" * 64,
            expected_runtime_closure_sha256="7" * 64,
            package_root=paths["package"],
            detached_seal=paths["seal"],
            expected_package_seal_sha256=paths["seal_sha"],
            request_json=paths["request"],
            output_root=paths["output"],
            python_executable=paths["python"],
            system_read_roots=[paths["system"]],
            forbidden_roots=[paths["scorer"]],
            command_runner=fake_run,
        )
    assert called is False
