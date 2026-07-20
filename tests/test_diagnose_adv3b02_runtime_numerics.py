from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

import pytest
import torch
from torch import nn

from scripts import diagnose_adv3b02_runtime_numerics as diagnostic


diagnostic._load_worker_dependencies()

SOURCE_COMMIT = "a" * 40


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TinyRuntime(nn.Module):
    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = rows[:, 0, :160]
        logits = torch.stack(
            (feature[:, 0], feature[:, 1], feature[:, 0] - feature[:, 1]), dim=1
        )
        return feature, logits


class OffsetRuntime(TinyRuntime):
    def forward(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature, logits = super().forward(rows)
        offset = logits.new_tensor([0.0, 0.02, -0.01])
        return feature + 0.01, logits + offset


def _assets(
    tmp_path: Path, runtime_model: nn.Module | None = None
) -> tuple[Path, Path, Path, Path]:
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"diagnostic-checkpoint")
    runtime = tmp_path / "runtime.ts"
    traced = torch.jit.trace(
        (runtime_model or TinyRuntime()).eval(),
        torch.randn(2, 2, diagnostic.INPUT_LEN),
        strict=False,
        check_trace=False,
    )
    torch.jit.save(traced, runtime)
    fresh_runtime = tmp_path / "fresh_runtime.ts"
    fresh_traced = torch.jit.trace(
        TinyRuntime().eval(),
        torch.randn(2, 2, diagnostic.INPUT_LEN),
        strict=False,
        check_trace=False,
    )
    torch.jit.save(fresh_traced, fresh_runtime)
    lineage = tmp_path / "lineage.json"
    lineage.write_bytes(b"unit-test-lineage")
    return checkpoint, runtime, fresh_runtime, lineage


def _patch_assets(
    monkeypatch: pytest.MonkeyPatch,
    checkpoint: Path,
    runtime: Path,
    lineage: Path,
) -> None:
    checkpoint_sha = _file_sha256(checkpoint)
    runtime_sha = _file_sha256(runtime)
    monkeypatch.setattr(diagnostic, "BASE_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(
        diagnostic,
        "RUNTIME_ARMS",
        {
            "b202": {
                "sha256": runtime_sha,
                "canonical_remote_path": str(runtime.resolve()),
                "lineage_evidence_path": str(lineage.resolve()),
                "lineage_evidence_sha256": _file_sha256(lineage),
                "lineage_scope": "unit_test_historical_lineage_only",
                "artifact_origin_receipt_sha256": "",
            },
            "f119": {
                "sha256": "f" * 64,
                "canonical_remote_path": "unit-test-f119",
                "lineage_evidence_path": "unit-test-f119-lineage",
                "lineage_evidence_sha256": "e" * 64,
                "lineage_scope": "unit_test_missing_lineage",
                "artifact_origin_receipt_sha256": "",
            },
        },
    )
    monkeypatch.setattr(
        diagnostic,
        "_load_checkpoint_bytes",
        lambda value: {"model": {"weight": torch.arange(4, dtype=torch.float32)}},
    )
    monkeypatch.setattr(
        diagnostic,
        "build_exact_ssdg_model_from_checkpoint",
        lambda *args, **kwargs: (TinyRuntime(), {"diagnostic": True}),
    )
    monkeypatch.setattr(diagnostic, "ADV3B02IdentityRuntime", lambda model: model)
    monkeypatch.setattr(
        diagnostic,
        "_git_audit",
        lambda: {
            "repository_root": "unit-test",
            "git_available": True,
            "commit": SOURCE_COMMIT,
            "dirty": False,
            "status_root_sha256": "1" * 64,
            "diff_root_sha256": "2" * 64,
            "cached_diff_root_sha256": "3" * 64,
            "untracked": [],
            "untracked_root_sha256": "4" * 64,
        },
    )


def _run_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    runtime_model: nn.Module | None = None,
    mode: str = "baseline",
    artifact_name: str = "worker.json",
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path, Path]:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path, runtime_model)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    if mode == "deterministic":
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    artifact = tmp_path / artifact_name
    summary = diagnostic._worker_diagnostic(
        checkpoint_path=checkpoint,
        runtime_path=runtime,
        lineage_evidence_path=lineage,
        artifact_origin_receipt_path=None,
        fresh_runtime_path=fresh_runtime,
        expected_fresh_runtime_sha256=_file_sha256(fresh_runtime),
        trace_builder_artifact_sha256="b" * 64,
        arm_id="b202",
        source_git_commit=SOURCE_COMMIT,
        source_archive_sha256="c" * 64,
        source_release_receipt_sha256="d" * 64,
        worker_artifact_out=artifact,
        device="cpu",
        worker_mode=mode,
        worker_scope="unit_test_only_cpu_primary",
        _allow_cpu_primary_for_tests=True,
    )
    return (
        summary,
        json.loads(artifact.read_text(encoding="utf-8")),
        artifact,
        checkpoint,
        runtime,
        lineage,
    )


def _worker_extra(fresh_runtime: Path, lineage: Path) -> dict[str, Any]:
    return {
        "lineage_evidence_path": lineage,
        "artifact_origin_receipt_path": None,
        "fresh_runtime_path": fresh_runtime,
        "expected_fresh_runtime_sha256": _file_sha256(fresh_runtime),
        "trace_builder_artifact_sha256": "b" * 64,
        "source_archive_sha256": "c" * 64,
        "source_release_receipt_sha256": "d" * 64,
    }


def _source_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    archive = tmp_path / "source.zip"
    source_member_bytes = b"unit source member\n"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("code/unit_source.py", source_member_bytes)
    source_members = [
        {
            "path": "code/unit_source.py",
            "bytes": len(source_member_bytes),
            "sha256": diagnostic._sha256_bytes(source_member_bytes),
        }
    ]
    body = {
        "schema": "cvs.development.source_archive_commit_receipt.v1",
        "issuer": diagnostic.SOURCE_RELEASE_ISSUER,
        "key_id": diagnostic.SOURCE_RELEASE_KEY_ID,
        "public_key_sha256": diagnostic.SOURCE_RELEASE_PUBLIC_KEY_SHA256,
        "source_archive_path": str(archive.resolve()),
        "source_archive_sha256": _file_sha256(archive),
        "source_git_commit": SOURCE_COMMIT,
        "source_members": source_members,
        "source_manifest_root_sha256": diagnostic._source_manifest_root(
            source_members
        ),
        "git_policy": {
            "mode": "git_exact",
            "commit": SOURCE_COMMIT,
            "dirty": False,
            "status_root_sha256": "1" * 64,
            "diff_root_sha256": "2" * 64,
            "cached_diff_root_sha256": "3" * 64,
            "untracked_root_sha256": "4" * 64,
        },
    }
    receipt = tmp_path / "source_release_receipt.json"
    receipt.write_text(
        json.dumps({**body, "signature_hex": "0" * 128}, sort_keys=True),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        diagnostic, "_verify_source_release_signature", lambda *args: None
    )
    return archive, receipt


def _attach_launch_audit(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    value = copy.deepcopy(payload)
    artifact_sha = "7" * 64
    artifact_bytes = 123
    value["orchestrator_launch_audit"] = {
        "returncode": 0,
        "stdout": "{}",
        "stderr": "",
        "stdout_summary": {
            "status": diagnostic.WORKER_STATUS,
            "artifact_path": "unit-worker-artifact",
            "artifact_sha256": artifact_sha,
            "artifact_bytes": artifact_bytes,
        },
        "worker_artifact_sha256": artifact_sha,
        "worker_artifact_bytes": artifact_bytes,
        "worker_artifact_path": "unit-worker-artifact",
        "environment_applied_before_worker_python_start": True,
        "deterministic_cublas_set_before_spawn": mode == "deterministic",
    }
    return value


def _trace_builder_from_worker(
    worker: dict[str, Any], *, fresh_sha: str, trace_sha: str
) -> dict[str, Any]:
    return {
        "schema": diagnostic.TRACE_BUILDER_SCHEMA,
        "status": diagnostic.TRACE_BUILDER_STATUS,
        "formal_authority": False,
        "parity_receipt_emitted": False,
        "checkpoint_sha256": worker["checkpoint_sha256"],
        "fixed_probe_spec": copy.deepcopy(worker["fixed_probe_spec"]),
        "fresh_runtime": copy.deepcopy(
            worker["suite"]["mode_result"]["fresh_trace"]
        ),
        "device": {"resolved_device": "cuda:0"},
        "source_release_binding": copy.deepcopy(worker["source_release_binding"]),
        "flags": {"cublas_workspace_config": None},
        "software": copy.deepcopy(worker["software"]),
        "checkpoint_model_state": copy.deepcopy(
            worker["suite"]["checkpoint_model_state"]
        ),
        "eager_tensor_registry": copy.deepcopy(
            worker["suite"]["eager_tensor_registry"]
        ),
        "orchestrator_launch_audit": {
            "returncode": 0,
            "stdout_summary": {
                "artifact_path": "unit-trace-artifact",
                "artifact_sha256": trace_sha,
                "fresh_runtime_path": "unit-fresh-runtime",
                "fresh_runtime_sha256": fresh_sha,
            },
            "trace_builder_artifact_sha256": trace_sha,
            "trace_builder_artifact_path": "unit-trace-artifact",
            "fresh_runtime_sha256": fresh_sha,
            "fresh_runtime_path": "unit-fresh-runtime",
            "environment_applied_before_worker_python_start": True,
            "cublas_removed_before_spawn": True,
        },
    }


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        return [item for pair in value.items() for item in _all_strings(pair)]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _all_strings(child)]
    return [value] if isinstance(value, str) else []


def test_worker_fixed_probe_repeatability_fresh_trace_and_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary, payload, _, _, _, _ = _run_worker(tmp_path, monkeypatch)
    assert summary["status"] == diagnostic.WORKER_STATUS
    assert payload["fixed_probe_spec"]["root_sha256"] == (
        diagnostic.EXPECTED_PROBE_ROOT_SHA256
    )
    assert [row["batch_size"] for row in payload["fixed_probe_spec"]["rows"]] == [
        1,
        8,
        256,
    ]
    suite = payload["suite"]
    mode = suite["mode_result"]
    assert mode["fresh_trace"]["bytes"] > 0
    assert mode["fresh_trace"]["selected_for_runtime"] is False
    assert suite["checkpoint_model_state"]["tensor_count"] == 1
    assert suite["eager_tensor_registry"]["state"]["root_sha256"]
    assert suite["existing_runtime_tensor_registry"]["state"]["root_sha256"]
    assert mode["fresh_trace"]["tensor_registry"]["state"]["root_sha256"]
    assert suite["existing_runtime_structure"]["runtime_structure_sha256"]
    assert payload["software"]["code_dependencies"]["root_sha256"]
    dependency_paths = {
        row["path"] for row in payload["software"]["code_dependencies"]["entries"]
    }
    assert {
        "code/model_dual_cvsincnet.py",
        "code/cvsrffi/somph_runtime_trust.py",
        "code/cvsrffi/phase1_center_lowrank_prototype_bundle.py",
        "code/cvsrffi/stage2_predictor_bundle.py",
        "code/SSDG/train_ssdg.py",
    } <= dependency_paths
    assert payload["resources"]["checkpoint_file_bytes"] > 0
    assert payload["resources"]["runtime_file_bytes"] > 0
    assert payload["resources"]["wall_time_seconds"] >= 0
    assert payload["resources"]["process_peak_rss"]["status"] in {
        "COMPLETE",
        "INCOMPLETE",
    }
    if payload["resources"]["process_peak_rss"]["status"] == "COMPLETE":
        assert payload["resources"]["process_peak_rss"]["peak_rss_bytes"] > 0
    for batch_size in ("1", "8", "256"):
        comparisons = mode["batches"][batch_size]["comparisons"]
        assert comparisons["eager_a_vs_eager_b"]["feature"]["max_abs"] == 0.0
        assert comparisons["existing_runtime_a_vs_existing_runtime_b"]["logits"][
            "max_abs"
        ] == 0.0
        assert comparisons["fresh_trace_a_vs_fresh_trace_b"]["feature"][
            "max_abs"
        ] == 0.0
        assert comparisons["eager_a_vs_fresh_trace_a"]["feature"]["max_abs"] == 0.0


def test_existing_runtime_difference_top1_and_margin_are_observable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, payload, _, _, _, _ = _run_worker(
        tmp_path, monkeypatch, runtime_model=OffsetRuntime()
    )
    comparisons = payload["suite"]["mode_result"]["batches"]["8"]["comparisons"]
    assert comparisons["eager_a_vs_existing_runtime_a"]["feature"]["max_abs"] > 0
    assert comparisons["fresh_trace_a_vs_existing_runtime_a"]["logits"]["max_abs"] > 0
    assert comparisons["eager_a_vs_fresh_trace_a"]["feature"]["max_abs"] == 0
    logits = comparisons["eager_a_vs_existing_runtime_a"]["logits"]
    assert "top1_disagreement_rate" in logits
    assert "reference_margin_max_abs_change" in logits


def test_worker_has_no_authority_or_pass_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary, payload, _, _, _, _ = _run_worker(tmp_path, monkeypatch)
    assert payload["status"] == diagnostic.WORKER_STATUS
    assert payload["formal_authority"] is False
    assert payload["resources"]["process_peak_rss"]["status"] in {
        "COMPLETE",
        "INCOMPLETE",
    }
    assert payload["parity_receipt_emitted"] is False
    assert payload["target_access"] is False
    assert payload["source_cache_access"] is False
    assert payload["runtime_selection_performed"] is False
    assert summary["parity_receipt_emitted"] is False
    strings = _all_strings(payload)
    assert "PASS" not in strings
    assert "cvs.phase1.runtime_checkpoint_parity_receipt.v1" not in strings
    assert "cvs.adv3b02_effective8_torchscript_parity.v1" not in strings


def test_arm_contract_rejects_third_arm_and_wrong_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    with pytest.raises(diagnostic.ADV3B02NumericalDiagnosticError, match="arm_id"):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="third-runtime",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=tmp_path / "third.json",
            device="cpu",
            worker_mode="baseline",
            worker_scope="unit_test_only_cpu_primary",
            _allow_cpu_primary_for_tests=True,
        )
    monkeypatch.setitem(diagnostic.RUNTIME_ARMS["b202"], "sha256", "0" * 64)
    with pytest.raises(
        diagnostic.ADV3B02NumericalDiagnosticError, match="preregistered arm"
    ):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=tmp_path / "wrong.json",
            device="cpu",
            worker_mode="baseline",
            worker_scope="unit_test_only_cpu_primary",
            _allow_cpu_primary_for_tests=True,
        )


def test_requested_cuda_never_falls_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(
        diagnostic.ADV3B02NumericalDiagnosticError,
        match="CPU fallback is forbidden",
    ):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=tmp_path / "must_not_exist.json",
            device="cuda:0",
            worker_mode="baseline",
            worker_scope="required_primary_cuda",
        )
    assert not (tmp_path / "must_not_exist.json").exists()


def test_deterministic_worker_requires_cublas_before_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(
        diagnostic.ADV3B02NumericalDiagnosticError, match="CUBLAS_WORKSPACE_CONFIG"
    ):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=tmp_path / "deterministic.json",
            device="cpu",
            worker_mode="deterministic",
            worker_scope="unit_test_only_cpu_primary",
            _allow_cpu_primary_for_tests=True,
        )


def test_flags_restore_after_deterministic_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = diagnostic._flag_snapshot()

    def fail_compare(*args, **kwargs):
        raise RuntimeError("synthetic compare failure")

    monkeypatch.setattr(diagnostic, "_run_device_suite", fail_compare)
    with pytest.raises(RuntimeError, match="synthetic compare"):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=tmp_path / "failed.json",
            device="cpu",
            worker_mode="deterministic",
            worker_scope="unit_test_only_cpu_primary",
            _allow_cpu_primary_for_tests=True,
        )
    assert diagnostic._flag_snapshot() == before
    assert not (tmp_path / "failed.json").exists()


def test_serialization_failure_never_creates_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    output = tmp_path / "serializer_attack.json"
    original = json.dumps

    def partial_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("partial serializer attack")

    monkeypatch.setattr(diagnostic.json, "dumps", partial_then_fail)
    with pytest.raises(RuntimeError, match="partial serializer attack"):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=output,
            device="cpu",
            worker_mode="baseline",
            worker_scope="unit_test_only_cpu_primary",
            _allow_cpu_primary_for_tests=True,
        )
    assert not output.exists()


def test_worker_and_final_artifacts_are_never_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    artifact = tmp_path / "existing.json"
    artifact.write_text("sentinel", encoding="utf-8")
    with pytest.raises(diagnostic.ADV3B02NumericalDiagnosticError, match="overwrite"):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=artifact,
            device="cpu",
            worker_mode="baseline",
            worker_scope="unit_test_only_cpu_primary",
            _allow_cpu_primary_for_tests=True,
        )
    assert artifact.read_text(encoding="utf-8") == "sentinel"


def test_launcher_sets_cublas_only_for_separate_deterministic_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []

    def fake_run(command, **kwargs):
        calls.append({"command": list(command), "env": dict(kwargs["env"])})
        output = Path(command[command.index("--worker-artifact-out") + 1])
        output.write_text("{}", encoding="utf-8")
        summary = {
            "status": diagnostic.WORKER_STATUS,
            "artifact_path": str(output),
            "artifact_sha256": _file_sha256(output),
            "artifact_bytes": output.stat().st_size,
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(summary), stderr=""
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    common = {
        "checkpoint": tmp_path / "checkpoint",
        "runtime": tmp_path / "runtime",
        "lineage_evidence": tmp_path / "lineage",
        "artifact_origin_receipt": None,
        "fresh_runtime": tmp_path / "fresh",
        "fresh_runtime_sha256": "1" * 64,
        "trace_builder_artifact_sha256": "2" * 64,
        "arm_id": "b202",
        "source_git_commit": SOURCE_COMMIT,
        "source_archive_sha256": "3" * 64,
        "source_release_receipt_sha256": "4" * 64,
        "source_execution_contract": tmp_path / "source-contract.json",
        "source_execution_contract_sha256": "5" * 64,
        "device": "cuda:0",
        "scope": "required_primary_cuda",
    }
    baseline_payload = diagnostic._launch_worker(
        **common, worker_output=tmp_path / "baseline.json", mode="baseline"
    )
    deterministic_payload = diagnostic._launch_worker(
        **common,
        worker_output=tmp_path / "deterministic.json",
        mode="deterministic",
    )
    assert len(calls) == 2
    assert "CUBLAS_WORKSPACE_CONFIG" not in calls[0]["env"] or calls[0]["env"].get(
        "CUBLAS_WORKSPACE_CONFIG"
    ) != ":4096:8"
    assert calls[1]["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert calls[0]["command"] != calls[1]["command"]
    for payload in (baseline_payload, deterministic_payload):
        audit = payload["orchestrator_launch_audit"]
        assert audit["returncode"] == 0
        assert "stdout" in audit and "stderr" in audit
        assert audit["environment_applied_before_worker_python_start"] is True


def test_orchestrator_merges_two_workers_without_running_modes_inline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, baseline, _, checkpoint, runtime, lineage = _run_worker(
        tmp_path, monkeypatch, artifact_name="template.json"
    )
    deterministic = copy.deepcopy(baseline)
    deterministic["worker"]["mode"] = "deterministic"
    deterministic["suite"]["mode_result"]["mode"] = "deterministic"
    deterministic["worker"]["startup_environment"][
        "CUBLAS_WORKSPACE_CONFIG"
    ] = ":4096:8"
    deterministic["suite"]["mode_result"]["flags"][
        "cublas_workspace_config"
    ] = ":4096:8"
    archive, source_receipt = _source_release(tmp_path, monkeypatch)
    source_receipt_sha = _file_sha256(source_receipt)
    release = diagnostic._validate_source_release(
        source_archive_path=archive,
        source_release_receipt_path=source_receipt,
        _unit_test_fixture=True,
    )
    execution_contract = diagnostic._source_execution_contract(release)
    execution_contract_sha = diagnostic._sha256_bytes(
        diagnostic._canonical_json_bytes(execution_contract)
    )
    source_manifest_root = release["source_manifest_root_sha256"]
    fresh_sha = "5" * 64
    trace_sha = "6" * 64
    for value, mode in ((baseline, "baseline"), (deterministic, "deterministic")):
        value["worker"]["scope"] = "required_primary_cuda"
        value["suite"]["device"]["resolved_device"] = "cuda:0"
        value["source_release_binding"] = {
            "source_git_commit": SOURCE_COMMIT,
            "source_archive_sha256": _file_sha256(archive),
            "source_release_receipt_sha256": source_receipt_sha,
            "source_execution_contract_sha256": execution_contract_sha,
            "source_manifest_root_sha256": source_manifest_root,
        }
        value["software"]["source_execution_binding"] = {
            "status": "SIGNED_MEMBER_MANIFEST_EXECUTION_CLOSED",
            "contract_sha256": execution_contract_sha,
            "source_manifest_root_sha256": source_manifest_root,
        }
        value["fresh_runtime_binding"]["sha256"] = fresh_sha
        value["fresh_runtime_binding"]["trace_builder_artifact_sha256"] = trace_sha
        value["suite"]["mode_result"]["fresh_trace"]["sha256"] = fresh_sha
        if mode == "baseline":
            value["worker"]["startup_environment"][
                "CUBLAS_WORKSPACE_CONFIG"
            ] = None
            value["suite"]["mode_result"]["flags"][
                "cublas_workspace_config"
            ] = None
    launched: list[str] = []

    def fake_launch_worker(**kwargs):
        launched.append(kwargs["mode"])
        return _attach_launch_audit(
            baseline if kwargs["mode"] == "baseline" else deterministic,
            mode=kwargs["mode"],
        )

    monkeypatch.setattr(diagnostic, "_launch_worker", fake_launch_worker)
    monkeypatch.setattr(
        diagnostic,
        "_launch_trace_builder",
        lambda **kwargs: _trace_builder_from_worker(
            baseline, fresh_sha=fresh_sha, trace_sha=trace_sha
        ),
    )
    output = tmp_path / "merged.json"
    summary = diagnostic.diagnose_runtime_numerics(
        checkpoint_path=checkpoint,
        runtime_path=runtime,
        lineage_evidence_path=lineage,
        artifact_origin_receipt_path=None,
        arm_id="b202",
        source_archive_path=archive,
        source_release_receipt_path=source_receipt,
        artifact_out=output,
        device="cuda:0",
        _allow_parent_torch_for_tests=True,
        _allow_unit_source_fixture=True,
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert launched == ["baseline", "deterministic"]
    assert set(payload["workers"]) == {
        "primary_cuda_baseline",
        "primary_cuda_deterministic",
    }
    assert payload["cross_worker_closure"]["fixed_probe_spec.root_sha256"] == (
        diagnostic.EXPECTED_PROBE_ROOT_SHA256
    )
    assert payload["cpu_control_can_substitute_cuda_or_authorize"] is False
    assert summary["formal_authority"] is False


def test_final_serialization_failure_never_creates_partial_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, template, _, checkpoint, runtime, lineage = _run_worker(
        tmp_path, monkeypatch, artifact_name="final_template.json"
    )

    def fake_launch_worker(**kwargs):
        return copy.deepcopy(template)

    monkeypatch.setattr(diagnostic, "_launch_worker", fake_launch_worker)
    monkeypatch.setattr(
        diagnostic,
        "_launch_trace_builder",
        lambda **kwargs: {
            "orchestrator_launch_audit": {
                "trace_builder_artifact_sha256": "6" * 64,
                "fresh_runtime_sha256": "5" * 64,
            },
            "software": {
                "nvidia_driver": {"status": "COMPLETE", "version": "unit"}
            },
        },
    )
    monkeypatch.setattr(diagnostic, "_validate_workers", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        diagnostic, "_validate_trace_builder", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        diagnostic,
        "_validate_source_release",
        lambda **kwargs: {
            "source_git_commit": SOURCE_COMMIT,
            "source_archive_sha256": _file_sha256(archive),
            "receipt_sha256": "9" * 64,
            "source_archive_path": str((tmp_path / "archive.zip").resolve()),
            "source_members": [
                {"path": "code/unit.py", "bytes": 1, "sha256": "8" * 64}
            ],
            "source_manifest_root_sha256": diagnostic._source_manifest_root(
                [{"path": "code/unit.py", "bytes": 1, "sha256": "8" * 64}]
            ),
            "git_policy": {"mode": "signed_manifest_only_no_git"},
        },
    )
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"archive")
    original = json.dumps

    def partial_then_fail(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("final partial serializer attack")

    monkeypatch.setattr(diagnostic.json, "dumps", partial_then_fail)
    output = tmp_path / "must_not_exist_final.json"
    with pytest.raises(RuntimeError, match="final partial serializer"):
        diagnostic.diagnose_runtime_numerics(
            checkpoint_path=checkpoint,
            runtime_path=runtime,
            lineage_evidence_path=lineage,
            artifact_origin_receipt_path=None,
            arm_id="b202",
            source_archive_path=archive,
            source_release_receipt_path=tmp_path / "ignored.json",
            artifact_out=output,
            device="cuda:0",
            _allow_parent_torch_for_tests=True,
        )
    assert not output.exists()


def test_parent_module_imports_no_torch_or_project_dependency() -> None:
    code = (
        "import sys; sys.path.insert(0, 'code'); "
        "import scripts.diagnose_adv3b02_runtime_numerics; "
        "assert 'torch' not in sys.modules; "
        "assert not any(name == 'cvsrffi' or name.startswith('cvsrffi.') "
        "for name in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(diagnostic.REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr


def test_same_runtime_bytes_at_unregistered_path_do_not_inherit_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, fresh_runtime, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    copied = tmp_path / "copied_runtime.ts"
    copied.write_bytes(runtime.read_bytes())
    with pytest.raises(
        diagnostic.ADV3B02NumericalDiagnosticError,
        match="origin receipt",
    ):
        diagnostic._worker_diagnostic(
            checkpoint_path=checkpoint,
            runtime_path=copied,
            **_worker_extra(fresh_runtime, lineage),
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            worker_artifact_out=tmp_path / "copy.json",
            device="cpu",
            worker_mode="baseline",
            worker_scope="unit_test_only_cpu_primary",
            _allow_cpu_primary_for_tests=True,
        )


def test_trace_builder_is_independent_and_writes_one_fresh_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint, runtime, _, lineage = _assets(tmp_path)
    _patch_assets(monkeypatch, checkpoint, runtime, lineage)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    fresh_output = tmp_path / "builder_fresh.ts"
    artifact_output = tmp_path / "builder.json"
    summary = diagnostic._trace_builder_diagnostic(
        checkpoint_path=checkpoint,
        fresh_runtime_out=fresh_output,
        trace_builder_artifact_out=artifact_output,
        device="cpu",
        source_git_commit=SOURCE_COMMIT,
        source_archive_sha256="c" * 64,
        source_release_receipt_sha256="d" * 64,
        _allow_cpu_for_tests=True,
    )
    payload = json.loads(artifact_output.read_text(encoding="utf-8"))
    assert summary["status"] == diagnostic.TRACE_BUILDER_STATUS
    assert _file_sha256(fresh_output) == summary["fresh_runtime_sha256"]
    assert payload["fresh_runtime"]["storage_scope"] == (
        "immutable_unique_trace_builder_output"
    )
    assert payload["formal_authority"] is False


def test_validate_workers_rejects_missing_launch_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, worker, _, checkpoint, runtime, _ = _run_worker(tmp_path, monkeypatch)
    worker["worker"]["scope"] = "required_primary_cuda"
    worker["suite"]["device"]["resolved_device"] = "cuda:0"
    with pytest.raises(
        diagnostic.ADV3B02NumericalDiagnosticError,
        match="authority/identity closure",
    ):
        diagnostic._validate_workers(
            {"primary_cuda_baseline": worker},
            contract=diagnostic._arm_contract("b202"),
            checkpoint_sha=_file_sha256(checkpoint),
            runtime_sha=_file_sha256(runtime),
            runtime_path=runtime,
            artifact_origin_receipt_path=None,
            normalized_device="cuda:0",
            source_git_commit=SOURCE_COMMIT,
            source_archive_sha256="c" * 64,
            source_release_receipt_sha256="d" * 64,
            source_execution_contract_sha256="e" * 64,
            source_manifest_root_sha256="f" * 64,
            fresh_runtime_sha256=worker["fresh_runtime_binding"]["sha256"],
            trace_builder_artifact_sha256="b" * 64,
        )


def test_per_channel_quantization_registry_keeps_scale_zero_point_and_axis() -> None:
    tensor = torch.quantize_per_channel(
        torch.tensor([[1.0, -1.0], [2.0, -2.0]]),
        scales=torch.tensor([0.1, 0.2], dtype=torch.float64),
        zero_points=torch.tensor([0, 1], dtype=torch.int64),
        axis=0,
        dtype=torch.qint8,
    )
    record = diagnostic._tensor_record("per_channel", tensor)
    quantization = record["quantization"]
    assert quantization["axis"] == 0
    assert quantization["scale_count"] == 2
    assert quantization["zero_point_count"] == 2
    assert len(quantization["scale_sha256"]) == 64
    assert len(quantization["zero_point_sha256"]) == 64


def test_nvidia_smi_failure_is_explicitly_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Device:
        type = "cuda"
        index = 0

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("nvidia-smi", 10)

    monkeypatch.setattr(diagnostic.subprocess, "run", timeout)
    audit = diagnostic._driver_version(Device())
    assert audit["status"] == "INCOMPLETE"
    assert audit["version"] is None


def test_source_release_receipt_requires_external_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, receipt = _source_release(tmp_path, monkeypatch)

    def reject(*args):
        raise ValueError("signature rejected")

    monkeypatch.setattr(diagnostic, "_verify_source_release_signature", reject)
    with pytest.raises(ValueError, match="signature rejected"):
        diagnostic._validate_source_release(
            source_archive_path=archive,
            source_release_receipt_path=receipt,
        )


def test_unit_source_signature_fixture_is_explicitly_not_authorized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, receipt = _source_release(tmp_path, monkeypatch)
    audit = diagnostic._validate_source_release(
        source_archive_path=archive,
        source_release_receipt_path=receipt,
        _unit_test_fixture=True,
    )
    assert audit["acceptance"] == "UNIT_TEST_SIGNATURE_FIXTURE_NOT_AUTHORIZED"
    assert audit["signature_verified"] is True


def test_nvidia_smi_incomplete_is_aggregated_across_trace_and_workers() -> None:
    trace = {"software": {"nvidia_driver": {"status": "COMPLETE"}}}
    workers = {
        "primary_cuda_baseline": {
            "software": {"nvidia_driver": {"status": "INCOMPLETE"}}
        },
        "primary_cuda_deterministic": {
            "software": {"nvidia_driver": {"status": "COMPLETE"}}
        },
    }
    summary = diagnostic._nvidia_completeness_summary(trace, workers)
    assert summary["diagnostic_completeness"] == (
        "INCOMPLETE_NVIDIA_DRIVER_AUDIT"
    )
    assert summary["incomplete_nvidia_smi_components"] == [
        "primary_cuda_baseline"
    ]


def test_worker_stdout_summary_must_bind_artifact_sha_and_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(command, **kwargs):
        output = Path(command[command.index("--worker-artifact-out") + 1])
        output.write_text("{}", encoding="utf-8")
        summary = {
            "status": diagnostic.WORKER_STATUS,
            "artifact_path": str(output),
            "artifact_sha256": "0" * 64,
            "artifact_bytes": output.stat().st_size,
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(summary), stderr=""
        )

    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    with pytest.raises(
        diagnostic.ADV3B02NumericalDiagnosticError,
        match="stdout/artifact closure",
    ):
        diagnostic._launch_worker(
            checkpoint=tmp_path / "checkpoint",
            runtime=tmp_path / "runtime",
            lineage_evidence=tmp_path / "lineage",
            artifact_origin_receipt=None,
            fresh_runtime=tmp_path / "fresh",
            fresh_runtime_sha256="1" * 64,
            trace_builder_artifact_sha256="2" * 64,
            arm_id="b202",
            source_git_commit=SOURCE_COMMIT,
            source_archive_sha256="3" * 64,
            source_release_receipt_sha256="4" * 64,
            source_execution_contract=tmp_path / "source-contract.json",
            source_execution_contract_sha256="5" * 64,
            worker_output=tmp_path / "worker.json",
            device="cuda:0",
            mode="baseline",
            scope="required_primary_cuda",
        )


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("existing_runtime_path",), "wrong-runtime"),
        (("canonical_origin_path",), "wrong-origin"),
        (("runtime_sha256",), "0" * 64),
        (("lineage_scope",), "wrong-scope"),
        (("scope",), "preregistered_read_only_copy"),
        (("receipt_required",), True),
        (("receipt_path",), "forged-receipt"),
        (("receipt_sha256",), "9" * 64),
        (("receipt_content",), {"forged": True}),
    ],
)
def test_validate_workers_recomputes_every_runtime_origin_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_path: tuple[str, ...],
    replacement: Any,
) -> None:
    _, worker, _, checkpoint, runtime, _ = _run_worker(tmp_path, monkeypatch)
    worker["worker"]["scope"] = "required_primary_cuda"
    worker["suite"]["device"]["resolved_device"] = "cuda:0"
    contract_sha = "e" * 64
    manifest_root = "f" * 64
    worker["source_release_binding"].update(
        {
            "source_execution_contract_sha256": contract_sha,
            "source_manifest_root_sha256": manifest_root,
        }
    )
    worker["software"]["source_execution_binding"] = {
        "status": "SIGNED_MEMBER_MANIFEST_EXECUTION_CLOSED",
        "contract_sha256": contract_sha,
        "source_manifest_root_sha256": manifest_root,
    }
    worker = _attach_launch_audit(worker, mode="baseline")
    valid_kwargs = {
        "contract": diagnostic._arm_contract("b202"),
        "checkpoint_sha": _file_sha256(checkpoint),
        "runtime_sha": _file_sha256(runtime),
        "runtime_path": runtime,
        "artifact_origin_receipt_path": None,
        "normalized_device": "cuda:0",
        "source_git_commit": SOURCE_COMMIT,
        "source_archive_sha256": "c" * 64,
        "source_release_receipt_sha256": "d" * 64,
        "source_execution_contract_sha256": contract_sha,
        "source_manifest_root_sha256": manifest_root,
        "fresh_runtime_sha256": worker["fresh_runtime_binding"]["sha256"],
        "trace_builder_artifact_sha256": "b" * 64,
    }
    diagnostic._validate_workers({"primary_cuda_baseline": worker}, **valid_kwargs)
    attacked = copy.deepcopy(worker)
    target = attacked["asset_lineage"]["runtime_origin"]
    target[field_path[0]] = replacement
    with pytest.raises(
        diagnostic.ADV3B02NumericalDiagnosticError,
        match="authority/identity closure",
    ):
        diagnostic._validate_workers(
            {"primary_cuda_baseline": attacked}, **valid_kwargs
        )


@pytest.mark.parametrize("attack", ["escape", "missing", "extra"])
def test_signed_source_archive_member_manifest_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, attack: str
) -> None:
    archive, receipt = _source_release(tmp_path, monkeypatch)
    body = json.loads(receipt.read_text(encoding="utf-8"))
    if attack == "escape":
        body["source_members"][0]["path"] = "../escape.py"
    elif attack == "missing":
        body["source_members"] = []
    else:
        body["source_members"].append(
            {"path": "code/extra.py", "bytes": 1, "sha256": "a" * 64}
        )
    body["source_members"] = sorted(
        body["source_members"], key=lambda row: row["path"]
    )
    body["source_manifest_root_sha256"] = diagnostic._source_manifest_root(
        body["source_members"]
    )
    receipt.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    with pytest.raises(diagnostic.ADV3B02NumericalDiagnosticError):
        diagnostic._validate_source_release(
            source_archive_path=archive,
            source_release_receipt_path=receipt,
            _unit_test_fixture=True,
        )


@pytest.mark.parametrize("attack", ["extra_import", "missing_import", "git_drift"])
def test_actual_execution_source_binding_rejects_manifest_or_git_drift(
    attack: str,
) -> None:
    signed = [{"path": "code/a.py", "bytes": 1, "sha256": "a" * 64}]
    observed = copy.deepcopy(signed)
    if attack == "extra_import":
        observed.append({"path": "code/b.py", "bytes": 1, "sha256": "b" * 64})
    elif attack == "missing_import":
        observed = []
    git = {
        "git_available": True,
        "commit": SOURCE_COMMIT,
        "dirty": False,
        "status_root_sha256": "1" * 64,
        "diff_root_sha256": "2" * 64,
        "cached_diff_root_sha256": "3" * 64,
        "untracked_root_sha256": "4" * 64,
    }
    policy = {
        "mode": "git_exact",
        "commit": SOURCE_COMMIT,
        "dirty": False,
        "status_root_sha256": "1" * 64,
        "diff_root_sha256": "2" * 64,
        "cached_diff_root_sha256": "3" * 64,
        "untracked_root_sha256": "4" * 64,
    }
    if attack == "git_drift":
        git["dirty"] = True
    contract = {
        "source_git_commit": SOURCE_COMMIT,
        "source_members": signed,
        "source_manifest_root_sha256": diagnostic._source_manifest_root(signed),
        "git_policy": policy,
        "contract_path": "unit-contract",
        "contract_sha256": "c" * 64,
    }
    dependencies = {"loaded_project_modules": observed}
    with pytest.raises(diagnostic.ADV3B02NumericalDiagnosticError):
        diagnostic._validate_execution_source_binding(
            dependencies=dependencies,
            git=git,
            source_git_commit=SOURCE_COMMIT,
            contract=contract,
        )


def test_cli_without_external_signed_source_receipt_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "diagnose",
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--runtime",
            str(tmp_path / "runtime"),
            "--lineage-evidence",
            str(tmp_path / "lineage"),
            "--arm-id",
            "b202",
            "--artifact-out",
            str(output),
            "--device",
            "cuda:0",
        ],
    )
    assert diagnostic.main() == 2
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == diagnostic.BLOCKED_SOURCE_STATUS
    assert summary["artifact_emitted"] is False
    assert not output.exists()


def test_quantized_tensor_bytes_include_scale_and_zero_point_storage() -> None:
    tensor = torch.quantize_per_channel(
        torch.tensor([[1.0, -1.0], [2.0, -2.0]]),
        scales=torch.tensor([0.1, 0.2], dtype=torch.float64),
        zero_points=torch.tensor([0, 1], dtype=torch.int64),
        axis=0,
        dtype=torch.qint8,
    )
    record = diagnostic._tensor_record("per_channel", tensor)
    assert record["tensor_bytes"] == (
        record["data_bytes"]
        + record["quantization"]["scale_bytes"]
        + record["quantization"]["zero_point_bytes"]
    )
    registry = diagnostic._tensor_registry({"per_channel": tensor})
    assert registry["tensor_bytes"] == record["tensor_bytes"]
    assert registry["quantization_parameter_bytes"] > 0


def test_worker_launcher_reads_artifact_snapshot_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = diagnostic._read_regular_bytes
    reads: dict[Path, int] = {}

    def counted(path: Path, name: str) -> bytes:
        resolved = Path(path).resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        return original_read(path, name)

    def fake_run(command, **kwargs):
        output = Path(command[command.index("--worker-artifact-out") + 1])
        artifact_bytes = diagnostic._canonical_json_bytes({})
        output.write_bytes(artifact_bytes)
        summary = {
            "status": diagnostic.WORKER_STATUS,
            "artifact_path": str(output),
            "artifact_sha256": diagnostic._sha256_bytes(artifact_bytes),
            "artifact_bytes": len(artifact_bytes),
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(summary), stderr=""
        )

    monkeypatch.setattr(diagnostic, "_read_regular_bytes", counted)
    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    output = tmp_path / "worker-once.json"
    diagnostic._launch_worker(
        checkpoint=tmp_path / "checkpoint",
        runtime=tmp_path / "runtime",
        lineage_evidence=tmp_path / "lineage",
        artifact_origin_receipt=None,
        fresh_runtime=tmp_path / "fresh",
        fresh_runtime_sha256="1" * 64,
        trace_builder_artifact_sha256="2" * 64,
        arm_id="b202",
        source_git_commit=SOURCE_COMMIT,
        source_archive_sha256="3" * 64,
        source_release_receipt_sha256="4" * 64,
        source_execution_contract=tmp_path / "contract",
        source_execution_contract_sha256="5" * 64,
        worker_output=output,
        device="cuda:0",
        mode="baseline",
        scope="required_primary_cuda",
    )
    assert reads == {output.resolve(): 1}


def test_trace_launcher_reads_each_output_snapshot_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_read = diagnostic._read_regular_bytes
    reads: dict[Path, int] = {}

    def counted(path: Path, name: str) -> bytes:
        resolved = Path(path).resolve()
        reads[resolved] = reads.get(resolved, 0) + 1
        return original_read(path, name)

    def fake_run(command, **kwargs):
        fresh = Path(command[command.index("--fresh-runtime-out") + 1])
        artifact = Path(command[command.index("--trace-builder-artifact-out") + 1])
        fresh_bytes = b"fresh-runtime"
        artifact_bytes = diagnostic._canonical_json_bytes(
            {
                "schema": diagnostic.TRACE_BUILDER_SCHEMA,
                "status": diagnostic.TRACE_BUILDER_STATUS,
            }
        )
        fresh.write_bytes(fresh_bytes)
        artifact.write_bytes(artifact_bytes)
        summary = {
            "status": diagnostic.TRACE_BUILDER_STATUS,
            "artifact_path": str(artifact),
            "artifact_sha256": diagnostic._sha256_bytes(artifact_bytes),
            "artifact_bytes": len(artifact_bytes),
            "fresh_runtime_path": str(fresh),
            "fresh_runtime_sha256": diagnostic._sha256_bytes(fresh_bytes),
            "fresh_runtime_bytes": len(fresh_bytes),
        }
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps(summary), stderr=""
        )

    monkeypatch.setattr(diagnostic, "_read_regular_bytes", counted)
    monkeypatch.setattr(diagnostic.subprocess, "run", fake_run)
    fresh = tmp_path / "fresh-once.ts"
    artifact = tmp_path / "trace-once.json"
    diagnostic._launch_trace_builder(
        checkpoint=tmp_path / "checkpoint",
        fresh_runtime_output=fresh,
        trace_builder_output=artifact,
        device="cuda:0",
        source_git_commit=SOURCE_COMMIT,
        source_archive_sha256="3" * 64,
        source_release_receipt_sha256="4" * 64,
        source_execution_contract=tmp_path / "contract",
        source_execution_contract_sha256="5" * 64,
    )
    assert reads == {artifact.resolve(): 1, fresh.resolve(): 1}
