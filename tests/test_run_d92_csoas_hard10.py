from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from scripts import run_d92_csoas_hard10 as runner  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _row(k_shot: int = 10, *, fallback: bool = False, codec_retry: int = 0, class_count: int = 11) -> dict[str, object]:
    active = k_shot > 2 and not fallback
    prefix = "d92_csoas_"
    row: dict[str, object] = {
        "scenario": runner.SCENES[0],
        "arm_id": runner.ARM_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "after_registered_d_mode_effective": "csoas_full" if k_shot > 2 else "d92_full_alias",
        "after_total_component_fit_count": 2 if k_shot > 2 else 3,
        "after_actual_component_inventory": {
            "actual_component_fit_count": 1 if k_shot > 2 else 3,
            "full_component_fit_count": 1 if k_shot > 2 else 3,
            "block3_component_fit_count": 0,
        },
        prefix + "active": active,
        prefix + "fallback_active": fallback,
        prefix + "fallback_reason": "NUMERIC_FALLBACK_EXACT_E0" if fallback else (None if k_shot > 2 else "K1_K2_EXACT_D92_FULL_ALIAS"),
        prefix + "candidate_attempt_fit_count": 1 if k_shot > 2 else 0,
        prefix + "fallback_reference_fit_count": 1 if fallback else 0,
        prefix + "candidate_statistic_receipt_available": active,
        prefix + "fallback_reference_full_head_byte_exact": True if fallback else None,
        prefix + "paired_e0_codec_state_equal": None,
        prefix + "g0_eligible": False,
        prefix + "g0_block_reason": "PENDING_DEPLOYED_CODEC_PAIRED_E0" if k_shot > 2 and not fallback else ("NUMERIC_FALLBACK_EXACT_E0" if fallback else "K1_K2_EXACT_D92_FULL_ALIAS"),
        prefix + "codec_retry_count": codec_retry,
        prefix + "query_rows_used": 0,
        "query_macs": class_count * 288,
        "after_state_bytes": 8583,
        "registered_class_count": class_count,
    }
    row["d92_e0d_csoas_g0_eligible"] = row.pop(prefix + "g0_eligible")
    row["d92_e0d_csoas_g0_block_reason"] = row.pop(prefix + "g0_block_reason")
    for field in runner.QUERY_ZERO_FIELDS:
        row[field] = False
    for field in runner.CSOAS_QUERY_ZERO_FIELDS:
        row[field] = False
    return row


@pytest.mark.parametrize("k_shot", [10, 1])
def test_fit_audit_accepts_real_csoas_active_and_exact_k1_alias(tmp_path: Path, k_shot: int) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(k_shot), "scenario": scene} for scene in runner.SCENES])
    runner._validate_fit_audit(path, k_shot=k_shot)


def test_fit_audit_rejects_numeric_fallback_for_formal_k_gt_2(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(10, fallback=True), "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="fallback"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_codec_retry_for_formal_k_gt_2(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, [{**_row(10, codec_retry=1), "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="retry"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_non_full1_two_state_inventory(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    rows = []
    for scene in runner.SCENES:
        row = _row(10)
        row["scenario"] = scene
        row["after_actual_component_inventory"] = {
            **row["after_actual_component_inventory"],
            "full_component_fit_count": 2,
        }
        rows.append(row)
    _write(path, rows)
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="FULL1"):
        runner._validate_fit_audit(path, k_shot=10)


def test_fit_audit_rejects_any_query_access(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    row = _row(10)
    row[runner.QUERY_ZERO_FIELDS[0]] = True
    _write(path, [{**row, "scenario": scene} for scene in runner.SCENES])
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="query access"):
        runner._validate_fit_audit(path, k_shot=10)


@pytest.mark.parametrize("class_count", [16, 26])
def test_fit_audit_accepts_hard9_new_class_counts(tmp_path: Path, class_count: int) -> None:
    path = tmp_path / f"fit_audit_{class_count}.json"
    _write(path, [{**_row(10, class_count=class_count), "scenario": scene} for scene in runner.SCENES])
    runner._validate_fit_audit(path, k_shot=10)


def test_runner_parser_exposes_prepare_smoke_and_eight_shard_commands() -> None:
    parser = runner.parser()
    assert set(parser._subparsers._group_actions[0].choices) >= {"prepare", "truth-free-smoke", "run-shard"}
    assert runner.SHARD_COUNT == 8
    assert runner._is_full_matrix({"job_count": 10, "jobs": [{}] * 10})


def test_shard_rewrite_only_touches_owned_job_roots_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    owned_job = output / "jobs" / "owned"
    foreign_job = output / "jobs" / "foreign"
    owned_receipt = owned_job / "job_receipt.json"
    foreign_receipt = foreign_job / "job_receipt.json"
    owned_summary = output / "summaries" / "shard_0.json"
    foreign_summary = output / "summaries" / "shard_1.json"
    payload = {
        "schema": "cvs.phase2.d92_pareto_distill_hard11.job_receipt.v1",
        "status": "PARETO_DISTILL_HARD11_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
    }
    for path in (owned_receipt, foreign_receipt):
        _write(path, payload)
    for path in (owned_summary, foreign_summary):
        _write(path, {**payload, "schema": "cvs.phase2.d92_pareto_distill_hard11.shard_summary.v1"})
    manifest = {
        "output_root": str(output),
        "jobs": [
            {"planned_shard_index": 0, "output_root": str(owned_job)},
            {"planned_shard_index": 1, "output_root": str(foreign_job)},
        ],
    }
    foreign_receipt_before = foreign_receipt.read_bytes()
    foreign_summary_before = foreign_summary.read_bytes()
    foreign_root_receipt = output / "jobs" / "foreign_root.json"
    shared_stop = output / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json"
    shared_ledger = output / "systemic_pre_prediction_failures" / "fp" / "outer" / "job.json"
    _write(foreign_root_receipt, payload)
    _write(shared_stop, {**payload, "schema": "cvs.phase2.d92_pareto_distill_hard11.systemic_stop.v1"})
    _write(shared_ledger, {**payload, "schema": "cvs.phase2.d92_pareto_distill_hard11.pre_prediction_failure.v1"})
    foreign_root_before = foreign_root_receipt.read_bytes()

    runner._rewrite_shard_output(manifest, shard_index=0)
    runner._rewrite_shared_failure_evidence(output)

    assert "csoas_hard10" in owned_receipt.read_text(encoding="utf-8")
    assert "csoas_hard10" in owned_summary.read_text(encoding="utf-8")
    assert foreign_receipt.read_bytes() == foreign_receipt_before
    assert foreign_summary.read_bytes() == foreign_summary_before
    assert foreign_root_receipt.read_bytes() == foreign_root_before
    assert "csoas_hard10" in shared_stop.read_text(encoding="utf-8")
    assert "csoas_hard10" in shared_ledger.read_text(encoding="utf-8")


def _shared_smoke_case(tmp_path: Path) -> tuple[dict[str, object], dict[str, object], dict[str, Path], str]:
    output = tmp_path / "matrix"
    smoke = output / "smoke"
    prediction_root = smoke / "diag"
    paths = {
        "before_prediction": prediction_root / "before" / "prediction_artifact.npz",
        "after_prediction": prediction_root / "after" / "prediction_artifact.npz",
        "before_commit": prediction_root / "before" / "COMMIT.json",
        "after_commit": prediction_root / "after" / "COMMIT.json",
        "before_fit_audit": prediction_root / "before" / "fit_audit.json",
        "after_fit_audit": prediction_root / "after" / "fit_audit.json",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(path.name.encode("utf-8"))
    job: dict[str, object] = {
        "outer_key": runner.SMOKE_OUTER_KEY,
        "outer_role": "performance",
        "k_shot": 5,
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
        "job_id": "smoke-job",
        "truth_sidecar_sha256": "b" * 64,
    }
    manifest: dict[str, object] = {
        "job_count": 10,
        "jobs": [job] + [{"job_id": f"job-{i}"} for i in range(9)],
        "output_root": str(output),
        "smoke_outer_key": runner.SMOKE_OUTER_KEY,
        "ground_component_dir": str(tmp_path / "ground"),
        "ground_manifest_sha256": "e" * 64,
    }
    expected_command = ["python", "predict", "--output-root", str(prediction_root)]
    hashes = {f"{field}_sha256": runner._base_runner._sha256_file(path) for field, path in paths.items()}
    receipt: dict[str, object] = {
        "schema": "cvs.phase2.d92_csoas_hard10.smoke_receipt.v1",
        "status": "D92_CSOAS_HARD10_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS",
        "matrix_manifest_sha256": "a" * 64,
        "selection_sha256": runner.CANONICAL_SELECTION_SHA256,
        "smoke_outer_key": runner.SMOKE_OUTER_KEY,
        "job_id": job["job_id"],
        "outer_role": job["outer_role"],
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
        "k_shot": job["k_shot"],
        "truth_sidecar_sha256": job["truth_sidecar_sha256"],
        "command": expected_command,
        **hashes,
        "prediction_closure": hashes,
        "truth_open": False,
        "query_truth_joined_only_after_immutable_predictions": True,
        "prediction_and_scorer_processes_isolated": True,
        **{field: False for field in runner.QUERY_ZERO_FIELDS},
    }
    _write(smoke / "smoke_receipt.json", receipt)
    return manifest, receipt, paths, str(output)


def test_shared_smoke_rejects_each_receipt_binding_tamper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    for field, value in (
        ("truth_sidecar_sha256", "c" * 64),
        ("command", ["tampered"]),
        ("before_prediction_sha256", "d" * 64),
        ("after_prediction_sha256", "d" * 64),
        ("before_commit_sha256", "d" * 64),
        ("after_commit_sha256", "d" * 64),
        ("before_fit_audit_sha256", "d" * 64),
        ("after_fit_audit_sha256", "d" * 64),
        ("fit_audit_sha256", "d" * 64),
        ("prediction_closure", {"tampered": "closure"}),
    ):
        case = tmp_path / field
        manifest, receipt, _paths, output = _shared_smoke_case(case)
        receipt[field] = value
        _write(Path(output) / "smoke" / "smoke_receipt.json", receipt)
        monkeypatch.setattr(runner, "_verify_manifest_artifacts", lambda *_a: None)
        monkeypatch.setattr(runner, "_validate_fit_audit", lambda *_a, **_k: None)
        monkeypatch.setattr(runner._base_runner, "_prediction_command", lambda *_a, **_k: ["python", "predict", "--output-root", str(Path(output) / "smoke" / "diag")])
        monkeypatch.setattr(runner._base_runner, "_prediction_closure_status", lambda *_a: ("closed", "closed"))
        with pytest.raises(runner.D92CSOASHard10RunnerError):
            runner._validate_shared_smoke(manifest, manifest_sha256="a" * 64, device="cpu")

    manifest, _receipt, _paths, _output = _shared_smoke_case(tmp_path / "closure_status")
    monkeypatch.setattr(runner, "_verify_manifest_artifacts", lambda *_a: None)
    monkeypatch.setattr(runner, "_validate_fit_audit", lambda *_a, **_k: None)
    monkeypatch.setattr(runner._base_runner, "_prediction_command", lambda *_a, **_k: ["python", "predict", "--output-root", str(Path(_output) / "smoke" / "diag")])
    monkeypatch.setattr(runner._base_runner, "_prediction_closure_status", lambda *_a: ("technical_failure", "tampered"))
    with pytest.raises(runner.D92CSOASHard10RunnerError, match="closure"):
        runner._validate_shared_smoke(manifest, manifest_sha256="a" * 64, device="cpu")
