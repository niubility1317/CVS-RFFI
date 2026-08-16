from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import threading
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from scripts import run_d92_ccoc_hard9_k1 as runner  # noqa: E402


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_bytes_readonly(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    os.chmod(path, stat.S_IREAD)
    return hashlib.sha256(value).hexdigest()


_CLOSURE_SHA_FIELDS = (
    "before_prediction_sha256",
    "after_prediction_sha256",
    "before_commit_sha256",
    "after_commit_sha256",
    "before_fit_audit_sha256",
    "after_fit_audit_sha256",
    "before_resource_audit_sha256",
    "after_resource_audit_sha256",
    "before_execution_receipt_sha256",
    "after_execution_receipt_sha256",
)


def _write_prediction_closure(prediction_root: Path) -> dict[str, Path]:
    for state in ("before", "after"):
        state_root = prediction_root / state
        for name in (
            "prediction_artifact.npz",
            "COMMIT.json",
            "fit_audit.json",
            "resource_audit.json",
            "execution_receipt.json",
        ):
            path = state_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{state}:{name}".encode("utf-8"))
    return runner._prediction_closure_paths(prediction_root)


def _score_fixture(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, Path], Path, str]:
    truth_path = tmp_path / "truth_sidecar.json"
    truth_sha256 = _write_bytes_readonly(truth_path, b"truth")
    paths = _write_prediction_closure(tmp_path / "diag")
    job: dict[str, object] = {
        "job_id": "job-1",
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "outer_role": "performance",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
        "truth_sidecar": str(truth_path),
        "truth_sidecar_sha256": truth_sha256,
    }
    score_path = tmp_path / "score.json"
    _write(
        score_path,
        {
            "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
            "candidate": runner.CANDIDATE_ID,
            "truth_sidecar_sha256": truth_sha256,
            "before_prediction_sha256": hashlib.sha256(
                paths["before_prediction"].read_bytes()
            ).hexdigest(),
            "after_prediction_sha256": hashlib.sha256(
                paths["after_prediction"].read_bytes()
            ).hexdigest(),
        },
    )
    return job, paths, score_path, truth_sha256


def _closure_sha_values(paths: dict[str, Path]) -> dict[str, str]:
    evidence_paths = {
        "before_prediction_sha256": paths["before_prediction"],
        "after_prediction_sha256": paths["after_prediction"],
        "before_commit_sha256": paths["before_commit"],
        "after_commit_sha256": paths["after_commit"],
        "before_fit_audit_sha256": paths["before_fit_audit"],
        "after_fit_audit_sha256": paths["after_fit_audit"],
        "before_resource_audit_sha256": paths["before_fit_audit"].with_name(
            "resource_audit.json"
        ),
        "after_resource_audit_sha256": paths["after_fit_audit"].with_name(
            "resource_audit.json"
        ),
        "before_execution_receipt_sha256": paths["before_fit_audit"].with_name(
            "execution_receipt.json"
        ),
        "after_execution_receipt_sha256": paths["after_fit_audit"].with_name(
            "execution_receipt.json"
        ),
    }
    return {
        field: hashlib.sha256(path.read_bytes()).hexdigest()
        for field, path in evidence_paths.items()
    }


def _reference_resources(*, peak: int = 10) -> dict[str, dict[str, int]]:
    return {
        scene: {
            "registration_wall_time_ns": 100_000_000,
            "registration_incremental_peak_working_set_bytes": peak,
            "query_macs": 11 * 288,
            "state_bytes": 11 * 289 * 4,
        }
        for scene in runner.SCENES
    }


def _row(
    scene: str,
    *,
    k_shot: int = 10,
    candidate_peak: int = 729_088,
    candidate_wall: int = 120_000_000,
    candidate_state_bytes: int = 11 * 289 * 4,
    candidate_query_macs: int = 11 * 288,
    postprocess_mode: object = None,
) -> dict[str, object]:
    active = k_shot > 2
    prefix = "d92_e0d_ccoc_"
    row: dict[str, object] = {
        "scenario": scene,
        "arm_id": runner.ARM_ID,
        "candidate_id": runner.CANDIDATE_ID,
        "after_registered_d_mode_effective": "ccoc_full" if active else "d92_full_alias",
        "after_state_postprocess_mode": postprocess_mode,
        "after_total_component_fit_count": 2 if active else 3,
        "after_actual_component_inventory": {
            "actual_component_fit_count": 1 if active else 3,
            "full_component_fit_count": 1 if active else 3,
        },
        "registered_class_count": 11,
        "query_macs": candidate_query_macs,
        "after_state_bytes": candidate_state_bytes,
        "after_registration_resource": {
            "registration_wall_time_ns": candidate_wall,
            "registration_incremental_peak_working_set_bytes": candidate_peak,
        },
        prefix + "active": active,
        prefix + "fallback_active": False,
        prefix + "fallback_reason": None if active else "K1_K2_EXACT_D92_FULL_ALIAS",
        prefix + "candidate_attempt_fit_count": 1 if active else 0,
        prefix + "fallback_reference_fit_count": 0,
        prefix + "candidate_statistic_receipt_available": active,
        prefix + "paired_e0_codec_state_equal": None,
        prefix + "g0_eligible": active,
        prefix + "g0_block_reason": None if active else "K1_K2_EXACT_D92_FULL_ALIAS",
        prefix + "query_rows_used": 0,
    }
    for field in runner.QUERY_ZERO_FIELDS:
        row[field] = False
        row[prefix + field] = False
    return row


def _fit_audit_rows(**kwargs: object) -> list[dict[str, object]]:
    return [_row(scene, **kwargs) for scene in runner.SCENES]


def test_fit_audit_accepts_ccoc_k_gt_2_and_k1_exact_alias(tmp_path: Path) -> None:
    for k_shot in (10, 1):
        path = tmp_path / f"fit_audit_k{k_shot}.json"
        _write(path, _fit_audit_rows(k_shot=k_shot))
        result = runner._validate_fit_audit(
            path,
            k_shot=k_shot,
            reference_resources=_reference_resources(),
        )
        assert result["scene_count"] == 3
        assert result["candidate_peak_hard_pass"] is True
        assert result["candidate_peak_target_pass"] is False


def test_query_access_zero_requires_base_and_approved_ccoc_mirror_only() -> None:
    prefix = "d92_e0d_ccoc_"
    row = {
        key: False
        for field in runner.QUERY_ZERO_FIELDS
        for key in (field, prefix + field)
    }

    assert runner._query_access_is_zero(row) is True
    for field in runner.QUERY_ZERO_FIELDS:
        for key in (field, prefix + field):
            missing = dict(row)
            del missing[key]
            assert runner._query_access_is_zero(missing) is False
            enabled = dict(row)
            enabled[key] = True
            assert runner._query_access_is_zero(enabled) is False


def test_candidate_peak_is_absolute_not_offset_by_e0_peak(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    _write(path, _fit_audit_rows(candidate_peak=729_088))

    low_reference = runner._validate_fit_audit(
        path,
        k_shot=10,
        reference_resources=_reference_resources(peak=1),
    )
    high_reference = runner._validate_fit_audit(
        path,
        k_shot=10,
        reference_resources=_reference_resources(peak=9_999_999),
    )

    assert low_reference["candidate_peak_hard_pass"] is True
    assert high_reference["candidate_peak_hard_pass"] is True
    assert low_reference["candidate_peak_target_pass"] is False
    assert high_reference["candidate_peak_target_pass"] is False
    assert low_reference["candidate_peak_max_bytes"] == high_reference[
        "candidate_peak_max_bytes"
    ] == 729_088


@pytest.mark.parametrize(
    ("field", "row_kwargs", "reference"),
    (
        (
            "wall",
            {"candidate_wall": 150_000_001},
            _reference_resources(),
        ),
        (
            "ratio",
            {"candidate_wall": 140_000_000},
            {
                scene: {
                    **resource,
                    "registration_wall_time_ns": 90_000_000,
                }
                for scene, resource in _reference_resources().items()
            },
        ),
        (
            "peak",
            {"candidate_peak": 1_048_577},
            _reference_resources(),
        ),
        (
            "query MAC",
            {"candidate_query_macs": 11 * 288 + 1},
            _reference_resources(),
        ),
        (
            "state",
            {"candidate_state_bytes": 11 * 289 * 4 + 1},
            _reference_resources(),
        ),
        (
            "postprocess",
            {"postprocess_mode": "unexpected_postprocess"},
            _reference_resources(),
        ),
    ),
)
def test_fit_audit_rejects_each_single_scene_resource_or_integrity_drift(
    tmp_path: Path,
    field: str,
    row_kwargs: dict[str, int],
    reference: dict[str, dict[str, int]],
) -> None:
    rows = _fit_audit_rows()
    rows[1] = _row(runner.SCENES[1], **row_kwargs)
    path = tmp_path / f"fit_audit_{field.replace(' ', '_')}.json"
    _write(path, rows)

    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match=field):
        runner._validate_fit_audit(
            path,
            k_shot=10,
            reference_resources=reference,
        )


def test_fit_audit_rejects_any_query_access(tmp_path: Path) -> None:
    rows = _fit_audit_rows()
    rows[0][runner.QUERY_ZERO_FIELDS[0]] = True
    path = tmp_path / "fit_audit_query.json"
    _write(path, rows)

    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="query access"):
        runner._validate_fit_audit(
            path,
            k_shot=10,
            reference_resources=_reference_resources(),
        )


def test_parser_has_only_prepare_smoke_and_shard_execution_boundaries() -> None:
    parser = runner.parser()
    commands = set(parser._subparsers._group_actions[0].choices)
    assert {"prepare", "smoke", "run-shard"} <= commands
    assert "truth" not in parser.format_help().lower()
    assert runner.SHARD_COUNT == 8


def test_prediction_manifest_check_defers_truth_until_score_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    truth_path = tmp_path / "truth_sidecar.json"
    truth_sha256 = _write_bytes_readonly(truth_path, b"score-only-truth")
    job_root = tmp_path / "job"
    job_root.mkdir()
    package_root = tmp_path / "package"
    package_root.mkdir()
    seal = tmp_path / "seal.json"
    seal_sha256 = _write_bytes_readonly(seal, b"seal")
    job: dict[str, object] = {
        "job_id": "truth-last-job",
        "outer_key": runner.SMOKE_OUTER_KEY,
        "outer_role": "performance",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
        "truth_sidecar": str(truth_path),
        "truth_sidecar_sha256": truth_sha256,
        "packages": {
            "sealed": {
                "package_root": str(package_root),
                "detached_seal_path": str(seal),
                "expected_seal_sha256": seal_sha256,
            }
        },
    }
    manifest = {"jobs": [job]}
    paths = _write_prediction_closure(job_root / "diag")
    truth_reads: list[Path] = []
    original_sha256_file = runner._sha256_file

    def observing_sha256(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved == truth_path.resolve():
            truth_reads.append(resolved)
        return original_sha256_file(Path(path))

    monkeypatch.setattr(runner, "_sha256_file", observing_sha256)
    runner._verify_manifest_artifacts(manifest)
    assert truth_reads == []

    runner._write_score_binding(
        job_root,
        job=job,
        matrix_manifest_sha256="c" * 64,
        method_lock_sha256="a" * 64,
        paths=paths,
        score_command=["score"],
    )
    assert truth_reads == [truth_path.resolve()]


def test_systemic_failure_needs_two_distinct_pre_prediction_outers(
    tmp_path: Path,
) -> None:
    first = {
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "job_id": "first",
        "arm_id": runner.ARM_ID,
    }
    second = {
        "outer_key": "rx_7_7__seed_713103__k_10__new_5",
        "job_id": "second",
        "arm_id": runner.ARM_ID,
    }

    assert runner._record_pre_prediction_failure(tmp_path, first, "same-error") is False
    assert runner._record_pre_prediction_failure(tmp_path, first, "same-error") is False
    assert runner._record_pre_prediction_failure(tmp_path, second, "same-error") is True

    stop = json.loads(
        (tmp_path / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").read_text(
            encoding="utf-8"
        )
    )
    assert stop["schema"] == "cvs.phase2.d92_ccoc_hard9_k1.systemic_failure.v1"
    assert stop["distinct_outer_count"] == 2


def test_score_evidence_requires_actual_readonly_sidecar_before_and_after_scoring(
    tmp_path: Path,
) -> None:
    truth_path = tmp_path / "truth_sidecar.json"
    expected_sha256 = _write_bytes_readonly(
        truth_path,
        b'{"schema":"cvs.phase2.query_truth_sidecar.v2","rows":[]}',
    )

    assert (
        runner._verify_truth_sidecar_snapshot(
            truth_path,
            expected_sha256=expected_sha256,
        )
        == expected_sha256
    )

    os.chmod(truth_path, stat.S_IWRITE | stat.S_IREAD)
    truth_path.write_bytes(b'{"schema":"cvs.phase2.query_truth_sidecar.v2","rows":[1]}')
    os.chmod(truth_path, stat.S_IREAD)
    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="truth sidecar"):
        runner._verify_truth_sidecar_snapshot(
            truth_path,
            expected_sha256=expected_sha256,
        )


def test_score_artifact_binds_actual_inputs_to_the_frozen_job_identity(
    tmp_path: Path,
) -> None:
    job, paths, score_path, truth_sha256 = _score_fixture(tmp_path)

    binding_path, binding_sha256 = runner._write_score_binding(
        tmp_path / "job",
        job=job,
        matrix_manifest_sha256="a" * 64,
        method_lock_sha256="b" * 64,
        paths=paths,
        score_command=["python", "score"],
    )
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    assert binding["outer_key"] == job["outer_key"]
    assert binding["arm_id"] == runner.ARM_ID
    assert binding["method_lock_sha256"] == "b" * 64
    assert binding_sha256 == hashlib.sha256(binding_path.read_bytes()).hexdigest()

    evidence = runner._validate_score_artifact(
        score_path,
        job=job,
        matrix_manifest_sha256="a" * 64,
        method_lock_sha256="b" * 64,
        truth_sidecar_sha256=truth_sha256,
        before_prediction_path=paths["before_prediction"],
        after_prediction_path=paths["after_prediction"],
        score_binding_path=binding_path,
    )
    assert evidence["job_id"] == job["job_id"]
    assert evidence["outer_key"] == job["outer_key"]
    assert evidence["matrix_manifest_sha256"] == "a" * 64

    _write(
        score_path,
        {
            "schema": "cvs.phase2.diag_cosine_dev_pair_score.v1",
            "candidate": "wrong-candidate",
            "truth_sidecar_sha256": truth_sha256,
            "before_prediction_sha256": hashlib.sha256(
                paths["before_prediction"].read_bytes()
            ).hexdigest(),
            "after_prediction_sha256": hashlib.sha256(
                paths["after_prediction"].read_bytes()
            ).hexdigest(),
        },
    )
    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="score artifact"):
        runner._validate_score_artifact(
            score_path,
            job=job,
            matrix_manifest_sha256="a" * 64,
            method_lock_sha256="b" * 64,
            truth_sidecar_sha256=truth_sha256,
            before_prediction_path=paths["before_prediction"],
            after_prediction_path=paths["after_prediction"],
            score_binding_path=binding_path,
        )


def test_score_binding_seals_all_before_after_closure_hashes_with_o_excl(
    tmp_path: Path,
) -> None:
    job, paths, _score_path, _truth_sha256 = _score_fixture(tmp_path)

    binding_path, binding_sha256 = runner._write_score_binding(
        tmp_path / "job",
        job=job,
        matrix_manifest_sha256="a" * 64,
        method_lock_sha256="b" * 64,
        paths=paths,
        score_command=["python", "score"],
    )

    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    expected_hashes = _closure_sha_values(paths)
    assert set(_CLOSURE_SHA_FIELDS) <= set(binding)
    assert {field: binding[field] for field in _CLOSURE_SHA_FIELDS} == expected_hashes
    assert binding_sha256 == hashlib.sha256(binding_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        runner._write_score_binding(
            tmp_path / "job",
            job=job,
            matrix_manifest_sha256="a" * 64,
            method_lock_sha256="b" * 64,
            paths=paths,
            score_command=["python", "score"],
        )


def test_outer_score_binding_rejects_fit_and_rewritten_commit_drift(
    tmp_path: Path,
) -> None:
    job, paths, score_path, truth_sha256 = _score_fixture(tmp_path)
    binding_path, _binding_sha256 = runner._write_score_binding(
        tmp_path / "job",
        job=job,
        matrix_manifest_sha256="a" * 64,
        method_lock_sha256="b" * 64,
        paths=paths,
        score_command=["python", "score"],
    )
    _write(paths["after_fit_audit"], {"tampered": "fit"})
    _write(paths["after_commit"], {"rewritten": "commit"})

    with pytest.raises(
        runner.D92CCOCHard9K1RunnerError, match="score binding closure"
    ):
        runner._validate_score_artifact(
            score_path,
            job=job,
            matrix_manifest_sha256="a" * 64,
            method_lock_sha256="b" * 64,
            truth_sidecar_sha256=truth_sha256,
            before_prediction_path=paths["before_prediction"],
            after_prediction_path=paths["after_prediction"],
            score_binding_path=binding_path,
        )


def test_final_job_receipt_seals_same_closure_hashes_and_rejects_rewrite(
    tmp_path: Path,
) -> None:
    job, paths, score_path, truth_sha256 = _score_fixture(tmp_path)
    binding_path, binding_sha256 = runner._write_score_binding(
        tmp_path / "job",
        job=job,
        matrix_manifest_sha256="a" * 64,
        method_lock_sha256="b" * 64,
        paths=paths,
        score_command=["python", "score"],
    )
    score_evidence = runner._validate_score_artifact(
        score_path,
        job=job,
        matrix_manifest_sha256="a" * 64,
        method_lock_sha256="b" * 64,
        truth_sidecar_sha256=truth_sha256,
        before_prediction_path=paths["before_prediction"],
        after_prediction_path=paths["after_prediction"],
        score_binding_path=binding_path,
    )
    closure_hashes = {
        field: score_evidence[field] for field in _CLOSURE_SHA_FIELDS
    }
    receipt_base = {
        "schema": runner.JOB_RECEIPT_SCHEMA,
        "status": "PREDICTIONS_AND_POST_PREDICTION_SCORE_COMPLETE",
        "job_id": job["job_id"],
        "outer_key": job["outer_key"],
        "outer_role": job["outer_role"],
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
        "matrix_manifest_sha256": "a" * 64,
        "method_lock_sha256": "b" * 64,
        "truth_sidecar_sha256": truth_sha256,
        "score_binding": str(binding_path),
        "score_binding_sha256": binding_sha256,
        "score_evidence": score_evidence,
    }
    receipt_path = tmp_path / "job" / "job_receipt.json"
    receipt_sha256 = runner._write_job_receipt(
        tmp_path / "job",
        receipt_base,
        closure_hashes=closure_hashes,
    )

    expected_hashes = _closure_sha_values(paths)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert {field: receipt[field] for field in _CLOSURE_SHA_FIELDS} == expected_hashes
    assert receipt["prediction_closure"] == expected_hashes
    assert receipt["truth_sidecar_sha256"] == truth_sha256
    assert receipt["score_binding_sha256"] == binding_sha256
    assert receipt_sha256 == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        runner._write_job_receipt(
            tmp_path / "job",
            receipt_base,
            closure_hashes=closure_hashes,
        )

    _write(paths["after_fit_audit"], {"tampered": "fit"})
    _write(paths["after_commit"], {"rewritten": "commit"})
    assert receipt["after_fit_audit_sha256"] != hashlib.sha256(
        paths["after_fit_audit"].read_bytes()
    ).hexdigest()
    assert receipt["after_commit_sha256"] != hashlib.sha256(
        paths["after_commit"].read_bytes()
    ).hexdigest()
    with pytest.raises(
        runner.D92CCOCHard9K1RunnerError, match="score binding closure"
    ):
        runner._validate_score_artifact(
            score_path,
            job=job,
            matrix_manifest_sha256="a" * 64,
            method_lock_sha256="b" * 64,
            truth_sidecar_sha256=truth_sha256,
            before_prediction_path=paths["before_prediction"],
            after_prediction_path=paths["after_prediction"],
            score_binding_path=binding_path,
        )


def test_nonzero_prediction_with_any_closure_artifact_is_post_prediction_failure(
    tmp_path: Path,
) -> None:
    prediction_root = tmp_path / "diag"

    assert runner._prediction_failure_stage(prediction_root) == "pre_prediction"

    artifact = prediction_root / "after" / "execution_receipt.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}", encoding="utf-8")
    assert runner._prediction_failure_stage(prediction_root) == "post_prediction"


def test_third_shard_shared_stop_after_prediction_prevents_scoring(
    tmp_path: Path,
) -> None:
    score_starts: list[str] = []
    _write(
        tmp_path / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json",
        {"schema": runner.SYSTEMIC_FAILURE_SCHEMA},
    )

    started, value = runner._start_score_unless_stopped(
        tmp_path,
        start=lambda: score_starts.append("score") or "started",
    )

    assert started is False
    assert value is None
    assert score_starts == []


def test_score_dispatch_barrier_rechecks_stop_in_check_start_window(
    tmp_path: Path,
) -> None:
    score_starts: list[str] = []

    started, value = runner._dispatch_score_under_stop_barrier(
        tmp_path,
        coordinator_id="shard-3",
        before_start=lambda: _write(
            tmp_path / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json",
            {"schema": runner.SYSTEMIC_FAILURE_SCHEMA},
        ),
        start=lambda: score_starts.append("score") or "started",
    )

    assert started is False
    assert value is None
    assert score_starts == []


def test_coordinator_stop_does_not_signal_foreign_pid(tmp_path: Path) -> None:
    job = {
        "job_id": "job-foreign",
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
    }
    _write(
        tmp_path / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json",
        {"schema": runner.SYSTEMIC_FAILURE_SCHEMA},
    )
    runner._write_active_process_receipt(
        tmp_path,
        job=job,
        shard_index=2,
        stage="prediction",
        pid=43210,
        parent_pid=12345,
        cwd=runner.CODE_ROOT,
        cmdline=("python", "prediction"),
    )
    signals: list[tuple[int, str]] = []
    result = runner.stop_verified_active_processes(
        tmp_path,
        coordinator_id="task3-coordinator",
        process_inspector=lambda _pid: {
            "pid": 43210,
            "parent_pid": 12345,
            "cwd": str(tmp_path / "foreign-cwd"),
            "cmdline": ["python", "prediction"],
        },
        signal_sender=lambda pid, signal: signals.append((pid, signal)),
        sleep_seconds=0.0,
    )

    assert result["verified_process_count"] == 0
    assert result["skipped_unverified_process_count"] == 1
    assert signals == []


def test_second_fingerprint_publishes_stop_and_automatically_stops_only_owned_child(
    tmp_path: Path,
) -> None:
    first = {
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "job_id": "first",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
    }
    second = {
        "outer_key": "rx_7_7__seed_713103__k_10__new_5",
        "job_id": "second",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
    }
    foreign = {
        "outer_key": "rx_8_8__seed_713103__k_5__new_20",
        "job_id": "foreign",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
    }
    runner._write_active_process_receipt(
        tmp_path,
        job=first,
        shard_index=0,
        stage="score",
        pid=101,
        parent_pid=201,
        cwd=runner.CODE_ROOT,
        cmdline=("python", "owned"),
    )
    runner._write_active_process_receipt(
        tmp_path,
        job=foreign,
        shard_index=1,
        stage="prediction",
        pid=102,
        parent_pid=202,
        cwd=runner.CODE_ROOT,
        cmdline=("python", "foreign"),
    )
    signals: list[tuple[int, str]] = []

    def inspect(pid: int) -> dict[str, object]:
        if pid == 101:
            return {
                "pid": 101,
                "parent_pid": 201,
                "cwd": str(runner.CODE_ROOT),
                "cmdline": ["python", "owned"],
            }
        return {
            "pid": 102,
            "parent_pid": 202,
            "cwd": str(tmp_path / "foreign-cwd"),
            "cmdline": ["python", "foreign"],
        }

    assert (
        runner._record_pre_prediction_failure(
            tmp_path,
            first,
            "same-error",
            coordinator_id="triggering-shard",
            process_inspector=inspect,
            signal_sender=lambda pid, signal: signals.append((pid, signal)),
            sleep_seconds=0.0,
        )
        is False
    )
    assert (
        runner._record_pre_prediction_failure(
            tmp_path,
            second,
            "same-error",
            coordinator_id="triggering-shard",
            process_inspector=inspect,
            signal_sender=lambda pid, signal: signals.append((pid, signal)),
            sleep_seconds=0.0,
        )
        is True
    )

    assert signals == [(101, "SIGTERM"), (101, "SIGKILL")]
    assert (tmp_path / "coordination" / "stop_action.json").is_file()


def test_prediction_dispatch_barrier_registers_popen_child_before_queued_stop_scan(
    tmp_path: Path,
) -> None:
    job = {
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "job_id": "prediction-race",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
    }
    popen_seen = threading.Event()
    stop_attempted = threading.Event()
    stop_errors: list[BaseException] = []
    signals: list[tuple[int, str]] = []

    class Child:
        pid = 703

        def wait(self) -> int:  # pragma: no cover - dispatch intentionally does not wait
            return 0

        def terminate(self) -> None:  # pragma: no cover - receipt write succeeds
            raise AssertionError("must not terminate during receipt registration")

    child = Child()

    def inspect(pid: int) -> dict[str, object]:
        assert pid == child.pid
        return {
            "pid": child.pid,
            "parent_pid": os.getpid(),
            "cwd": str(runner.CODE_ROOT),
            "cmdline": ["python", "prediction"],
        }

    def publish_stop() -> None:
        assert popen_seen.wait(1.0)
        stop_attempted.set()
        try:
            runner._publish_systemic_stop_and_terminate(
                tmp_path,
                coordinator_id="second-fingerprint-shard",
                fingerprint="a" * 64,
                distinct_outer_count=2,
                process_inspector=inspect,
                signal_sender=lambda pid, signal: signals.append((pid, signal)),
                sleep_seconds=0.0,
            )
        except BaseException as error:  # make thread failure observable to pytest
            stop_errors.append(error)

    worker = threading.Thread(target=publish_stop)

    def start_prediction() -> Child:
        worker.start()
        return runner._start_shard_child(
            ["python", "prediction"],
            output_root=tmp_path,
            job=job,
            shard_index=3,
            stage="prediction",
            stdout=object(),
            stderr=object(),
            env={},
            popen=lambda *_args, **_kwargs: child,
            after_popen_before_receipt=lambda: (
                popen_seen.set(),
                stop_attempted.wait(1.0),
            ),
        )

    started, registered_child = runner._dispatch_prediction_under_stop_barrier(
        tmp_path,
        coordinator_id="prediction-shard",
        start=start_prediction,
    )
    worker.join(timeout=2.0)

    assert started is True
    assert registered_child is child
    assert not worker.is_alive()
    assert stop_errors == []
    assert list((tmp_path / "active_processes" / "shard_3").glob("*.json"))
    assert signals == [(child.pid, "SIGTERM"), (child.pid, "SIGKILL")]
    assert (tmp_path / "SYSTEMIC_TECHNICAL_FAILURE_STOP.json").is_file()


def test_shard_child_writes_exclusive_process_receipt_before_wait(
    tmp_path: Path,
) -> None:
    job = {
        "job_id": "job-active",
        "outer_key": "rx_7_7__seed_713104__k_5__new_20",
        "arm_id": runner.ARM_ID,
        "candidate": runner.CANDIDATE_ID,
    }

    class Child:
        pid = 54321

        def wait(self) -> int:
            receipts = list((tmp_path / "active_processes" / "shard_2").glob("*.json"))
            assert len(receipts) == 1
            receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
            assert receipt["pid"] == self.pid
            assert receipt["job_id"] == job["job_id"]
            assert receipt["cwd"] == str(runner.CODE_ROOT.resolve())
            assert receipt["cmdline"] == ["python", "prediction"]
            return 0

        def terminate(self) -> None:  # pragma: no cover - receipt write succeeds
            raise AssertionError("must not terminate a correctly receipted child")

    assert (
        runner._run_shard_child(
            ["python", "prediction"],
            output_root=tmp_path,
            job=job,
            shard_index=2,
            stage="prediction",
            stdout=object(),
            stderr=object(),
            env={},
            popen=lambda *_args, **_kwargs: Child(),
        )
        == 0
    )


def test_runtime_source_lock_closes_scientific_entry_and_rejects_file_drift(
    tmp_path: Path,
) -> None:
    lock = json.loads(
        (ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v8.json").read_text(
            encoding="utf-8"
        )
    )
    protected = {
        "scripts/run_d92_e0d_prediction.py",
        "scripts/score_d92_be_prediction.py",
        "scripts/probe_d92_registration_balanced_covariance.py",
        "cvsrffi/stage2_d92_cross_class_offblock_consensus.py",
        "cvsrffi/stage2_d92_e0d_slim.py",
        "cvsrffi/stage2_d92_e0d_query_evaluation.py",
        "cvsrffi/stage2_d42_unified_shrinkage_lda.py",
    }
    assert protected <= set(lock["runtime_source"]["files"])
    assert runner._verify_runtime_source_lock(lock, code_root=ROOT / "code")[
        "scientific_entry_commit"
    ] == "053ef7d006b05d4cb00c593e9b694669c0ecb005"

    source = tmp_path / "code" / "cvsrffi" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen")
    source_lock = {
        "scientific_entry_commit": "930c5d644c323bab94deece9a08fdfb09f565399",
        "files": {
            "cvsrffi/runtime.py": {
                "git_blob": "a" * 40,
                "sha256": hashlib.sha256(b"frozen").hexdigest(),
            }
        },
    }

    def temp_git(_repo_root: Path, _arguments: tuple[str, ...]) -> str:
        return "" if _arguments[0] == "cat-file" else "a" * 40

    assert runner._verify_runtime_source_files(
        source_lock,
        code_root=tmp_path / "code",
        git_runner=temp_git,
    )["file_count"] == 1
    source.write_bytes(b"drift")
    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="runtime source"):
        runner._verify_runtime_source_files(
            source_lock,
            code_root=tmp_path / "code",
            git_runner=temp_git,
        )


def _run_extracted_archive_prepare_probe(
    tmp_path: Path,
    *,
    drift_locked_source: bool,
) -> subprocess.CompletedProcess[str]:
    """Run real ``prepare`` from an archive-shaped tree with no Git metadata."""

    archive = (
        ROOT
        / "automation_reports"
        / "CV-SincNet"
        / "d92_e0_full_ccoc_hard9k1_20260817_v2"
        / "runtime"
        / "d92_ccoc_hard9_k1_source_fe9033be_20260817_v2.tar.gz"
    )
    extracted = tmp_path / "extracted"
    extracted.mkdir()
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            path = Path(member.name)
            assert not path.is_absolute() and ".." not in path.parts
            assert not member.issym() and not member.islnk()
        bundle.extractall(extracted)

    extracted_config = (
        extracted / "configs" / "stage2_d92_ccoc_hard9_k1_v8.json"
    )
    extracted_config.write_bytes(
        (ROOT / "configs" / "stage2_d92_ccoc_hard9_k1_v8.json").read_bytes()
    )
    config = json.loads(extracted_config.read_text(encoding="utf-8"))
    for relative_path in config["runtime_source"]["files"]:
        archived_source = extracted / "code" / relative_path
        archived_source.write_bytes((ROOT / "code" / relative_path).read_bytes())

    # Use the exact prospective v5 runner/matrix bytes inside the proven
    # runtime closure, still without a .git directory.
    extracted_runner = extracted / "code" / "scripts" / "run_d92_ccoc_hard9_k1.py"
    extracted_runner.write_bytes(
        (ROOT / "code" / "scripts" / "run_d92_ccoc_hard9_k1.py").read_bytes()
    )
    extracted_matrix = (
        extracted / "code" / "cvsrffi" / "stage2_d92_ccoc_hard9_k1.py"
    )
    extracted_matrix.write_bytes(
        (ROOT / "code" / "cvsrffi" / "stage2_d92_ccoc_hard9_k1.py").read_bytes()
    )
    if drift_locked_source:
        locked = extracted / "code" / "cvsrffi" / "stage2_d92_e0d_slim.py"
        locked.write_bytes(locked.read_bytes() + b"\n# byte drift\n")

    probe = """
import argparse
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root / "code"))
from scripts import run_d92_ccoc_hard9_k1 as runner

config = root / "configs" / "stage2_d92_ccoc_hard9_k1_v8.json"
output_root = root / "prepare_output"
runner.build_hard9_k1_manifest = lambda _config, require_package_files: {
    "method_lock": str(config),
    "jobs": [],
    "output_root": str(output_root),
}
try:
    value = runner.prepare(argparse.Namespace(config=str(config)))
except Exception as error:
    print(f"{type(error).__name__}: {error}", file=sys.stderr)
    raise SystemExit(2)
print(json.dumps(value, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(extracted / "code")
    return subprocess.run(
        [sys.executable, "-c", probe, str(extracted)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_extracted_archive_prepare_uses_sha_only_without_git_metadata(
    tmp_path: Path,
) -> None:
    result = _run_extracted_archive_prepare_probe(
        tmp_path,
        drift_locked_source=False,
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    assert receipt["runtime_source_verification_mode"] == "sha256_only"


def test_extracted_archive_prepare_still_rejects_locked_source_byte_drift(
    tmp_path: Path,
) -> None:
    result = _run_extracted_archive_prepare_probe(
        tmp_path,
        drift_locked_source=True,
    )
    assert result.returncode == 2
    assert "runtime source SHA drift" in result.stderr
    assert "frozen commit is unavailable" not in result.stderr


def test_runtime_source_gate_requires_frozen_commit_and_both_git_blob_views(
    tmp_path: Path,
) -> None:
    source = tmp_path / "code" / "cvsrffi" / "runtime.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen")
    commit = "c" * 40
    blob = "d" * 40
    source_lock = {
        "scientific_entry_commit": commit,
        "files": {
            "cvsrffi/runtime.py": {
                "git_blob": blob,
                "sha256": hashlib.sha256(b"frozen").hexdigest(),
            }
        },
    }

    def git_ok(_repo_root: Path, arguments: tuple[str, ...]) -> str:
        values = {
            ("cat-file", "-e", f"{commit}^{{commit}}") : "",
            ("rev-parse", f"{commit}:code/cvsrffi/runtime.py"): blob,
            ("rev-parse", "HEAD:code/cvsrffi/runtime.py"): blob,
        }
        return values[arguments]

    assert runner._verify_runtime_source_files(
        source_lock,
        code_root=tmp_path / "code",
        git_runner=git_ok,
    )["file_count"] == 1

    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="frozen commit"):
        runner._verify_runtime_source_files(
            source_lock,
            code_root=tmp_path / "code",
            git_runner=lambda _root, _arguments: (_ for _ in ()).throw(
                runner.D92CCOCHard9K1RunnerError("frozen commit missing")
            ),
        )

    def frozen_blob_drift(_repo_root: Path, arguments: tuple[str, ...]) -> str:
        if arguments == ("cat-file", "-e", f"{commit}^{{commit}}"):
            return ""
        if arguments == ("rev-parse", f"{commit}:code/cvsrffi/runtime.py"):
            return "e" * 40
        return blob

    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="frozen blob"):
        runner._verify_runtime_source_files(
            source_lock,
            code_root=tmp_path / "code",
            git_runner=frozen_blob_drift,
        )

    def head_blob_drift(_repo_root: Path, arguments: tuple[str, ...]) -> str:
        if arguments == ("cat-file", "-e", f"{commit}^{{commit}}"):
            return ""
        if arguments == ("rev-parse", f"{commit}:code/cvsrffi/runtime.py"):
            return blob
        return "f" * 40

    with pytest.raises(runner.D92CCOCHard9K1RunnerError, match="HEAD blob"):
        runner._verify_runtime_source_files(
            source_lock,
            code_root=tmp_path / "code",
            git_runner=head_blob_drift,
        )


def test_e0_resource_records_use_embedded_projection_without_historical_file(
    tmp_path: Path,
) -> None:
    outer_key = "rx_7_7__seed_713104__k_5__new_20"
    job = {
        "outer_key": outer_key,
        "k_shot": 5,
        "new_class_count": 20,
    }
    fit_audit = (
        tmp_path
        / "jobs"
        / outer_key
        / "E0_FULL_ONLY"
        / "diag"
        / "after"
        / "fit_audit.json"
    )
    embedded_scenes: dict[str, dict[str, int]] = {}
    for index, scene in enumerate(runner.SCENES):
        embedded_scenes[scene] = {
            "registration_wall_time_ns": 100 + index,
            "registration_incremental_peak_working_set_bytes": 200 + index,
            "query_macs": 7_488,
            "state_bytes": 18_498,
        }
    job["e0_resource"] = {
        "fit_audit": {
            "path": str(fit_audit),
            "sha256": "a" * 64,
        },
        "scenes": embedded_scenes,
    }

    observation = runner._load_verified_e0_resource_records(job)
    assert observation == {
        "source_mode": "embedded_preregistered_projection",
        "fit_audit_declared_sha256": "a" * 64,
        "scenes": embedded_scenes,
    }
    binding = runner._bind_e0_resource_observations({"jobs": [job]})
    assert binding == {
        "source_mode": "embedded_preregistered_projection",
        "fit_audit_declared_sha256": {outer_key: "a" * 64},
    }
    assert not fit_audit.exists()


def test_embedded_e0_resource_projection_rejects_identity_and_value_tamper() -> None:
    outer_key = "rx_7_7__seed_713104__k_5__new_20"
    job = {
        "outer_key": outer_key,
        "k_shot": 5,
        "new_class_count": 20,
        "e0_resource": {
            "fit_audit": {
                "path": f"/missing/jobs/{outer_key}/E0_FULL_ONLY/diag/after/fit_audit.json",
                "sha256": "a" * 64,
            },
            "scenes": {
                scene: {
                    "registration_wall_time_ns": 100,
                    "registration_incremental_peak_working_set_bytes": 0,
                    "query_macs": 7_488,
                    "state_bytes": 1,
                }
                for scene in runner.SCENES
            },
        },
    }

    cases: list[tuple[str, dict[str, object], str]] = []
    scene_drift = copy.deepcopy(job)
    del scene_drift["e0_resource"]["scenes"][runner.SCENES[0]]
    cases.append(("scene", scene_drift, "scene identity"))
    query_drift = copy.deepcopy(job)
    query_drift["e0_resource"]["scenes"][runner.SCENES[0]]["query_macs"] += 1
    cases.append(("query", query_drift, "query MAC identity"))
    state_drift = copy.deepcopy(job)
    state_drift["e0_resource"]["scenes"][runner.SCENES[0]]["state_bytes"] = 0
    cases.append(("state", state_drift, "query/state"))
    wall_drift = copy.deepcopy(job)
    wall_drift["e0_resource"]["scenes"][runner.SCENES[0]][
        "registration_wall_time_ns"
    ] = 0
    cases.append(("wall", wall_drift, "wall"))
    peak_drift = copy.deepcopy(job)
    peak_drift["e0_resource"]["scenes"][runner.SCENES[0]][
        "registration_incremental_peak_working_set_bytes"
    ] = -1
    cases.append(("peak", peak_drift, "peak"))
    identity_drift = copy.deepcopy(job)
    identity_drift["outer_key"] = "rx_7_7__seed_713104__k_10__new_20"
    cases.append(("identity", identity_drift, "job identity"))

    for label, tampered, error in cases:
        with pytest.raises(
            runner.D92CCOCHard9K1RunnerError,
            match=error,
        ):
            runner._load_verified_e0_resource_records(tampered)
