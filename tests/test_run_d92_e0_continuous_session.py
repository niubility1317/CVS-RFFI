from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_d92_e0_continuous_session as runner
from cvsrffi import stage2_d92_continuous_session_matrix as matrix


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "stage2_d92_e0_continuous_session_v1.json"


def _prepare(tmp_path: Path) -> Path:
    result = runner.prepare(
        method_lock_path=CONFIG,
        output_root=tmp_path / "matrix",
    )
    return Path(result["manifest_path"])


def _prepare_deltas(manifest_path: Path) -> dict[str, object]:
    calls: list[dict[str, object]] = []

    def fake_delta(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        Path(str(kwargs["output_root"])).mkdir(parents=True, exist_ok=True)
        return {"status": "MOCK_DELTAS"}

    result = runner.prepare_deltas(
        manifest_path=manifest_path,
        delta_entry=fake_delta,
    )
    result["calls"] = calls
    return result


def test_prepare_writes_exactly_five_jobs_without_truth_fields(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == "matrix_manifest.json"
    assert len(manifest["jobs"]) == 5
    assert manifest["job_count"] == 5
    assert manifest["schedules"] == list(matrix.SCHEDULE_NAMES)
    encoded = json.dumps(manifest, ensure_ascii=False).lower()
    assert "truth_sidecar" not in encoded
    assert "truth_payload" not in encoded

    with pytest.raises(FileExistsError, match="manifest|output"):
        runner.prepare(method_lock_path=CONFIG, output_root=tmp_path / "matrix")


def test_smoke_uses_first_job_and_only_two_truth_free_schedules(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    delta_result = _prepare_deltas(manifest_path)
    calls: list[dict[str, object]] = []

    def fake_prediction(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"status": "MOCK_PREDICTION", "schedules": kwargs["schedules"]}

    receipt = runner.smoke(
        manifest_path=manifest_path,
        prediction_entry=fake_prediction,
        device="cpu",
    )

    assert receipt["job_id"] == json.loads(manifest_path.read_text())["jobs"][0]["job_id"]
    assert receipt["schedules"] == ["batch_5", "singleton_forward"]
    assert len(calls) == 1
    assert list(calls[0]["schedules"]) == ["batch_5", "singleton_forward"]
    assert calls[0]["prepared_delta_root"]
    assert "after_enrollment_package_root" not in calls[0]
    assert not any(
        any(token in str(key).lower() for token in ("truth", "score", "role", "quota"))
        for key in calls[0]
    )


def test_run_job_calls_all_four_schedules_once_without_truth(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    _prepare_deltas(manifest_path)
    calls: list[dict[str, object]] = []

    def fake_prediction(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"status": "MOCK_PREDICTION", "schedules": kwargs["schedules"]}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = runner.run_job(
        manifest_path=manifest_path,
        job_id=manifest["jobs"][0]["job_id"],
        prediction_entry=fake_prediction,
        device="cpu",
    )

    assert receipt["status"] == "PREDICTION_COMPLETE_TRUTH_FREE"
    assert receipt["schedules"] == list(matrix.SCHEDULE_NAMES)
    assert len(calls) == 1
    assert list(calls[0]["schedules"]) == list(matrix.SCHEDULE_NAMES)
    assert calls[0]["prepared_delta_root"]
    assert "after_enrollment_package_root" not in calls[0]


def test_prepare_deltas_calls_public_delta_builder_once_per_job(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    result = _prepare_deltas(manifest_path)

    assert result["status"] == "DELTAS_PREPARED"
    assert result["job_count"] == 5
    calls = result["calls"]
    assert len(calls) == 5
    assert all("before_enrollment_package_root" in call for call in calls)
    assert all("after_enrollment_package_root" in call for call in calls)
    assert all("after_apply_package_root" not in call for call in calls)
    assert all(
        not any(token in str(key).lower() for token in ("truth", "query", "score"))
        for call in calls
        for key in call
    )
    assert all(Path(str(call["output_root"])).is_dir() for call in calls)


def test_status_reports_only_technical_counts(tmp_path: Path) -> None:
    manifest_path = _prepare(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    job_root = Path(manifest["jobs"][0]["output_root"])
    (job_root / "batch_5").mkdir(parents=True)
    (job_root / "batch_5" / "prediction_artifact.npz").write_bytes(b"mock")
    (job_root / "job_receipt.json").write_text(
        json.dumps({"status": "PREDICTION_COMPLETE_TRUTH_FREE"}),
        encoding="utf-8",
    )

    status = runner.status(manifest_path=manifest_path)

    assert status["job_count"] == 5
    assert status["completed_job_count"] == 1
    assert status["prediction_artifact_count"] == 1
    assert "accuracy" not in json.dumps(status).lower()
    assert "truth" not in json.dumps(status).lower()


def test_cli_help_exposes_bounded_subcommands() -> None:
    parser = runner.build_parser()
    help_text = parser.format_help()
    for name in ("prepare", "prepare-deltas", "smoke", "run-job", "status"):
        assert name in help_text
