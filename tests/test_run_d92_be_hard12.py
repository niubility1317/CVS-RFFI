from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

from scripts import run_d92_be_hard12 as runner


def _write_readonly(path: Path, data: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, stat.S_IREAD)


def _manifest(tmp_path: Path) -> tuple[Path, str, dict]:
    output = tmp_path / "matrix"
    truth = tmp_path / "source" / "truth_sidecar.json"
    truth.parent.mkdir(parents=True)
    truth.write_text("{}", encoding="utf-8")
    packages = {}
    for name in (
        "before_enrollment",
        "before_apply",
        "after_enrollment",
        "after_apply",
    ):
        packages[name] = {
            "package_root": str(tmp_path / "source" / name),
            "detached_seal_path": str(tmp_path / "source" / f"{name}.seal.json"),
            "expected_seal_sha256": name[0] * 64,
        }
    job = {
        "index": 0,
        "planned_shard_index": 0,
        "job_id": "rx_3_19__seed_713104__k_1__new_20__arm_full",
        "outer_key": "rx_3_19__seed_713104__k_1__new_20",
        "outer_role": "liveness",
        "receiver": "3-19",
        "seed": 713104,
        "k_shot": 1,
        "new_class_count": 20,
        "arm_id": "FULL",
        "candidate": "d92_be_full",
        "packages": packages,
        "truth_sidecar": str(truth),
        "output_root": str(output / "jobs" / "outer" / "FULL"),
    }
    payload = {
        "schema": "cvs.phase2.d92_be_hard12.matrix.v1",
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "protocol_schema": "p2_min_v1",
        "selection_sha256": runner.CANONICAL_SELECTION_SHA256,
        "context_sha256": runner.CONTEXT_SHA256,
        "method_lock_sha256": "f" * 64,
        "ground_component_dir": str(tmp_path / "ground"),
        "ground_manifest_sha256": "e" * 64,
        "output_root": str(output),
        "shard_count": 8,
        "job_count": 1,
        "jobs": [job],
    }
    manifest = tmp_path / "matrix_manifest.json"
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    manifest.write_bytes(raw)
    return manifest, hashlib.sha256(raw).hexdigest(), job


def test_prediction_and_score_commands_keep_truth_out_of_predictor(tmp_path: Path):
    _manifest_path, _digest, job = _manifest(tmp_path)
    prediction = runner._prediction_command(
        job,
        ground_component_dir="ground",
        ground_manifest_sha256="e" * 64,
        device="cuda:0",
    )
    score = runner._score_command(job)
    assert all("truth" not in token.lower() for token in prediction)
    assert "--truth-sidecar" in score
    assert all("enrollment" not in token and "apply-package" not in token for token in score)
    assert prediction.index("--arm") < prediction.index("--output-root")


def test_run_shard_commits_prediction_before_starting_scorer(monkeypatch, tmp_path: Path):
    manifest, digest, job = _manifest(tmp_path)
    calls: list[list[str]] = []

    def fake_subprocess(command, **_kwargs):
        calls.append(list(command))
        if command[1].endswith("run_d92_be_prediction.py"):
            output = Path(command[command.index("--output-root") + 1])
            for state in ("before", "after"):
                _write_readonly(output / state / "prediction_artifact.npz")
                _write_readonly(output / state / "COMMIT.json")
        else:
            assert (Path(job["output_root"]) / "diag" / "before" / "COMMIT.json").is_file()
            assert (Path(job["output_root"]) / "diag" / "after" / "COMMIT.json").is_file()
            output = Path(command[command.index("--output-path") + 1])
            _write_readonly(output, b"{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess)
    result = runner.run_shard(
        argparse.Namespace(
            matrix_manifest=str(manifest),
            matrix_manifest_sha256=digest,
            shard_index=0,
            shard_count=8,
            device="cuda:0",
            cpu_threads=2,
        )
    )
    assert result["status"] == "PASS"
    assert result["completed_job_count"] == 1
    assert len(calls) == 2
    assert calls[0][1].endswith("run_d92_be_prediction.py")
    assert calls[1][1].endswith("score_d92_be_prediction.py")
    receipt = json.loads(
        (Path(job["output_root"]) / "job_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["truth_sidecar_exposed_to_predictor"] is False
    assert receipt["query_truth_joined_only_after_immutable_predictions"] is True


def test_run_shard_refuses_existing_job_output_without_launch(monkeypatch, tmp_path: Path):
    manifest, digest, job = _manifest(tmp_path)
    Path(job["output_root"]).mkdir(parents=True)
    called = []
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: called.append(True),
    )
    result = runner.run_shard(
        argparse.Namespace(
            matrix_manifest=str(manifest),
            matrix_manifest_sha256=digest,
            shard_index=0,
            shard_count=8,
            device="cuda:0",
            cpu_threads=2,
        )
    )
    assert result["status"] == "PARTIAL_FAILURE"
    assert result["failed_job_count"] == 1
    assert called == []
