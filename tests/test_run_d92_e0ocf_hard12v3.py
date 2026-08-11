from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_d92_e0ocf_hard12v3 as runner

QUERY_ZERO_FIELDS = (
    "query_truth_access", "query_fit_access", "query_update_access", "query_selection_access",
    "query_role_oracle_access", "query_class_quota_access", "query_global_reassignment",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _full_manifest(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    output = tmp_path / "matrix"
    jobs: list[dict[str, object]] = []
    outer_keys = [
        runner.SMOKE_OUTER_KEY,
        "rx_7_14__seed_713105__k_1__new_20",
        *[f"outer_{outer_index:02d}" for outer_index in range(2, 12)],
    ]
    for outer_index in range(12):
        outer_key = outer_keys[outer_index]
        role = "liveness" if outer_index < 2 else "performance"
        k_shot = 1 if outer_index < 2 else 5
        for arm_index, arm_id in enumerate(runner.ARM_ORDER):
            root = output / "jobs" / outer_key / arm_id
            package = {
                f"{state}_{phase}": {
                    "package_root": str(root / "package"),
                    "detached_seal_path": str(root / f"{state}_{phase}.seal.json"),
                    "expected_seal_sha256": "a" * 64,
                }
                for state in ("before", "after")
                for phase in ("enrollment", "apply")
            }
            jobs.append({
                "index": len(jobs),
                "outer_index": outer_index,
                "arm_position": arm_index,
                "planned_shard_index": outer_index % 8,
                "job_id": f"{outer_key}__arm_{arm_id.lower()}",
                "outer_key": outer_key,
                "outer_role": role,
                "k_shot": k_shot,
                "arm_id": arm_id,
                "candidate": f"candidate_{arm_id.lower()}",
                "role": "primary" if arm_id == "E0_OCF25" else "diagnostic_only" if arm_id == "E0_OCF50" else "baseline",
                "packages": package,
                "truth_sidecar": str(root / "truth_sidecar.json"),
                "output_root": str(root),
            })
    manifest = {
        "schema": "cvs.phase2.d92_e0ocf_hard12v3.matrix.v1",
        "status": "FROZEN_DEVELOPMENT_MATRIX",
        "protocol_schema": "p2_min_v1",
        "selection_sha256": runner.CANONICAL_SELECTION_SHA256,
        "context_sha256": runner.CONTEXT_SHA256,
        "ground_component_dir": str(tmp_path / "ground"),
        "ground_manifest_sha256": "b" * 64,
        "output_root": str(output),
        "shard_count": 8,
        "job_count": 60,
        "arms": list(runner.ARM_ORDER),
        "primary_arm": runner.PRIMARY_ARM,
        "smoke_outer_key": runner.SMOKE_OUTER_KEY,
        "jobs": jobs,
    }
    manifest_path = tmp_path / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return manifest_path, digest, manifest


def _fake_child_run(command: list[str], **_: object) -> SimpleNamespace:
    if "run_d92_e0d_prediction.py" in str(command[1]):
        root = Path(command[command.index("--output-root") + 1])
        for state in ("before", "after"):
            state_root = root / state
            state_root.mkdir(parents=True, exist_ok=True)
            (state_root / "prediction_artifact.npz").write_bytes(f"{state}-prediction".encode())
            (state_root / "COMMIT.json").write_text("{}", encoding="utf-8")
        rows = [{
            "scenario": "leo_clear_weak",
            "query_truth_access": False,
            "query_fit_access": False,
            "query_update_access": False,
            "query_selection_access": False,
            "query_role_oracle_access": False,
            "query_class_quota_access": False,
            "query_global_reassignment": False,
        }]
        _write_json(root / "after" / "fit_audit.json", rows)
    else:
        path = Path(command[command.index("--output-path") + 1])
        _write_json(path, {"status": "PASS"})
    return SimpleNamespace(returncode=0)


def test_shared_stop_counts_distinct_outers_not_arms(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    output.mkdir()
    fp = "a" * 64
    first = {"job_id": "outer_a__arm_full", "outer_key": "outer_a", "arm_id": "D92_FULL"}
    same = {"job_id": "outer_a__arm_ocf25", "outer_key": "outer_a", "arm_id": "E0_OCF25"}
    second = {"job_id": "outer_b__arm_full", "outer_key": "outer_b", "arm_id": "D92_FULL"}
    assert runner._record_shared_pre_prediction_failure(output, first, fp) is False
    assert runner._record_shared_pre_prediction_failure(output, same, fp) is False
    assert runner._record_shared_pre_prediction_failure(output, second, fp) is True


def test_cli_parser_exposes_prepare_smoke_and_run_shard() -> None:
    parser = runner.parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0


def test_full_matrix_first_smoke_publishes_exact_shared_receipt(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", _fake_child_run)
    args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, output_root=str(Path(manifest["output_root"]) / "smoke"), device="cpu", cpu_threads=1)
    receipt = runner.truth_free_smoke(args)
    smoke_root = Path(str(manifest["output_root"])) / "smoke"
    assert (smoke_root / "smoke_receipt.json").is_file()
    assert receipt["matrix_manifest_sha256"] == digest
    assert receipt["selection_sha256"] == runner.CANONICAL_SELECTION_SHA256
    assert receipt["outer_key"] == runner.SMOKE_OUTER_KEY
    assert receipt["job_id"] == f"{runner.SMOKE_OUTER_KEY}__arm_d92_full"
    assert receipt["arm_id"] == "D92_FULL"
    assert receipt["k_shot"] == 1
    assert receipt["truth_open"] is False
    assert all(receipt[field] is False for field in QUERY_ZERO_FIELDS)


def test_full_matrix_run_shard_rejects_absent_or_tampered_smoke_before_events(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, shard_index=0, shard_count=8, device="cpu", cpu_threads=1)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("child dispatch must not occur")))
    with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="smoke"):
        runner.run_shard(args)
    smoke_root = Path(str(manifest["output_root"])) / "smoke"
    smoke_root.mkdir(parents=True)
    _write_json(smoke_root / "smoke_receipt.json", {"status": "tampered"})
    with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="smoke"):
        runner.run_shard(args)
    assert not (Path(str(manifest["output_root"])) / "events").exists()


def test_full_matrix_run_shard_accepts_valid_shared_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", _fake_child_run)
    smoke_args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, output_root=str(Path(manifest["output_root"]) / "smoke"), device="cpu", cpu_threads=1)
    runner.truth_free_smoke(smoke_args)
    shard_args = SimpleNamespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, shard_index=0, shard_count=8, device="cpu", cpu_threads=1)
    summary = runner.run_shard(shard_args)
    assert summary["status"] == "PASS"
    expected_ids = [job["job_id"] for job in manifest["jobs"] if job["planned_shard_index"] == 0]
    assert summary["selected_job_count"] == len(expected_ids) == 10
    assert summary["completed_job_ids"] == expected_ids


def test_full_manifest_rejects_missing_or_tampered_smoke_outer_key(tmp_path: Path) -> None:
    manifest_path, _, manifest = _full_manifest(tmp_path)
    for value in (None, "rx_7_14__seed_713105__k_1__new_20"):
        payload = dict(manifest)
        if value is None:
            payload.pop("smoke_outer_key", None)
        else:
            payload["smoke_outer_key"] = value
        _write_json(manifest_path, payload)
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        with pytest.raises(runner.D92E0OCFHard12V3RunnerError, match="smoke_outer_key"):
            runner._load_manifest(manifest_path, digest)
