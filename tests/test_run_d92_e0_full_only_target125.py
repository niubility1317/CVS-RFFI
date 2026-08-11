from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_d92_e0_full_only_target125 as runner


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d108_cbrrc_smme_target125_20260801_r3\artifacts\remote_r1\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_e0_full_only_target125_v1.json").resolve()
SCENES = runner.SCENES


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_prediction_closure(root: Path) -> None:
    for state in ("before", "after"):
        state_root = root / state
        state_root.mkdir(parents=True, exist_ok=True)
        tokens = np.asarray(["old_clear", "old_low", "old_rain"], dtype="U16")
        scenarios = np.asarray(SCENES, dtype="U24")
        if state == "after":
            tokens = np.asarray(["old_clear", "old_low", "old_rain", "new_clear"], dtype="U16")
            scenarios = np.asarray([*SCENES, SCENES[0]], dtype="U24")
        np.savez(
            state_root / "prediction_artifact.npz",
            query_tokens=tokens,
            scenarios=scenarios,
            predicted_class_handles=np.asarray([f"pred_{i}" for i in range(len(tokens))]),
        )
        rows = [
            {"scenario": scene, **{field: False for field in runner.QUERY_ZERO_FIELDS}}
            for scene in SCENES
        ]
        _write_json(state_root / "fit_audit.json", rows)
        _write_json(state_root / "resource_audit.json", {"state": state})
        _write_json(state_root / "execution_receipt.json", {"schema": "cvs.phase2.diag_cosine_exploration_receipt.v1"})
        names = ("execution_receipt.json", "fit_audit.json", "prediction_artifact.npz", "resource_audit.json")
        members = [{"relative_path": name, "sha256": _sha256(state_root / name), "size_bytes": (state_root / name).stat().st_size} for name in names]
        _write_json(
            state_root / "COMMIT.json",
            {
                "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
                "members": members,
                "artifact_root_sha256": hashlib.sha256(_canonical_bytes(members)).hexdigest(),
                "execution_receipt_sha256": members[0]["sha256"],
                "prediction_artifact_sha256": members[2]["sha256"],
            },
        )


def _fake_child_run(command: list[str], **_: object) -> SimpleNamespace:
    if "run_d92_e0d_prediction.py" in str(command[1]):
        _write_prediction_closure(Path(command[command.index("--output-root") + 1]))
    else:
        _write_json(Path(command[command.index("--output-path") + 1]), {"status": "PASS"})
    return SimpleNamespace(returncode=0)


def _full_manifest(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    manifest = runner.build_target125_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    for job in manifest["jobs"]:
        for package in job["packages"].values():
            package["expected_seal_sha256"] = "a" * 64
    path = tmp_path / "matrix_manifest.json"
    _write_json(path, manifest)
    return path, _sha256(path), manifest


def test_cli_exposes_prepare_smoke_and_run_shard() -> None:
    parser = runner.parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0


def test_run_shard_rejects_without_valid_first_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("dispatch before smoke")))
    with pytest.raises(runner.Target125RunnerError, match="smoke"):
        runner.run_shard(argparse.Namespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, shard_index=0, shard_count=8, device="cpu", cpu_threads=1))
    assert not (Path(str(manifest["output_root"])) / "events").exists()


def test_full_matrix_smoke_then_normal_shard_dispatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", _fake_child_run)
    smoke = runner.truth_free_smoke(argparse.Namespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, output_root=str(Path(str(manifest["output_root"])) / "smoke"), device="cpu", cpu_threads=1))
    assert smoke["arm_id"] == runner.ARM_ID
    summary = runner.run_shard(argparse.Namespace(matrix_manifest=str(manifest_path), matrix_manifest_sha256=digest, shard_index=0, shard_count=8, device="cpu", cpu_threads=1))
    expected = [job["job_id"] for job in manifest["jobs"] if job["planned_shard_index"] == 0]
    assert summary["status"] == "PASS"
    assert summary["selected_job_count"] == len(expected)
    assert summary["completed_job_ids"] == expected


def test_shared_failure_stops_after_two_distinct_outers(tmp_path: Path) -> None:
    output = tmp_path / "matrix"
    fingerprint = "a" * 64
    assert runner._record_shared_pre_prediction_failure(output, {"job_id": "a", "outer_key": "outer_a", "arm_id": runner.ARM_ID}, fingerprint) is False
    assert runner._record_shared_pre_prediction_failure(output, {"job_id": "b", "outer_key": "outer_a", "arm_id": runner.ARM_ID}, fingerprint) is False
    assert runner._record_shared_pre_prediction_failure(output, {"job_id": "c", "outer_key": "outer_b", "arm_id": runner.ARM_ID}, fingerprint) is True
    assert runner._shared_systemic_stop_path(output).is_file()


def test_closure_reuses_proven_e0ocf_validator(tmp_path: Path) -> None:
    root = tmp_path / "diag"
    _write_prediction_closure(root)
    assert runner._prediction_closure_status(root) == ("closed", "closed")

