from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_d92_newguard_hard11 as runner


CONTEXT = Path(
    r"E:\type10-7\automation_reports\CV-SincNet\d108_cbrrc_smme_target125_20260801_r3\artifacts\remote_r1\prepared\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_full_bidirectional_newguard_hard11_v1.json").resolve()
SCENES = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _full_manifest(tmp_path: Path) -> tuple[Path, str, dict[str, object]]:
    output = tmp_path / "matrix"
    manifest = runner.build_hard11_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=output,
        require_package_files=False,
    )
    for job in manifest["jobs"]:
        for package in job["packages"].values():
            package["expected_seal_sha256"] = "a" * 64
    manifest_path = tmp_path / "matrix_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest_path, _sha256(manifest_path), manifest


def _write_prediction_closure(root: Path) -> None:
    k1 = "__k_1__" in str(root)
    mode = "d92_full_alias" if k1 else "newguard_maxmin"
    total, actual = (3, 3) if k1 else (2, 1)
    for state in ("before", "after"):
        state_root = root / state
        state_root.mkdir(parents=True, exist_ok=True)
        np.savez(
            state_root / "prediction_artifact.npz",
            query_tokens=np.asarray(["old_clear", "old_low", "old_rain"]),
            scenarios=np.asarray(SCENES),
            predicted_class_handles=np.asarray(["pred_0", "pred_1", "pred_2"]),
        )
        _write_json(state_root / "fit_audit.json", [{
            "scenario": scene,
            "arm_id": runner.ARM_ID,
            "candidate_id": runner.CANDIDATE_ID,
            "after_registered_d_mode_effective": mode,
            "after_total_component_fit_count": total,
            "after_actual_component_inventory": {"actual_component_fit_count": actual},
            "after_registration_resource": {"registration_wall_time_ns": 100.0, "registration_incremental_peak_working_set_bytes": 100.0},
            "query_macs": 7488,
            "after_state_bytes": 18503,
            **{field: False for field in runner.QUERY_ZERO_FIELDS},
        } for scene in SCENES])
        _write_json(state_root / "execution_receipt.json", {"status": "PASS"})
        _write_json(state_root / "resource_audit.json", {"status": "PASS"})
        names = ("execution_receipt.json", "fit_audit.json", "prediction_artifact.npz", "resource_audit.json")
        members = [{"relative_path": name, "sha256": _sha256(state_root / name), "size_bytes": (state_root / name).stat().st_size} for name in names]
        artifact_root_sha = hashlib.sha256(json.dumps(members, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        _write_json(state_root / "COMMIT.json", {
            "schema": "cvs.phase2.diag_cosine_exploration_commit.v1",
            "members": members,
            "artifact_root_sha256": artifact_root_sha,
            "execution_receipt_sha256": members[0]["sha256"],
            "prediction_artifact_sha256": members[2]["sha256"],
        })


def _fake_child_run(command: list[str], **_: object) -> SimpleNamespace:
    if "run_d92_e0d_prediction.py" in str(command[1]):
        root = Path(command[command.index("--output-root") + 1])
        _write_prediction_closure(root)
    else:
        path = Path(command[command.index("--output-path") + 1])
        _write_json(path, {"status": "PASS"})
    return SimpleNamespace(returncode=0)


def test_cli_parser_exposes_commands() -> None:
    parser = runner.parser()
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--help"])
    assert error.value.code == 0


def test_k_gt_2_smoke_precedes_shards_and_k1_alias_is_liveness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", _fake_child_run)
    monkeypatch.setattr(runner, "_verify_manifest_artifacts", lambda *_a: None)
    smoke = runner.truth_free_smoke(
        SimpleNamespace(
            matrix_manifest=str(manifest_path),
            matrix_manifest_sha256=digest,
            output_root=str(Path(manifest["output_root"]) / "smoke"),
            device="cpu",
            cpu_threads=1,
        )
    )
    assert smoke["outer_key"] == runner.SMOKE_OUTER_KEY
    assert smoke["k_shot"] > 2
    assert smoke["outer_role"] == "performance"
    assert smoke["truth_open"] is False
    completed: list[str] = []
    for shard in range(8):
        summary = runner.run_shard(
            SimpleNamespace(
                matrix_manifest=str(manifest_path),
                matrix_manifest_sha256=digest,
                shard_index=shard,
                shard_count=8,
                device="cpu",
                cpu_threads=1,
            )
        )
        assert summary["status"] == "PASS"
        completed.extend(summary["completed_job_ids"])
    assert len(completed) == 11
    assert len(set(completed)) == 11
    assert any(job["k_shot"] == 1 and job["outer_role"] == "liveness" for job in manifest["jobs"])


def test_runner_refuses_tampered_smoke_before_prediction_dispatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    manifest_path, digest, manifest = _full_manifest(tmp_path)
    smoke_root = Path(manifest["output_root"]) / "smoke"
    smoke_root.mkdir(parents=True)
    _write_json(smoke_root / "smoke_receipt.json", {"status": "tampered"})
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    monkeypatch.setattr(runner, "_verify_manifest_artifacts", lambda *_a: None)
    with pytest.raises(runner.D92NewGuardHard11RunnerError, match="smoke"):
        runner.run_shard(
            SimpleNamespace(
                matrix_manifest=str(manifest_path),
                matrix_manifest_sha256=digest,
                shard_index=0,
                shard_count=8,
                device="cpu",
                cpu_threads=1,
            )
        )


def test_fit_audit_gate_requires_newguard_mode_inventory_and_query_zero(tmp_path: Path) -> None:
    path = tmp_path / "fit_audit.json"
    rows = []
    for scene in SCENES:
        rows.append({
            "scenario": scene,
            "arm_id": runner.ARM_ID,
            "candidate_id": runner.CANDIDATE_ID,
            "after_registered_d_mode_effective": "wrong_mode",
            "after_total_component_fit_count": 4,
            "after_actual_component_inventory": {"actual_component_fit_count": 2},
            **{field: False for field in runner.QUERY_ZERO_FIELDS},
        })
    _write_json(path, rows)
    with pytest.raises(runner.D92NewGuardHard11RunnerError, match="fit audit"):
        runner._validate_fit_audit(path, k_shot=10)


def test_manifest_artifact_hashes_are_checked_on_real_paths(tmp_path: Path) -> None:
    package_root = tmp_path / "package"
    package_root.mkdir()
    seal = tmp_path / "seal.json"
    truth = tmp_path / "truth_sidecar.json"
    seal.write_text("seal", encoding="utf-8")
    truth.write_text("{}", encoding="utf-8")
    job = {
        "source_job_root": str(tmp_path),
        "truth_sidecar": str(truth),
        "packages": {"before_enrollment": {"package_root": str(package_root), "detached_seal_path": str(seal), "expected_seal_sha256": _sha256(seal)}},
    }
    runner._verify_manifest_artifacts({"jobs": [job]})
    seal.write_text("tampered", encoding="utf-8")
    with pytest.raises(runner.D92NewGuardHard11RunnerError, match="SHA"):
        runner._verify_manifest_artifacts({"jobs": [job]})
