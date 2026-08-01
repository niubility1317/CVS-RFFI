"""Focused matrix and truth-free input tests for D108 Target125."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cvsrffi import stage2_d108_target125_inputs as inputs
from cvsrffi.stage2_d108_matrix_protocol import (
    ARMS,
    CANDIDATE_ID,
    OUTER_JOB_COUNT,
    PHASES,
    SCENES,
    canonical_bytes,
    freeze_d108_target125_matrix,
)


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_json(path: Path, value: Any) -> str:
    raw = canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha_bytes(raw)


def _d92_job_id(receiver: str, seed: int, k_shot: int, new_count: int) -> str:
    return f"d92-rx-{receiver}__seed-{seed}__k-{k_shot}__new-{new_count}"


def _make_prepare_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    matrix = freeze_d108_target125_matrix()
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"d108-checkpoint")
    checkpoint_sha = _sha_bytes(checkpoint.read_bytes())
    output_root = tmp_path / "d92-output"
    jobs_root = output_root / "jobs"
    jobs_root.mkdir(parents=True)
    jobs: list[dict[str, Any]] = []
    for outer in matrix.outer_rows:
        job_id = _d92_job_id(outer.receiver, outer.seed, outer.k_shot, outer.new_count)
        job_root = jobs_root / job_id
        for phase in PHASES:
            for profile, leaf in (
                ("enrollment", "enrollment_only"),
                ("apply", "apply_only_staging"),
            ):
                (job_root / "offline" / "predictor" / phase / leaf).mkdir(
                    parents=True, exist_ok=True
                )
                seal_root = (
                    job_root / "offline" / "seals"
                    if profile == "enrollment"
                    else job_root / "apply_seals"
                )
                seal_root.mkdir(parents=True, exist_ok=True)
                seal_leaf = (
                    f"{phase}_enrollment.seal.json"
                    if profile == "enrollment"
                    else f"{phase}_apply.seal.json"
                )
                (seal_root / seal_leaf).write_bytes(
                    f"{job_id}:{phase}:{profile}".encode("utf-8")
                )
        authority = tmp_path / "authority" / job_id
        authority.mkdir(parents=True)
        authority_commit = authority / "COMMIT.json"
        authority_sha = _write_json(
            authority_commit,
            {"schema": "synthetic.d92.authority.v1", "job_id": job_id},
        )
        jobs.append(
            {
                "candidate": inputs.D92_CANDIDATE,
                "job_id": job_id,
                "receiver": outer.receiver,
                "seed": outer.seed,
                "k_shot": outer.k_shot,
                "new_class_count": outer.new_count,
                "output_root": str(job_root.resolve()),
                "scenarios": list(SCENES),
                "authority_bundle": str(authority.resolve()),
                "authority_commit_path": str(authority_commit.resolve()),
                "authority_commit_sha256": authority_sha,
            }
        )
    manifest = {
        "schema": inputs.D92_MATRIX_SCHEMA,
        "candidate": inputs.D92_CANDIDATE,
        "job_count": OUTER_JOB_COUNT,
        "receivers": ["20-1", "3-19", "7-14", "7-7", "8-8"],
        "phase1_checkpoint_sha256": checkpoint_sha,
        "sealed_runtime_sha256": _sha_bytes(b"d92-runtime"),
        "phase2_contract": {
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "phase2_clean_dataset_reachable": False,
            "phase2_query_batch_global_assignment": False,
            "phase2_query_class_quota_access": False,
            "phase2_query_role_oracle_access": False,
            "phase2_query_true_batch_class_count_access": False,
        },
        "jobs": jobs,
    }
    manifest_path = tmp_path / "d92_manifest.json"
    manifest_sha = _write_json(manifest_path, manifest)
    ground = tmp_path / "int8_component"
    ground.mkdir()
    ground_sha = _write_json(ground / "manifest.json", {"component": "synthetic-d19"})
    monkeypatch.setattr(inputs, "D92_MATRIX_MANIFEST_SHA256", manifest_sha)
    monkeypatch.setattr(inputs, "D19_GROUND_MANIFEST_SHA256", ground_sha)
    method_lock = {
        "schema": inputs.D108_METHOD_LOCK_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "protocol_schema": "p2_min_v1",
        "d92_matrix_manifest_sha256": manifest_sha,
        "feature_view": "d92_288d_relu160_fft96_rf32",
        "feature_width": 288,
        "arms": list(ARMS),
        "ground_component": {
            "path": inputs.D19_GROUND_COMPONENT_PATH,
            "manifest_sha256": ground_sha,
        },
    }
    lock_path = tmp_path / "d108_lock.json"
    lock_sha = _write_json(lock_path, method_lock)
    return {
        "d92_matrix_manifest_path": manifest_path.resolve(),
        "expected_d92_matrix_manifest_sha256": manifest_sha,
        "d92_output_root": output_root.resolve(),
        "checkpoint_path": checkpoint.resolve(),
        "expected_checkpoint_sha256": checkpoint_sha,
        "d108_method_lock_path": lock_path.resolve(),
        "expected_d108_method_lock_sha256": lock_sha,
        "ground_component_dir": ground.resolve(),
        "expected_ground_manifest_sha256": ground_sha,
        "source_jobs": jobs,
    }


def test_target125_matrix_counts_ids_and_fixed_four_arms() -> None:
    matrix = freeze_d108_target125_matrix()
    assert len(matrix.outer_rows) == 125
    assert len(matrix.scene_rows) == 375
    assert len(matrix.arm_pairs) == 1500
    assert len(matrix.surfaces) == 3000
    assert matrix.outer_rows[0].outer_id == "d108-rx-20-1__seed-713102__k-10__new-5"
    assert tuple(surface.arm for surface in matrix.surfaces[:8]) == (
        "M0",
        "M0",
        "M_DA",
        "M_DA",
        "M_HEAD",
        "M_HEAD",
        "M_JOINT",
        "M_JOINT",
    )
    assert "ROUTED" not in ARMS


def test_prepare_binds_d19_ground_and_every_source_authority_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _make_prepare_inputs(tmp_path, monkeypatch)
    jobs = arguments.pop("source_jobs")
    result = inputs.prepare_d108_target125_inputs(
        **arguments, output_dir=tmp_path / "prepared"
    )
    assert result["outer_job_count"] == 125
    plan = json.loads(Path(result["plan_manifest"]).read_text(encoding="utf-8"))
    assert set(plan["identity"]["ground_component"]) == {
        "directory",
        "manifest_path",
        "manifest_sha256",
    }
    assert plan["identity"]["ground_component"]["manifest_sha256"] == arguments[
        "expected_ground_manifest_sha256"
    ]
    assert all(set(row["authority_bundle"]) == {"directory", "commit_path", "commit_sha256"} for row in plan["rows"])
    rows = {
        (row["receiver"], row["seed"], row["k_shot"], row["new_count"]): row
        for row in plan["rows"]
    }
    for receiver in ("20-1", "3-19", "7-14", "7-7", "8-8"):
        for seed in (713102, 713103, 713104, 713105, 713106):
            k5 = rows[(receiver, seed, 5, 20)]
            k10 = rows[(receiver, seed, 10, 20)]
            assert k5["source_d92_job_id"] == k10["source_d92_job_id"]
            assert k5["authority_bundle"] == k10["authority_bundle"]
    assert len(jobs) == 125


def test_prepare_rejects_authority_commit_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _make_prepare_inputs(tmp_path, monkeypatch)
    jobs = arguments.pop("source_jobs")
    Path(jobs[0]["authority_commit_path"]).write_bytes(b"tampered")
    with pytest.raises(inputs.D108Target125InputError, match="COMMIT SHA mismatch"):
        inputs.prepare_d108_target125_inputs(
            **arguments, output_dir=tmp_path / "prepared"
        )


def test_release_lock_pins_real_d92_and_d19_assets() -> None:
    lock = json.loads(
        Path("configs/stage2_d108_cbrrc_smme_r1.json").read_text(encoding="utf-8")
    )
    assert lock["candidate_id"] == CANDIDATE_ID
    assert lock["d92_matrix_manifest_sha256"] == (
        "b70045e7cd45a6029bc0a1a47ada0bb72d16fdb6bc7662c43bd253bfc7e4bc5c"
    )
    assert lock["ground_component"] == {
        "path": inputs.D19_GROUND_COMPONENT_PATH,
        "manifest_sha256": (
            "15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c"
        ),
        "basis_loader": "D81_ground_basis_from_sealed_D19_int8_component",
    }
