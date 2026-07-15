from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from paper_reproduction.scripts.authorize_cvs_stage2c_effective8_strict_plan import authorize
from paper_reproduction.scripts.build_cvs_stage2c_effective8_strict_plan import generate_strict_plan
from paper_reproduction.scripts.run_cvs_stage2c_effective8_strict_plan import run_matrix_shard


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "paper_reproduction/configs/cvs_stage2c_effective8_formal_matrix_20260715.json"


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def test_matrix_rejects_pre_smoke_manifest(tmp_path: Path) -> None:
    plan = generate_strict_plan(
        PLAN, out_dir=tmp_path / "strict", runtime_project_root="/srv/CV-SincNet",
        runtime_artifact_root="/srv/CV-SincNet/runs/v14/runtime_artifacts",
        expected_candidate_capsule_sha256="d" * 64,
    )
    with pytest.raises(RuntimeError, match="fail-closed"):
        run_matrix_shard(
            plan, project_root=tmp_path, device="cuda:0", shard_index=0,
            shard_count=8, log_dir=tmp_path / "logs", state_path=tmp_path / "state.json",
        )


def test_authority_requires_bound_three_scenario_protocol_valid_smoke(tmp_path: Path) -> None:
    plan = generate_strict_plan(
        PLAN, out_dir=tmp_path / "strict", runtime_project_root="/srv/CV-SincNet",
        runtime_artifact_root="/srv/CV-SincNet/runs/v14/runtime_artifacts",
        expected_candidate_capsule_sha256="e" * 64,
    )
    cell = {
        "status": "PROTOCOL_VALID", "package_id": plan["smoke_package_id"],
        "k_shot": 1, "candidate_capsule_sha256": "e" * 64,
        "formal_scenario_row_count": 3,
    }
    smoke = {
        "schema": "cvs.stage2c.effective8.n607_landlock_smoke.v1",
        "status": "PASS", "matrix_launch_authority_recommended": True,
        "candidate_capsule_sha256": "e" * 64,
        "package_id": plan["smoke_package_id"], "k_shot": 1,
        "cell_receipt": cell,
        "cell_receipt_sha256": hashlib.sha256(_canonical(cell)).hexdigest(),
    }
    authorized = authorize(plan, smoke)
    assert authorized["launch_authority"] is True
    assert authorized["authority_state"] == "N607_LANDLOCK_SMOKE_PASS"
    tampered = dict(smoke)
    tampered["cell_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="not authority-bearing"):
        authorize(plan, tampered)
