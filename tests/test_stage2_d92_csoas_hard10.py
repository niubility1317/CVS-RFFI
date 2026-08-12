from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi import stage2_d92_csoas_hard10 as matrix  # noqa: E402


METHOD_LOCK = ROOT / "configs" / "stage2_d92_csoas_hard10_v1.json"
G0_OUTER = "rx_7_7__seed_713106__k_10__new_5"


def test_csoas_hard10_identity_is_hard9_plus_k1_and_new_selection() -> None:
    assert matrix.ARM_ID == "E0_FULL_CSOAS"
    assert matrix.CANDIDATE_ID == "d92_e0_full_csoas"
    assert matrix.REGISTERED_MODE == "csoas_full"
    assert matrix.CLAIM_SCOPE == "DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN"
    assert matrix.SCENES == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    assert matrix.SHARD_COUNT == 8
    assert G0_OUTER in matrix.EXCLUDED_OUTER_KEYS
    assert all(row["outer_key"] != G0_OUTER for row in matrix.HARD10_ROWS)
    assert sum(row["role"] == "performance" for row in matrix.HARD10_ROWS) == 9
    assert matrix.HARD10_ROWS[-1]["role"] == "liveness"
    assert matrix.HARD10_ROWS[-1]["outer_key"] == matrix.LIVENESS_OUTER_KEY
    assert matrix.canonical_selection_sha256() == matrix.CANONICAL_SELECTION_SHA256
    assert matrix.CANONICAL_SELECTION_SHA256 != "4fc836fbe3960cf95bfdf9db9eba1d311fb47fa4cc2ff89b64acab7e88f8e61"


def test_csoas_manifest_has_ten_jobs_and_new_schema(tmp_path: Path) -> None:
    manifest = matrix.build_hard10_manifest(
        context_path=tmp_path / "context.json",
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["schema"] == "cvs.phase2.d92_csoas_hard10.matrix.v1"
    assert manifest["job_count"] == 10
    assert manifest["performance_outer_count"] == 9
    assert manifest["liveness_outer_count"] == 1
    assert manifest["scene_arm_count"] == 30
    assert manifest["shard_count"] == 8
    assert manifest["candidate_ids"] == {"E0_FULL_CSOAS": "d92_e0_full_csoas"}
    assert all(job["arm_id"] == "E0_FULL_CSOAS" for job in manifest["jobs"])
    assert all(G0_OUTER not in job["outer_key"] for job in manifest["jobs"])
    assert matrix.validate_hard10_manifest(manifest)["job_count"] == 10


def test_csoas_method_lock_freezes_protocol_fit_and_resource_contract() -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["schema"] == "cvs.phase2.d92_csoas_hard10.method_lock.v1"
    assert lock["registered_mode"] == "csoas_full"
    assert lock["fit_gate"]["k_gt_2_total"] == 2
    assert lock["fit_gate"]["k_gt_2_actual"] == 1
    assert lock["fit_gate"]["postprocess_fit"] == 0
    assert lock["query_contract"]["truth_access"] is False
    assert lock["resource_gate"]["registration_wall_p90_max_ns"] == 150_000_000
    assert lock["resource_gate"]["registration_wall_ratio_max"] == 1.5
    assert matrix.validate_method_lock(lock)["claim_scope"] == matrix.CLAIM_SCOPE


def test_csoas_manifest_rejects_excluded_outer_and_count_drift(tmp_path: Path) -> None:
    manifest = matrix.build_hard10_manifest(
        context_path=tmp_path / "context.json",
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    jobs = list(manifest["jobs"])
    jobs[0] = {**jobs[0], "outer_key": G0_OUTER}
    with pytest.raises(ValueError):
        matrix.validate_hard10_manifest({**manifest, "jobs": jobs})
    with pytest.raises(ValueError):
        matrix.validate_hard10_manifest({**manifest, "job_count": 11})
