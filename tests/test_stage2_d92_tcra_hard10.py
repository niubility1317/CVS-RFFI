from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "code") not in sys.path:
    sys.path.insert(0, str(ROOT / "code"))

from cvsrffi.stage2_d92_tcra_hard10 import (  # noqa: E402
    ARM_ID,
    ARM_ORDER,
    CANDIDATE_ID,
    CLAIM_SCOPE,
    HARD10_ROWS,
    LIVENESS_OUTER_KEY,
    SCENES,
    SHARD_COUNT,
    SMOKE_OUTER_KEY,
    build_hard10_manifest,
    canonical_selection_sha256,
    validate_hard10_manifest,
    validate_method_lock,
)


METHOD_LOCK = ROOT / "configs" / "stage2_d92_tcra_safe_v2_hard10_v1.json"


def test_hard10_identity_excludes_g0_and_keeps_k1_liveness() -> None:
    assert ARM_ID == "E0_FULL_D42_TAIL_CLASS_ROW_ASCENT"
    assert CANDIDATE_ID == "d92_e0_full_d42_tail_class_row_ascent"
    assert ARM_ORDER == (ARM_ID,)
    assert tuple(row["outer_key"] for row in HARD10_ROWS[:-1]) == (
        "rx_7_7__seed_713104__k_5__new_20",
        "rx_7_7__seed_713103__k_10__new_5",
        "rx_8_8__seed_713103__k_5__new_20",
        "rx_8_8__seed_713103__k_10__new_5",
        "rx_8_8__seed_713106__k_5__new_20",
        "rx_7_14__seed_713104__k_10__new_10",
        "rx_3_19__seed_713102__k_10__new_5",
        "rx_7_7__seed_713105__k_10__new_20",
        "rx_7_7__seed_713104__k_10__new_5",
    )
    assert HARD10_ROWS[-1]["outer_key"] == LIVENESS_OUTER_KEY
    assert sum(row["role"] == "performance" for row in HARD10_ROWS) == 9
    assert sum(row["role"] == "liveness" for row in HARD10_ROWS) == 1
    assert SMOKE_OUTER_KEY == HARD10_ROWS[0]["outer_key"]
    assert SHARD_COUNT == 8
    assert SCENES == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    assert CLAIM_SCOPE == "DEVELOPMENT_ONLY_DISJOINT_FROM_G0_HARD_SCREEN"
    assert re.fullmatch(r"[0-9a-f]{64}", canonical_selection_sha256())


def test_manifest_has_10_jobs_and_rejects_excluded_row(tmp_path: Path) -> None:
    manifest = build_hard10_manifest(
        context_path=tmp_path / "context.json",
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["outer_count"] == 10
    assert manifest["performance_outer_count"] == 9
    assert manifest["liveness_outer_count"] == 1
    assert manifest["job_count"] == 10
    assert manifest["scene_count"] == 3
    assert manifest["scene_arm_count"] == 30
    assert manifest["shard_count"] == 8
    assert manifest["smoke_outer_key"] == SMOKE_OUTER_KEY
    assert all("rx_7_7__seed_713106__k_10__new_5" not in job["outer_key"] for job in manifest["jobs"])
    assert validate_hard10_manifest(manifest)["job_count"] == 10
    with pytest.raises(ValueError):
        validate_hard10_manifest({**manifest, "job_count": 11})


def test_method_lock_freezes_tcra_receipt_and_direction_gates() -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["selection_sha256"] == canonical_selection_sha256()
    assert lock["state_postprocess_mode"] == "d42_tcra"
    assert lock["gate"]["revision"] == "safe_directional_v2"
    assert lock["fit_gate"]["k_gt_2_total"] == 2
    assert lock["fit_gate"]["k_gt_2_actual"] == 1
    assert lock["fit_gate"]["postprocess_fit"] == 0
    assert lock["fit_gate"]["k1_total"] == 3
    assert lock["fit_gate"]["k1_actual"] == 3
    assert lock["fit_gate"]["k1_alias"] == "real_inventory"
    assert lock["query_contract"] == {
        "decision": "per_sample_all_registered_classes",
        "truth_access": False,
        "fit_access": False,
        "update_access": False,
        "selection_access": False,
        "role_oracle_access": False,
        "class_quota_access": False,
        "global_reassignment": False,
    }
    assert validate_method_lock(lock)["claim_scope"] == CLAIM_SCOPE
