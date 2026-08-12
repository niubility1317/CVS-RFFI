from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from cvsrffi.stage2_d92_tpce_hard11 import (
    ARM_ID,
    ARM_ORDER,
    CANDIDATE_ID,
    FIT_GATE,
    HARD11_ROWS,
    LIVENESS_OUTER_KEY,
    RESOURCE_GATE,
    SCENES,
    SHARD_COUNT,
    SMOKE_OUTER_KEY,
    build_hard11_manifest,
    canonical_selection_sha256,
    validate_hard11_manifest,
    validate_method_lock,
)


CONTEXT = Path(
    r"E:\\type10-7\\automation_reports\\CV-SincNet\\d108_cbrrc_smme_target125_20260801_r3\\artifacts\\remote_r1\\prepared\\target125_context.json"
)
METHOD_LOCK = Path("configs/stage2_d92_full_d42_tpce_hard11_v1.json")


def test_tpce_identity_and_fit_gate() -> None:
    assert ARM_ID == "E0_FULL_D42_TAIL_PAIR_CODE_EXCHANGE"
    assert CANDIDATE_ID == "d92_e0_full_d42_tail_pair_code_exchange"
    assert ARM_ORDER == (ARM_ID,)
    assert len(HARD11_ROWS) == 11
    assert sum(row["role"] == "performance" for row in HARD11_ROWS) == 10
    assert HARD11_ROWS[-1]["outer_key"] == LIVENESS_OUTER_KEY
    assert SHARD_COUNT == 8
    assert SCENES == ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
    assert SMOKE_OUTER_KEY == HARD11_ROWS[0]["outer_key"]
    assert FIT_GATE == {"k_gt_2_total": 2, "k_gt_2_actual": 1, "k1_alias": "real_inventory"}
    assert RESOURCE_GATE["registration_wall_p90_max_ns"] == 150_000_000


def test_manifest_expands_to_frozen_hard11(tmp_path: Path) -> None:
    manifest = build_hard11_manifest(
        context_path=CONTEXT,
        method_lock_path=METHOD_LOCK,
        output_root=tmp_path / "matrix",
        require_package_files=False,
    )
    assert manifest["schema"] == "cvs.phase2.d92_tpce_hard11.matrix.v1"
    assert manifest["arms"] == [ARM_ID]
    assert manifest["candidate_ids"] == {ARM_ID: CANDIDATE_ID}
    assert manifest["job_count"] == 11
    assert manifest["scene_arm_count"] == 33
    assert manifest["shard_count"] == 8
    assert {tuple(job["scenarios"]) for job in manifest["jobs"]} == {SCENES}
    assert validate_hard11_manifest(manifest)["job_count"] == 11


def test_method_lock_rejects_tpce_identity_drift() -> None:
    lock = json.loads(METHOD_LOCK.read_text(encoding="utf-8"))
    assert lock["arms"][ARM_ID]["registered_mode"] == "full_only"
    assert lock["state_postprocess_mode"] == "d42_tpce"
    assert lock["deployment_policy"]["codec"] == "post_compile_direct_coef2_qint8_pair_exchange"
    assert lock["deployment_policy"]["selection"] == "synchronous_int32_sum_then_single_boundary_check"
    assert lock["deployment_policy"]["quantization"] == "no_requantize_no_scan"
    assert validate_method_lock(lock)["primary_arm"] == ARM_ID
    with pytest.raises(ValueError):
        validate_method_lock({**lock, "primary_arm": "wrong"})


def test_selection_hash_is_frozen() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", canonical_selection_sha256())
