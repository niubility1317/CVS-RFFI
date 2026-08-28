from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "code" / "scripts" / "run_sf_r5d92_adapt.py"
_SPEC = importlib.util.spec_from_file_location("run_sf_r5d92_adapt", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_build_adapt_config = _MODULE._build_adapt_config


def test_build_r5d92_adapt_config_reuses_data_plan_and_locks_e0() -> None:
    plan = {
        "schema": "cvs.sf_d3_erbt.plan.v1",
        "run_id": "fixture",
        "capsule_id": "capsule",
        "base_checkpoint_path": "/checkpoint.pth",
        "phase1_bundle": {},
        "d3_config": {},
        "scenes": {
            "leo_clear_weak": {
                "gpu": 0,
                "split_id": "split",
                "old_support_input": "/unused-old.npz",
                "registered_support_input": "/unused-registered.npz",
                "query_input": "/unused-query.npz",
                "old_support_output": "/old.npz",
                "registered_support_output": "/registered.npz",
                "query_output": "/query.npz",
            }
        },
    }
    matrix = {
        "schema": "cvs.sf_tapft.slim_matrix.v1",
        "base_sf_tapft": {
            "trainability_profile": "p1_head_norm",
            "norm_rules": [["t3", "weight_bias"]],
            "phase_steps": [300, 150, 70],
        },
    }

    result = _build_adapt_config(plan, matrix, "leo_clear_weak", "J3_R5D92_G")

    assert result["support_path"] == "/old.npz"
    assert result["candidate_id"] == "J3_R5D92_G_LEO_CLEAR_WEAK"
    assert result["sf_tapft"]["phase_steps"] == [300, 150, 70]
    assert result["sf_tapft"]["validation_steps"] == []
    assert result["sf_tapft"]["oof_temperature_calibration"] is False
