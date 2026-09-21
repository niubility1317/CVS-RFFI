import pytest
from pathlib import Path
from scripts.core90_cross_response_matrix import build_matrix, build_confirmation_matrix, V2_VARIANTS

CONFIG = Path(__file__).resolve().parents[1] / "code/configs/phase1_core90_cross_response_v2.json"


def arguments(tmp_path):
    return dict(config_path=CONFIG,wisig_pkl="future-authorized-source.pkl",output_root=tmp_path,
        roles=dict(wisig_train_rxs="1,3,4,6,8",wisig_test_rxs="0,2,5,7,9,10,11",
                   wisig_train_days="1,2,3",wisig_test_days="0"),
        seeds=[101,102],variants=["U1","U3","U1_mask_off"])


def test_v2_paired_seed_data_and_budget_registration(tmp_path):
    rows = build_confirmation_matrix(**arguments(tmp_path),confirmation_boundary={
        "previous_target_results_used":False,"independent_confirmation_scope":"synthetic test only",
        "frozen_before_launch":True})
    assert len(rows) == 6 and not any(r["launch"] for r in rows)
    assert {r["data_seed"] for r in rows} == {392005}
    for r in rows:
        assert r["baseline_args"]["epochs"] == 200
        assert r["baseline_args"]["batch_size"] == 128
        assert r["baseline_args"]["run_id"] == "core90_cross_response_v2"
    a,b = rows[:2]
    changed = {k for k in a["cross_response"] if a["cross_response"][k] != b["cross_response"][k]}
    assert changed == {"decision_enabled"}
    assert rows[2]["cross_response"]["mixstyle_role_policy"] == "original_mask"


def test_confirmation_cannot_infer_boundary_or_seeds(tmp_path):
    with pytest.raises(ValueError,match="boundary"):
        build_confirmation_matrix(**arguments(tmp_path),confirmation_boundary={})
    args = arguments(tmp_path)
    args["seeds"] = [101]
    with pytest.raises(ValueError,match="multiple"):
        build_confirmation_matrix(**args,confirmation_boundary={})


def test_all_candidates_registered_without_authorizing_source_thresholds(tmp_path):
    args = arguments(tmp_path)
    args["variants"] = V2_VARIANTS
    rows = build_matrix(**args)
    assert len(rows) == 2 * len(V2_VARIANTS)
    assert all(not r["launch"] for r in rows)
    assert all(r["cross_response"]["decision_calibration"] is None for r in rows)
