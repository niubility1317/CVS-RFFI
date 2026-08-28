import json
from pathlib import Path

from cvsrffi.sf_tapft_slim_matrix import build_row_config, validate_slim_matrix
from cvsrffi.target_only_progressive_runner import _parse_config


REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = (
    REPO_ROOT
    / "configs"
    / "stage2_sf_tapft_p1_compact_deploy_replay_rx20_1_s392002_20260828.json"
)


def _matrix():
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def test_p1_matrix_is_exact_d0_to_d4_replay_without_promotion_claim():
    matrix = validate_slim_matrix(_matrix())
    assert [row["row_id"] for row in matrix["rows"]] == [
        "D0",
        "D1",
        "D2",
        "D3",
        "D4",
    ]
    assert matrix["shared_config"]["permission"] == "DIAGNOSTIC_NON_FORMAL"
    assert matrix["shared_config"]["protocol_schema"] == "p2_min_v1"
    assert matrix["shared_config"]["phase2_data_status"] == "VALIDATED_ONCE"


def test_p1_matrix_freezes_report_candidate_semantics():
    matrix = _matrix()
    rows = {
        row_id: build_row_config(matrix, row_id)[0]["sf_tapft"]
        for row_id in ("D0", "D1", "D2", "D3", "D4")
    }

    assert rows["D0"]["phase_steps"] == [300, 150, 70]
    assert rows["D0"]["norm_rules"] == [["t3", "weight_bias"]]
    assert rows["D1"]["phase_steps"] == [503, 0, 0]
    assert rows["D1"]["norm_rules"] == [
        ["t3", "weight_bias"],
        ["t2", "weight"],
    ]
    assert rows["D2"]["phase_steps"] == [231, 0, 0]
    assert rows["D2"]["norm_rules"][-2:] == [
        ["t1", "weight"],
        ["time_fuse", "weight"],
    ]
    assert rows["D3"]["phase_steps"] == [327, 0, 0]
    assert rows["D3"]["oof_temperature_calibration"] is True
    for row_id in ("D1", "D2", "D3"):
        assert rows[row_id]["cache_storage_dtype"] == "off"
        assert rows[row_id]["suffix_compute_dtype"] == "off"
    assert rows["D4"]["head_polish_steps"] == 100
    assert rows["D4"]["head_cvar_steps"] == 30
    assert rows["D4"]["head_cvar_weight"] == 0.03
    assert rows["D4"]["head_cvar_top_k"] == 2
    assert all(row["hard_pair_weight"] == 0 for row in rows.values())


def test_p1_matrix_uses_one_time_fp32_compute_cache_controls():
    matrix = _matrix()
    for row_id in ("D0", "D4"):
        row = build_row_config(matrix, row_id)[0]["sf_tapft"]
        assert row["cache_storage_dtype"] == "float32"
        assert row["suffix_compute_dtype"] == "float32"
        assert row["cache_device"] == "model"


def test_every_p1_row_passes_runner_config_contract_before_remote_launch():
    matrix = _matrix()
    for row_id in ("D0", "D1", "D2", "D3", "D4"):
        resolved, method_config = _parse_config(build_row_config(matrix, row_id)[0])
        assert resolved["candidate_id"]
        assert method_config.hard_pair_weight == 0
        if row_id in {"D0", "D4"}:
            assert method_config.cache_storage_dtype == "float32"
        else:
            assert method_config.cache_storage_dtype == "off"
