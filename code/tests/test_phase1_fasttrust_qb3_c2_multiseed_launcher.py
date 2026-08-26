import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.json"
LAUNCHER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.sh"


def test_c2_multiseed_matrix_only_fills_the_two_missing_causal_rows():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert data["schema"] == "cvs.phase1.fasttrust_qb3_matrix.v2"
    assert data["run_id"] == "phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826_r1"
    assert data["epochs"] == 200
    assert data["unlabeled_batch_size"] == 256
    assert data["source_roles"] == {
        "L_s": 0.07,
        "U_s": 0.63,
        "V_cal": 0.15,
        "V_select": 0.15,
    }
    assert data["training_schedule"] == "LEO_WEAK"
    rows = data["rows"]
    assert {(row["seed"], row["variant"]) for row in rows} == {
        (713101, "C2"),
        (713102, "C2"),
    }
    assert {row["gpu"] for row in rows} == {4, 5}
    for row in rows:
        assert row["hard"] is True
        assert row["partial"] is True
        assert row["partial_set"] is True
        assert row["partial_conditional"] is False
        assert row["negative"] is False
        assert row["feature_anchor"] == 0.0
        assert row["eval_batch_size"] == 512
        assert row["recovery_checkpoint_interval"] == 1


def test_c2_dedicated_launcher_binds_the_new_matrix_instead_of_speed_profile():
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.json" in launcher
    assert "phase1_adv3b02_fasttrust_qb3_speed_profile" not in launcher
