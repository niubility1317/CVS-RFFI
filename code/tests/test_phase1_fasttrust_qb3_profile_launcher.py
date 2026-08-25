import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826.json"
LAUNCHER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_fasttrust_qb3_matrix_20260826.sh"


def test_qb3_speed_profile_is_same_seed_factorial_without_algorithm_changes():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert data["schema"] == "cvs.phase1.fasttrust_qb3_matrix.v2"
    assert data["run_id"] == "phase1_adv3b02_fasttrust_qb3_speed_profile_s392002_20260826_r2"
    assert data["epochs"] == 21
    assert data["unlabeled_batch_size"] == 256
    assert data["source_roles"] == {
        "L_s": 0.07,
        "U_s": 0.63,
        "V_cal": 0.15,
        "V_select": 0.15,
    }
    rows = data["rows"]
    assert len(rows) == 4
    assert {row["seed"] for row in rows} == {392002}
    assert {row["gpu"] for row in rows} == {4, 5, 6, 7}
    assert data["muse_schedule"] == {
        "s2a_start": 17,
        "s2b_start": 18,
        "s3a_start": 19,
        "s3b_start": 20,
        "s3c_start": 21,
    }
    assert {
        (row["eval_batch_size"], row["recovery_checkpoint_interval"])
        for row in rows
    } == {(512, 1), (512, 5), (1024, 1), (1024, 5)}
    assert all(row["hard"] and row["partial_set"] and row["partial_conditional"] for row in rows)
    assert all(not row["negative"] and row["feature_anchor"] == 0.0 for row in rows)


def test_qb3_matrix_launcher_exists_for_remote_behavioral_dry_run():
    assert LAUNCHER.is_file()
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert 'RC4_LAMBDA_HARD="${hard_lambda}"' in launcher
