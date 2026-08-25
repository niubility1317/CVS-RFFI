import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826.json"


def test_qb3_multiseed_matrix_freezes_c0_c3_and_only_adds_new_seeds():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert data["schema"] == "cvs.phase1.fasttrust_qb3_matrix.v2"
    assert data["run_id"] == "phase1_adv3b02_fasttrust_qb3_c0c3_ms_e200_20260826_r1"
    assert data["epochs"] == 200
    assert data["unlabeled_batch_size"] == 256
    assert data["muse_schedule"] == {
        "s2a_start": 17,
        "s2b_start": 41,
        "s3a_start": 69,
        "s3b_start": 161,
        "s3c_start": 181,
    }
    assert data["source_roles"] == {
        "L_s": 0.07,
        "U_s": 0.63,
        "V_cal": 0.15,
        "V_select": 0.15,
    }

    rows = data["rows"]
    assert len(rows) == 4
    assert {(row["seed"], row["variant"]) for row in rows} == {
        (713101, "C0"),
        (713101, "C3"),
        (713102, "C0"),
        (713102, "C3"),
    }
    assert {row["gpu"] for row in rows} == {0, 1, 2, 3}
    assert all(row["eval_batch_size"] == 512 for row in rows)
    assert all(row["recovery_checkpoint_interval"] == 1 for row in rows)
    assert all(not row["negative"] and row["feature_anchor"] == 0.0 for row in rows)

    for row in rows:
        if row["variant"] == "C0":
            assert not any(
                row[key]
                for key in ("hard", "partial", "partial_set", "partial_conditional")
            )
        else:
            assert all(
                row[key]
                for key in ("hard", "partial", "partial_set", "partial_conditional")
            )
