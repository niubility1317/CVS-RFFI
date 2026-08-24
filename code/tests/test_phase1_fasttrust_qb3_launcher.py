import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824.json"
LAUNCHER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_fasttrust_qb3_bc_hps_e200_20260824.sh"
WORKER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_muse_ssdg_20260819.sh"


def test_qb3_matrix_freezes_full_e200_causal_design_and_speed_controls():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert data["run_id"] == "phase1_adv3b02_fasttrust_qb3_bc_hps_e200_s392002_20260824_r1"
    assert data["seed"] == 392002
    assert data["epochs"] == 200
    assert data["unlabeled_batch_size"] == 256
    assert data["identity_domain_objective_mode"] == "bounded_confusion"
    assert data["hard_effective_budget"] == 0.05
    assert data["partial_effective_budget"] == 0.10
    assert data["partial_threshold_scope"] == "global"
    assert data["decouple_partial_negative_aps"] is True
    assert data["eval_batch_size"] == 1024
    assert data["source_val_heavy_eval"] == {
        "start_epoch": 1,
        "interval": 10,
        "final_window": 20,
        "final_interval": 1,
    }
    rows = data["rows"]
    assert [row["gpu"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["candidate"] for row in rows] == [
        "E200_C0_BC_NO_U_ID",
        "E200_C1_BC_H",
        "E200_C2_BC_H_PSET",
        "E200_C3_BC_H_PSET_PCOND",
        "E200_C4_BC_U_FEATURE_ANCHOR",
    ]
    assert [row["hard"] for row in rows] == [False, True, True, True, False]
    assert [row["partial_set"] for row in rows] == [False, False, True, True, False]
    assert [row["partial_conditional"] for row in rows] == [False, False, False, True, False]
    assert rows[4]["feature_anchor"] > 0.0


def test_qb3_launcher_and_worker_propagate_bounded_confusion_and_recovery():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")

    assert 'RESOURCE_SLOT_LIMIT="${RESOURCE_SLOT_LIMIT:-1}"' in launcher
    assert "IDENTITY_DOMAIN_OBJECTIVE_MODE=bounded_confusion" in launcher
    assert "RC4_RECOVERY_CHECKPOINT_INTERVAL=1" in launcher
    assert "RC4_PARTIAL_THRESHOLD_SCOPE=global" in launcher
    assert "SOURCE_VAL_HEAVY_EVAL_INTERVAL" in launcher
    for argument in (
        "--identity_domain_objective_mode",
        "--rc4_decouple_partial_negative_aps",
        "--rc4_partial_threshold_scope",
        "--rc4_hard_effective_budget",
        "--rc4_class_receiver_effective_budget",
        "--rc4_lambda_discriminator",
        "--rc4_lambda_confusion",
        "--rc4_lambda_partial_set",
        "--rc4_lambda_partial_conditional",
        "--rc4_lambda_feature_anchor",
        "--rc4_recovery_checkpoint_interval",
    ):
        assert argument in worker
