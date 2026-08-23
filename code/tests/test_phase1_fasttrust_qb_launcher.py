import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823.json"
LAUNCHER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_fasttrust_rc4_qb_e200_20260823.sh"
WORKER = ROOT / "code" / "scripts" / "launch_phase1_adv3b02_muse_ssdg_20260819.sh"


def test_qb_matrix_is_minimal_same_row_e200_and_quality_budgeted():
    data = json.loads(MATRIX.read_text(encoding="utf-8"))

    assert data["run_id"] == "phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1"
    assert data["seed"] == 392002
    assert data["epochs"] == 200
    assert data["unlabeled_batch_size"] == 256
    assert data["total_identity_effective_budget"] == 0.15
    assert data["rc4_lambda_domain"] == 0.16
    assert data["eval_batch_size"] == 512
    assert data["source_val_heavy_eval"] == {
        "start_epoch": 1,
        "interval": 5,
        "final_window": 20,
        "final_interval": 1,
    }
    assert [row["candidate"] for row in data["rows"]] == [
        "E200_QB0_NO_U_ID_SAFE",
        "E200_QB1_STRICT_H_SAFE",
        "E200_QB2_H_PRESID_B15",
    ]
    assert [row["gpu"] for row in data["rows"]] == [0, 1, 2]
    assert data["rows"][0]["hard"] is False and data["rows"][0]["partial"] is False
    assert data["rows"][1]["hard"] is True and data["rows"][1]["partial"] is False
    assert data["rows"][2]["hard"] is True and data["rows"][2]["partial"] is True
    assert [row["total_budget"] for row in data["rows"]] == [0.0, 0.15, 0.15]
    assert all(row["negative"] is False for row in data["rows"])


def test_qb_launcher_uses_one_process_per_gpu_and_propagates_safe_speed_controls():
    launcher = LAUNCHER.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")

    assert 'RESOURCE_SLOT_LIMIT="${RESOURCE_SLOT_LIMIT:-1}"' in launcher
    assert "RC4_TOTAL_IDENTITY_EFFECTIVE_BUDGET" in launcher
    assert "RC4_USE_CALIBRATED_PARTIAL_THRESHOLD" in launcher
    assert "RC4_LAMBDA_DOMAIN" in launcher
    assert "SOURCE_VAL_HEAVY_EVAL_INTERVAL" in launcher
    assert "EVAL_BATCH_SIZE" in launcher
    assert '--rc4_total_identity_effective_budget "${RC4_TOTAL_IDENTITY_EFFECTIVE_BUDGET}"' in worker
    assert '--rc4_use_calibrated_partial_threshold "${RC4_USE_CALIBRATED_PARTIAL_THRESHOLD}"' in worker
    assert '--rc4_lambda_domain "${RC4_LAMBDA_DOMAIN}"' in worker
    assert '--source_val_heavy_eval_interval "${SOURCE_VAL_HEAVY_EVAL_INTERVAL}"' in worker
    assert '--eval_batch_size "${EVAL_BATCH_SIZE}"' in worker
