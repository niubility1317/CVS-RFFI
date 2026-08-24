import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = PROJECT_ROOT / "code" / "scripts" / "launch_phase1_ccoi_pa_v1_20260824.sh"
CONFIG = PROJECT_ROOT / "docs" / "experiments" / "PHASE1_CCOI_PA_V1_CONFIG_20260824.json"


def test_launcher_is_smoke_first_source_only_and_four_scenario_complete():
    text = LAUNCHER.read_text(encoding="utf-8")
    smoke_pos = text.index("SMOKE_ROOT")
    full_pos = text.index("FULL MATRIX", smoke_pos)

    assert smoke_pos < full_pos
    assert "C0,C1,C2,C3,C4" in text
    assert "leo_clear_weak" in text
    assert "leo_low_elev_weak" in text
    assert "leo_rain_weak" in text
    assert "best_joint_safe_ssdg.pth" in text
    assert "Dataset_WigSig/ManySig.pkl" in text
    assert "score_phase1_ccoi_pa.py" in text
    assert "--target" not in text and "--query" not in text


def test_launcher_bytes_are_utf8_lf_without_bom():
    raw = LAUNCHER.read_bytes()

    assert raw.startswith(b"#!/usr/bin/env bash\n")
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" not in raw


def test_config_freezes_one_seed_roles_and_same_capacity_rows():
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert config["seed"] == 20260824
    assert config["source_roles"] == {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15}
    assert config["scenarios"] == ["clean", "leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
    assert config["rows"] == ["C0", "C1", "C2", "C3", "C4"]
    assert config["same_capacity_rows"] == ["C1", "C2", "C3", "C4"]
    assert config["target_or_query_training_access"] is False
