from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "code/scripts/launch_phase1_adv3b02_nmfdu_gate_v1_queue_20260901.sh"
CONFIG = ROOT / "code/configs/phase1_adv3b02_nmfdu_gate_v1.json"


def test_launcher_freezes_report_matrix_and_current_source_protocol() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for row, mode in (
        ("M1", "equal"),
        ("M2", "i_only"),
        ("M3", "physical_full"),
        ("M4", "full"),
    ):
        assert f"launch_row {row} {mode}" in text
    assert "execution=historical_checkpoint_eval_only" in text
    assert "--labeled_ratio 0.07" in text
    assert "--unlabeled_ratio 0.63" in text
    assert "--source_val_ratio 0.30" in text
    assert "--epochs 200" in text
    assert '--seed "${SEED}"' in text
    assert "--lambda_sat_cls 0.68" in text
    assert "--lambda_sat_cons 0" in text
    assert "leo_clear_weak,leo_low_elev_weak,leo_rain_weak" in text


def test_launcher_has_no_phase2_or_query_data_path_and_refuses_overwrite() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    command_region = text[text.index("CMD=(env") : text.index("launch_row()")]
    assert "phase2_" not in command_region.lower()
    assert "query" not in command_region.lower()
    assert "refusing to overwrite" in text
    assert '[[ ! -e "${RUNS_ROOT}" && ! -e "${LOG_ROOT}" ]]' in text


def test_json_matrix_matches_launcher_ablation_contract() -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = {item["row"]: item for item in payload["single_seed_matrix"]}
    assert tuple(rows) == ("M0", "M1", "M2", "M3", "M4")
    assert rows["M0"]["execution"] == "historical_checkpoint_eval_only"
    assert {rows[row]["nmfdu_ablation_mode"] for row in ("M1", "M2", "M3", "M4")} == {
        "equal",
        "i_only",
        "physical_full",
        "full",
    }
    assert payload["phase2_access"] is False
    assert payload["query_access"] is False
