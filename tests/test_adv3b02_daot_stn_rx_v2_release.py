from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh"
LAUNCHER = (
    ROOT
    / "code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_p1_p2_manysig_s392005_20260903.sh"
)
EXPANDED_LAUNCHER = (
    ROOT
    / "code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_p1_p5_manysig_s392005_20260903.sh"
)
OPTIONAL_LAUNCHER = (
    ROOT
    / "code/scripts/launch_phase1_adv3b02_daot_stn_rx_v2_e1_r1_manysig_s392005_20260903.sh"
)
CONFIG = ROOT / "configs/phase1_adv3b02_daot_stn_rx_v2_s392005.json"


def test_worker_maps_explicit_rx_v2_switch_and_loss_weights() -> None:
    worker = WORKER.read_text(encoding="utf-8")

    assert 'DAOT_RX_V2="${DAOT_RX_V2:-false}"' in worker
    assert '--use_adv3b02_daot_stn_rx_v2 true' in worker
    for name in (
        "TANGENT",
        "ROUTE",
        "RX",
        "TAIL",
        "NUISANCE",
        "FINGERPRINT",
        "SUBSPACE",
    ):
        assert f'DAOT_LAMBDA_{name}="${{DAOT_LAMBDA_{name}:-}}"' in worker
        assert f'--daot_lambda_{name.lower()}' in worker


def test_release_is_two_row_same_seed_adjacent_tangent_comparison() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert "ROWS=(V2-P1 V2-P2)" in launcher
    assert "GPUS=(1 7)" in launcher
    assert '--only=V2-P1|--only=V2-P2' in launcher
    assert "SEED=\"${SEED:-392005}\"" in launcher
    assert "DAOT_RX_V2=true" in launcher
    assert 'tangent="0"' in launcher
    assert 'tangent="0.035"' in launcher
    for setting in (
        "DAOT_LAMBDA_ROUTE=0",
        "DAOT_LAMBDA_RX=0",
        "DAOT_LAMBDA_TAIL=0",
        "DAOT_LAMBDA_NUISANCE=0",
        "DAOT_LAMBDA_FINGERPRINT=0",
        "DAOT_LAMBDA_SUBSPACE=0",
    ):
        assert setting in launcher
    assert "baseline=excluded_by_user" in launcher
    assert "non_leo_weak=excluded_by_user" in launcher


def test_release_freezes_single_v_protocol_and_required_eval_scenarios() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))

    assert "PHASE1_SOURCE_ROLE_PROTOCOL=legacy_l_u_v" in launcher
    assert "SOURCE_VAL_RATIO=0.30" in launcher
    assert "SOURCE_CAL_RATIO=0" in launcher
    assert "SOURCE_SELECT_RATIO=0" in launcher
    assert config["dataset"]["roles"] == {"L_s": 6300, "U_s": 56700, "V": 27000}
    assert config["final_evaluation"] == [
        "clean",
        "leo_clear_weak",
        "leo_low_elev_weak",
        "leo_rain_weak",
    ]


def test_expanded_release_follows_the_nested_p1_to_p5_design_matrix() -> None:
    launcher = EXPANDED_LAUNCHER.read_text(encoding="utf-8")

    assert "ROWS=(V2-P1 V2-P2 V2-P3 V2-P4 V2-P5)" in launcher
    assert "GPUS=(1 7 0 2 4)" in launcher
    assert "P3 adds randomized single-TX plus routing" in launcher
    assert "P4 adds TX-conditioned RX alignment" in launcher
    assert "P5 adds receiver-by-channel tail CVaR" in launcher
    assert 'V2-P1) tangent="0"; route="0"; rx="0"; tail="0" ;;' in launcher
    assert 'V2-P2) tangent="0.035"; route="0"; rx="0"; tail="0" ;;' in launcher
    assert 'V2-P3) tangent="0.035"; route="0.05"; rx="0"; tail="0" ;;' in launcher
    assert 'V2-P4) tangent="0.035"; route="0.05"; rx="0.075"; tail="0" ;;' in launcher
    assert 'V2-P5) tangent="0.035"; route="0.05"; rx="0.075"; tail="0.10" ;;' in launcher
    assert "DAOT_LAMBDA_SUBSPACE=0" in launcher
    assert "baseline=excluded_by_user" in launcher
    assert "non_leo_weak=excluded_by_user" in launcher


def test_optional_release_freezes_e1_and_r1_without_touching_mainline_rows() -> None:
    worker = WORKER.read_text(encoding="utf-8")
    launcher = OPTIONAL_LAUNCHER.read_text(encoding="utf-8")

    assert 'DAOT_EFFICIENCY_MODE="${DAOT_EFFICIENCY_MODE:-legacy}"' in worker
    assert '--daot_efficiency_mode "${DAOT_EFFICIENCY_MODE}"' in worker
    assert 'DAOT_SUBSPACE_RANK="${DAOT_SUBSPACE_RANK:-8}"' in worker
    assert 'DAOT_SUBSPACE_UPDATE_INTERVAL="${DAOT_SUBSPACE_UPDATE_INTERVAL:-5}"' in worker
    assert '--daot_subspace_rank "${DAOT_SUBSPACE_RANK}"' in worker
    assert '--daot_subspace_update_interval "${DAOT_SUBSPACE_UPDATE_INTERVAL}"' in worker
    assert "ROWS=(V2-E1 V2-R1)" in launcher
    assert "GPUS=(3 5)" in launcher
    assert 'V2-E1) subspace="0" ;;' in launcher
    assert 'V2-R1) subspace="0.05" ;;' in launcher
    assert "DAOT_EFFICIENCY_MODE=e1" in launcher
    assert "DAOT_SUBSPACE_RANK=8" in launcher
    assert "DAOT_SUBSPACE_UPDATE_INTERVAL=5" in launcher
    for setting in (
        "DAOT_LAMBDA_TANGENT=0.035",
        "DAOT_LAMBDA_ROUTE=0.05",
        "DAOT_LAMBDA_RX=0.075",
        "DAOT_LAMBDA_TAIL=0.10",
        "DAOT_LAMBDA_NUISANCE=0",
        "DAOT_LAMBDA_FINGERPRINT=0",
    ):
        assert setting in launcher
    assert "baseline=excluded_by_user" in launcher
    assert "non_leo_weak=excluded_by_user" in launcher
