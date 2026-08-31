from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "launch_phase1_bicad_xr_quick24_n607_20260831.sh"
)
R3_SCRIPT = SCRIPT.with_name("launch_phase1_bicad_xr_quick24_n607_20260831_r3.sh")
PAIRBICAD_SCRIPT = SCRIPT.with_name("launch_phase1_pairbicad_p0p4_n607_20260831.sh")
CONVERGENCE_SCRIPT = SCRIPT.with_name(
    "launch_phase1_pairbicad_convergence_n607_20260831.sh"
)
FINAL_SCRIPT = SCRIPT.with_name("launch_phase1_pairbicad_final_n607_20260831.sh")


def test_n607_launch_script_is_fixed_to_registered_quick24_matrix() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r2" in source
    assert "--stage quick" in source
    assert "--max-jobs-per-gpu 3" in source
    assert "--formal" in source
    assert "--wisig-pkl" in source
    assert "nohup" in source
    assert "dispatcher.pid" in source


def test_n607_launch_script_refuses_output_or_log_overwrite() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert 'test ! -e "${RUN_ROOT}"' in source
    assert 'test ! -e "${DISPATCH_LOG}"' in source
    assert 'test ! -e "${PID_FILE}"' in source
    assert "rm " not in source
    assert "pkill" not in source
    assert "killall" not in source


def test_r3_n607_launch_script_uses_a_fresh_nonoverwriting_run() -> None:
    source = R3_SCRIPT.read_text(encoding="utf-8")

    assert "phase1_adv3b02_bicad_xr_quick24_seed3_u5000_20260831_r3" in source
    assert "phase1_bicad_xr_quick24_20260831_r3" in source
    assert "--max-jobs-per-gpu 3" in source
    assert 'test ! -e "${RUN_ROOT}"' in source
    assert 'test ! -e "${DISPATCH_LOG}"' in source
    assert 'test ! -e "${PID_FILE}"' in source
    assert "rm " not in source
    assert "pkill" not in source


def test_pairbicad_n607_launch_script_contains_the_frozen_30_row_command() -> None:
    source = PAIRBICAD_SCRIPT.read_text(encoding="utf-8")

    assert "phase1_adv3b02_pairbicad_p0p4_loro2_seed3_u4000_20260831_r1" in source
    assert "phase1_pairbicad_p0p4_loro2_20260831_r1" in source
    assert "--stage pairbicad" in source
    assert "--candidates P0,P1,P2,P3,P4" in source
    assert "--folds 1,8" in source
    assert "--seeds 392001,392002,392003" in source
    assert "--optimizer-updates 4000" in source
    assert "--max-jobs-per-gpu 2" in source
    assert "--batch_size 48" not in source
    assert "--formal" in source
    assert "--wisig-pkl" in source
    assert "nohup" in source
    assert "dispatcher.pid" in source


def test_pairbicad_n607_launch_script_is_source_only_and_nonoverwriting() -> None:
    source = PAIRBICAD_SCRIPT.read_text(encoding="utf-8")

    assert 'test ! -e "${RUN_ROOT}"' in source
    assert 'test ! -e "${DISPATCH_LOG}"' in source
    assert 'test ! -e "${PID_FILE}"' in source
    assert "rm " not in source
    assert "pkill" not in source
    assert "killall" not in source
    forbidden = ("phase2", "target", "support", "query", "truth", "admin")
    assert not any(token in source.lower() for token in forbidden)


def test_pairbicad_convergence_n607_script_uses_explicit_top_candidates() -> None:
    source = CONVERGENCE_SCRIPT.read_text(encoding="utf-8")

    assert "PAIRBICAD_CONVERGENCE_CANDIDATES" in source
    assert ": \"${PAIRBICAD_CONVERGENCE_CANDIDATES:?" in source
    assert "--stage pairbicad_convergence" in source
    assert '--candidates "${PAIRBICAD_CONVERGENCE_CANDIDATES}"' in source
    assert "--folds 1,8" in source
    assert "--seeds 392001,392002,392003" in source
    assert "--optimizer-updates 9000" in source
    assert "--max-jobs-per-gpu 2" in source
    assert 'test ! -e "${RUN_ROOT}"' in source
    assert 'test ! -e "${DISPATCH_LOG}"' in source
    assert 'test ! -e "${PID_FILE}"' in source
    assert "nohup" in source
    forbidden = ("phase2", "target", "support", "query", "truth", "admin")
    assert not any(token in source.lower() for token in forbidden)


def test_pairbicad_final_n607_script_uses_explicit_candidate_and_budget() -> None:
    source = FINAL_SCRIPT.read_text(encoding="utf-8")

    assert "PAIRBICAD_FINAL_CANDIDATE" in source
    assert ": \"${PAIRBICAD_FINAL_CANDIDATE:?" in source
    assert "PAIRBICAD_FINAL_OPTIMIZER_UPDATES" in source
    assert ": \"${PAIRBICAD_FINAL_OPTIMIZER_UPDATES:?" in source
    assert "--stage pairbicad_final" in source
    assert '--candidates "${PAIRBICAD_FINAL_CANDIDATE}"' in source
    assert "--folds 1,2,3,4,5" in source
    assert "--seeds 392001,392002,392003" in source
    assert '--optimizer-updates "${PAIRBICAD_FINAL_OPTIMIZER_UPDATES}"' in source
    assert "--max-jobs-per-gpu 2" in source
    assert 'test ! -e "${RUN_ROOT}"' in source
    assert 'test ! -e "${DISPATCH_LOG}"' in source
    assert 'test ! -e "${PID_FILE}"' in source
    assert "nohup" in source
    forbidden = ("phase2", "target", "support", "query", "truth", "admin")
    assert not any(token in source.lower() for token in forbidden)
