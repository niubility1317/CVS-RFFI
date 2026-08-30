from __future__ import annotations

from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "launch_phase1_bicad_xr_quick24_n607_20260831.sh"
)
R3_SCRIPT = SCRIPT.with_name("launch_phase1_bicad_xr_quick24_n607_20260831_r3.sh")


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
