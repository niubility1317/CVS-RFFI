from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


REPAIR_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "launch_phase1_pairbicad_cv2_screen24_20260901.py"
)
HISTORICAL_SCRIPT = REPAIR_SCRIPT.with_name(
    "launch_phase1_pairbicad_cv2_screen24_20260831.py"
)


def _load_repair_launcher():
    if not REPAIR_SCRIPT.is_file():
        pytest.fail("E200 repair launcher is not implemented yet")
    spec = importlib.util.spec_from_file_location("pairbicad_cv2_e200", REPAIR_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e200_repair_launcher_is_a_new_file_and_keeps_historical_launcher() -> None:
    assert REPAIR_SCRIPT.is_file()
    assert HISTORICAL_SCRIPT.is_file()


def test_e200_repair_launcher_has_no_update_budget_cli() -> None:
    launcher = _load_repair_launcher()
    source = REPAIR_SCRIPT.read_text(encoding="utf-8")

    assert launcher.RUN_ID_DEFAULT == (
        "phase1_pairbicad_cv2_fixed11_e200_seed392002_20260901_r1"
    )
    assert "--epochs" in source
    assert "--bicad_optimizer_updates" not in source
    assert "optimizer_updates" not in source


def test_e200_repair_launcher_is_source_only_and_caps_gpu_slots() -> None:
    launcher = _load_repair_launcher()
    parser_destinations = {action.dest for action in launcher.build_parser()._actions}

    assert launcher.CV2_CANDIDATE_IDS == tuple(
        f"CV2-{family}{index}"
        for family in ("B", "D", "T")
        for index in range(4)
    )
    assert launcher.FOLDS == (1, 8)
    assert launcher.SEED == 392002
    assert launcher.TRAIN_DAYS == (1, 2, 3)
    assert launcher.MAX_ACTIVE_PER_GPU == 2
    assert "candidates" not in parser_destinations
    assert "folds" not in parser_destinations
    assert "seed" not in parser_destinations
