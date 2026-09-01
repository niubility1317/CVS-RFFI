from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = PROJECT_ROOT / "code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.schedule import build_ecrs_stage_state, ecrs_stage_for_epoch  # noqa: E402


def test_report_stage0_to_stage6_matrix_and_v1_defaults() -> None:
    stage0 = build_ecrs_stage_state(0)
    stage1 = build_ecrs_stage_state(1)
    stage2 = build_ecrs_stage_state(2)
    stage3 = build_ecrs_stage_state(3)
    stage4 = build_ecrs_stage_state(4, progress=0.5)
    stage5 = build_ecrs_stage_state(5)
    stage6 = build_ecrs_stage_state(6)

    assert stage0["known_excitation"] and not stage0["canonical"]
    assert stage1["adv3b02_only"] and not stage1["canonical"]
    assert stage2["canonical"] and stage2["content"] and not stage2["diff_tx"]
    assert stage3["split_fit"] and stage3["pair_cross"] and not stage3["resp_cls"]
    assert stage4["resp_cls"] and stage4["same_tx_cross"] and stage4["diff_tx"]
    assert 0.0 <= stage4["active_rho_max"] <= 0.2
    assert not stage5["learnable_basis"]
    assert not stage6["fasttrust"]


def test_e200_epoch_mapping_keeps_advanced_stages_deferred() -> None:
    assert ecrs_stage_for_epoch(1)["stage"] == 2
    assert ecrs_stage_for_epoch(40)["stage"] == 2
    assert ecrs_stage_for_epoch(41)["stage"] == 3
    assert ecrs_stage_for_epoch(90)["stage"] == 3
    assert ecrs_stage_for_epoch(91)["stage"] == 4
    assert ecrs_stage_for_epoch(200)["stage"] == 4
    assert ecrs_stage_for_epoch(200)["learnable_basis"] is False


def test_stage5_and_stage6_require_explicit_report_conditions() -> None:
    assert build_ecrs_stage_state(5, enable_learnable_basis=True)["learnable_basis"]
    assert not build_ecrs_stage_state(6, enable_fasttrust=True, teacher_stable=False)["fasttrust"]
    assert build_ecrs_stage_state(6, enable_fasttrust=True, teacher_stable=True)["fasttrust"]
