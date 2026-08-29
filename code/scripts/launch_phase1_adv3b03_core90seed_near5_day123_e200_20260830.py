#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


_BASE_PATH = Path(__file__).with_name(
    "launch_phase1_adv3b03_src5_day123_seed16_e200_20260829.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location("phase1_adv3b03_day123_base_near5", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load base launcher: {_BASE_PATH}")
_base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_base)


CORE90_ORIGINAL_SEED = 392002
RUN_ID_DEFAULT = "phase1_adv3b03_core90seed_near5_day123_e200_20260830_r1"
FORMAL_SEEDS = (392000, 392004, 391999, 392005, 391998)
FORMAL_GPUS = (3, 4, 5, 6, 7)
_GPU_BY_SEED = dict(zip(FORMAL_SEEDS, FORMAL_GPUS))

PlanRow = _base.PlanRow
SCENARIOS = _base.SCENARIOS
SAT_SCHEDULE = _base.SAT_SCHEDULE
build_train_command = _base.build_train_command
build_eval_command = _base.build_eval_command
run_row = _base.run_row


def build_plan(seeds: Sequence[int] | None = None) -> list[PlanRow]:
    selected = list(seeds) if seeds is not None else list(FORMAL_SEEDS)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("seeds must be a non-empty unique sequence")
    if any(int(seed) not in _GPU_BY_SEED for seed in selected):
        raise ValueError("seed is outside the frozen near5 matrix")
    return [
        PlanRow(
            seed=int(seed),
            gpu=_GPU_BY_SEED[int(seed)],
            candidate_id=f"S{int(seed)}_ADV3B03_MU10_ALPHA20_E200",
        )
        for seed in selected
    ]


_base.RUN_ID_DEFAULT = RUN_ID_DEFAULT
_base.FORMAL_SEEDS = FORMAL_SEEDS
_base.build_plan = build_plan


def main(argv: Sequence[str] | None = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
