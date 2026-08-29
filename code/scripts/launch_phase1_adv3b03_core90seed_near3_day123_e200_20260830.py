#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Sequence


_BASE_PATH = Path(__file__).with_name(
    "launch_phase1_adv3b03_src5_day123_seed16_e200_20260829.py"
)
_BASE_SPEC = importlib.util.spec_from_file_location("phase1_adv3b03_day123_base", _BASE_PATH)
if _BASE_SPEC is None or _BASE_SPEC.loader is None:
    raise RuntimeError(f"cannot load base launcher: {_BASE_PATH}")
_base = importlib.util.module_from_spec(_BASE_SPEC)
_BASE_SPEC.loader.exec_module(_base)


CORE90_ORIGINAL_SEED = 392002
RUN_ID_DEFAULT = "phase1_adv3b03_core90seed_near3_day123_e200_20260830_r1"
FORMAL_SEEDS = (CORE90_ORIGINAL_SEED - 1, CORE90_ORIGINAL_SEED, CORE90_ORIGINAL_SEED + 1)

# Reuse the already verified ADV3B03 day1/2/3 source-only implementation. Only
# the immutable run identity and preregistered seed matrix differ.
_base.RUN_ID_DEFAULT = RUN_ID_DEFAULT
_base.FORMAL_SEEDS = FORMAL_SEEDS

PlanRow = _base.PlanRow
SCENARIOS = _base.SCENARIOS
SAT_SCHEDULE = _base.SAT_SCHEDULE
build_plan = _base.build_plan
build_train_command = _base.build_train_command
build_eval_command = _base.build_eval_command
run_row = _base.run_row


def main(argv: Sequence[str] | None = None) -> int:
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
