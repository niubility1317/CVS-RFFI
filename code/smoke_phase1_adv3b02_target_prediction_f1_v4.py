"""Fresh v4 identity for the frozen F1 ADV blind-prediction smoke."""

from __future__ import annotations

from collections.abc import Sequence

import smoke_phase1_adv3b02_target_prediction_f1 as _smoke


SMOKE_SCHEMA = "cvs.phase1.adv3b02_target_prediction_technical_smoke.v4"
SMOKE_RUN_ID = "phase1_adv3b02_target_prediction_20260816_v4"

_smoke.SMOKE_SCHEMA = SMOKE_SCHEMA
_smoke.SMOKE_RUN_ID = SMOKE_RUN_ID

ADV3B02TargetSmokeError = _smoke.ADV3B02TargetSmokeError
run_f1_technical_smoke = _smoke.run_f1_technical_smoke
validate_f1_technical_smoke_receipt = _smoke.validate_f1_technical_smoke_receipt
build_arg_parser = _smoke.build_arg_parser


def main(argv: Sequence[str] | None = None) -> int:
    return _smoke.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
