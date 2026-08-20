#!/usr/bin/env python3
"""Truth-last scorer for the complete B0-B3 M2.5 matrix."""

from __future__ import annotations

from cvsrffi.stage2_m24_safe_residual import D1
from cvsrffi.stage2_m25_anchored_residual import B1, B2, B3
from scripts import score_m24_d1_refit_matrix as shared
from scripts.run_m24_d1_refit_matrix import DEFAULT_CONDITIONS, DEFAULT_RECEIVERS, DEFAULT_SEEDS
from scripts.run_m25_anchored_residual_full125 import (
    EVIDENCE_ARMS,
    EXPECTED_INPUT_IDENTITIES,
    EXPECTED_METHOD_ROWS,
)


shared.D0 = D1
shared.CANDIDATE_ARMS = (B1, B2, B3)
shared.EVIDENCE_ARMS = EVIDENCE_ARMS
shared.EXPECTED_INPUT_IDENTITIES = EXPECTED_INPUT_IDENTITIES
shared.EXPECTED_METHOD_ROWS = EXPECTED_METHOD_ROWS
shared.DEFAULT_CONDITIONS = DEFAULT_CONDITIONS
shared.DEFAULT_RECEIVERS = DEFAULT_RECEIVERS
shared.DEFAULT_SEEDS = DEFAULT_SEEDS
shared.SCORED_MATRIX_SCHEMA = "cvs.erbt_idr.m25.anchored_residual_scored_matrix.v1"


if __name__ == "__main__":
    raise SystemExit(shared.main())

