#!/usr/bin/env python3
"""Truth-last scorer for the complete G0-G4 M2.4 matrix."""

from __future__ import annotations

from cvsrffi.stage2_m24_invariance_breaking import G1, G2, G3, G4
from cvsrffi.stage2_m24_safe_residual import D0
from scripts import score_m24_d1_refit_matrix as shared
from scripts.run_m24_d1_refit_matrix import DEFAULT_CONDITIONS, DEFAULT_RECEIVERS, DEFAULT_SEEDS
from scripts.run_m24_invariance_breaking_full125 import (
    EVIDENCE_ARMS,
    EXPECTED_INPUT_IDENTITIES,
    EXPECTED_METHOD_ROWS,
)


shared.D0 = D0
shared.CANDIDATE_ARMS = (G1, G2, G3, G4)
shared.EVIDENCE_ARMS = EVIDENCE_ARMS
shared.EXPECTED_INPUT_IDENTITIES = EXPECTED_INPUT_IDENTITIES
shared.EXPECTED_METHOD_ROWS = EXPECTED_METHOD_ROWS
shared.DEFAULT_CONDITIONS = DEFAULT_CONDITIONS
shared.DEFAULT_RECEIVERS = DEFAULT_RECEIVERS
shared.DEFAULT_SEEDS = DEFAULT_SEEDS
shared.SCORED_MATRIX_SCHEMA = "cvs.erbt_idr.m24.invariance_breaking_scored_matrix.v1"


if __name__ == "__main__":
    raise SystemExit(shared.main())

