#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_qb3_c2_ms_e200_20260826.json}"
export ROOT CODE_ROOT MATRIX
exec bash "${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_fasttrust_qb3_matrix_20260826.sh" "$@"
