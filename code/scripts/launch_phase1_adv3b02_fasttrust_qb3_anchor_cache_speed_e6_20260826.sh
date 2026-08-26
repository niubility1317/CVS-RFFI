#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CODE_ROOT="${CODE_ROOT:-${SCRIPT_ROOT}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_qb3_anchor_cache_speed_e6_20260826.json}"
export ROOT CODE_ROOT MATRIX
exec bash "${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_fasttrust_qb3_matrix_20260826.sh" "$@"
