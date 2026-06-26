#!/usr/bin/env bash
set -uo pipefail

# Launch the FJMP-v2 safe residual ablation batch (V2-01 ... V2-06).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-runs/fjmp_v2_safe_residual}"
LOG_ROOT="${LOG_ROOT:-logs/fjmp_v2_safe_residual}"
PLAN="${PLAN:-FJMP-V2}"

exec bash "${SCRIPT_DIR}/run_fjmp_sgv_bp_8gpu.sh" \
  --plan "${PLAN}" \
  --run-root "${RUN_ROOT}" \
  --log-root "${LOG_ROOT}" \
  "$@"
