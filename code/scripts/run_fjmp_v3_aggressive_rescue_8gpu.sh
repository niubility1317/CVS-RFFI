#!/usr/bin/env bash
set -uo pipefail

# Launch the FJMP-v3 aggressive rescue residual batch (V3-01 ... V3-08).
# GPUs 0-7 are used by default through the shared dynamic queue launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_ROOT="${RUN_ROOT:-runs/fjmp_v3_aggressive_rescue}"
LOG_ROOT="${LOG_ROOT:-logs/fjmp_v3_aggressive_rescue}"
PLAN="${PLAN:-FJMP-V3}"

exec bash "${SCRIPT_DIR}/run_fjmp_sgv_bp_8gpu.sh" \
  --plan "${PLAN}" \
  --run-root "${RUN_ROOT}" \
  --log-root "${LOG_ROOT}" \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  "$@"
