#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_fjmp_sgv_bp_8gpu.sh" \
  --plan A03-A06-REPRO \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --run-root "${RUN_ROOT:-runs/fjmp_a03_a06_repro}" \
  --log-root "${LOG_ROOT:-logs/fjmp_a03_a06_repro}" \
  "$@"
