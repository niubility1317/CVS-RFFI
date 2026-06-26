#!/usr/bin/env bash
set -euo pipefail

# Focused 8-GPU dynamic queue for the post-5.16 FJMP loss-design matrix.
#
# Usage:
#   bash scripts/run_fjmp_loss_design_8gpu.sh \
#     --base-ckpt runs/cvs_rffi_staged/B3b_stable_sat07_cls_only/latest_model.pth
#
# Useful options are forwarded to run_fjmp_sgv_bp_8gpu.sh:
#   --gpu-ids 0,1,2,3,4,5,6,7
#   --python /path/to/python
#   --dry-run
#   --extra-args "--amp false --eval_max_batches 0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "${SCRIPT_DIR}/run_fjmp_sgv_bp_8gpu.sh" \
  --plan LOSS-DESIGN \
  --gpu-ids "${GPU_IDS:-0,1,2,3,4,5,6,7}" \
  --run-root "${RUN_ROOT:-runs/fjmp_loss_design}" \
  --log-root "${LOG_ROOT:-logs/fjmp_loss_design}" \
  "$@"
