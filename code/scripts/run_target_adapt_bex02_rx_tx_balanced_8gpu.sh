#!/usr/bin/env bash
set -uo pipefail

# RX x TX balanced target-domain adaptation experiments.
# For each target receiver, select K samples from each transmitter.
#
# Examples:
#   bash code/scripts/run_target_adapt_bex02_rx_tx_balanced_8gpu.sh --plan CORE --dry-run
#   bash code/scripts/run_target_adapt_bex02_rx_tx_balanced_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGET_SAMPLES_PER_RX_TX="${TARGET_SAMPLES_PER_RX_TX:-2,3}"
EXP_PREFIX="${EXP_PREFIX:-BEX02_tadapt_rxtx}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/target_adapt_bex02_rx_tx_balanced_8gpu}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/target_adapt_bex02_rx_tx_balanced_8gpu}"

export TARGET_SAMPLES_PER_RX_TX EXP_PREFIX RUN_ROOT LOG_ROOT

exec bash "${SCRIPT_DIR}/run_target_adapt_bex02_sweep_8gpu.sh" "$@"
