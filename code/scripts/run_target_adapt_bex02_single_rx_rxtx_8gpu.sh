#!/usr/bin/env bash
set -uo pipefail

# Single-target-receiver RX x TX balanced adaptation experiments.
# Runs RX 7,8,9,10,11 separately with 5/10 samples per transmitter for 30 epochs.
#
# Examples:
#   bash code/scripts/run_target_adapt_bex02_single_rx_rxtx_8gpu.sh --plan CORE --dry-run
#   bash code/scripts/run_target_adapt_bex02_single_rx_rxtx_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGET_LOADERS="${TARGET_LOADERS:-test_unseen_day_rx_7,test_unseen_day_rx_8,test_unseen_day_rx_9,test_unseen_day_rx_10,test_unseen_day_rx_11}"
TARGET_SAMPLES_PER_RX_TX="${TARGET_SAMPLES_PER_RX_TX:-5,10}"
TARGET_LABEL_MODES="${TARGET_LABEL_MODES:-labeled,unlabeled}"
EPOCHS="${EPOCHS:-30}"
ADAPT_WEIGHTS="${ADAPT_WEIGHTS:-base}"
EXP_PREFIX="${EXP_PREFIX:-BEX02_tadapt_single_rx_rxtx}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/target_adapt_bex02_single_rx_rxtx_8gpu}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/target_adapt_bex02_single_rx_rxtx_8gpu}"

export TARGET_LOADERS TARGET_SAMPLES_PER_RX_TX TARGET_LABEL_MODES EPOCHS ADAPT_WEIGHTS EXP_PREFIX RUN_ROOT LOG_ROOT

exec bash "${SCRIPT_DIR}/run_target_adapt_bex02_sweep_8gpu.sh" "$@"
