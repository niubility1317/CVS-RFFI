#!/usr/bin/env bash
set -uo pipefail

# Clean-target control for BEX02 target-domain adaptation.
# This keeps the 8GPU sweep matrix identical to run_target_adapt_bex02_sweep_8gpu.sh,
# but adapts on clean target-domain samples instead of provided satellite samples.
#
# Examples:
#   bash code/scripts/run_target_adapt_bex02_clean_target_8gpu.sh --plan CORE --dry-run
#   bash code/scripts/run_target_adapt_bex02_clean_target_8gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5,6,7

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TARGET_CHANNEL_VIEW="${TARGET_CHANNEL_VIEW:-clean}"
EXP_PREFIX="${EXP_PREFIX:-BEX02_tadapt_clean}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/target_adapt_bex02_clean_target_8gpu}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/target_adapt_bex02_clean_target_8gpu}"

export TARGET_CHANNEL_VIEW EXP_PREFIX RUN_ROOT LOG_ROOT

exec bash "${SCRIPT_DIR}/run_target_adapt_bex02_sweep_8gpu.sh" "$@"
