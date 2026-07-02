#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SHARD_COUNT="${SHARD_COUNT:-8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_leo_feature_adapter_matrix_20260703}"
SCRIPT="${SCRIPT:-${ROOT}/code/scripts/sweep_phase1_adv3b02_leo_feature_adapter_20260703.sh}"

mkdir -p "${LOG_ROOT}"

for g in $(seq 0 $((SHARD_COUNT - 1))); do
  nohup env \
    GPU="${g}" \
    CELL_SHARD_INDEX="${g}" \
    CELL_SHARD_COUNT="${SHARD_COUNT}" \
    PYTHON="${PYTHON}" \
    ROOT="${ROOT}" \
    bash "${SCRIPT}" \
    > "${LOG_ROOT}/driver_shard${g}.out" 2>&1 &
  echo "shard=${g} pid=$!"
done
