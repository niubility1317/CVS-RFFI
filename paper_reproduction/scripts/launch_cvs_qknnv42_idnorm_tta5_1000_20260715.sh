#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-qknnv42_idnorm_tta5_1000_20260715_v1}"
RUN_ROOT="${ROOT}/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/${RUN_ID}"
CONFIG="${ROOT}/paper_reproduction/configs/cvs_qknnv42_idnorm_tta5_1000_stage2c_20260715_n607.json"
CKPT="${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
SHARD_COUNT="${SHARD_COUNT:-8}"

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}/pids"
cd "${ROOT}"

[[ -s "${CKPT}" ]] || { echo "missing strict ADV3B02 checkpoint" >&2; exit 4; }
for scenario in leo_clear_weak leo_low_elev_weak leo_rain_weak; do
  base="${ROOT}/runs/qknnv42_support_only_taskadapt_875_20260715_v1/base_features/${scenario}.npz"
  [[ -s "${base}" ]] || { echo "missing one-view LEO base cache ${base}" >&2; exit 4; }
done

"${PYTHON}" -u -m paper_reproduction.scripts.run_cvs_qknnv42_idnorm_tta5_1000 \
  --config "${CONFIG}" --ckpt "${CKPT}" \
  --adapter-root "${RUN_ROOT}/adapters" --output-root "${RUN_ROOT}/results" \
  --log-root "${LOG_ROOT}" --manifest "${RUN_ROOT}/matrix_manifest.json" \
  --prepare-only

for shard in $(seq 0 $((SHARD_COUNT - 1))); do
  gpu=$((shard % 8))
  pid_file="${LOG_ROOT}/pids/shard_${shard}.pid"
  if [[ -s "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" 2>/dev/null; then
    echo "shard ${shard} already active pid=$(cat "${pid_file}")" >&2
    exit 5
  fi
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    -m paper_reproduction.scripts.run_cvs_qknnv42_idnorm_tta5_1000 \
    --config "${CONFIG}" --ckpt "${CKPT}" \
    --adapter-root "${RUN_ROOT}/adapters" --output-root "${RUN_ROOT}/results" \
    --log-root "${LOG_ROOT}" --manifest "${RUN_ROOT}/matrix_manifest.json" \
    --shard-index "${shard}" --shard-count "${SHARD_COUNT}" \
    --device cuda:0 --execute \
    >"${LOG_ROOT}/shard_${shard}.out" 2>&1 </dev/null &
  echo "$!" >"${pid_file}"
  echo "[SHARD-LAUNCHED] shard=${shard} gpu=${gpu} pid=$!"
done

echo "[QKNN-IDNORM-TTA5-1000-LAUNCHED] run_root=${RUN_ROOT} shards=${SHARD_COUNT}"
