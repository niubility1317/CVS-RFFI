#!/usr/bin/env bash
set -euo pipefail

# SSDG semi-supervised launch script.
#
# The labeled split is forced to 0.1 by the SSDG presets. The remaining
# train-days/train-RXs pool, except validation, is used as unlabeled data with
# TX labels hidden and RX/day domain labels retained.
#
# Examples:
#   bash run_ssdg_experiments.sh
#   SSDG_PRESETS=ssdg_r19_pseudo_cons_strict bash run_ssdg_experiments.sh
#   GPU_IDS=0,1,2,3 STOP_ON_FAIL=1 bash run_ssdg_experiments.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
BASE_ARGS="${BASE_ARGS:---dataset wisig --wisig_domain rx_day --batch_size 256 --primary_udu_weight 0.65}"
EPOCHS="${EPOCHS:-200}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
SSDG_PRESETS="${SSDG_PRESETS:-ssdg_r19_pseudo_cons,ssdg_r19_pseudo_cons_strict,ssdg_r25_pseudo_cons,ssdg_r19_pseudo_cons_fishr}"

mkdir -p logs ssdg_runs
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a PRESET_LIST <<< "${SSDG_PRESETS}"

if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "GPU_IDS is empty." >&2
  exit 1
fi

launch_one() {
  local gpu_id="$1"
  local preset="$2"
  local run_dir="ssdg_runs/${preset}"
  local stamp
  local log_path

  stamp="$(date +%Y%m%d_%H%M%S)"
  log_path="logs/ssdg_${preset}_${stamp}.log"
  mkdir -p "${run_dir}"

  echo "[SSDG] launch preset=${preset} gpu=${gpu_id} log=${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 \
  nohup "${PYTHON_BIN}" -u train.py \
    ${BASE_ARGS} \
    --preset "${preset}" \
    --epochs "${EPOCHS}" \
    --latest_save_path "${run_dir}/latest_model.pth" \
    --best_save_path "${run_dir}/best_model.pth" \
    > "${log_path}" 2>&1 &

  echo "$!"
}

pids=()
tags=()
slot=0
status=0

for raw_preset in "${PRESET_LIST[@]}"; do
  preset="$(echo "${raw_preset}" | xargs)"
  [ -z "${preset}" ] && continue

  gpu="${GPU_LIST[$slot]}"
  pid="$(launch_one "${gpu}" "${preset}" | tail -n 1)"
  pids+=("${pid}")
  tags+=("${preset}")

  slot=$((slot + 1))
  if [ "$slot" -ge "${#GPU_LIST[@]}" ]; then
    for i in "${!pids[@]}"; do
      if wait "${pids[$i]}"; then
        echo "[SSDG] done preset=${tags[$i]} pid=${pids[$i]}"
      else
        echo "[SSDG] failed preset=${tags[$i]} pid=${pids[$i]}" >&2
        status=1
        [ "${STOP_ON_FAIL}" = "1" ] && exit 1
      fi
    done
    pids=()
    tags=()
    slot=0
  fi
done

for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[SSDG] done preset=${tags[$i]} pid=${pids[$i]}"
  else
    echo "[SSDG] failed preset=${tags[$i]} pid=${pids[$i]}" >&2
    status=1
  fi
done

exit "${status}"
