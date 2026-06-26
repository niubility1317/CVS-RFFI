#!/usr/bin/env bash
set -uo pipefail

# Best-base exploration launcher after N04.
# Defaults to GPUs 6 and 7, with a dynamic two-GPU queue.
#
# Examples:
#   bash code/scripts/run_best_base_explore_gpu67.sh --dry-run
#   bash code/scripts/run_best_base_explore_gpu67.sh
#   GPU_IDS=6,7 PYTHON_BIN=python3 bash code/scripts/run_best_base_explore_gpu67.sh --plan P0
#   STREAM_LOGS=1 bash code/scripts/run_best_base_explore_gpu67.sh --plan FULL

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-6,7}"
PLAN="${PLAN:-FULL}"
PYTHON_BIN="${PYTHON_BIN:-}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/best_base_explore}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/best_base_explore}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"

usage() {
  sed -n '1,14p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3 python python.exe py; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN//,/}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

append_rows_for_plan() {
  local plan_name
  plan_name="$(echo "$1" | tr '[:lower:]' '[:upper:]')"
  case "${plan_name}" in
    P0)
      cat <<'EOF' >> "${QUEUE_FILE}"
BEX01_fishr002_storm|P0|Fishr=0.02 plus single storm_mp SAT; combines N04 Fishr with N24 storm signal.|--sat_train_scenario storm_mp --lambda_fishr 0.02 --fishr_min_domains 4
BEX02_fishr002_mixed_e170|P0|N04 repeated with 170 epochs to reduce late E200 decay while preserving best-epoch region.|--epochs 170 --sat_train_scenario mixed_orbit --lambda_fishr 0.02 --fishr_min_domains 4
BEX03_fishr002_mixed_swa|P0|N04 plus SWA from E140 every 5 epochs; tests late-checkpoint smoothing.|--use_swa_ckpt --swa_start_epoch 140 --swa_interval 5 --sat_train_scenario mixed_orbit --lambda_fishr 0.02 --fishr_min_domains 4
BEX04_fishr002_all5_tinycons|P0|N04 Fishr plus all-five SAT cycle and tiny consistency; stability probe from N18 direction.|--sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit --lambda_sat_cons 0.005 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    FISHR)
      cat <<'EOF' >> "${QUEUE_FILE}"
BEX05_fishr0018_mixed|FISHR|Fine scan just below N04; checks if 0.02 is slightly over-regularized.|--sat_train_scenario mixed_orbit --lambda_fishr 0.018 --fishr_min_domains 4
BEX06_fishr0022_mixed|FISHR|Fine scan just above N04; most likely positive Fishr neighbor.|--sat_train_scenario mixed_orbit --lambda_fishr 0.022 --fishr_min_domains 4
BEX07_fishr0025_mixed|FISHR|Upper Fishr neighbor; detects whether the optimum extends past 0.02.|--sat_train_scenario mixed_orbit --lambda_fishr 0.025 --fishr_min_domains 4
EOF
      ;;
    MIXSAT)
      cat <<'EOF' >> "${QUEUE_FILE}"
BEX08_fishr002_mixed_rain|MIXSAT|Mixed-orbit plus rain with Fishr=0.02; rain is the strict-UDU-correlated SAT view.|--sat_train_scenarios mixed_orbit,rain_leo --lambda_fishr 0.02 --fishr_min_domains 4
BEX09_fishr002_mixed_storm_e170|MIXSAT|Mixed-orbit plus storm at 170 epochs; targeted bottleneck plus shorter training.|--epochs 170 --sat_train_scenarios mixed_orbit,storm_mp --lambda_fishr 0.02 --fishr_min_domains 4
BEX10_fishr002_late80|MIXSAT|N04 with earlier MixStyle anneal; tests whether late augmentation pressure causes E200 decay.|--sat_train_scenario mixed_orbit --mixstyle_late_start 80 --mixstyle_late_ramp_epochs 40 --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    SEED)
      cat <<'EOF' >> "${QUEUE_FILE}"
BEX11_n04_seed2026|SEED|N04 seed robustness check for a less lucky seed than 1337.|--seed 2026 --sat_train_scenario mixed_orbit --lambda_fishr 0.02 --fishr_min_domains 4
BEX12_storm_seed2026|SEED|Best P0 storm candidate seed robustness check.|--seed 2026 --sat_train_scenario storm_mp --lambda_fishr 0.02 --fishr_min_domains 4
EOF
      ;;
    *)
      echo "ERROR: unknown plan '${plan_name}'. Use P0,FISHR,MIXSAT,SEED,FULL." >&2
      exit 2
      ;;
  esac
}

generate_queue() {
  : > "${QUEUE_FILE}"
  local plan_upper
  plan_upper="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
  if [ "${plan_upper}" = "FULL" ] || [ "${plan_upper}" = "ALL" ]; then
    append_rows_for_plan P0
    append_rows_for_plan FISHR
    append_rows_for_plan MIXSAT
    append_rows_for_plan SEED
  else
    IFS=',' read -r -a plans <<< "${PLAN}"
    for p in "${plans[@]}"; do
      append_rows_for_plan "${p}"
    done
  fi
}

generate_queue
TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: selected plan produced an empty queue." >&2
  exit 2
fi

BASE_ARGS=(
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.2
  --primary_udu_weight 0.65
  --epochs 200
  --eval_sat_channel
  --eval_sat_on test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --test_eval_policy every_epoch
  --slim_group none
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
  --use_mixstyle
  --use_sat_consistency
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
  --seed 1337
)

RUNNING_PIDS=()
RUNNING_TAGS=()
RUNNING_GPUS=()
FREE_GPUS=("${GPU_LIST[@]}")
NEXT_INDEX=0
STATUS=0

queue_line_at() {
  sed -n "$(($1 + 1))p" "${QUEUE_FILE}"
}

launch_one() {
  local gpu_id="$1" exp_id="$2" group="$3" purpose="$4" extra_args="$5"
  local out_dir="${RUN_ROOT}/${exp_id}"
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/best_primary_ood_model.pth" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  mkdir -p "${out_dir}"
  local log="${LOG_ROOT}/${exp_id}_$(date +%Y%m%d_%H%M%S).log"
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u train.py "${BASE_ARGS[@]}" \
    --run_name "${exp_id}" \
    --latest_save_path "${out_dir}/latest_model.pth" \
    --best_save_path "${out_dir}/best_val_model.pth" \
    --best_primary_save_path "${out_dir}/best_primary_ood_model.pth" \
    --best_unseen_day_unseen_rx_save_path "${out_dir}/best_strict_udu_model.pth" \
    --best_test_save_path "${out_dir}/best_test_overall_model.pth" \
    --best_worst_rx_save_path "${out_dir}/best_worst_rx_model.pth")"
  cmd="${cmd}${extra_args}"

  {
    echo "EXP_ID=${exp_id}"
    echo "GROUP=${group}"
    echo "PURPOSE=${purpose}"
    echo "GPU=${gpu_id}"
    echo "RUN_DIR=${out_dir}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${cmd}"
  } > "${log}"

  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] gpu=${gpu_id} exp=${exp_id} group=${group} cmd=${cmd}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  if [ "${STREAM_LOGS}" = "1" ]; then
    (
      CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" 2>&1
      status="$?"
      echo "__EXIT_STATUS__=${status}"
      exit "${status}"
    ) | sed -u "s/^/[${exp_id}|GPU${gpu_id}] /" | tee -a "${log}" &
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 bash -lc "${cmd}" >> "${log}" 2>&1 &
  fi
  RUNNING_PIDS+=("$!")
  RUNNING_TAGS+=("${exp_id}")
  RUNNING_GPUS+=("${gpu_id}")
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} group=${group} pid=${RUNNING_PIDS[-1]} log=${log}"
}

start_until_full() {
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local line exp_id group purpose extra_args
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS='|' read -r exp_id group purpose extra_args <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${group}" "${purpose}" "${extra_args}"
  done
}

reap_one() {
  wait -n
  local status=$?
  local i
  for i in "${!RUNNING_PIDS[@]}"; do
    if ! kill -0 "${RUNNING_PIDS[$i]}" 2>/dev/null; then
      local gpu="${RUNNING_GPUS[$i]}" tag="${RUNNING_TAGS[$i]}"
      unset 'RUNNING_PIDS[i]' 'RUNNING_TAGS[i]' 'RUNNING_GPUS[i]'
      RUNNING_PIDS=("${RUNNING_PIDS[@]}")
      RUNNING_TAGS=("${RUNNING_TAGS[@]}")
      RUNNING_GPUS=("${RUNNING_GPUS[@]}")
      FREE_GPUS+=("${gpu}")
      if [ "${status}" -ne 0 ]; then
        STATUS=1
      fi
      log_msg "[FINISHED] exp=${tag} status=${status} freed_gpu=${gpu}"
      return
    fi
  done
}

log_msg "Best-base exploration launcher"
log_msg "PLAN=${PLAN} TOTAL_JOBS=${TOTAL_JOBS} GPU_IDS=${GPU_IDS_CSV}"
log_msg "PYTHON_BIN=${PYTHON_BIN}"
log_msg "RUN_ROOT=${RUN_ROOT}"
log_msg "LOG_ROOT=${LOG_ROOT}"
log_msg "QUEUE_FILE=${QUEUE_FILE}"
log_msg "DRY_RUN=${DRY_RUN} SKIP_DONE=${SKIP_DONE} STOP_ON_FAIL=${STOP_ON_FAIL} STREAM_LOGS=${STREAM_LOGS}"

start_until_full
while [ "${#RUNNING_PIDS[@]}" -gt 0 ] || [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
  if [ "${#RUNNING_PIDS[@]}" -gt 0 ]; then
    reap_one
    if [ "${STATUS}" -ne 0 ] && [ "${STOP_ON_FAIL}" = "1" ]; then
      log_msg "Stopping early because a job failed and STOP_ON_FAIL=1."
      exit 1
    fi
  fi
  start_until_full
done

log_msg "Best-base exploration queue finished status=${STATUS}"
exit "${STATUS}"
