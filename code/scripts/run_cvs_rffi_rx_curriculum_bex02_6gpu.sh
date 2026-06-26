#!/usr/bin/env bash
set -uo pipefail

# CVS-RFFI receiver-curriculum launcher using the BEX02_fishr002_mixed_e170
# configuration. The receiver matrix mirrors run_baseline_pseudo_rx_curriculum_6gpu.sh.
#
# Examples:
#   bash code/scripts/run_cvs_rffi_rx_curriculum_bex02_6gpu.sh --plan SMOKE --dry-run
#   bash code/scripts/run_cvs_rffi_rx_curriculum_bex02_6gpu.sh --plan CORE --gpu-ids 0,1,2,3,4,5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5}"
PLAN="${PLAN:-CORE}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-${WORKSPACE_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/cvs_rffi_bex02_rx_curriculum}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/cvs_rffi_bex02_rx_curriculum}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"

usage() {
  sed -n '1,10p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids CSV        GPUs to use, default 0,1,2,3,4,5
  --plan NAME          SMOKE, CORE, or FULL
  --wisig-pkl PATH     Dataset_WigSig/ManySig.pkl path
  --python PATH        Python executable
  --run-root PATH      Output checkpoint root
  --log-root PATH      Log root
  --no-skip-done       Re-run even when latest_model.pth exists
  --stop-on-fail       Stop queue after first failure
  --stream-logs        Stream job logs to scheduler stdout
  --dry-run            Print commands only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
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

if [ "${DRY_RUN}" != "1" ] && { [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; }; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${WISIG_PKL}" ]; then
  echo "ERROR: WISIG_PKL not found: ${WISIG_PKL}" >&2
  echo "Set WISIG_PKL=/path/to/Dataset_WigSig/ManySig.pkl or pass --wisig-pkl." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${PLAN}_${STAMP}.log"
QUEUE_FILE="${LOG_ROOT}/queue_${PLAN}_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

rx_label() {
  case "$1" in
    0) echo "1-1" ;;
    1) echo "1-19" ;;
    2) echo "14-7" ;;
    3) echo "18-2" ;;
    4) echo "19-2" ;;
    5) echo "2-1" ;;
    6) echo "2-19" ;;
    7) echo "20-1" ;;
    8) echo "3-19" ;;
    9) echo "7-14" ;;
    10) echo "7-7" ;;
    11) echo "8-8" ;;
    *) echo "$1" ;;
  esac
}

csv_has() {
  local csv="$1" needle="$2" item
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    if [ "${item}" = "${needle}" ]; then
      return 0
    fi
  done
  return 1
}

csv_len() {
  local csv="$1"
  IFS=',' read -r -a items <<< "${csv}"
  echo "${#items[@]}"
}

csv_labels() {
  local csv="$1" out="" item label
  IFS=',' read -r -a items <<< "${csv}"
  for item in "${items[@]}"; do
    label="$(rx_label "${item}")"
    if [ -z "${out}" ]; then
      out="${label}"
    else
      out="${out},${label}"
    fi
  done
  echo "${out}"
}

append_rx() {
  local csv="$1" rx="$2"
  if [ -z "${csv}" ]; then
    echo "${rx}"
  else
    echo "${csv},${rx}"
  fi
}

build_train_csv() {
  local pair_csv="$1" universe_csv="$2" level="$3" train_csv="${pair_csv}" rx
  IFS=',' read -r -a universe <<< "${universe_csv}"
  for rx in "${universe[@]}"; do
    if [ "$(csv_len "${train_csv}")" -ge "${level}" ]; then
      break
    fi
    if ! csv_has "${train_csv}" "${rx}"; then
      train_csv="$(append_rx "${train_csv}" "${rx}")"
    fi
  done
  if [ "$(csv_len "${train_csv}")" -ne "${level}" ]; then
    echo "ERROR: could not build K=${level} train set from ${pair_csv}" >&2
    return 2
  fi
  echo "${train_csv}"
}

build_test_csv() {
  local train_csv="$1" test_csv="" rx
  for rx in 0 1 2 3 4 5 6 7 8 9 10 11; do
    if ! csv_has "${train_csv}" "${rx}"; then
      test_csv="$(append_rx "${test_csv}" "${rx}")"
    fi
  done
  echo "${test_csv}"
}

pair_csv_for() {
  local target_name="$1" pair_name="$2"
  case "${target_name}:${pair_name}" in
    T1:P01) echo "0,10" ;;
    T1:P02) echo "0,11" ;;
    T1:P03) echo "0,2" ;;
    T1:P04) echo "10,11" ;;
    T1:P05) echo "10,2" ;;
    T1:P06) echo "11,2" ;;
    T14:P01) echo "0,1" ;;
    T14:P02) echo "0,10" ;;
    T14:P03) echo "0,11" ;;
    T14:P04) echo "1,10" ;;
    T14:P05) echo "1,11" ;;
    T14:P06) echo "10,11" ;;
    *) echo "ERROR: unknown pair ${target_name}:${pair_name}" >&2; return 2 ;;
  esac
}

target_label_for() {
  case "$1" in
    T1) echo "1-19" ;;
    T14) echo "14-7" ;;
    *) echo "$1" ;;
  esac
}

target_idx_for() {
  case "$1" in
    T1) echo "1" ;;
    T14) echo "2" ;;
    *) echo "-1" ;;
  esac
}

universe_for() {
  case "$1" in
    T1) echo "0,10,11,2,5,6,3" ;;
    T14) echo "0,1,10,11,5,6,3" ;;
    *) echo "ERROR: unknown target ${1}" >&2; return 2 ;;
  esac
}

generate_queue() {
  local plan_upper target_names pair_names levels target_name pair_name level pair_csv universe_csv
  local train_csv test_csv train_dash train_labels test_labels exp_id desc target_label target_idx
  plan_upper="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
  : > "${QUEUE_FILE}"
  case "${plan_upper}" in
    SMOKE)
      target_names="T1"
      pair_names="P01"
      levels="2 7"
      ;;
    CORE|FULL)
      target_names="T1 T14"
      pair_names="P01 P02 P03 P04 P05 P06"
      levels="2 3 4 5 6 7"
      ;;
    *)
      echo "ERROR: unknown plan: ${PLAN}" >&2
      return 2
      ;;
  esac

  for target_name in ${target_names}; do
    universe_csv="$(universe_for "${target_name}")" || return 2
    target_label="$(target_label_for "${target_name}")"
    target_idx="$(target_idx_for "${target_name}")"
    for pair_name in ${pair_names}; do
      pair_csv="$(pair_csv_for "${target_name}" "${pair_name}")" || return 2
      for level in ${levels}; do
        train_csv="$(build_train_csv "${pair_csv}" "${universe_csv}" "${level}")" || return 2
        test_csv="$(build_test_csv "${train_csv}")"
        train_dash="${train_csv//,/-}"
        train_labels="$(csv_labels "${train_csv}")"
        test_labels="$(csv_labels "${test_csv}")"
        exp_id="BEX02_fishr002_mixed_e170_plain_${target_name}_${pair_name}_K${level}_train-${train_dash}_test-rest"
        desc="config=BEX02_fishr002_mixed_e170; mode=plain; design_target=Rx(${target_label}) idx=${target_idx}; image_pair=${pair_name}; K=${level}; train_rx=${train_labels}; test_all_remaining_rx=${test_labels}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${exp_id}" "${target_name}" "${test_csv}" "${train_csv}" "${level}" "${desc}" >> "${QUEUE_FILE}"
      done
    done
  done
}

if ! generate_queue; then
  echo "ERROR: failed to generate receiver-curriculum queue." >&2
  exit 2
fi
TOTAL_JOBS="$(wc -l < "${QUEUE_FILE}" | tr -d ' ')"
if [ "${TOTAL_JOBS}" -lt 1 ]; then
  echo "ERROR: selected plan produced an empty queue." >&2
  exit 2
fi

BASE_ARGS=(
  --batch_size 256
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx_day
  --wisig_out_len 256
  --wisig_train_ratio 0.2
  --wisig_guard_gap 8
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_max_day123_per_combo 0
  --wisig_max_train_per_combo 0
  --wisig_max_val_per_combo 0
  --wisig_max_test_per_combo 0
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor 2
  --primary_udu_weight 0.65
  --epochs 170
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --test_eval_policy val_improved_final
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
  --lambda_fishr 0.02
  --fishr_min_domains 4
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
  local gpu_id="$1" exp_id="$2" target_name="$3" test_rxs="$4" train_rxs="$5" level="$6" desc="$7"
  local out_dir="${RUN_ROOT}/${exp_id}"
  local log="${LOG_ROOT}/${exp_id}_${STAMP}.log"

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/latest_model.pth" ]; then
    log_msg "[SKIP-DONE] exp=${exp_id} out_dir=${out_dir}"
    FREE_GPUS+=("${gpu_id}")
    return
  fi

  mkdir -p "${out_dir}"
  local cmd
  cmd="$(printf '%q ' "${PYTHON_BIN}" -u train.py \
    "${BASE_ARGS[@]}" \
    --wisig_train_rxs "${train_rxs}" \
    --wisig_test_rxs "${test_rxs}" \
    --run_name "${exp_id}" \
    --latest_save_path "${out_dir}/latest_model.pth" \
    --best_save_path "${out_dir}/best_val_model.pth" \
    --best_primary_save_path "${out_dir}/best_primary_ood_model.pth" \
    --best_unseen_day_unseen_rx_save_path "${out_dir}/best_strict_udu_model.pth" \
    --best_test_save_path "${out_dir}/best_test_overall_model.pth" \
    --best_worst_rx_save_path "${out_dir}/best_worst_rx_model.pth")"

  {
    echo "EXP_ID=${exp_id}"
    echo "CONFIG=BEX02_fishr002_mixed_e170"
    echo "TARGET=${target_name}"
    echo "TRAIN_RXS=${train_rxs}"
    echo "TEST_RXS=${test_rxs}"
    echo "LEVEL=${level}"
    echo "DESCRIPTION=${desc}"
    echo "GPU=${gpu_id}"
    echo "RUN_DIR=${out_dir}"
    echo "CMD=CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${cmd}"
  } > "${log}"

  if [ "${DRY_RUN}" = "1" ]; then
    log_msg "[DRY-RUN] gpu=${gpu_id} exp=${exp_id} ${desc}"
    log_msg "CMD=${cmd}"
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
  log_msg "[LAUNCHED] gpu=${gpu_id} exp=${exp_id} pid=${RUNNING_PIDS[-1]} log=${log}"
}

start_until_full() {
  while [ "${#FREE_GPUS[@]}" -gt 0 ] && [ "${NEXT_INDEX}" -lt "${TOTAL_JOBS}" ]; do
    local gpu_id="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    local line exp_id target_name test_rxs train_rxs level desc
    line="$(queue_line_at "${NEXT_INDEX}")"
    NEXT_INDEX=$((NEXT_INDEX + 1))
    IFS=$'\t' read -r exp_id target_name test_rxs train_rxs level desc <<< "${line}"
    launch_one "${gpu_id}" "${exp_id}" "${target_name}" "${test_rxs}" "${train_rxs}" "${level}" "${desc}"
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

log_msg "CVS-RFFI BEX02 receiver-curriculum launcher"
log_msg "PLAN=${PLAN} TOTAL_JOBS=${TOTAL_JOBS} GPU_IDS=${GPU_IDS_CSV}"
log_msg "PYTHON_BIN=${PYTHON_BIN}"
log_msg "WISIG_PKL=${WISIG_PKL}"
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

log_msg "CVS-RFFI BEX02 receiver-curriculum queue finished status=${STATUS}"
exit "${STATUS}"
