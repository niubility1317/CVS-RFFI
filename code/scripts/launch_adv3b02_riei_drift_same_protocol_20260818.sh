#!/usr/bin/env bash
set -euo pipefail

# Direct Phase1 comparison on one shared WiSig/ManySig input surface.
# Each method keeps its paper-native encoder/objective; the data path, receiver/day
# split, equalization, crop and output length are shared.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-phase1_adv3b02_riei_drift_same_protocol_20260818_v1}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
SEED="${SEED:-713101}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
STAGE="matrix"
SKIP_DONE="${SKIP_DONE:-1}"

usage() {
  cat <<'EOF'
Usage: launch_adv3b02_riei_drift_same_protocol_20260818.sh [options]

Stages:
  --stage smoke       Run one rx7_d01 row per method (three jobs).
  --stage matrix      Run six profiles x three methods (18 jobs).

Options:
  --root PATH         Project root.
  --python PATH       Python executable.
  --wisig-pkl PATH    ManySig compact pickle.
  --run-root PATH     Immutable output root.
  --log-root PATH     Log root.
  --gpu-ids CSV       GPU indices, default 0,1,2,3,4,5,6,7.
  --max-per-gpu N     Concurrent jobs per GPU, default 2.
  --no-skip-done      Do not skip an output that already has metrics.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --root) ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --max-per-gpu) MAX_PER_GPU="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "[ERROR] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${STAGE}" in
  smoke|matrix) ;;
  *) echo "[ERROR] --stage must be smoke or matrix" >&2; exit 2 ;;
esac

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[ERROR] python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if [ ! -f "${WISIG_PKL}" ]; then
  echo "[ERROR] ManySig pickle not found: ${WISIG_PKL}" >&2
  exit 2
fi
if [ "${MAX_PER_GPU}" -lt 1 ]; then
  echo "[ERROR] --max-per-gpu must be positive" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAGE}_${STAMP}.log"
MANIFEST="${RUN_ROOT}/manifest_${STAGE}_${STAMP}.tsv"
export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
if [ "${#GPU_IDS[@]}" -lt 1 ]; then
  echo "[ERROR] no GPUs supplied" >&2
  exit 2
fi

log() {
  echo "[$(date +%F_%T)] $*" | tee -a "${SCHED_LOG}"
}

# name|train_days|test_days|train_rxs|test_rxs
PROFILES=(
  "rx3_d0|0|2,3|0,3,6|7,8,9,10,11"
  "rx3_d01|0,1|2,3|0,3,6|7,8,9,10,11"
  "rx5_d01|0,1|2,3|0,1,2,3,4|7,8,9,10,11"
  "rx7_d0|0|2,3|0,1,2,3,4,5,6|7,8,9,10,11"
  "rx7_d01|0,1|2,3|0,1,2,3,4,5,6|7,8,9,10,11"
  "rx7_d012|0,1,2|3|0,1,2,3,4,5,6|7,8,9,10,11"
)

declare -A PROFILE_TRAIN_DAYS PROFILE_TEST_DAYS PROFILE_TRAIN_RXS PROFILE_TEST_RXS
for row in "${PROFILES[@]}"; do
  IFS='|' read -r name train_days test_days train_rxs test_rxs <<< "${row}"
  PROFILE_TRAIN_DAYS["${name}"]="${train_days}"
  PROFILE_TEST_DAYS["${name}"]="${test_days}"
  PROFILE_TRAIN_RXS["${name}"]="${train_rxs}"
  PROFILE_TEST_RXS["${name}"]="${test_rxs}"
done

METHODS=(adv3b02 riei_fd drift)
if [ "${STAGE}" = "smoke" ]; then
  JOBS=("adv3b02|rx7_d01" "riei_fd|rx7_d01" "drift|rx7_d01")
else
  JOBS=()
  for row in "${PROFILES[@]}"; do
    IFS='|' read -r profile _rest <<< "${row}"
    for method in "${METHODS[@]}"; do
      JOBS+=("${method}|${profile}")
    done
  done
fi

printf 'job_id\tmethod\tprofile\tgpu\toutput_dir\tlog_file\tcommand\n' > "${MANIFEST}"

build_command() {
  local method="$1" profile="$2" out_dir="$3"
  local train_days="${PROFILE_TRAIN_DAYS[${profile}]}"
  local test_days="${PROFILE_TEST_DAYS[${profile}]}"
  local train_rxs="${PROFILE_TRAIN_RXS[${profile}]}"
  local test_rxs="${PROFILE_TEST_RXS[${profile}]}"
  CMD=()
  if [ "${method}" = "adv3b02" ]; then
    CMD=("${PYTHON_BIN}" "${ROOT}/code/SSDG/train_ssdg.py"
      --baseline_ckpt "" --from_scratch true
      --split_mode tx_rx_day_1_7_2
      --source_split_seed "${SEED}"
      --labeled_ratio 0.07 --unlabeled_ratio 0.63 --source_val_ratio 0.30
      --wisig_pkl "${WISIG_PKL}" --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256
      --wisig_train_days "${train_days}" --wisig_test_days "${test_days}"
      --wisig_train_rxs "${train_rxs}" --wisig_test_rxs "${test_rxs}"
      --seed "${SEED}" --output_dir "${out_dir}"
      --epochs 200 --checkpoint_selection final_only
      --test_eval_policy val_improved_final --test_eval_start_epoch 999999
      --batch_size 128 --eval_batch_size 256 --num_workers 0 --device cuda:0)
  elif [ "${method}" = "riei_fd" ]; then
    CMD=("${PYTHON_BIN}" -m baselines.riei_fd.train_cvs
      --wisig_pkl "${WISIG_PKL}" --wisig_protocol cvs_day_rx
      --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256
      --wisig_train_days "${train_days}" --wisig_test_days "${test_days}"
      --wisig_train_rxs "${train_rxs}" --wisig_test_rxs "${test_rxs}"
      --use_source_ssl_split --wisig_labeled_ratio 0.07 --wisig_unlabeled_ratio 0.63
      --wisig_source_val_ratio 0.30 --wisig_cap_strategy front --wisig_split_seed "${SEED}"
      --seed "${SEED}" --epochs 200 --device cuda:0 --output_dir "${out_dir}"
      --eval_sat_channel --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
      --eval_sat_on all --no_test_on_val_improve --test_eval_start_epoch 999999
      --paper_eval_last_n 0 --num_workers 0)
  else
    CMD=("${PYTHON_BIN}" -m baselines.drift.train_cvs
      --wisig_pkl "${WISIG_PKL}" --wisig_protocol cvs_day_rx
      --wisig_equalized 1 --wisig_domain rx_day --wisig_out_len 256
      --wisig_train_days "${train_days}" --wisig_test_days "${test_days}"
      --wisig_train_rxs "${train_rxs}" --wisig_test_rxs "${test_rxs}"
      --use_source_ssl_split --wisig_labeled_ratio 0.07 --wisig_unlabeled_ratio 0.63
      --wisig_source_val_ratio 0.30 --wisig_cap_strategy front --wisig_split_seed "${SEED}"
      --seed "${SEED}" --epochs 200 --device cuda:0 --output_dir "${out_dir}"
      --eval_sat_channel --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
      --eval_sat_on all --no_test_on_val_improve --test_eval_start_epoch 999999
      --paper_eval_last_n 0 --num_workers 0)
  fi
}

run_job() {
  local method="$1" profile="$2" gpu="$3"
  local job_id="${method}__${profile}__seed${SEED}"
  local out_dir="${RUN_ROOT}/${job_id}"
  local log_file="${LOG_ROOT}/${job_id}.log"
  if [ "${SKIP_DONE}" = "1" ] && { [ -f "${out_dir}/metrics.json" ] || [ -f "${out_dir}/phase1_terminal_status.json" ]; }; then
    log "SKIP job=${job_id} existing_output=${out_dir}"
    return 0
  fi
  build_command "${method}" "${profile}" "${out_dir}"
  local pretty
  printf -v pretty '%q ' "${CMD[@]}"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${job_id}" "${method}" "${profile}" "${gpu}" "${out_dir}" "${log_file}" "${pretty}" >> "${MANIFEST}"
  log "START job=${job_id} gpu=${gpu} out=${out_dir}"
  mkdir -p "${out_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" > "${log_file}" 2>&1
  local rc=$?
  log "DONE job=${job_id} gpu=${gpu} rc=${rc}"
  return "${rc}"
}

log "RUN_ID=${RUN_ID} stage=${STAGE} seed=${SEED} jobs=${#JOBS[@]} max_per_gpu=${MAX_PER_GPU}"
log "shared_input=ManySig equalized=1 out_len=256 domain=rx_day crop=center normalize=true"

job_index=0
wave=0
overall_rc=0
while [ "${job_index}" -lt "${#JOBS[@]}" ]; do
  pids=()
  labels=()
  for gpu in "${GPU_IDS[@]}"; do
    for _slot in $(seq 1 "${MAX_PER_GPU}"); do
      [ "${job_index}" -lt "${#JOBS[@]}" ] || break 2
      IFS='|' read -r method profile <<< "${JOBS[${job_index}]}"
      (
        run_job "${method}" "${profile}" "${gpu}"
      ) &
      pids+=("$!")
      labels+=("${method}__${profile}")
      job_index=$((job_index + 1))
    done
  done
  log "WAVE_START wave=${wave} count=${#pids[@]}"
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      overall_rc=1
      log "WAVE_JOB_FAILED wave=${wave} job=${labels[$i]}"
    fi
  done
  log "WAVE_DONE wave=${wave}"
  wave=$((wave + 1))
done

log "RUN_DONE stage=${STAGE} rc=${overall_rc} manifest=${MANIFEST}"
exit "${overall_rc}"
