#!/usr/bin/env bash
set -euo pipefail

# Unified launcher for paper baselines under the same CVS-RFFI/WiSig split.
#
# Example:
#   METHODS=cvcnn,riei,drift,receiver_agnostic,tifs2025 GPU_IDS=0,1,2,3 \
#     nohup bash run_cvs_baseline_queue.sh > baseline_logs/cvs_baselines_$(date +%Y%m%d_%H%M%S).nohup.log 2>&1 &
#
# Dry run:
#   DRY_RUN=1 bash run_cvs_baseline_queue.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
METHODS_CSV="${METHODS:-cvcnn,riei,drift,receiver_agnostic,tifs2025}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"
PARALLEL="${PARALLEL:-1}"
GLOBAL_SEED="${SEED:-1337}"

ROOT_DIR="$(pwd)"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-baseline_runs/cvs_baseline_queue_${RUN_STAMP}}"
LOG_ROOT="${LOG_ROOT:-baseline_logs}"
LOG_DIR="${LOG_DIR:-${LOG_ROOT}/cvs_baseline_queue_${RUN_STAMP}}"

# CVS-RFFI split parameters. Keep these synchronized with the main CVS-RFFI
# experiments unless you are intentionally running an ablation.
WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
WISIG_EQUALIZED="${WISIG_EQUALIZED:-1}"
WISIG_DOMAIN="${WISIG_DOMAIN:-rx_day}"
WISIG_OUT_LEN="${WISIG_OUT_LEN:-256}"
WISIG_TRAIN_RATIO="${WISIG_TRAIN_RATIO:-0.2}"
WISIG_VAL_RATIO="${WISIG_VAL_RATIO:--1.0}"
WISIG_GUARD_GAP="${WISIG_GUARD_GAP:-8}"
WISIG_TRAIN_DAYS="${WISIG_TRAIN_DAYS:-0,1}"
WISIG_TEST_DAYS="${WISIG_TEST_DAYS:-2,3}"
WISIG_TRAIN_RXS="${WISIG_TRAIN_RXS:-0,1,2,3,4,5,6}"
WISIG_TEST_RXS="${WISIG_TEST_RXS:-7,8,9,10,11}"
WISIG_MAX_DAY123_PER_COMBO="${WISIG_MAX_DAY123_PER_COMBO:-0}"
WISIG_MAX_TRAIN_PER_COMBO="${WISIG_MAX_TRAIN_PER_COMBO:-0}"
WISIG_MAX_VAL_PER_COMBO="${WISIG_MAX_VAL_PER_COMBO:-0}"
WISIG_MAX_TEST_PER_COMBO="${WISIG_MAX_TEST_PER_COMBO:-0}"

# Do not override paper-stated training hyperparameters by default. Set the
# method-specific *_EPOCHS variables only when intentionally running an ablation.
CVCNN_EPOCHS="${CVCNN_EPOCHS:-200}"
RIEI_EPOCHS="${RIEI_EPOCHS:-200}"
DRIFT_EPOCHS="${DRIFT_EPOCHS:-200}"
RA_EPOCHS="${RA_EPOCHS:-}"
TIFS_EPOCHS="${TIFS_EPOCHS:-}"
TIFS_PRETRAIN_EPOCHS="${TIFS_PRETRAIN_EPOCHS:-}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

# Match CVS-RFFI satellite-channel evaluation dimensions by default.
EVAL_SAT_CHANNEL="${EVAL_SAT_CHANNEL:-1}"
SAT_EVAL_ON="${SAT_EVAL_ON:-main}"
SAT_EVAL_SCENARIOS="${SAT_EVAL_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:-0}"
SAT_SEED="${SAT_SEED:-2027}"
SAT_FS_HZ="${SAT_FS_HZ:-25e6}"
SAT_FC_HZ="${SAT_FC_HZ:-2.462e9}"

# Paper-specific optional settings.
RA_FUSION="${RA_FUSION:-soft}"
FINETUNE_TARGET_TEST="${FINETUNE_TARGET_TEST:-test_seen_day_unseen_rx}"
FINETUNE_SHOTS_PER_CLASS="${FINETUNE_SHOTS_PER_CLASS:-20}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-20}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${LOG_DIR}"
MANIFEST="${RUN_ROOT}/manifest.tsv"
: > "${MANIFEST}"

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a METHOD_LIST <<< "${METHODS_CSV}"

if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "[BASELINE-QUEUE] GPU_IDS is empty." >&2
  exit 1
fi

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

append_extra_args() {
  local array_name="$1"
  local extra="${2:-}"
  local -n target_array="${array_name}"
  local extra_args=()
  if [ -n "$(trim "${extra}")" ]; then
    read -r -a extra_args <<< "${extra}"
    target_array+=("${extra_args[@]}")
  fi
}

append_optional_arg() {
  local array_name="$1"
  local flag="$2"
  local value="${3:-}"
  local -n target_array="${array_name}"
  if [ -n "$(trim "${value}")" ]; then
    target_array+=("${flag}" "${value}")
  fi
}

common_cvs_args=(
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized "${WISIG_EQUALIZED}"
  --wisig_domain "${WISIG_DOMAIN}"
  --wisig_out_len "${WISIG_OUT_LEN}"
  --wisig_train_ratio "${WISIG_TRAIN_RATIO}"
  --wisig_val_ratio "${WISIG_VAL_RATIO}"
  --wisig_guard_gap "${WISIG_GUARD_GAP}"
  --wisig_train_days "${WISIG_TRAIN_DAYS}"
  --wisig_test_days "${WISIG_TEST_DAYS}"
  --wisig_train_rxs "${WISIG_TRAIN_RXS}"
  --wisig_test_rxs "${WISIG_TEST_RXS}"
  --wisig_max_day123_per_combo "${WISIG_MAX_DAY123_PER_COMBO}"
  --wisig_max_train_per_combo "${WISIG_MAX_TRAIN_PER_COMBO}"
  --wisig_max_val_per_combo "${WISIG_MAX_VAL_PER_COMBO}"
  --wisig_max_test_per_combo "${WISIG_MAX_TEST_PER_COMBO}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --prefetch_factor "${PREFETCH_FACTOR}"
  --seed "${GLOBAL_SEED}"
)

sat_eval_args=()
if [ "${EVAL_SAT_CHANNEL}" = "1" ]; then
  sat_eval_args=(
    --eval_sat_channel
    --eval_sat_on "${SAT_EVAL_ON}"
    --eval_sat_scenarios "${SAT_EVAL_SCENARIOS}"
    --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}"
    --sat_seed "${SAT_SEED}"
    --sat_fs_hz "${SAT_FS_HZ}"
    --sat_fc_hz "${SAT_FC_HZ}"
  )
fi

run_cmd() {
  local gpu="$1"
  local name="$2"
  local log_path="$3"
  shift 3
  echo "[BASELINE-QUEUE][${name}][GPU ${gpu}] log=${log_path}"
  echo "[BASELINE-QUEUE][${name}][GPU ${gpu}] $*"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "$@" 2>&1 | tee "${log_path}"
}

run_method() {
  local method="$1"
  local gpu="$2"
  local module=""
  local run_dir="${RUN_ROOT}/${method}_seed${GLOBAL_SEED}"
  local log_path="${LOG_DIR}/cvs_baseline_${method}_seed${GLOBAL_SEED}_${RUN_STAMP}.log"
  local cmd=()
  mkdir -p "${run_dir}"

  case "${method}" in
    cvcnn)
      module="baselines.cvcnn.train"
      cmd=("${PYTHON_BIN}" -u -m "${module}" "${common_cvs_args[@]}" "${sat_eval_args[@]}" --output_dir "${run_dir}")
      append_optional_arg cmd --epochs "${CVCNN_EPOCHS}"
      append_extra_args cmd "${CVCNN_EXTRA_ARGS:-}"
      ;;
    riei)
      module="baselines.riei.train"
      cmd=("${PYTHON_BIN}" -u -m "${module}" "${common_cvs_args[@]}" "${sat_eval_args[@]}" --output_dir "${run_dir}")
      append_optional_arg cmd --epochs "${RIEI_EPOCHS}"
      append_extra_args cmd "${RIEI_EXTRA_ARGS:-}"
      ;;
    drift)
      module="baselines.drift.train"
      cmd=("${PYTHON_BIN}" -u -m "${module}" "${common_cvs_args[@]}" "${sat_eval_args[@]}" --output_dir "${run_dir}")
      append_optional_arg cmd --epochs "${DRIFT_EPOCHS}"
      append_extra_args cmd "${DRIFT_EXTRA_ARGS:-}"
      ;;
    receiver_agnostic|ra|receiver_agnostic_rffi)
      method="receiver_agnostic"
      module="baselines.receiver_agnostic_rffi.train"
      run_dir="${RUN_ROOT}/${method}_seed${GLOBAL_SEED}"
      log_path="${LOG_DIR}/cvs_baseline_${method}_seed${GLOBAL_SEED}_${RUN_STAMP}.log"
      mkdir -p "${run_dir}"
      cmd=(
        "${PYTHON_BIN}" -u -m "${module}" "${common_cvs_args[@]}" "${sat_eval_args[@]}"
        --collaborative_fusion "${RA_FUSION}"
        --output_dir "${run_dir}"
      )
      append_optional_arg cmd --epochs "${RA_EPOCHS}"
      append_extra_args cmd "${RA_EXTRA_ARGS:-}"
      ;;
    tifs2025|tifs2025_channel_receiver_rffi)
      method="tifs2025"
      module="baselines.tifs2025_channel_receiver_rffi.train"
      run_dir="${RUN_ROOT}/${method}_seed${GLOBAL_SEED}"
      log_path="${LOG_DIR}/cvs_baseline_${method}_seed${GLOBAL_SEED}_${RUN_STAMP}.log"
      mkdir -p "${run_dir}"
      cmd=(
        "${PYTHON_BIN}" -u -m "${module}" "${common_cvs_args[@]}" "${sat_eval_args[@]}"
        --output_dir "${run_dir}"
      )
      append_optional_arg cmd --pretrain_epochs "${TIFS_PRETRAIN_EPOCHS}"
      append_optional_arg cmd --epochs "${TIFS_EPOCHS}"
      append_extra_args cmd "${TIFS_EXTRA_ARGS:-}"
      ;;
    receiver_agnostic_finetune|ra_finetune)
      method="receiver_agnostic_finetune"
      module="baselines.receiver_agnostic_rffi.finetune_cvs"
      run_dir="${RUN_ROOT}/${method}_seed${GLOBAL_SEED}"
      log_path="${LOG_DIR}/cvs_baseline_${method}_seed${GLOBAL_SEED}_${RUN_STAMP}.log"
      mkdir -p "${run_dir}"
      local checkpoint="${RA_FINETUNE_CHECKPOINT:-${RUN_ROOT}/receiver_agnostic_seed${GLOBAL_SEED}/best_by_val.pt}"
      cmd=(
        "${PYTHON_BIN}" -u -m "${module}" "${common_cvs_args[@]}" "${sat_eval_args[@]}"
        --checkpoint "${checkpoint}"
        --target_test "${FINETUNE_TARGET_TEST}"
        --shots_per_class "${FINETUNE_SHOTS_PER_CLASS}"
        --epochs "${FINETUNE_EPOCHS}"
        --output_dir "${run_dir}"
      )
      append_extra_args cmd "${RA_FINETUNE_EXTRA_ARGS:-}"
      ;;
    *)
      echo "[BASELINE-QUEUE] Unknown method '${method}'." >&2
      return 2
      ;;
  esac

  printf '%s\t%s\t%s\t%s\n' "${method}" "${module}" "${log_path}" "${run_dir}" >> "${MANIFEST}"
  run_cmd "${gpu}" "${method}" "${log_path}" "${cmd[@]}"
}

echo "[BASELINE-QUEUE] root=${ROOT_DIR}"
echo "[BASELINE-QUEUE] run_root=${RUN_ROOT}"
echo "[BASELINE-QUEUE] methods=${METHODS_CSV}"
echo "[BASELINE-QUEUE] gpus=${GPU_IDS_CSV}"
echo "[BASELINE-QUEUE] parallel=${PARALLEL}"
echo "[BASELINE-QUEUE] split train_ratio=${WISIG_TRAIN_RATIO} guard_gap=${WISIG_GUARD_GAP} train_days=${WISIG_TRAIN_DAYS} test_days=${WISIG_TEST_DAYS} train_rxs=${WISIG_TRAIN_RXS} test_rxs=${WISIG_TEST_RXS}"
echo "[BASELINE-QUEUE] sat_eval=${EVAL_SAT_CHANNEL} on=${SAT_EVAL_ON} scenarios=${SAT_EVAL_SCENARIOS} max_batches=${SAT_EVAL_MAX_BATCHES}"

gpu_i=0
job_pids=()
job_names=()
for raw_method in "${METHOD_LIST[@]}"; do
  method="$(trim "${raw_method}")"
  if [ -z "${method}" ]; then
    continue
  fi
  gpu="${GPU_LIST[$((gpu_i % ${#GPU_LIST[@]}))]}"
  if [ "${PARALLEL}" = "1" ]; then
    run_method "${method}" "${gpu}" &
    job_pids+=("$!")
    job_names+=("${method}")
  else
    if run_method "${method}" "${gpu}"; then
      :
    else
      status=$?
      echo "[BASELINE-QUEUE] method=${method} failed with status=${status}" >&2
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        exit "${status}"
      fi
    fi
  fi
  gpu_i=$((gpu_i + 1))
done

if [ "${PARALLEL}" = "1" ]; then
  failed=0
  for i in "${!job_pids[@]}"; do
    pid="${job_pids[$i]}"
    name="${job_names[$i]}"
    if wait "${pid}"; then
      echo "[BASELINE-QUEUE] method=${name} finished."
    else
      status=$?
      failed=1
      echo "[BASELINE-QUEUE] method=${name} failed with status=${status}" >&2
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        exit "${status}"
      fi
    fi
  done
  if [ "${failed}" = "1" ] && [ "${STOP_ON_FAIL}" = "1" ]; then
    exit 1
  fi
fi

echo "[BASELINE-QUEUE] finished. manifest=${MANIFEST}"
