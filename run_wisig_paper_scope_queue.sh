#!/usr/bin/env bash
set -uo pipefail

# Paper-scope WiSig launcher for the current comparison:
#   RIEI-FD, DRIFT, and CVS-RFFI CEN_A31_a22_satboost_ce1p28_stack only.
# ERM/DANN/MTL/RA-Collab are intentionally excluded.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"
cd "${WORKSPACE_ROOT}" || exit 1

METHODS_CSV="${METHODS:-riei_fd,drift,cvsrffi_cen_a31}"
WISIG_PROTOCOL="${WISIG_PROTOCOL:-drift_day1}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
RUN_ROOT_USER_SET="${RUN_ROOT+x}"
LOG_ROOT_USER_SET="${LOG_ROOT+x}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/wisig_paper_scope_${WISIG_PROTOCOL}_seed1337}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/wisig_paper_scope_${WISIG_PROTOCOL}_seed1337}"
CVS_TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"

SEED="${SEED:-1337}"
WISIG_SPLIT_SEED="${WISIG_SPLIT_SEED:-${SEED}}"
TRAIN_RATIO="${TRAIN_RATIO:-0.1}"
VAL_RATIO="${VAL_RATIO:-0.9}"
GUARD_GAP="${GUARD_GAP:-8}"
TRAIN_DAYS="${TRAIN_DAYS:-0}"
TEST_DAYS="${TEST_DAYS:-0}"
WISIG_EQUALIZED="${WISIG_EQUALIZED:-1}"
WISIG_DOMAIN="${WISIG_DOMAIN:-rx_day}"
WISIG_OUT_LEN="${WISIG_OUT_LEN:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
CEN_EPOCHS="${CEN_EPOCHS:-170}"
RIEI_PAPER_EVAL_LAST_N="${RIEI_PAPER_EVAL_LAST_N:-}"
DRIFT_PAPER_EVAL_LAST_N="${DRIFT_PAPER_EVAL_LAST_N:-5}"
RIEI_LAMBDA_FEATURE_NORM="${RIEI_LAMBDA_FEATURE_NORM:-0}"
RIEI_OPTIMIZER="${RIEI_OPTIMIZER:-adam}"
RIEI_SGD_MOMENTUM="${RIEI_SGD_MOMENTUM:-0}"
RIEI_CE_REDUCTION="${RIEI_CE_REDUCTION:-sum}"
RIEI_MI_REDUCTION="${RIEI_MI_REDUCTION:-sum}"
RIEI_IE_REDUCTION="${RIEI_IE_REDUCTION:-sum}"
RIEI_WISIG_RMS_NORMALIZE="${RIEI_WISIG_RMS_NORMALIZE:-0}"
RIEI_TEST_EVAL_INTERVAL="${RIEI_TEST_EVAL_INTERVAL:-0}"
RIEI_FED_VARIANT="${RIEI_FED_VARIANT:-imagenet1d}"
DRIFT_MSE_CAP="${DRIFT_MSE_CAP:-0}"
DRIFT_LAMBDA_MSE="${DRIFT_LAMBDA_MSE:-0.02}"
DRIFT_LAMBDA_FEATURE_NORM="${DRIFT_LAMBDA_FEATURE_NORM:-0}"
DRIFT_BATCH_SIZE="${DRIFT_BATCH_SIZE:-64}"
DRIFT_MSE_REDUCTION="${DRIFT_MSE_REDUCTION:-sum}"
DRIFT_GRAD_CLIP_NORM="${DRIFT_GRAD_CLIP_NORM:-0}"
DRIFT_PAPER_SAMPLE_STRATEGY="${DRIFT_PAPER_SAMPLE_STRATEGY:-front}"
DRIFT_WISIG_RMS_NORMALIZE="${DRIFT_WISIG_RMS_NORMALIZE:-1}"
SAT_EVAL="${SAT_EVAL:-0}"
SAT_EVAL_ON="${SAT_EVAL_ON:-test_seen_day_unseen_rx}"
SAT_SCENARIOS="${SAT_SCENARIOS:-clear_leo,low_elev_leo,rain_leo}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:--1}"
CEN_DRIFT_DAY1_DOMAIN_MIN="${CEN_DRIFT_DAY1_DOMAIN_MIN:-3}"

usage() {
  cat <<'EOF'
Options:
  --methods CSV          Methods: riei_fd,drift,cvsrffi_cen_a31
  --wisig-protocol NAME  drift_day1 or riei_original
  --gpu-ids CSV          GPUs to use
  --wisig-pkl PATH       Dataset_WigSig/ManySig.pkl path
  --python PATH          Python executable
  --run-root PATH        Output root
  --log-root PATH        Log root
  --dry-run              Print commands only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --wisig-protocol) WISIG_PROTOCOL="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; RUN_ROOT_USER_SET=1; shift 2 ;;
    --log-root) LOG_ROOT="$2"; LOG_ROOT_USER_SET=1; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${WISIG_PROTOCOL}" in
  drift_day1)
    TRAIN_RXS="${TRAIN_RXS:-1-1,14-7,7-7}"
    TEST_RXS="${TEST_RXS:-1-19,19-2,2-1,2-19,20-1,7-14,8-8}"
    PAPER_DAY="${PAPER_DAY:-0}"
    PAPER_TRAIN_SAMPLES_PER_COMBO="${PAPER_TRAIN_SAMPLES_PER_COMBO:-800}"
    PAPER_VAL_SAMPLES_PER_COMBO="${PAPER_VAL_SAMPLES_PER_COMBO:-200}"
    PAPER_TEST_SAMPLES_PER_COMBO="${PAPER_TEST_SAMPLES_PER_COMBO:-200}"
    ;;
  riei_original)
    TRAIN_RXS="${TRAIN_RXS:-1-1,7-7}"
    TEST_RXS="${TEST_RXS:-1-19}"
    PAPER_DAY="${PAPER_DAY:-0}"
    PAPER_TRAIN_SAMPLES_PER_COMBO="${PAPER_TRAIN_SAMPLES_PER_COMBO:-2400}"
    PAPER_VAL_SAMPLES_PER_COMBO="${PAPER_VAL_SAMPLES_PER_COMBO:-800}"
    PAPER_TEST_SAMPLES_PER_COMBO="${PAPER_TEST_SAMPLES_PER_COMBO:-800}"
    ;;
  *)
    echo "ERROR: WISIG_PROTOCOL must be drift_day1 or riei_original, got ${WISIG_PROTOCOL}" >&2
    exit 2
    ;;
esac
if [ -z "${RIEI_PAPER_EVAL_LAST_N}" ]; then
  RIEI_PAPER_EVAL_LAST_N=5
fi
if [ -z "${RUN_ROOT_USER_SET}" ]; then
  RUN_ROOT="${WORKSPACE_ROOT}/runs/wisig_paper_scope_${WISIG_PROTOCOL}_seed${SEED}"
fi
if [ -z "${LOG_ROOT_USER_SET}" ]; then
  LOG_ROOT="${WORKSPACE_ROOT}/logs/wisig_paper_scope_${WISIG_PROTOCOL}_seed${SEED}"
fi

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3 python python.exe py; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi
if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no python executable found. Pass --python or set PYTHON_BIN." >&2
  exit 2
fi
if [ -z "${CVS_TRAIN_SCRIPT}" ]; then
  if [ -f "${WORKSPACE_ROOT}/train.py" ]; then
    CVS_TRAIN_SCRIPT="${WORKSPACE_ROOT}/train.py"
  else
    CVS_TRAIN_SCRIPT="${WORKSPACE_ROOT}/code/train.py"
  fi
fi
if [ "${DRY_RUN}" != "1" ] && [ ! -f "${WISIG_PKL}" ]; then
  echo "ERROR: WISIG_PKL not found: ${WISIG_PKL}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/code:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
MANIFEST="${RUN_ROOT}/manifest_${STAMP}.tsv"
: > "${MANIFEST}"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

format_cmd() {
  printf "%q " "$@"
}

canonical_method() {
  case "$1" in
    riei|riei_fd) echo "riei_fd" ;;
    drift) echo "drift" ;;
    cen_a31|cvsrffi|cvsrffi_cen_a31) echo "cvsrffi_cen_a31" ;;
    *) return 1 ;;
  esac
}

append_baseline_common_args() {
  CMD+=(
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol "${WISIG_PROTOCOL}"
    --wisig_equalized "${WISIG_EQUALIZED}"
    --wisig_domain "${WISIG_DOMAIN}"
    --wisig_out_len "${WISIG_OUT_LEN}"
    --wisig_train_ratio "${TRAIN_RATIO}"
    --wisig_val_ratio "${VAL_RATIO}"
    --wisig_guard_gap "${GUARD_GAP}"
    --wisig_train_days "${TRAIN_DAYS}"
    --wisig_test_days "${TEST_DAYS}"
    --wisig_train_rxs "${TRAIN_RXS}"
    --wisig_test_rxs "${TEST_RXS}"
    --wisig_paper_day "${PAPER_DAY}"
    --wisig_paper_train_samples_per_combo "${PAPER_TRAIN_SAMPLES_PER_COMBO}"
    --wisig_paper_val_samples_per_combo "${PAPER_VAL_SAMPLES_PER_COMBO}"
    --wisig_paper_test_samples_per_combo "${PAPER_TEST_SAMPLES_PER_COMBO}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --seed "${SEED}"
    --wisig_split_seed "${WISIG_SPLIT_SEED}"
  )
  if [ "${SAT_EVAL}" = "1" ]; then
    CMD+=(--eval_sat_channel --eval_sat_on "${SAT_EVAL_ON}" --eval_sat_scenarios "${SAT_SCENARIOS}" --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}")
  fi
}

append_cen_a31_args() {
  local run_dir="$1"
  local cen_fishr_min_domains=4
  local -a cen_extra_domain_args=()
  if [ "${WISIG_PROTOCOL}" = "drift_day1" ]; then
    cen_fishr_min_domains="${CEN_DRIFT_DAY1_DOMAIN_MIN}"
    cen_extra_domain_args=(--group_ce_min_domains "${CEN_DRIFT_DAY1_DOMAIN_MIN}")
  fi
  CMD+=(
    "${CVS_TRAIN_SCRIPT}"
    --train_mode centralized
    --dataset wisig
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol "${WISIG_PROTOCOL}"
    --wisig_equalized "${WISIG_EQUALIZED}"
    --wisig_domain "${WISIG_DOMAIN}"
    --wisig_out_len "${WISIG_OUT_LEN}"
    --wisig_train_ratio "${TRAIN_RATIO}"
    --wisig_val_ratio "${VAL_RATIO}"
    --wisig_train_days "${TRAIN_DAYS}"
    --wisig_test_days "${TEST_DAYS}"
    --wisig_train_rxs "${TRAIN_RXS}"
    --wisig_test_rxs "${TEST_RXS}"
    --wisig_paper_day "${PAPER_DAY}"
    --wisig_paper_train_samples_per_combo "${PAPER_TRAIN_SAMPLES_PER_COMBO}"
    --wisig_paper_val_samples_per_combo "${PAPER_VAL_SAMPLES_PER_COMBO}"
    --wisig_paper_test_samples_per_combo "${PAPER_TEST_SAMPLES_PER_COMBO}"
    --batch_size 256
    --eval_batch_size 256
    --num_workers 4
    --prefetch_factor 2
    --epochs "${CEN_EPOCHS}"
    --test_eval_policy every_epoch
    --test_eval_start_epoch 81
    --slim_group none
    --branch_ablation no_dac
    --domain_branch_ablation no_stats
    --domain_enhancer rcn_stats
    --domain_enhancer_strength 0.35
    --exp_group s3_rxrobust_no_dac
    --model_variant lite_d
    --use_mixstyle
    --mixstyle_layers time_down,t1
    --mixstyle_mix same_tx_crossdomain
    --mixstyle_fallback skip
    --mixstyle_strength 0.70
    --mixstyle_p 0.18
    --mixstyle_late_start 110
    --mixstyle_late_ramp_epochs 40
    --mixstyle_late_min_p 0.05
    --mixstyle_late_min_strength 0.32
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --concat_sat_start_epoch 1
    --concat_sat_ce_weight 1.28
    --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
    --sat_view_prob 1.00
    --lambda_sat_cls 0.00
    --lambda_sat_cons 0.00
    --lambda_fishr 0.005
    --fishr_min_domains "${cen_fishr_min_domains}"
    --domain_freq_stability_mode dsq
    --freq_stability_channels 2
    --lambda_group_ce 0.06
    --group_ce_mode smooth_dro_capped
    --group_ce_top_frac 0.35
    --groupdro_tau 0.50
    --groupdro_cap 0.65
    --use_proto_memory
    --lambda_proto 0.015
    --proto_momentum 0.95
    --lambda_supcon_id 0.02
    --supcon_temp 0.12
    --generalization_feature z_id
    --seed "${SEED}"
    --run_name "CEN_A31_a22_satboost_ce1p28_stack_${WISIG_PROTOCOL}_seed${SEED}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_seen_day_unseen_rx_save_path "${run_dir}/best_seen_day_unseen_rx_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_protocol_primary_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )
  CMD+=("${cen_extra_domain_args[@]}")
  if [ "${SAT_EVAL}" = "1" ]; then
    CMD+=(--eval_sat_channel --eval_sat_on "${SAT_EVAL_ON}" --eval_sat_scenarios "${SAT_SCENARIOS}" --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}")
  else
    CMD+=(--no_eval_sat_channel)
  fi
}

run_one() {
  local method="$1"
  local gpu="$2"
  local run_name="${method}_${WISIG_PROTOCOL}_seed${SEED}"
  local out_dir="${RUN_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}_${STAMP}.log"
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/metrics.json" ]; then
    log_msg "[PAPER-SCOPE][${method}] skip existing metrics: ${out_dir}/metrics.json"
    return 0
  fi

  if [ "${method}" = "cvsrffi_cen_a31" ]; then
    CMD=("${PYTHON_BIN}" -u)
    append_cen_a31_args "${out_dir}"
  else
    local module
    if [ "${method}" = "riei_fd" ]; then
      module="baselines.riei_fd.train"
    else
      module="baselines.drift.train"
    fi
    CMD=("${PYTHON_BIN}" -u -m "${module}")
    append_baseline_common_args
    CMD+=(--output_dir "${out_dir}" --epochs "${BASELINE_EPOCHS}")
    if [ "${method}" = "riei_fd" ]; then
      CMD+=(--batch_size 64 --lr_all 0.0001 --lr_fed 0.0001 --lambda_mi 1.2 --lambda_ie 1.2)
      CMD+=(--optimizer "${RIEI_OPTIMIZER}" --sgd_momentum "${RIEI_SGD_MOMENTUM}")
      CMD+=(--ce_reduction "${RIEI_CE_REDUCTION}" --mi_reduction "${RIEI_MI_REDUCTION}" --ie_reduction "${RIEI_IE_REDUCTION}")
      CMD+=(--lambda_feature_norm "${RIEI_LAMBDA_FEATURE_NORM}")
      CMD+=(--test_eval_interval "${RIEI_TEST_EVAL_INTERVAL}")
      CMD+=(--fed_variant "${RIEI_FED_VARIANT}")
      if [ "${RIEI_WISIG_RMS_NORMALIZE}" = "1" ]; then
        CMD+=(--wisig_rms_normalize)
      else
        CMD+=(--no-wisig_rms_normalize)
      fi
      CMD+=(--paper_eval_last_n "${RIEI_PAPER_EVAL_LAST_N}" --paper_eval_name "riei_last${RIEI_PAPER_EVAL_LAST_N}")
    else
      CMD+=(
        --batch_size "${DRIFT_BATCH_SIZE}"
        --lr 0.0001
        --lambda_grl 1.0
        --grl_coeff 1.0
        --lambda_center 0.01
        --center_mode batch
        --lambda_mse "${DRIFT_LAMBDA_MSE}"
        --mse_reduction "${DRIFT_MSE_REDUCTION}"
        --mse_cap "${DRIFT_MSE_CAP}"
        --lambda_feature_norm "${DRIFT_LAMBDA_FEATURE_NORM}"
        --grad_clip_norm "${DRIFT_GRAD_CLIP_NORM}"
        --wisig_paper_sample_strategy "${DRIFT_PAPER_SAMPLE_STRATEGY}"
        --no-normalize_features_for_mse
        --domain_discriminator_layers 2
        --grl_schedule constant
      )
      if [ "${DRIFT_WISIG_RMS_NORMALIZE}" = "1" ]; then
        CMD+=(--wisig_rms_normalize)
      else
        CMD+=(--no-wisig_rms_normalize)
      fi
      CMD+=(--paper_eval_last_n "${DRIFT_PAPER_EVAL_LAST_N}" --paper_eval_name "drift_last${DRIFT_PAPER_EVAL_LAST_N}")
    fi
  fi

  printf "%s\t%s\t%s\t%s\n" "${method}" "${log_file}" "${out_dir}" "$(format_cmd "${CMD[@]}")" >> "${MANIFEST}"
  log_msg "[PAPER-SCOPE][${method}][GPU ${gpu}] log=${log_file}"
  log_msg "[PAPER-SCOPE][${method}][GPU ${gpu}] $(format_cmd "${CMD[@]}")"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  if [ "${STREAM_LOGS}" = "1" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "${log_file}"
    return "${PIPESTATUS[0]}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" > "${log_file}" 2>&1
}

log_msg "[PAPER-SCOPE] root=${WORKSPACE_ROOT}"
log_msg "[PAPER-SCOPE] methods=${METHODS_CSV} protocol=${WISIG_PROTOCOL} seed=${SEED}"
log_msg "[PAPER-SCOPE] split train_rxs=${TRAIN_RXS} test_rxs=${TEST_RXS} samples train=${PAPER_TRAIN_SAMPLES_PER_COMBO} val=${PAPER_VAL_SAMPLES_PER_COMBO} test=${PAPER_TEST_SAMPLES_PER_COMBO}"
log_msg "[PAPER-SCOPE] run_root=${RUN_ROOT} log_root=${LOG_ROOT} gpus=${GPU_IDS_CSV} sat_eval=${SAT_EVAL}"
if [ "${WISIG_PROTOCOL}" = "drift_day1" ]; then
  log_msg "[PAPER-SCOPE] cen_drift_day1_domain_min=${CEN_DRIFT_DAY1_DOMAIN_MIN} (CEN_A31 only: group_ce/fishr)"
fi

declare -a PIDS=()
declare -a NAMES=()
status=0
gpu_i=0
for raw_method in "${METHODS[@]}"; do
  requested_method="$(echo "${raw_method}" | xargs)"
  [ -z "${requested_method}" ] && continue
  if ! method="$(canonical_method "${requested_method}")"; then
    log_msg "ERROR: unknown method '${requested_method}'"
    exit 2
  fi
  gpu="${GPU_LIST[$((gpu_i % ${#GPU_LIST[@]}))]}"
  gpu_i=$((gpu_i + 1))
  if [ "${DRY_RUN}" = "1" ]; then
    run_one "${method}" "${gpu}" || status=$?
    continue
  fi
  run_one "${method}" "${gpu}" &
  PIDS+=("$!")
  NAMES+=("${method}")
done

if [ "${DRY_RUN}" != "1" ]; then
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "${pid}"; then
      log_msg "[PAPER-SCOPE][${name}] done"
    else
      rc=$?
      log_msg "[PAPER-SCOPE][${name}] failed rc=${rc}"
      status="${rc}"
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        break
      fi
    fi
  done
fi

log_msg "[PAPER-SCOPE] finished status=${status} manifest=${MANIFEST}"
exit "${status}"
