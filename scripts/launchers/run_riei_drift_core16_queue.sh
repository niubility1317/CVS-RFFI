#!/usr/bin/env bash
set -uo pipefail

# Core RIEI/DRIFT reproduction exploration queue.
# This launcher keeps each paper method's core modules and losses enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="${SCRIPT_DIR}"
cd "${WORKSPACE_ROOT}" || exit 1

WISIG_PROTOCOL="${WISIG_PROTOCOL:-drift_day1}"
GPU_IDS_CSV="${GPU_IDS:-0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7}"
VARIANTS_CSV="${VARIANTS:-RIEI_C01_sum_base,RIEI_C02_sum_clip1,RIEI_C03_sum_clip5,RIEI_C04_sum_lr5e5,RIEI_C05_sum_wd1e4,RIEI_C06_sum_featnorm1e4,RIEI_C07_sum_ie0p8,RIEI_C08_sum_mi_square,DRIFT_C01_raw_base,DRIFT_C02_raw_featnorm1e4,DRIFT_C03_raw_featnorm1e5,DRIFT_C04_raw_cap10000,DRIFT_C05_raw_cap5000,DRIFT_C06_raw_mse0p01,DRIFT_C07_raw_dann_grl,DRIFT_C08_norm_control}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
RUN_ROOT_USER_SET="${RUN_ROOT+x}"
LOG_ROOT_USER_SET="${LOG_ROOT+x}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/riei_drift_core16_${WISIG_PROTOCOL}_seed1337}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/riei_drift_core16_${WISIG_PROTOCOL}_seed1337}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"
STREAM_LOGS="${STREAM_LOGS:-0}"
SKIP_DONE="${SKIP_DONE:-1}"

SEED="${SEED:-1337}"
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
RIEI_PAPER_EVAL_LAST_N="${RIEI_PAPER_EVAL_LAST_N:-}"
DRIFT_PAPER_EVAL_LAST_N="${DRIFT_PAPER_EVAL_LAST_N:-5}"

usage() {
  cat <<'EOF'
Options:
  --variants CSV         Default 16 variants, 8 RIEI + 8 DRIFT
  --wisig-protocol NAME  drift_day1 or riei_original
  --gpu-ids CSV          Must provide one GPU entry per variant; default maps two jobs per GPU
  --wisig-pkl PATH       Dataset_WigSig/ManySig.pkl path
  --python PATH          Python executable
  --run-root PATH        Output root
  --log-root PATH        Log root
  --no-skip-done         Run even if an output directory already has completion markers
  --stop-on-fail         Stop waiting after first failed child
  --stream-logs          Run foreground with tee
  --dry-run              Print commands and create manifest only
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --variants) VARIANTS_CSV="$2"; shift 2 ;;
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
  if [ "${WISIG_PROTOCOL}" = "drift_day1" ]; then
    RIEI_PAPER_EVAL_LAST_N=5
  else
    RIEI_PAPER_EVAL_LAST_N=10
  fi
fi
if [ -z "${RUN_ROOT_USER_SET}" ]; then
  RUN_ROOT="${WORKSPACE_ROOT}/runs/riei_drift_core16_${WISIG_PROTOCOL}_seed${SEED}"
fi
if [ -z "${LOG_ROOT_USER_SET}" ]; then
  LOG_ROOT="${WORKSPACE_ROOT}/logs/riei_drift_core16_${WISIG_PROTOCOL}_seed${SEED}"
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
if [ "${DRY_RUN}" != "1" ] && [ ! -f "${WISIG_PKL}" ]; then
  echo "ERROR: WISIG_PKL not found: ${WISIG_PKL}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a VARIANT_IDS <<< "${VARIANTS_CSV}"
if [ "${#GPU_LIST[@]}" -ne "${#VARIANT_IDS[@]}" ]; then
  echo "ERROR: gpu count (${#GPU_LIST[@]}) must equal variant count (${#VARIANT_IDS[@]})." >&2
  exit 2
fi

export PYTHONPATH="${WORKSPACE_ROOT}:${WORKSPACE_ROOT}/code:${PYTHONPATH:-}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
MANIFEST="${RUN_ROOT}/manifest_${STAMP}.tsv"
printf "variant_id\tmethod\tgpu\tscope\tcore_modules\tcore_losses\tlog_file\toutput_dir\tcommand\n" > "${MANIFEST}"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

format_cmd() {
  printf "%q " "$@"
}

variant_method() {
  case "$1" in
    RIEI_*) echo "riei_fd" ;;
    DRIFT_*) echo "drift" ;;
    *) return 1 ;;
  esac
}

variant_scope() {
  case "$1" in
    RIEI_C01_sum_base|DRIFT_C01_raw_base) echo "paper_literal_baseline" ;;
    DRIFT_C08_norm_control) echo "control_normalized_mse_not_paper_literal" ;;
    *) echo "paper_core_stability_exploration" ;;
  esac
}

variant_core_modules() {
  case "$1" in
    RIEI_*) echo "FED+EC+RC+alternating_training" ;;
    DRIFT_*) echo "tx_rx_split+tx_classifier+rx_classifier+domain_discriminator+receiver_centers" ;;
    *) return 1 ;;
  esac
}

variant_core_losses() {
  case "$1" in
    RIEI_*) echo "CE+lambda_mi*MI-lambda_ie*IE" ;;
    DRIFT_*) echo "tx_CE+rx_CE+lambda_grl*domain_CE+lambda_center*center+lambda_mse*negative_MSE" ;;
    *) return 1 ;;
  esac
}

append_common_args() {
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
  )
}

append_riei_base_args() {
  CMD+=(
    --batch_size 64
    --lr_all "${RIEI_LR_ALL:-0.0001}"
    --lr_fed "${RIEI_LR_FED:-0.0001}"
    --lambda_mi 1.2
    --lambda_ie "${RIEI_LAMBDA_IE:-1.2}"
    --ce_reduction sum
    --mi_reduction sum
    --ie_reduction sum
    --paper_eval_last_n "${RIEI_PAPER_EVAL_LAST_N}"
    --paper_eval_name "riei_last${RIEI_PAPER_EVAL_LAST_N}"
  )
  CMD+=("${RIEI_EXTRA_ARGS[@]}")
}

append_drift_base_args() {
  CMD+=(
    --batch_size 64
    --lr 0.0001
    --lambda_grl 1.0
    --grl_coeff 1.0
    --lambda_center 0.01
    --center_mode ema
    --center_momentum 0.95
    --lambda_mse "${DRIFT_LAMBDA_MSE:-0.02}"
    "${DRIFT_MSE_NORM_FLAG:---no-normalize_features_for_mse}"
    --mse_reduction sum
    --domain_discriminator_layers 2
    --grl_schedule "${DRIFT_GRL_SCHEDULE:-constant}"
    --paper_eval_last_n "${DRIFT_PAPER_EVAL_LAST_N}"
    --paper_eval_name "drift_last${DRIFT_PAPER_EVAL_LAST_N}"
  )
  CMD+=("${DRIFT_EXTRA_ARGS[@]}")
}

append_variant_args() {
  case "$1" in
    RIEI_C01_sum_base) ;;
    RIEI_C02_sum_clip1) RIEI_EXTRA_ARGS+=(--grad_clip_norm 1.0) ;;
    RIEI_C03_sum_clip5) RIEI_EXTRA_ARGS+=(--grad_clip_norm 5.0) ;;
    RIEI_C04_sum_lr5e5) RIEI_LR_ALL=0.00005; RIEI_LR_FED=0.00005 ;;
    RIEI_C05_sum_wd1e4) RIEI_EXTRA_ARGS+=(--weight_decay_all 0.0001 --weight_decay_fed 0.0001) ;;
    RIEI_C06_sum_featnorm1e4) RIEI_EXTRA_ARGS+=(--lambda_feature_norm 0.0001) ;;
    RIEI_C07_sum_ie0p8) RIEI_LAMBDA_IE=0.8 ;;
    RIEI_C08_sum_mi_square) RIEI_EXTRA_ARGS+=(--mi_mode cosine_square) ;;
    DRIFT_C01_raw_base) ;;
    DRIFT_C02_raw_featnorm1e4) DRIFT_EXTRA_ARGS+=(--lambda_feature_norm 0.0001) ;;
    DRIFT_C03_raw_featnorm1e5) DRIFT_EXTRA_ARGS+=(--lambda_feature_norm 0.00001) ;;
    DRIFT_C04_raw_cap10000) DRIFT_EXTRA_ARGS+=(--mse_cap 10000) ;;
    DRIFT_C05_raw_cap5000) DRIFT_EXTRA_ARGS+=(--mse_cap 5000) ;;
    DRIFT_C06_raw_mse0p01) DRIFT_LAMBDA_MSE=0.01 ;;
    DRIFT_C07_raw_dann_grl) DRIFT_GRL_SCHEDULE=dann ;;
    DRIFT_C08_norm_control) DRIFT_MSE_NORM_FLAG=--normalize_features_for_mse ;;
    *) echo "ERROR: unknown variant: $1" >&2; exit 2 ;;
  esac
}

run_one() {
  local variant_id="$1"
  local gpu="$2"
  local method
  local module
  local out_dir
  local log_file
  local scope
  local core_modules
  local core_losses
  local RIEI_LR_ALL=0.0001
  local RIEI_LR_FED=0.0001
  local RIEI_LAMBDA_IE=1.2
  local -a RIEI_EXTRA_ARGS=()
  local DRIFT_LAMBDA_MSE=0.02
  local DRIFT_GRL_SCHEDULE=constant
  local DRIFT_MSE_NORM_FLAG=--no-normalize_features_for_mse
  local -a DRIFT_EXTRA_ARGS=()

  if ! method="$(variant_method "${variant_id}")"; then
    echo "ERROR: unsupported variant id: ${variant_id}" >&2
    exit 2
  fi
  if [ "${method}" = "riei_fd" ]; then
    module="baselines.riei_fd.train"
  else
    module="baselines.drift.train"
  fi

  out_dir="${RUN_ROOT}/${variant_id}"
  log_file="${LOG_ROOT}/${variant_id}.log"
  scope="$(variant_scope "${variant_id}")"
  core_modules="$(variant_core_modules "${variant_id}")"
  core_losses="$(variant_core_losses "${variant_id}")"

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/metrics.json" ]; then
    log_msg "[CORE16][${variant_id}] skip existing metrics: ${out_dir}/metrics.json"
    return 0
  fi

  append_variant_args "${variant_id}"
  mkdir -p "${out_dir}" "$(dirname "${log_file}")"
  CMD=("${PYTHON_BIN}" -u -m "${module}")
  append_common_args
  CMD+=(--output_dir "${out_dir}" --epochs "${BASELINE_EPOCHS}")
  if [ "${method}" = "riei_fd" ]; then
    append_riei_base_args
  else
    append_drift_base_args
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${variant_id}" "${method}" "${gpu}" "${scope}" "${core_modules}" "${core_losses}" \
    "${log_file}" "${out_dir}" "$(format_cmd "${CMD[@]}")" >> "${MANIFEST}"

  log_msg "[CORE16][${variant_id}][${method}][GPU ${gpu}] scope=${scope}"
  log_msg "[CORE16][${variant_id}] core_modules=${core_modules}"
  log_msg "[CORE16][${variant_id}] core_losses=${core_losses}"
  log_msg "[CORE16][${variant_id}] log=${log_file}"
  log_msg "[CORE16][${variant_id}] $(format_cmd "${CMD[@]}")"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  if [ "${STREAM_LOGS}" = "1" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "${log_file}"
    return "${PIPESTATUS[0]}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" > "${log_file}" 2>&1
}

log_msg "[CORE16] root=${WORKSPACE_ROOT}"
log_msg "[CORE16] protocol=${WISIG_PROTOCOL} seed=${SEED} epochs=${BASELINE_EPOCHS}"
log_msg "[CORE16] split train_rxs=${TRAIN_RXS} test_rxs=${TEST_RXS} train_ratio=${TRAIN_RATIO} val_ratio=${VAL_RATIO}"
log_msg "[CORE16] samples train=${PAPER_TRAIN_SAMPLES_PER_COMBO} val=${PAPER_VAL_SAMPLES_PER_COMBO} test=${PAPER_TEST_SAMPLES_PER_COMBO}"
log_msg "[CORE16] run_root=${RUN_ROOT} log_root=${LOG_ROOT}"
log_msg "[CORE16] variants=${VARIANTS_CSV}"
log_msg "[CORE16] gpus=${GPU_IDS_CSV}"
log_msg "[CORE16] invariant=RIEI keeps CE+MI+IE; DRIFT keeps GRL+center+negative_MSE."

declare -a PIDS=()
declare -a NAMES=()
status=0
for i in "${!VARIANT_IDS[@]}"; do
  variant_id="${VARIANT_IDS[$i]}"
  gpu="${GPU_LIST[$i]}"
  if [ "${DRY_RUN}" = "1" ]; then
    run_one "${variant_id}" "${gpu}" || status=$?
    continue
  fi
  run_one "${variant_id}" "${gpu}" &
  PIDS+=("$!")
  NAMES+=("${variant_id}")
done

if [ "${DRY_RUN}" != "1" ]; then
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "${pid}"; then
      log_msg "[CORE16][${name}] done"
    else
      rc=$?
      log_msg "[CORE16][${name}] failed rc=${rc}"
      status="${rc}"
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        break
      fi
    fi
  done
fi

log_msg "[CORE16] finished status=${status} manifest=${MANIFEST}"
exit "${status}"
