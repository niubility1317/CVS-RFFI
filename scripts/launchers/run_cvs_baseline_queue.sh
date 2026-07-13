#!/usr/bin/env bash
set -uo pipefail

# CVS-RFFI paper-baseline queue launcher.
#
# This public launcher executes from the repository root even though it lives in
# scripts/launchers/. It writes run outputs only under runs/ and logs/, both
# ignored by git.
#
# Typical uses:
#   conda activate ssr-gpu
#   bash scripts/launchers/run_cvs_baseline_queue.sh --dry-run
#   bash scripts/launchers/run_cvs_baseline_queue.sh --methods riei_fd --wisig-protocol riei_original
#   bash scripts/launchers/run_cvs_baseline_queue.sh --methods drift --wisig-protocol drift_day1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

METHODS_CSV="${METHODS:-riei_fd,drift}"
GPU_IDS_CSV="${GPU_IDS:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-${REPO_ROOT}/Dataset_WigSig/ManySig.pkl}"
WISIG_PROTOCOL="${WISIG_PROTOCOL:-cvs_day_rx}"
RUN_ROOT="${RUN_ROOT:-}"
LOG_ROOT="${LOG_ROOT:-}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

SEED="${SEED:-1337}"
TRAIN_RATIO="${TRAIN_RATIO:-0.1}"
VAL_RATIO="${VAL_RATIO:-0.9}"
SSL_MODE="${SSL_MODE:-none}"
LABELED_RATIO="${LABELED_RATIO:-0.1}"
UNLABELED_RATIO="${UNLABELED_RATIO:-0.6}"
SOURCE_VAL_RATIO="${SOURCE_VAL_RATIO:-0.3}"
PSEUDO_START_EPOCH="${PSEUDO_START_EPOCH:-1}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.95}"
PSEUDO_MARGIN="${PSEUDO_MARGIN:-0.0}"
LAMBDA_PSEUDO="${LAMBDA_PSEUDO:-1.0}"
CONSISTENCY_START_EPOCH="${CONSISTENCY_START_EPOCH:-1}"
CONSISTENCY_TEMPERATURE="${CONSISTENCY_TEMPERATURE:-1.0}"
LAMBDA_CONSISTENCY="${LAMBDA_CONSISTENCY:-1.0}"
GUARD_GAP="${GUARD_GAP:-8}"
TRAIN_DAYS_USER_SET="${TRAIN_DAYS+x}"
TEST_DAYS_USER_SET="${TEST_DAYS+x}"
TRAIN_RXS_USER_SET="${TRAIN_RXS+x}"
TEST_RXS_USER_SET="${TEST_RXS+x}"
TRAIN_DAYS="${TRAIN_DAYS:-0,1}"
TEST_DAYS="${TEST_DAYS:-2,3}"
TRAIN_RXS="${TRAIN_RXS:-0,1,2,3,4,5,6}"
TEST_RXS="${TEST_RXS:-7,8,9,10,11}"
PAPER_DAY="${PAPER_DAY:-0}"
PAPER_TRAIN_SAMPLES_PER_COMBO="${PAPER_TRAIN_SAMPLES_PER_COMBO:-800}"
PAPER_VAL_SAMPLES_PER_COMBO="${PAPER_VAL_SAMPLES_PER_COMBO:-200}"
PAPER_TEST_SAMPLES_PER_COMBO="${PAPER_TEST_SAMPLES_PER_COMBO:-200}"
WISIG_EQUALIZED="${WISIG_EQUALIZED:-1}"
WISIG_DOMAIN="${WISIG_DOMAIN:-rx_day}"
WISIG_OUT_LEN="${WISIG_OUT_LEN:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

STANDARD_EPOCHS="${STANDARD_EPOCHS:-200}"
RIEI_PAPER_EVAL_LAST_N="${RIEI_PAPER_EVAL_LAST_N:-10}"
DRIFT_PAPER_EVAL_LAST_N="${DRIFT_PAPER_EVAL_LAST_N:-5}"
DEFAULT_PAPER_EVAL_LAST_N="${DEFAULT_PAPER_EVAL_LAST_N:-5}"

SAT_EVAL="${SAT_EVAL:-1}"
SAT_EVAL_ON="${SAT_EVAL_ON:-main}"
SAT_SCENARIOS="${SAT_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:-0}"
SAT_SEED="${SAT_SEED:-2027}"
SAT_FS_HZ="${SAT_FS_HZ:-25e6}"
SAT_FC_HZ="${SAT_FC_HZ:-2.462e9}"

SAT_VIEW_AUG="${SAT_VIEW_AUG:-0}"
SAT_TRAIN_SCENARIO="${SAT_TRAIN_SCENARIO:-clear_leo}"
SAT_TRAIN_SCENARIOS="${SAT_TRAIN_SCENARIOS:-}"
SAT_VIEW_PROB="${SAT_VIEW_PROB:-1.0}"
SAT_VIEW_SEED="${SAT_VIEW_SEED:-2027}"

usage() {
  sed -n '1,16p' "$0"
  cat <<'EOF'

Options:
  --methods CSV          Methods: cvcnn_ce,riei_fd,drift,ra_collab
  --gpu-ids CSV          CUDA ids assigned round-robin
  --wisig-pkl PATH       Dataset_WigSig/ManySig.pkl path
  --wisig-protocol NAME  cvs_day_rx, riei_original, or drift_day1
  --python PATH          Python executable
  --run-root PATH        Output checkpoint/metrics root
  --log-root PATH        Log root
  --seed N               Random seed, default 1337
  --sat-view-aug         Enable source-derived satellite-channel train view
  --sat-train-scenario NAME
                         Train-view scenario when --sat-view-aug is enabled
  --sat-train-scenarios CSV
                         Cycle multiple train-view scenarios
  --no-sat-eval          Disable satellite-channel OOD evaluation
  --no-skip-done         Re-run even when metrics.json exists
  --stop-on-fail         Stop after first failed job
  --dry-run              Print commands without checking dataset or launching
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --wisig-protocol|--protocol) WISIG_PROTOCOL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --sat-view-aug) SAT_VIEW_AUG=1; shift ;;
    --sat-train-scenario) SAT_TRAIN_SCENARIO="$2"; shift 2 ;;
    --sat-train-scenarios) SAT_TRAIN_SCENARIOS="$2"; shift 2 ;;
    --no-sat-eval) SAT_EVAL=0; shift ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${WISIG_PROTOCOL}" in
  cvs_day_rx|riei_original|drift_day1) ;;
  *) echo "ERROR: unsupported --wisig-protocol '${WISIG_PROTOCOL}'." >&2; exit 2 ;;
esac

case "${SSL_MODE}" in
  none|pseudo_label|augmentation_consistency) ;;
  *) echo "ERROR: unsupported SSL_MODE '${SSL_MODE}'." >&2; exit 2 ;;
esac

if [ "${WISIG_PROTOCOL}" = "drift_day1" ]; then
  [ -z "${TRAIN_DAYS_USER_SET}" ] && TRAIN_DAYS="0"
  [ -z "${TEST_DAYS_USER_SET}" ] && TEST_DAYS="0"
  [ -z "${TRAIN_RXS_USER_SET}" ] && TRAIN_RXS="1-1,14-7,7-7"
  [ -z "${TEST_RXS_USER_SET}" ] && TEST_RXS="1-19,19-2,2-1,2-19,20-1,7-14,8-8"
elif [ "${WISIG_PROTOCOL}" = "riei_original" ]; then
  [ -z "${TRAIN_DAYS_USER_SET}" ] && TRAIN_DAYS="0"
  [ -z "${TEST_DAYS_USER_SET}" ] && TEST_DAYS="0"
  [ -z "${TRAIN_RXS_USER_SET}" ] && TRAIN_RXS="1-1,7-7"
  [ -z "${TEST_RXS_USER_SET}" ] && TEST_RXS="1-19"
  PAPER_TRAIN_SAMPLES_PER_COMBO="2400"
  PAPER_VAL_SAMPLES_PER_COMBO="800"
  PAPER_TEST_SAMPLES_PER_COMBO="800"
fi

RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/baseline_paper_audit_${WISIG_PROTOCOL}_seed${SEED}}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/baseline_paper_audit_${WISIG_PROTOCOL}_seed${SEED}}"

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
  echo "Set WISIG_PKL=/path/to/Dataset_WigSig/ManySig.pkl or pass --wisig-pkl." >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ "${CONDA_DEFAULT_ENV:-}" != "ssr-gpu" ]; then
  echo "WARNING: CONDA_DEFAULT_ENV='${CONDA_DEFAULT_ENV:-}'. Project instructions expect: conda activate ssr-gpu" >&2
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/code:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
MANIFEST="${RUN_ROOT}/manifest_${STAMP}.tsv"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

canonical_method() {
  case "$1" in
    cvcnn_ce|riei_fd|drift|ra_collab) echo "$1" ;;
    *) return 1 ;;
  esac
}

module_for_method() {
  case "$1" in
    cvcnn_ce) echo "baselines.cvcnn_ce.train" ;;
    riei_fd) echo "baselines.riei_fd.train" ;;
    drift) echo "baselines.drift.train" ;;
    ra_collab) echo "baselines.ra_collab.train" ;;
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
    --wisig_guard_gap "${GUARD_GAP}"
    --wisig_train_days "${TRAIN_DAYS}"
    --wisig_test_days "${TEST_DAYS}"
    --wisig_train_rxs "${TRAIN_RXS}"
    --wisig_test_rxs "${TEST_RXS}"
    --wisig_max_day123_per_combo 0
    --wisig_max_train_per_combo 0
    --wisig_max_val_per_combo 0
    --wisig_max_test_per_combo 0
    --wisig_paper_day "${PAPER_DAY}"
    --wisig_paper_train_samples_per_combo "${PAPER_TRAIN_SAMPLES_PER_COMBO}"
    --wisig_paper_val_samples_per_combo "${PAPER_VAL_SAMPLES_PER_COMBO}"
    --wisig_paper_test_samples_per_combo "${PAPER_TEST_SAMPLES_PER_COMBO}"
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --seed "${SEED}"
  )
  if [ "${SSL_MODE}" = "none" ]; then
    CMD+=(--wisig_train_ratio "${TRAIN_RATIO}" --wisig_val_ratio "${VAL_RATIO}")
  else
    CMD+=(
      --use_source_ssl_split
      --wisig_labeled_ratio "${LABELED_RATIO}"
      --wisig_unlabeled_ratio "${UNLABELED_RATIO}"
      --wisig_source_val_ratio "${SOURCE_VAL_RATIO}"
    )
  fi
  if [ "${SAT_EVAL}" = "1" ]; then
    CMD+=(
      --eval_sat_channel
      --eval_sat_on "${SAT_EVAL_ON}"
      --eval_sat_scenarios "${SAT_SCENARIOS}"
      --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}"
      --sat_seed "${SAT_SEED}"
      --sat_fs_hz "${SAT_FS_HZ}"
      --sat_fc_hz "${SAT_FC_HZ}"
    )
  fi
  if [ "${SAT_VIEW_AUG}" = "1" ]; then
    CMD+=(--use_sat_channel_view_aug)
    if [ -n "${SAT_TRAIN_SCENARIOS}" ]; then
      CMD+=(--sat_train_scenarios "${SAT_TRAIN_SCENARIOS}")
    else
      CMD+=(--sat_train_scenario "${SAT_TRAIN_SCENARIO}")
    fi
    CMD+=(--sat_view_prob "${SAT_VIEW_PROB}" --sat_view_seed "${SAT_VIEW_SEED}")
  fi
}

append_method_args() {
  local method="$1"
  CMD+=(--epochs "${STANDARD_EPOCHS}")
  case "${method}" in
    riei_fd)
      CMD+=(--paper_eval_last_n "${RIEI_PAPER_EVAL_LAST_N}" --paper_eval_name "riei_last${RIEI_PAPER_EVAL_LAST_N}") ;;
    drift)
      CMD+=(--paper_eval_last_n "${DRIFT_PAPER_EVAL_LAST_N}" --paper_eval_name "drift_last${DRIFT_PAPER_EVAL_LAST_N}") ;;
    cvcnn_ce|ra_collab)
      CMD+=(--paper_eval_last_n "${DEFAULT_PAPER_EVAL_LAST_N}" --paper_eval_name "aligned_wisig_last${DEFAULT_PAPER_EVAL_LAST_N}") ;;
  esac
  if [ "${method}" = "ra_collab" ]; then
    CMD+=(--collaborative_fusion soft)
  fi
  if [ "${SSL_MODE}" = "pseudo_label" ]; then
    CMD+=(
      --use_pseudo_labels
      --pseudo_start_epoch "${PSEUDO_START_EPOCH}"
      --pseudo_threshold "${PSEUDO_THRESHOLD}"
      --pseudo_margin "${PSEUDO_MARGIN}"
      --lambda_pseudo "${LAMBDA_PSEUDO}"
    )
  elif [ "${SSL_MODE}" = "augmentation_consistency" ]; then
    CMD+=(
      --use_augmentation_consistency
      --consistency_start_epoch "${CONSISTENCY_START_EPOCH}"
      --consistency_temperature "${CONSISTENCY_TEMPERATURE}"
      --lambda_consistency "${LAMBDA_CONSISTENCY}"
    )
  fi
}

format_cmd() {
  printf "%q " "$@"
}

run_one() {
  local method="$1"
  local gpu="$2"
  local module="$3"
  local out_dir="${RUN_ROOT}/${method}_${WISIG_PROTOCOL}_seed${SEED}"
  local log_file="${LOG_ROOT}/baseline_${method}_${WISIG_PROTOCOL}_seed${SEED}_${STAMP}.log"

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/metrics.json" ]; then
    log_msg "[BASELINE-QUEUE][${method}] skip existing metrics: ${out_dir}/metrics.json"
    return 0
  fi

  CMD=("${PYTHON_BIN}" -u -m "${module}")
  append_common_args
  CMD+=(--output_dir "${out_dir}")
  append_method_args "${method}"

  local printable
  printable="$(format_cmd "${CMD[@]}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${method}" "${WISIG_PROTOCOL}" "${module}" "${log_file}" "${out_dir}" "${printable}" >> "${MANIFEST}"
  log_msg "[BASELINE-QUEUE][${method}][GPU ${gpu}] log=${log_file}"
  log_msg "[BASELINE-QUEUE][${method}][GPU ${gpu}] ${printable}"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" > "${log_file}" 2>&1
}

IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
: > "${MANIFEST}"

log_msg "[BASELINE-QUEUE] root=${REPO_ROOT}"
log_msg "[BASELINE-QUEUE] run_root=${RUN_ROOT}"
log_msg "[BASELINE-QUEUE] log_root=${LOG_ROOT}"
log_msg "[BASELINE-QUEUE] methods=${METHODS_CSV}"
log_msg "[BASELINE-QUEUE] gpus=${GPU_IDS_CSV}"
if [ "${SSL_MODE}" = "none" ]; then
  log_msg "[BASELINE-QUEUE] protocol=${WISIG_PROTOCOL} train_ratio=${TRAIN_RATIO} val_ratio=${VAL_RATIO} guard_gap=${GUARD_GAP}"
else
  log_msg "[BASELINE-QUEUE] protocol=${WISIG_PROTOCOL} source_ssl=1 guard_gap=${GUARD_GAP}"
fi
log_msg "[BASELINE-QUEUE] ssl_mode=${SSL_MODE} labeled_ratio=${LABELED_RATIO} unlabeled_ratio=${UNLABELED_RATIO} source_val_ratio=${SOURCE_VAL_RATIO}"
log_msg "[BASELINE-QUEUE] train_days=${TRAIN_DAYS} test_days=${TEST_DAYS} train_rxs=${TRAIN_RXS} test_rxs=${TEST_RXS}"
log_msg "[BASELINE-QUEUE] paper_samples train=${PAPER_TRAIN_SAMPLES_PER_COMBO} val=${PAPER_VAL_SAMPLES_PER_COMBO} test=${PAPER_TEST_SAMPLES_PER_COMBO}"
log_msg "[BASELINE-QUEUE] paper_eval_last_n default=${DEFAULT_PAPER_EVAL_LAST_N} riei=${RIEI_PAPER_EVAL_LAST_N} drift=${DRIFT_PAPER_EVAL_LAST_N}"
log_msg "[BASELINE-QUEUE] seed=${SEED} sat_eval=${SAT_EVAL} eval_on=${SAT_EVAL_ON} scenarios=${SAT_SCENARIOS}"
log_msg "[BASELINE-QUEUE] sat_view_aug=${SAT_VIEW_AUG} train_scenario=${SAT_TRAIN_SCENARIO} train_scenarios=${SAT_TRAIN_SCENARIOS:-<single>}"

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
  if ! module="$(module_for_method "${method}")"; then
    log_msg "ERROR: unknown canonical method '${method}'"
    exit 2
  fi
  gpu="${GPU_LIST[$((gpu_i % ${#GPU_LIST[@]}))]}"
  gpu_i=$((gpu_i + 1))
  if [ "${DRY_RUN}" = "1" ]; then
    run_one "${method}" "${gpu}" "${module}" || status=$?
    continue
  fi
  run_one "${method}" "${gpu}" "${module}" &
  PIDS+=("$!")
  NAMES+=("${method}")
done

if [ "${DRY_RUN}" != "1" ]; then
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    name="${NAMES[$i]}"
    if wait "${pid}"; then
      log_msg "[BASELINE-QUEUE][${name}] done"
    else
      rc=$?
      log_msg "[BASELINE-QUEUE][${name}] failed rc=${rc}"
      status="${rc}"
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        break
      fi
    fi
  done
fi

log_msg "[BASELINE-QUEUE] finished status=${status} manifest=${MANIFEST}"
exit "${status}"
