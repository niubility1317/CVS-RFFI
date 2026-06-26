#!/usr/bin/env bash
set -euo pipefail

# CVS data ratio sweep for fixed RIEI/DRIFT paper-core variants.
# Default: satellite channel view augmentation and satellite evaluation enabled.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "${ROOT}" || exit 1

RUN_ID="${RUN_ID:-cvs_fixed_riei_drift_ratio_sweep_20260608_214039}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RATIOS_CSV="${TRAIN_RATIOS:-0.1,0.05,0.03,0.02,0.01,0.005}"
METHODS_CSV="${METHODS:-riei_fixed_sat,drift_fixed_sat}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7,0,1,2,3}"
PYTHON_BIN="${PYTHON_BIN:-}"
WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
STREAM_LOGS="${STREAM_LOGS:-0}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
QUEUE_SLOT_POLL_SECONDS="${QUEUE_SLOT_POLL_SECONDS:-60}"

SEED="${SEED:-1337}"
WISIG_PROTOCOL="${WISIG_PROTOCOL:-cvs_day_rx}"
VAL_RATIO="${VAL_RATIO:--1.0}"
GUARD_GAP="${GUARD_GAP:-8}"
TRAIN_DAYS="${TRAIN_DAYS:-0,1}"
TEST_DAYS="${TEST_DAYS:-2,3}"
TRAIN_RXS="${TRAIN_RXS:-0,1,2,3,4,5,6}"
TEST_RXS="${TEST_RXS:-7,8,9,10,11}"
WISIG_SPLIT_STRATEGY="${WISIG_SPLIT_STRATEGY:-random}"
WISIG_CAP_STRATEGY="${WISIG_CAP_STRATEGY:-random}"
WISIG_EQUALIZED="${WISIG_EQUALIZED:-1}"
WISIG_DOMAIN="${WISIG_DOMAIN:-rx_day}"
WISIG_OUT_LEN="${WISIG_OUT_LEN:-256}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
PAPER_EVAL_LAST_N="${PAPER_EVAL_LAST_N:-1}"

SAT_EVAL="${SAT_EVAL:-1}"
SAT_EVAL_ON="${SAT_EVAL_ON:-main}"
SAT_SCENARIOS="${SAT_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:-0}"
SAT_TRAIN_SCENARIOS="${SAT_TRAIN_SCENARIOS:-${SAT_SCENARIOS}}"
SAT_TRAIN_AUG="${SAT_TRAIN_AUG:-1}"
SAT_VIEW_PROB="${SAT_VIEW_PROB:-1.00}"
SAT_VIEW_SEED="${SAT_VIEW_SEED:-2027}"

usage() {
  cat <<'EOF'
Options:
  --ratios CSV           Train ratios; default 0.1,0.05,0.03,0.02,0.01,0.005
  --methods CSV          riei_fixed_sat,drift_fixed_sat,riei_paper_nosat,drift_paper_nosat,riei_paper_sat,drift_paper_sat
  --gpu-ids CSV          One GPU per generated job; default 12-job mapping
  --python PATH          Python executable
  --wisig-pkl PATH       Dataset_WigSig/ManySig.pkl path
  --run-id NAME          Run id under runs/ and logs/
  --run-root PATH        Output root
  --log-root PATH        Log root
  --max-train-per-gpu N  Queue capacity guard; default 2
  --no-skip-done         Re-run even when metrics.json exists
  --stream-logs          Run foreground logs inside queue jobs
  --no-sat-train-aug     Disable satellite channel view augmentation during training
  --dry-run              Print commands and write manifest only

Method label convention:
  riei_paper_* / drift_paper_*  = original paper method tags.
  riei_fixed_* / drift_fixed_*  = optimized implementation tags.
  RIEI fixed keeps CE+MI-IE and adds lambda_feature_norm=0.0001.
  DRIFT fixed keeps raw negative-MSE separation and adds mse_cap=4000.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --ratios) RATIOS_CSV="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; RUN_ROOT="${ROOT}/runs/${RUN_ID}"; LOG_ROOT="${ROOT}/logs/${RUN_ID}"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --max-train-per-gpu) MAX_TRAIN_PER_GPU="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
    --no-sat-train-aug) SAT_TRAIN_AUG=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ "${WISIG_PROTOCOL}" != "cvs_day_rx" ]; then
  echo "ERROR: this launcher is for CVS data only: WISIG_PROTOCOL=cvs_day_rx, got ${WISIG_PROTOCOL}" >&2
  exit 2
fi
if [ "${VAL_RATIO}" != "-1.0" ]; then
  echo "ERROR: VAL_RATIO must stay -1.0 so wisig_train_ratio is not overwritten by cvs_data.py; got ${VAL_RATIO}" >&2
  exit 2
fi

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python python3 python python.exe py; do
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

IFS=',' read -r -a RATIOS <<< "${RATIOS_CSV}"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"

if [ "${#RATIOS[@]}" -lt 1 ] || [ "${#METHODS[@]}" -lt 1 ]; then
  echo "ERROR: ratios and methods must be non-empty." >&2
  exit 2
fi

expected_jobs=$(( ${#RATIOS[@]} * ${#METHODS[@]} ))
if [ "${#GPU_LIST[@]}" -ne "${expected_jobs}" ]; then
  echo "ERROR: GPU_IDS count (${#GPU_LIST[@]}) must equal ratio*method jobs (${expected_jobs})." >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
MANIFEST="${RUN_ROOT}/manifest_${STAMP}.tsv"
printf "job_id\tmethod\tratio\tgpu\tscope\tcore_modules\tcore_losses\tsat_aug\tlog_file\toutput_dir\tcommand\n" > "${MANIFEST}"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

format_cmd() {
  printf "%q " "$@"
}

ratio_tag() {
  case "$1" in
    0.1|.1) echo "r010" ;;
    0.05|.05) echo "r005" ;;
    0.03|.03) echo "r003" ;;
    0.02|.02) echo "r002" ;;
    0.01|.01) echo "r001" ;;
    0.005|.005) echo "r0005" ;;
    *) echo "r$(echo "$1" | tr '.' 'p')" ;;
  esac
}

canonical_method() {
  case "$1" in
    riei|riei_fixed|riei_fixed_sat|RIEI_FIXED_SAT) echo "riei_fixed_sat" ;;
    riei_nosat|riei_fixed_nosat|RIEI_FIXED_NOSAT) echo "riei_fixed_nosat" ;;
    riei_paper|riei_paper_nosat|RIEI_PAPER_NOSAT) echo "riei_paper_nosat" ;;
    riei_paper_sat|RIEI_PAPER_SAT) echo "riei_paper_sat" ;;
    drift|drfit|drift_fixed|drift_fixed_sat|DRIFT_FIXED_SAT) echo "drift_fixed_sat" ;;
    drift_nosat|drfit_nosat|drift_fixed_nosat|DRIFT_FIXED_NOSAT) echo "drift_fixed_nosat" ;;
    drift_paper|drfit_paper|drift_paper_nosat|DRIFT_PAPER_NOSAT) echo "drift_paper_nosat" ;;
    drift_paper_sat|drfit_paper_sat|DRIFT_PAPER_SAT) echo "drift_paper_sat" ;;
    *) return 1 ;;
  esac
}

append_shared_cvs_args() {
  local ratio="$1"
  CMD+=(
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol "${WISIG_PROTOCOL}"
    --wisig_equalized "${WISIG_EQUALIZED}"
    --wisig_domain "${WISIG_DOMAIN}"
    --wisig_out_len "${WISIG_OUT_LEN}"
    --wisig_train_ratio "${ratio}"
    --wisig_val_ratio "${VAL_RATIO}"
    --wisig_guard_gap "${GUARD_GAP}"
    --wisig_train_days "${TRAIN_DAYS}"
    --wisig_test_days "${TEST_DAYS}"
    --wisig_train_rxs "${TRAIN_RXS}"
    --wisig_test_rxs "${TEST_RXS}"
    --wisig_split_strategy "${WISIG_SPLIT_STRATEGY}"
    --wisig_cap_strategy "${WISIG_CAP_STRATEGY}"
    --wisig_max_day123_per_combo 0
    --wisig_max_train_per_combo 0
    --wisig_max_val_per_combo 0
    --wisig_max_test_per_combo 0
    --eval_batch_size "${EVAL_BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --seed "${SEED}"
  )
}

append_sat_args() {
  local method="$1"
  if [ "${SAT_EVAL}" = "1" ]; then
    CMD+=(--eval_sat_channel --eval_sat_on "${SAT_EVAL_ON}" --eval_sat_scenarios "${SAT_SCENARIOS}" --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}")
  else
    CMD+=(--no_eval_sat_channel)
  fi
  case "${method}" in
    *_sat)
      if [ "${SAT_TRAIN_AUG}" = "1" ]; then
        CMD+=(
          --use_sat_channel_view_aug
          --sat_train_scenarios "${SAT_TRAIN_SCENARIOS}"
          --sat_view_prob "${SAT_VIEW_PROB}"
          --sat_view_seed "${SAT_VIEW_SEED}"
        )
      fi
      ;;
    *_nosat)
      ;;
    *)
      echo "ERROR: unsupported satellite augmentation method tag: ${method}" >&2
      exit 2
      ;;
  esac
}

build_cmd() {
  local method="$1"
  local ratio="$2"
  local out_dir="$3"
  if [ "${method}" = "riei_fixed_sat" ] || [ "${method}" = "riei_fixed_nosat" ] || [ "${method}" = "riei_paper_nosat" ] || [ "${method}" = "riei_paper_sat" ]; then
    CMD=("${PYTHON_BIN}" -u -m baselines.riei_fd.train)
    append_shared_cvs_args "${ratio}"
    append_sat_args "${method}"
    CMD+=(
      --output_dir "${out_dir}"
      --epochs "${BASELINE_EPOCHS}"
      --no_test_on_val_improve
      --batch_size 64
      --lr_all 0.0001
      --lr_fed 0.0001
      --lambda_mi 1.2
      --lambda_ie 1.2
      --ce_reduction sum
      --mi_reduction sum
      --ie_reduction sum
      --paper_eval_last_n "${PAPER_EVAL_LAST_N}"
    )
    if [ "${method}" = "riei_fixed_sat" ] || [ "${method}" = "riei_fixed_nosat" ]; then
      CMD+=(--lambda_feature_norm 0.0001)
      CMD+=(--paper_eval_name "cvs_riei_fixed_${method##*_}_last${PAPER_EVAL_LAST_N}")
    else
      CMD+=(--paper_eval_name "cvs_riei_paper_${method##*_}_last${PAPER_EVAL_LAST_N}")
    fi
  elif [ "${method}" = "drift_fixed_sat" ] || [ "${method}" = "drift_fixed_nosat" ] || [ "${method}" = "drift_paper_nosat" ] || [ "${method}" = "drift_paper_sat" ]; then
    CMD=("${PYTHON_BIN}" -u -m baselines.drift.train)
    append_shared_cvs_args "${ratio}"
    append_sat_args "${method}"
    CMD+=(
      --output_dir "${out_dir}"
      --epochs "${BASELINE_EPOCHS}"
      --no_test_on_val_improve
      --batch_size 64
      --lr 0.0001
      --lambda_grl 1.0
      --grl_coeff 1.0
      --lambda_center 0.01
      --center_mode ema
      --center_momentum 0.95
      --lambda_mse 0.02
      --no-normalize_features_for_mse
      --mse_reduction sum
      --domain_discriminator_layers 2
      --grl_schedule constant
      --paper_eval_last_n "${PAPER_EVAL_LAST_N}"
    )
    if [ "${method}" = "drift_fixed_sat" ] || [ "${method}" = "drift_fixed_nosat" ]; then
      CMD+=(--mse_cap 4000)
      CMD+=(--paper_eval_name "cvs_drift_fixed_${method##*_}_last${PAPER_EVAL_LAST_N}")
    else
      CMD+=(--paper_eval_name "cvs_drift_paper_${method##*_}_last${PAPER_EVAL_LAST_N}")
    fi
  else
    echo "ERROR: unsupported method: ${method}" >&2
    exit 2
  fi
}

method_scope() {
  case "$1" in
    riei_fixed_sat) echo "cvs_ratio_sweep_fixed_riei_sat_view" ;;
    riei_fixed_nosat) echo "cvs_ratio_sweep_fixed_riei_no_sat_view" ;;
    riei_paper_sat) echo "cvs_ratio_sweep_paper_original_riei_sat_view" ;;
    riei_paper_nosat) echo "cvs_ratio_sweep_paper_original_riei_no_sat_view" ;;
    drift_fixed_sat) echo "cvs_ratio_sweep_fixed_drift_sat_view" ;;
    drift_fixed_nosat) echo "cvs_ratio_sweep_fixed_drift_no_sat_view" ;;
    drift_paper_sat) echo "cvs_ratio_sweep_paper_original_drift_sat_view" ;;
    drift_paper_nosat) echo "cvs_ratio_sweep_paper_original_drift_no_sat_view" ;;
  esac
}

method_modules() {
  case "$1" in
    riei_fixed_sat) echo "FED+EC+RC+alternating_training+sat_channel_view_aug" ;;
    riei_fixed_nosat) echo "FED+EC+RC+alternating_training" ;;
    riei_paper_sat) echo "FED+EC+RC+alternating_training+sat_channel_view_aug" ;;
    riei_paper_nosat) echo "FED+EC+RC+alternating_training" ;;
    drift_fixed_sat) echo "tx_rx_split+tx_classifier+rx_classifier+domain_discriminator+receiver_centers+sat_channel_view_aug" ;;
    drift_fixed_nosat) echo "tx_rx_split+tx_classifier+rx_classifier+domain_discriminator+receiver_centers" ;;
    drift_paper_sat) echo "tx_rx_split+tx_classifier+rx_classifier+domain_discriminator+receiver_centers+sat_channel_view_aug" ;;
    drift_paper_nosat) echo "tx_rx_split+tx_classifier+rx_classifier+domain_discriminator+receiver_centers" ;;
  esac
}

method_losses() {
  case "$1" in
    riei_fixed_sat|riei_fixed_nosat) echo "CE+lambda_mi*MI-lambda_ie*IE+lambda_feature_norm" ;;
    riei_paper_sat|riei_paper_nosat) echo "CE+lambda_mi*MI-lambda_ie*IE" ;;
    drift_fixed_sat|drift_fixed_nosat) echo "tx_CE+rx_CE+lambda_grl*domain_CE+lambda_center*center+lambda_mse*negative_MSE_cap4000" ;;
    drift_paper_sat|drift_paper_nosat) echo "tx_CE+rx_CE+lambda_grl*domain_CE+lambda_center*center+lambda_mse*raw_negative_MSE" ;;
  esac
}

method_version_note() {
  case "$1" in
    riei_paper_sat|riei_paper_nosat)
      echo "version=original_paper method_tag=riei_paper_* note=paper-original RIEI: CE+lambda_mi*MI-lambda_ie*IE"
      ;;
    drift_paper_sat|drift_paper_nosat)
      echo "version=original_paper method_tag=drift_paper_* note=paper-original DRIFT: raw negative-MSE separation"
      ;;
    riei_fixed_sat|riei_fixed_nosat)
      echo "version=fix_optimized method_tag=riei_fixed_* note=optimized RIEI: keeps CE+MI-IE and adds lambda_feature_norm=0.0001"
      ;;
    drift_fixed_sat|drift_fixed_nosat)
      echo "version=fix_optimized method_tag=drift_fixed_* note=optimized DRIFT: keeps raw negative-MSE separation and adds mse_cap=4000"
      ;;
  esac
}

QUEUE_DIR="${LOG_ROOT}/queues_${STAMP}"
declare -A QUEUE_FILES=()
QUEUE_GPUS=()
QUEUE_FILE_RESULT=""

queue_file_for_gpu() {
  local gpu="$1"
  if [ -n "${QUEUE_FILES[$gpu]:-}" ]; then
    QUEUE_FILE_RESULT="${QUEUE_FILES[$gpu]}"
    return 0
  fi
  mkdir -p "${QUEUE_DIR}"
  local queue_file="${QUEUE_DIR}/gpu_${gpu}.sh"
  QUEUE_FILES[$gpu]="${queue_file}"
  QUEUE_GPUS+=("${gpu}")
  cat > "${queue_file}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
GPU_ID="${gpu}"
MAX_ACTIVE="${MAX_TRAIN_PER_GPU}"
POLL_SECONDS="${QUEUE_SLOT_POLL_SECONDS}"

gpu_process_count_queue() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  nvidia-smi --id="\${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

wait_for_one_job() {
  if ! wait -n; then
    FAILED=1
  fi
  if (( RUNNING_COUNT > 0 )); then
    RUNNING_COUNT=\$(( RUNNING_COUNT - 1 ))
  fi
}

wait_for_gpu_slot() {
  local visible_count observed_external_count allowed_count
  while true; do
    visible_count="\$(gpu_process_count_queue)"
    observed_external_count=\$(( visible_count - RUNNING_COUNT ))
    if (( observed_external_count < 0 )); then
      observed_external_count=0
    fi
    KNOWN_EXTERNAL_COUNT="\${observed_external_count}"
    allowed_count=\$(( MAX_ACTIVE - KNOWN_EXTERNAL_COUNT ))
    if (( allowed_count < 0 )); then
      allowed_count=0
    fi
    if (( RUNNING_COUNT < allowed_count )); then
      break
    fi
    if (( RUNNING_COUNT > 0 )); then
      wait_for_one_job
    else
      echo "[QUEUE][GPU \${GPU_ID}] waiting visible_count=\${visible_count} external_count=\${KNOWN_EXTERNAL_COUNT} running_count=\${RUNNING_COUNT} max=\${MAX_ACTIVE} at \$(date +%F_%T)" >&2
      sleep "\${POLL_SECONDS}"
    fi
  done
}

INITIAL_EXTERNAL_COUNT="\$(gpu_process_count_queue)"
KNOWN_EXTERNAL_COUNT="\${INITIAL_EXTERNAL_COUNT}"
echo "[QUEUE][GPU \${GPU_ID}] queue_start initial_external_count=\${INITIAL_EXTERNAL_COUNT} at \$(date +%F_%T)"
RUNNING_COUNT=0
FAILED=0
EOF
  chmod +x "${queue_file}"
  QUEUE_FILE_RESULT="${queue_file}"
}

append_to_gpu_queue() {
  local gpu="$1"
  local job_id="$2"
  local log_file="$3"
  local out_dir="$4"
  local queue_file
  queue_file_for_gpu "${gpu}"
  queue_file="${QUEUE_FILE_RESULT}"
  {
    printf '\necho "[QUEUE][GPU %s] job_start %s at $(date +%%F_%%T)"\n' "${gpu}" "${job_id}"
    printf 'wait_for_gpu_slot\n'
    printf 'mkdir -p %q\n' "${out_dir}"
    printf '(\n'
    printf '  set +e\n'
    printf '  echo "[QUEUE][GPU %s] job_exec %s at $(date +%%F_%%T)"\n' "${gpu}" "${job_id}"
    if [ "${STREAM_LOGS}" = "1" ]; then
      printf '  CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "${gpu}"
      format_cmd "${CMD[@]}"
      printf '2>&1 | tee %q\n' "${log_file}"
      printf '  rc=${PIPESTATUS[0]}\n'
    else
      printf '  CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "${gpu}"
      format_cmd "${CMD[@]}"
      printf '> %q 2>&1\n' "${log_file}"
      printf '  rc=$?\n'
    fi
    printf '  echo "[QUEUE][GPU %s] job_done %s rc=${rc} at $(date +%%F_%%T)"\n' "${gpu}" "${job_id}"
    printf '  exit "${rc}"\n'
    printf ') &\n'
    printf 'pid=$!\n'
    printf 'RUNNING_COUNT=$(( RUNNING_COUNT + 1 ))\n'
    printf 'echo "[QUEUE][GPU %s] job_pid %s pid=${pid}"\n' "${gpu}" "${job_id}"
  } >> "${queue_file}"
}

finalize_refill_queues() {
  local gpu queue_file
  for gpu in "${QUEUE_GPUS[@]}"; do
    queue_file="${QUEUE_FILES[$gpu]}"
    cat >> "${queue_file}" <<'EOF'

while (( RUNNING_COUNT > 0 )); do
  wait_for_one_job
done
echo "[QUEUE][GPU ${GPU_ID}] queue_done failed=${FAILED} at $(date +%F_%T)"
exit "${FAILED}"
EOF
  done
}

launch_refill_queues() {
  local gpu queue_file queue_log pid
  finalize_refill_queues
  : > "${LOG_ROOT}/launch_pids.tsv"
  for gpu in "${QUEUE_GPUS[@]}"; do
    queue_file="${QUEUE_FILES[$gpu]}"
    queue_log="${LOG_ROOT}/gpu_${gpu}_queue_${STAMP}.log"
    nohup bash "${queue_file}" > "${queue_log}" 2>&1 &
    pid="$!"
    printf "queue\tgpu_%s\t%s\t%s\t%s\t%s\n" "${gpu}" "${pid}" "${queue_log}" "${queue_file}" "${QUEUE_DIR}" \
      | tee -a "${LOG_ROOT}/launch_pids.tsv"
    log_msg "[CVS-FIXED][GPU ${gpu}] refill_queue_pid=${pid} queue=${queue_file} log=${queue_log}"
  done
}

launch_one() {
  local method="$1"
  local ratio="$2"
  local gpu="$3"
  local tag job_id out_dir log_file scope modules losses version_note
  tag="$(ratio_tag "${ratio}")"
  job_id="${method}_${tag}_seed${SEED}"
  out_dir="${RUN_ROOT}/${job_id}"
  log_file="${LOG_ROOT}/${job_id}.log"
  scope="$(method_scope "${method}")"
  modules="$(method_modules "${method}")"
  losses="$(method_losses "${method}")"
  version_note="$(method_version_note "${method}")"
  build_cmd "${method}" "${ratio}" "${out_dir}"
  local sat_aug_desc
  case "${method}" in
    *_sat) sat_aug_desc="use_sat_channel_view_aug;scenarios=${SAT_TRAIN_SCENARIOS};prob=${SAT_VIEW_PROB}" ;;
    *_nosat) sat_aug_desc="disabled" ;;
  esac
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${job_id}" "${method}" "${ratio}" "${gpu}" "${scope}" "${modules}" "${losses}" \
    "${sat_aug_desc}" \
    "${log_file}" "${out_dir}" "$(format_cmd "${CMD[@]}")" >> "${MANIFEST}"
  log_msg "[CVS-FIXED][${job_id}][GPU ${gpu}] ratio=${ratio} scope=${scope}"
  log_msg "[CVS-FIXED][${job_id}] ${version_note}"
  log_msg "[CVS-FIXED][${job_id}] modules=${modules}"
  log_msg "[CVS-FIXED][${job_id}] losses=${losses}"
  log_msg "[CVS-FIXED][${job_id}] $(format_cmd "${CMD[@]}")"
  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/metrics.json" ]; then
    log_msg "[CVS-FIXED][${job_id}] skip existing metrics=${out_dir}/metrics.json"
    return 0
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  mkdir -p "${out_dir}" "$(dirname "${log_file}")"
  append_to_gpu_queue "${gpu}" "${job_id}" "${log_file}" "${out_dir}"
}

log_msg "[CVS-FIXED] run_id=${RUN_ID} root=${ROOT}"
log_msg "[CVS-FIXED] ratios=${RATIOS_CSV} methods=${METHODS_CSV} gpus=${GPU_IDS_CSV}"
log_msg "[CVS-FIXED] method_label_convention paper=original_paper fixed=fix_optimized"
log_msg "[CVS-FIXED] fixed_delta riei=lambda_feature_norm=0.0001 drift=mse_cap=4000"
log_msg "[CVS-FIXED] cvs_split protocol=${WISIG_PROTOCOL} train_days=${TRAIN_DAYS} test_days=${TEST_DAYS} train_rxs=${TRAIN_RXS} test_rxs=${TEST_RXS} split_strategy=${WISIG_SPLIT_STRATEGY} cap_strategy=${WISIG_CAP_STRATEGY}"
log_msg "[CVS-FIXED] val_ratio=${VAL_RATIO} epochs=${BASELINE_EPOCHS} seed=${SEED} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
log_msg "[CVS-FIXED] sat_train_aug=${SAT_TRAIN_AUG:-1} sat_view scenarios=${SAT_TRAIN_SCENARIOS} prob=${SAT_VIEW_PROB} seed=${SAT_VIEW_SEED} sat_eval=${SAT_EVAL}"

job_i=0
for ratio in "${RATIOS[@]}"; do
  ratio="$(echo "${ratio}" | xargs)"
  [ -n "${ratio}" ] || continue
  for raw_method in "${METHODS[@]}"; do
    raw_method="$(echo "${raw_method}" | xargs)"
    [ -n "${raw_method}" ] || continue
    if ! method="$(canonical_method "${raw_method}")"; then
      echo "ERROR: unsupported method: ${raw_method}" >&2
      exit 2
    fi
    gpu="${GPU_LIST[$job_i]}"
    launch_one "${method}" "${ratio}" "${gpu}"
    job_i=$((job_i + 1))
  done
done

if [ "${DRY_RUN}" != "1" ]; then
  launch_refill_queues
fi

log_msg "[CVS-FIXED] queued_jobs=${job_i} manifest=${MANIFEST}"
exit 0
