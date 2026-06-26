#!/usr/bin/env bash
set -euo pipefail

# Unified WiSig ratio-0.1 queue for:
#   - CVS-RFFI CEN_A31 full stack and focused ablations
#   - RIEI-FD and DRIFT/DRFIT comparison baselines; comparison defaults use satellite-view enhanced variants
#
# Default split is the project CVS-RFFI ratio-0.1 WiSig split:
#   train days 0,1; test days 2,3; train receivers 0..6; test receivers 7..11.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${ROOT}" || exit 1

PLAN="${PLAN:-all}"
METHODS_CSV="${METHODS:-}"
SPLITS_CSV="${SPLITS:-rx7_d01}"
ABLATION_SPLITS_CSV="${ABLATION_SPLITS:-rx7_d01}"
JOBS_CSV="${JOBS:-}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
PYTHON_BIN="${PYTHON_BIN:-}"
CVS_TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-}"
WISIG_PKL="${WISIG_PKL:-./Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-cvsrffi_riei_drift_r010_seed1337}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_DONE="${SKIP_DONE:-1}"
GPU_REFILL_QUEUE="${GPU_REFILL_QUEUE:-1}"
FINAL_ONLY_TEST="${FINAL_ONLY_TEST:-1}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
QUEUE_SLOT_POLL_SECONDS="${QUEUE_SLOT_POLL_SECONDS:-60}"
STREAM_LOGS="${STREAM_LOGS:-0}"

SEED="${SEED:-1337}"
WISIG_PROTOCOL="${WISIG_PROTOCOL:-cvs_day_rx}"
TRAIN_RATIO="${TRAIN_RATIO:-0.1}"
VAL_RATIO="${VAL_RATIO:-0.9}"
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
CVS_NUM_WORKERS="${CVS_NUM_WORKERS:-4}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"

BASELINE_EPOCHS="${BASELINE_EPOCHS:-200}"
CVS_EPOCHS="${CVS_EPOCHS:-170}"
CVS_TEST_START_EPOCH="${CVS_TEST_START_EPOCH:-81}"
CVS_FINAL_ONLY_START_EPOCH="${CVS_FINAL_ONLY_START_EPOCH:-999999}"
RIEI_PAPER_EVAL_LAST_N="${RIEI_PAPER_EVAL_LAST_N:-0}"
DRIFT_PAPER_EVAL_LAST_N="${DRIFT_PAPER_EVAL_LAST_N:-0}"
SAT_EVAL="${SAT_EVAL:-1}"
SAT_EVAL_ON="${SAT_EVAL_ON:-main}"
SAT_SCENARIOS="${SAT_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:-0}"
BASELINE_SAT_TRAIN_SCENARIOS="${BASELINE_SAT_TRAIN_SCENARIOS:-${SAT_SCENARIOS}}"
BASELINE_SAT_VIEW_PROB="${BASELINE_SAT_VIEW_PROB:-1.00}"
BASELINE_SAT_VIEW_SEED="${BASELINE_SAT_VIEW_SEED:-2027}"

usage() {
  cat <<'EOF'
Options:
  --plan NAME            all, comparison, ablation, comparison_plus_ablation, or sat_comparison
  --methods CSV          Methods/ablations to run
                         Defaults:
                           all: riei_fd_sat,drift_sat,cvs_full,cvs_no_satboost,cvs_no_mixstyle,cvs_no_fishr,cvs_no_group_ce,cvs_no_proto_supcon,cvs_no_dsq
                           comparison: riei_fd_sat,drift_sat,cvs_full
                           ablation: cvs_full,cvs_no_satboost,cvs_no_mixstyle,cvs_no_fishr,cvs_no_group_ce,cvs_no_proto_supcon,cvs_no_dsq
                           comparison_plus_ablation: satellite-enhanced comparison on every split; CVS ablations only on ABLATION_SPLITS.
                           sat_comparison: riei_fd_sat,drift_sat
  --splits CSV           Split profiles:
                         rx1_d0, rx1_d01, rx1_d012,
                         rx3_d0, rx3_d01, rx3_d012,
                         rx5_d0, rx5_d01, rx5_d012,
                         rx7_d0, rx7_d01, rx7_d012, custom
                         custom uses TRAIN_DAYS/TEST_DAYS/TRAIN_RXS/TEST_RXS env vars.
  --ablation-splits CSV  Split profiles/labels where CVS ablations run under comparison_plus_ablation
  --jobs CSV             Exact split:method list. Example: rx7_d01:riei_fd,rx7_d012:cvs_full
  --gpu-ids CSV          GPUs to use
  --wisig-pkl PATH       Dataset_WigSig/ManySig.pkl path
  --python PATH          Python executable
  --cvs-train-script P   CVS-RFFI train.py path
  --run-root PATH        Output root
  --log-root PATH        Log root
  --dry-run              Print commands only
  --no-skip-done         Re-run even when metrics.json exists
  --stream-logs          Stream foreground logs instead of redirecting
Environment:
  GPU_REFILL_QUEUE=1     Build one refill queue per GPU; each GPU runs up to MAX_TRAIN_PER_GPU jobs.
  FINAL_ONLY_TEST=1      Disable in-training test evaluation; final checkpoint tests still run.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --plan) PLAN="$2"; shift 2 ;;
    --methods) METHODS_CSV="$2"; shift 2 ;;
    --splits) SPLITS_CSV="$2"; shift 2 ;;
    --ablation-splits) ABLATION_SPLITS_CSV="$2"; shift 2 ;;
    --jobs) JOBS_CSV="$2"; shift 2 ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --cvs-train-script) CVS_TRAIN_SCRIPT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --no-skip-done) SKIP_DONE=0; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${PLAN}" in
  all)
    DEFAULT_METHODS="riei_fd_sat,drift_sat,cvs_full,cvs_no_satboost,cvs_no_mixstyle,cvs_no_fishr,cvs_no_group_ce,cvs_no_proto_supcon,cvs_no_dsq"
    ;;
  comparison)
    DEFAULT_METHODS="riei_fd_sat,drift_sat,cvs_full"
    ;;
  ablation)
    DEFAULT_METHODS="cvs_full,cvs_no_satboost,cvs_no_mixstyle,cvs_no_fishr,cvs_no_group_ce,cvs_no_proto_supcon,cvs_no_dsq"
    ;;
  comparison_plus_ablation)
    DEFAULT_METHODS="riei_fd_sat,drift_sat,cvs_full,cvs_no_satboost,cvs_no_mixstyle,cvs_no_fishr,cvs_no_group_ce,cvs_no_proto_supcon,cvs_no_dsq"
    ;;
  sat_comparison)
    DEFAULT_METHODS="riei_fd_sat,drift_sat"
    ;;
  *)
    echo "ERROR: --plan must be all, comparison, ablation, comparison_plus_ablation, or sat_comparison; got ${PLAN}" >&2
    exit 2
    ;;
esac

METHODS_CSV="${METHODS_CSV:-${DEFAULT_METHODS}}"

if [ "${WISIG_PROTOCOL}" != "cvs_day_rx" ]; then
  echo "ERROR: this ratio-0.1 queue intentionally supports WISIG_PROTOCOL=cvs_day_rx only; got ${WISIG_PROTOCOL}" >&2
  echo "Use run_wisig_paper_scope_queue.sh for DRIFT/RIEI original-paper fixed-sample protocols." >&2
  exit 2
fi
if [ "${TRAIN_RATIO}" != "0.1" ]; then
  echo "ERROR: TRAIN_RATIO must stay 0.1 for this queue; got ${TRAIN_RATIO}" >&2
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
if [ -z "${CVS_TRAIN_SCRIPT}" ]; then
  if [ -f "${ROOT}/code/train.py" ]; then
    CVS_TRAIN_SCRIPT="${ROOT}/code/train.py"
  else
    CVS_TRAIN_SCRIPT="${ROOT}/train.py"
  fi
fi
if [ ! -f "${CVS_TRAIN_SCRIPT}" ]; then
  echo "ERROR: CVS_TRAIN_SCRIPT not found: ${CVS_TRAIN_SCRIPT}" >&2
  exit 2
fi
if [ "${DRY_RUN}" != "1" ] && [ ! -f "${WISIG_PKL}" ]; then
  echo "ERROR: WISIG_PKL not found: ${WISIG_PKL}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV}"
IFS=',' read -r -a SPLITS <<< "${SPLITS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi
if [ "${#SPLITS[@]}" -lt 1 ]; then
  echo "ERROR: SPLITS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"
MANIFEST="${RUN_ROOT}/manifest_${STAMP}.tsv"
: > "${MANIFEST}"

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

format_cmd() {
  printf "%q " "$@"
}

gpu_process_count() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return 0
  fi
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
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

wait_for_gpu_slot() {
  local visible_count observed_external_count external_count allowed_count
  while true; do
    visible_count="\$(gpu_process_count_queue)"
    observed_external_count=\$(( visible_count - RUNNING_COUNT ))
    if (( observed_external_count < 0 )); then
      observed_external_count=0
    fi
    if (( RUNNING_COUNT == 0 && observed_external_count < KNOWN_EXTERNAL_COUNT )); then
      KNOWN_EXTERNAL_COUNT="\${observed_external_count}"
    fi
    if (( observed_external_count > KNOWN_EXTERNAL_COUNT )); then
      KNOWN_EXTERNAL_COUNT="\${observed_external_count}"
    fi
    external_count="\${KNOWN_EXTERNAL_COUNT}"
    allowed_count=\$(( MAX_ACTIVE - external_count ))
    if (( allowed_count < 0 )); then
      allowed_count=0
    fi
    if (( RUNNING_COUNT < allowed_count )); then
      break
    fi
    if (( RUNNING_COUNT > 0 )); then
      wait_for_one_job
    else
      echo "[QUEUE][GPU \${GPU_ID}] waiting visible_count=\${visible_count} external_count=\${external_count} running_count=\${RUNNING_COUNT} max=\${MAX_ACTIVE} at \$(date +%F_%T)" >&2
      sleep "\${POLL_SECONDS}"
    fi
  done
}

wait_for_one_job() {
  if ! wait -n; then
    FAILED=1
  fi
  if (( RUNNING_COUNT > 0 )); then
    RUNNING_COUNT=\$(( RUNNING_COUNT - 1 ))
  fi
}

INITIAL_EXTERNAL_COUNT="\$(gpu_process_count_queue)"
KNOWN_EXTERNAL_COUNT="\${INITIAL_EXTERNAL_COUNT}"
echo "[QUEUE][GPU \${GPU_ID}] queue_start initial_external_count=\${INITIAL_EXTERNAL_COUNT} at \$(date +%F_%T)"
RUNNING_PIDS=()
RUNNING_COUNT=0
FAILED=0
EOF
  chmod +x "${queue_file}"
  QUEUE_FILE_RESULT="${queue_file}"
}

append_to_gpu_queue() {
  local gpu="$1"
  local run_name="$2"
  local log_file="$3"
  local out_dir="$4"
  local queue_file
  queue_file_for_gpu "${gpu}"
  queue_file="${QUEUE_FILE_RESULT}"
  {
    printf '\necho "[QUEUE][GPU %s] job_start %s at $(date +%%F_%%T)"\n' "${gpu}" "${run_name}"
    printf 'wait_for_gpu_slot\n'
    printf 'mkdir -p %q\n' "${out_dir}"
    printf '(\n'
    printf '  set +e\n'
    printf '  echo "[QUEUE][GPU %s] job_exec %s at $(date +%%F_%%T)"\n' "${gpu}" "${run_name}"
    printf '  CUDA_VISIBLE_DEVICES=%q PYTHONUNBUFFERED=1 ' "${gpu}"
    format_cmd "${CMD[@]}"
    printf '> %q 2>&1\n' "${log_file}"
    printf '  rc=$?\n'
    printf '  echo "[QUEUE][GPU %s] job_done %s rc=${rc} at $(date +%%F_%%T)"\n' "${gpu}" "${run_name}"
    printf '  exit "${rc}"\n'
    printf ') &\n'
    printf 'pid=$!\n'
    printf 'RUNNING_PIDS+=("${pid}")\n'
    printf 'RUNNING_COUNT=$(( RUNNING_COUNT + 1 ))\n'
    printf 'echo "[QUEUE][GPU %s] job_pid %s pid=${pid}"\n' "${gpu}" "${run_name}"
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
  for gpu in "${QUEUE_GPUS[@]}"; do
    queue_file="${QUEUE_FILES[$gpu]}"
    queue_log="${LOG_ROOT}/gpu_${gpu}_queue_${STAMP}.log"
    nohup bash "${queue_file}" > "${queue_log}" 2>&1 &
    pid="$!"
    printf "queue\tgpu_%s\t%s\t%s\t%s\t%s\n" "${gpu}" "${pid}" "${queue_log}" "${queue_file}" "${QUEUE_DIR}" \
      | tee -a "${LOG_ROOT}/launch_pids.tsv"
    log_msg "[R010-QUEUE][GPU ${gpu}] refill_queue_pid=${pid} queue=${queue_file} log=${queue_log}"
  done
}

canonical_method() {
  case "$1" in
    riei|riei_fd) echo "riei_fd" ;;
    riei_sat|riei_fd_sat|riei_sat5|riei_fd_sat5) echo "riei_fd_sat" ;;
    drift|drfit) echo "drift" ;;
    drift_sat|drfit_sat|drift_sat5|drfit_sat5) echo "drift_sat" ;;
    cvs|cvs_full|cvsrffi|cvsrffi_full|cvsrffi_cen_a31) echo "cvs_full" ;;
    cvs_no_sat|cvs_no_satboost|no_satboost) echo "cvs_no_satboost" ;;
    cvs_no_mix|cvs_no_mixstyle|no_mixstyle) echo "cvs_no_mixstyle" ;;
    cvs_no_fishr|no_fishr) echo "cvs_no_fishr" ;;
    cvs_no_group_ce|no_group_ce) echo "cvs_no_group_ce" ;;
    cvs_no_proto|cvs_no_proto_supcon|no_proto_supcon) echo "cvs_no_proto_supcon" ;;
    cvs_no_dsq|no_dsq|cvs_no_domain_dsq) echo "cvs_no_dsq" ;;
    *) return 1 ;;
  esac
}

is_cvs_method() {
  case "$1" in
    cvs_*) return 0 ;;
    *) return 1 ;;
  esac
}

split_allows_ablation() {
  local token norm
  local -a ABLATION_SPLIT_LIST
  IFS=',' read -r -a ABLATION_SPLIT_LIST <<< "${ABLATION_SPLITS_CSV}"
  for token in "${ABLATION_SPLIT_LIST[@]}"; do
    token="$(echo "${token}" | xargs)"
    [ -z "${token}" ] && continue
    norm="${token//_/}"
    if [ "${norm}" = "${SPLIT_LABEL}" ]; then
      return 0
    fi
  done
  return 1
}

apply_day_profile() {
  local day_profile="$1"
  case "${day_profile}" in
    0)
      CURRENT_TRAIN_DAYS="0"
      CURRENT_TEST_DAYS="2,3"
      ;;
    01)
      CURRENT_TRAIN_DAYS="0,1"
      CURRENT_TEST_DAYS="2,3"
      ;;
    012)
      CURRENT_TRAIN_DAYS="0,1,2"
      CURRENT_TEST_DAYS="3"
      ;;
    *)
      echo "ERROR: unknown day profile '${day_profile}'" >&2
      exit 2
      ;;
  esac
}

seq_csv() {
  local start="$1"
  local end="$2"
  local out=""
  local i
  for ((i = start; i <= end; i++)); do
    if [ -z "${out}" ]; then
      out="${i}"
    else
      out="${out},${i}"
    fi
  done
  echo "${out}"
}

receiver_set_for_variant() {
  local count="$1"
  local variant="$2"
  local start
  case "${variant}" in
    lo)
      seq_csv 0 $((count - 1))
      ;;
    hi)
      start=$((7 - count))
      seq_csv "${start}" 6
      ;;
    sp)
      case "${count}" in
        2) echo "0,6" ;;
        3) echo "0,3,6" ;;
        4) echo "0,2,4,6" ;;
        5) echo "0,1,3,5,6" ;;
        6) echo "0,1,2,4,5,6" ;;
        7) echo "0,1,2,3,4,5,6" ;;
        *) echo "ERROR: receiver count must be 2..7; got ${count}" >&2; exit 2 ;;
      esac
      ;;
    all)
      if [ "${count}" != "7" ]; then
        echo "ERROR: receiver variant all is only valid for rx7; got rx${count}" >&2
        exit 2
      fi
      echo "0,1,2,3,4,5,6"
      ;;
    *)
      echo "ERROR: unknown receiver variant '${variant}'" >&2
      exit 2
      ;;
  esac
}

apply_split_profile() {
  local profile="$1"
  if [[ "${profile}" =~ ^rx([2-7])(lo|sp|hi|all)_d(0|01|012)$ ]]; then
    local rx_count="${BASH_REMATCH[1]}"
    local rx_variant="${BASH_REMATCH[2]}"
    local day_profile="${BASH_REMATCH[3]}"
    SPLIT_LABEL="rx${rx_count}${rx_variant}d${day_profile}"
    apply_day_profile "${day_profile}"
    CURRENT_TRAIN_RXS="$(receiver_set_for_variant "${rx_count}" "${rx_variant}")"
    CURRENT_TEST_RXS="7,8,9,10,11"
    return 0
  fi
  case "${profile}" in
    rx3_d0)
      SPLIT_LABEL="rx3d0"
      CURRENT_TRAIN_DAYS="0"
      CURRENT_TEST_DAYS="2,3"
      CURRENT_TRAIN_RXS="0,3,6"
      CURRENT_TEST_RXS="7,8,9,10,11"
      ;;
    rx3_d01)
      SPLIT_LABEL="rx3d01"
      CURRENT_TRAIN_DAYS="0,1"
      CURRENT_TEST_DAYS="2,3"
      CURRENT_TRAIN_RXS="0,3,6"
      CURRENT_TEST_RXS="7,8,9,10,11"
      ;;
    rx5_d01)
      SPLIT_LABEL="rx5d01"
      CURRENT_TRAIN_DAYS="0,1"
      CURRENT_TEST_DAYS="2,3"
      CURRENT_TRAIN_RXS="0,1,2,3,4"
      CURRENT_TEST_RXS="7,8,9,10,11"
      ;;
    rx7_d0)
      SPLIT_LABEL="rx7d0"
      CURRENT_TRAIN_DAYS="0"
      CURRENT_TEST_DAYS="2,3"
      CURRENT_TRAIN_RXS="0,1,2,3,4,5,6"
      CURRENT_TEST_RXS="7,8,9,10,11"
      ;;
    rx7_d01)
      SPLIT_LABEL="rx7d01"
      CURRENT_TRAIN_DAYS="0,1"
      CURRENT_TEST_DAYS="2,3"
      CURRENT_TRAIN_RXS="0,1,2,3,4,5,6"
      CURRENT_TEST_RXS="7,8,9,10,11"
      ;;
    rx7_d012)
      SPLIT_LABEL="rx7d012"
      CURRENT_TRAIN_DAYS="0,1,2"
      CURRENT_TEST_DAYS="3"
      CURRENT_TRAIN_RXS="0,1,2,3,4,5,6"
      CURRENT_TEST_RXS="7,8,9,10,11"
      ;;
    custom)
      SPLIT_LABEL="custom"
      CURRENT_TRAIN_DAYS="${TRAIN_DAYS}"
      CURRENT_TEST_DAYS="${TEST_DAYS}"
      CURRENT_TRAIN_RXS="${TRAIN_RXS}"
      CURRENT_TEST_RXS="${TEST_RXS}"
      ;;
    *)
      echo "ERROR: unknown split profile '${profile}'" >&2
      exit 2
      ;;
  esac
}

run_name_for_method() {
  case "$1" in
    riei_fd) echo "riei_fd_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    drift) echo "drift_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    riei_fd_sat) echo "riei_fd_sat5_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    drift_sat) echo "drift_sat5_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    cvs_full) echo "CEN_ABL_FULL_a31_stack_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    cvs_no_satboost) echo "CEN_ABL_NO_SATBOOST_a31_stack_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    cvs_no_mixstyle) echo "CEN_ABL_NO_MIXSTYLE_a31_stack_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    cvs_no_fishr) echo "CEN_ABL_NO_FISHR_a31_stack_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    cvs_no_group_ce) echo "CEN_ABL_NO_GROUPCE_a31_stack_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    cvs_no_proto_supcon) echo "CEN_ABL_NO_PROTO_SUPCON_a31_stack_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    cvs_no_dsq) echo "CEN_ABL_NO_DSQ_DOMAIN_a31_stack_${SPLIT_LABEL}_r010_seed${SEED}" ;;
    *) return 1 ;;
  esac
}

append_shared_wisig_args() {
  CMD+=(
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol "${WISIG_PROTOCOL}"
    --wisig_equalized "${WISIG_EQUALIZED}"
    --wisig_domain "${WISIG_DOMAIN}"
    --wisig_out_len "${WISIG_OUT_LEN}"
    --wisig_train_ratio "${TRAIN_RATIO}"
    --wisig_val_ratio "${VAL_RATIO}"
    --wisig_guard_gap "${GUARD_GAP}"
    --wisig_train_days "${CURRENT_TRAIN_DAYS}"
    --wisig_test_days "${CURRENT_TEST_DAYS}"
    --wisig_train_rxs "${CURRENT_TRAIN_RXS}"
    --wisig_test_rxs "${CURRENT_TEST_RXS}"
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

append_sat_eval_args() {
  if [ "${SAT_EVAL}" = "1" ]; then
    CMD+=(--eval_sat_channel --eval_sat_on "${SAT_EVAL_ON}" --eval_sat_scenarios "${SAT_SCENARIOS}" --sat_eval_max_batches "${SAT_EVAL_MAX_BATCHES}")
  else
    CMD+=(--no_eval_sat_channel)
  fi
}

build_baseline_cmd() {
  local method="$1"
  local out_dir="$2"
  local base_method="${method}"
  local module
  if [ "${method}" = "riei_fd_sat" ]; then
    base_method="riei_fd"
  elif [ "${method}" = "drift_sat" ]; then
    base_method="drift"
  fi
  if [ "${base_method}" = "riei_fd" ]; then
    module="baselines.riei_fd.train"
  else
    module="baselines.drift.train"
  fi
  CMD=("${PYTHON_BIN}" -u -m "${module}")
  append_shared_wisig_args
  append_sat_eval_args
  CMD+=(--output_dir "${out_dir}" --epochs "${BASELINE_EPOCHS}" --no_test_on_val_improve)
  if [ "${method}" = "riei_fd_sat" ] || [ "${method}" = "drift_sat" ]; then
    CMD+=(
      --use_sat_channel_view_aug
      --sat_train_scenarios "${BASELINE_SAT_TRAIN_SCENARIOS}"
      --sat_view_prob "${BASELINE_SAT_VIEW_PROB}"
      --sat_view_seed "${BASELINE_SAT_VIEW_SEED}"
    )
  fi
  if [ "${base_method}" = "riei_fd" ]; then
    CMD+=(--batch_size 64 --lr_all 0.0001 --lr_fed 0.0001 --lambda_mi 1.2 --lambda_ie 1.2)
    CMD+=(--paper_eval_last_n "${RIEI_PAPER_EVAL_LAST_N}" --paper_eval_name "riei_last${RIEI_PAPER_EVAL_LAST_N}")
  else
    CMD+=(
      --batch_size 64
      --lr 0.0001
      --lambda_grl 1.0
      --lambda_center 0.01
      --lambda_mse 0.02
      --no-normalize_features_for_mse
      --grl_schedule constant
    )
    CMD+=(--paper_eval_last_n "${DRIFT_PAPER_EVAL_LAST_N}" --paper_eval_name "drift_last${DRIFT_PAPER_EVAL_LAST_N}")
  fi
}

append_cvs_base_args() {
  local run_name="$1"
  local run_dir="$2"
  CMD+=(
    --train_mode centralized
    --dataset wisig
    --batch_size 256
    --eval_batch_size 256
    --num_workers "${CVS_NUM_WORKERS}"
    --prefetch_factor "${PREFETCH_FACTOR}"
    --wisig_pkl "${WISIG_PKL}"
    --wisig_protocol "${WISIG_PROTOCOL}"
    --wisig_equalized "${WISIG_EQUALIZED}"
    --wisig_domain "${WISIG_DOMAIN}"
    --wisig_out_len "${WISIG_OUT_LEN}"
    --wisig_train_ratio "${TRAIN_RATIO}"
    --wisig_val_ratio "${VAL_RATIO}"
    --wisig_guard_gap "${GUARD_GAP}"
    --wisig_train_days "${CURRENT_TRAIN_DAYS}"
    --wisig_test_days "${CURRENT_TEST_DAYS}"
    --wisig_train_rxs "${CURRENT_TRAIN_RXS}"
    --wisig_test_rxs "${CURRENT_TEST_RXS}"
    --wisig_split_strategy "${WISIG_SPLIT_STRATEGY}"
    --wisig_cap_strategy "${WISIG_CAP_STRATEGY}"
    --wisig_max_day123_per_combo 0
    --wisig_max_train_per_combo 0
    --wisig_max_val_per_combo 0
    --wisig_max_test_per_combo 0
    --epochs "${CVS_EPOCHS}"
    --slim_group none
    --branch_ablation no_dac
    --domain_branch_ablation no_stats
    --exp_group s3_rxrobust_no_dac
    --model_variant lite_d
    --primary_udu_weight 0.70
    --lambda_sat_cls 0.00
    --lambda_sat_cons 0.00
    --seed "${SEED}"
    --run_name "${run_name}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_seen_day_unseen_rx_save_path "${run_dir}/best_seen_day_unseen_rx_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )
  if [ "${FINAL_ONLY_TEST}" = "1" ]; then
    CMD+=(--test_eval_policy val_improved_final --test_eval_start_epoch "${CVS_FINAL_ONLY_START_EPOCH}")
  else
    CMD+=(--test_eval_policy every_epoch --test_eval_start_epoch "${CVS_TEST_START_EPOCH}")
  fi
  append_sat_eval_args
}

append_cvs_mixstyle_args() {
  CMD+=(
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
  )
}

append_cvs_satboost_args() {
  CMD+=(
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --concat_sat_start_epoch 1
    --concat_sat_ce_weight 1.28
    --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
    --sat_view_prob 1.00
  )
}

append_cvs_domain_args() {
  local method="$1"
  if [ "${method}" = "cvs_no_dsq" ]; then
    CMD+=(--domain_enhancer off --domain_enhancer_strength 0.0 --domain_freq_stability_mode off --freq_stability_channels 0)
  else
    CMD+=(--domain_enhancer rcn_stats --domain_enhancer_strength 0.35 --domain_freq_stability_mode dsq --freq_stability_channels 2)
  fi
}

append_cvs_group_args() {
  local method="$1"
  if [ "${method}" = "cvs_no_group_ce" ]; then
    CMD+=(--lambda_group_ce 0.00 --group_ce_min_domains 4)
  else
    CMD+=(
      --lambda_group_ce 0.06
      --group_ce_mode smooth_dro_capped
      --group_ce_top_frac 0.35
      --group_ce_min_domains 4
      --groupdro_tau 0.50
      --groupdro_cap 0.65
    )
  fi
}

append_cvs_fishr_args() {
  local method="$1"
  if [ "${method}" = "cvs_no_fishr" ]; then
    CMD+=(--lambda_fishr 0.00 --fishr_min_domains 4)
  else
    CMD+=(--lambda_fishr 0.005 --fishr_min_domains 4)
  fi
}

append_cvs_proto_supcon_args() {
  local method="$1"
  if [ "${method}" = "cvs_no_proto_supcon" ]; then
    CMD+=(--no_use_proto_memory --lambda_proto 0.00 --lambda_supcon_id 0.00 --generalization_feature z_id)
  else
    CMD+=(
      --use_proto_memory
      --lambda_proto 0.015
      --proto_momentum 0.95
      --lambda_supcon_id 0.02
      --supcon_temp 0.12
      --generalization_feature z_id
    )
  fi
}

build_cvs_cmd() {
  local method="$1"
  local run_name="$2"
  local run_dir="$3"
  CMD=("${PYTHON_BIN}" -u "${CVS_TRAIN_SCRIPT}")
  append_cvs_base_args "${run_name}" "${run_dir}"
  if [ "${method}" = "cvs_no_mixstyle" ]; then
    CMD+=(--no_use_mixstyle)
  else
    append_cvs_mixstyle_args
  fi
  if [ "${method}" = "cvs_no_satboost" ]; then
    CMD+=(--no_use_concat_sat_channel_aug --no_concat_sat_ce_only --concat_sat_ce_weight 0.00 --sat_view_prob 0.00)
  else
    append_cvs_satboost_args
  fi
  append_cvs_domain_args "${method}"
  append_cvs_group_args "${method}"
  append_cvs_fishr_args "${method}"
  append_cvs_proto_supcon_args "${method}"
}

launch_one() {
  local method="$1"
  local gpu="$2"
  local run_name
  run_name="$(run_name_for_method "${method}")"
  local out_dir="${RUN_ROOT}/${run_name}"
  local log_file="${LOG_ROOT}/${run_name}_${STAMP}.log"

  if [ "${SKIP_DONE}" = "1" ] && [ -f "${out_dir}/metrics.json" ]; then
    log_msg "[R010-QUEUE][${method}] skip existing metrics: ${out_dir}/metrics.json"
    return 0
  fi

  if is_cvs_method "${method}"; then
    build_cvs_cmd "${method}" "${run_name}" "${out_dir}"
  else
    build_baseline_cmd "${method}" "${out_dir}"
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${SPLIT_LABEL}" "${method}" "${gpu}" "${log_file}" "${out_dir}" "$(format_cmd "${CMD[@]}")" >> "${MANIFEST}"
  log_msg "[R010-QUEUE][${SPLIT_LABEL}][${method}][GPU ${gpu}] run=${run_name} log=${log_file}"
  log_msg "[R010-QUEUE][${SPLIT_LABEL}][${method}][GPU ${gpu}] $(format_cmd "${CMD[@]}")"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  if [ "${GPU_REFILL_QUEUE}" = "1" ]; then
    mkdir -p "${out_dir}"
    append_to_gpu_queue "${gpu}" "${run_name}" "${log_file}" "${out_dir}"
    return 0
  fi

  local count
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_TRAIN_PER_GPU )); then
    printf "%s\t%s\tgpu=%s active_count=%s max=%s\n" "${method}" "${run_name}" "${gpu}" "${count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${out_dir}"
  if [ "${STREAM_LOGS}" = "1" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" 2>&1 | tee "${log_file}"
    return "${PIPESTATUS[0]}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${CMD[@]}" > "${log_file}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${method}" "${run_name}" "${gpu}" "${pid}" "${log_file}" "${out_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

schedule_method_for_current_split() {
  local requested_method="$1"
  local method gpu
  requested_method="$(echo "${requested_method}" | xargs)"
  [ -z "${requested_method}" ] && return 0
  if ! method="$(canonical_method "${requested_method}")"; then
    log_msg "ERROR: unknown method '${requested_method}'"
    exit 2
  fi
  if [ "${method}" != "${requested_method}" ]; then
    log_msg "[ALIAS] ${requested_method} -> ${method}"
  fi
  if [ "${PLAN}" = "comparison_plus_ablation" ] && is_cvs_method "${method}" && [ "${method}" != "cvs_full" ] && ! split_allows_ablation; then
    log_msg "[R010-QUEUE][${SPLIT_LABEL}][${method}] skip ablation outside ABLATION_SPLITS=${ABLATION_SPLITS_CSV}"
    return 0
  fi
  gpu="${GPU_LIST[$((gpu_i % ${#GPU_LIST[@]}))]}"
  gpu_i=$((gpu_i + 1))
  launch_one "${method}" "${gpu}" || status=$?
}

log_msg "[R010-QUEUE] root=${ROOT}"
log_msg "[R010-QUEUE] plan=${PLAN} methods=${METHODS_CSV} splits=${SPLITS_CSV} ablation_splits=${ABLATION_SPLITS_CSV} jobs=${JOBS_CSV} seed=${SEED}"
log_msg "[R010-QUEUE] protocol=${WISIG_PROTOCOL} train_ratio=${TRAIN_RATIO} val_ratio=${VAL_RATIO} guard_gap=${GUARD_GAP} split_strategy=${WISIG_SPLIT_STRATEGY} cap_strategy=${WISIG_CAP_STRATEGY}"
log_msg "[R010-QUEUE] run_root=${RUN_ROOT} log_root=${LOG_ROOT} gpus=${GPU_IDS_CSV} dry_run=${DRY_RUN}"
log_msg "[R010-QUEUE] gpu_refill_queue=${GPU_REFILL_QUEUE} max_train_per_gpu=${MAX_TRAIN_PER_GPU} final_only_test=${FINAL_ONLY_TEST}"
log_msg "[R010-QUEUE] baseline_sat_view scenarios=${BASELINE_SAT_TRAIN_SCENARIOS} prob=${BASELINE_SAT_VIEW_PROB} seed=${BASELINE_SAT_VIEW_SEED}"

status=0
gpu_i=0
if [ -n "${JOBS_CSV}" ]; then
  IFS=',' read -r -a JOB_LIST <<< "${JOBS_CSV}"
  for raw_job in "${JOB_LIST[@]}"; do
    job="$(echo "${raw_job}" | xargs)"
    [ -z "${job}" ] && continue
    if [[ "${job}" != *:* ]]; then
      log_msg "ERROR: --jobs entry must be split:method, got '${job}'"
      exit 2
    fi
    split_profile="${job%%:*}"
    requested_method="${job#*:}"
    apply_split_profile "${split_profile}"
    log_msg "[R010-QUEUE][${SPLIT_LABEL}] train_days=${CURRENT_TRAIN_DAYS} test_days=${CURRENT_TEST_DAYS} train_rxs=${CURRENT_TRAIN_RXS} test_rxs=${CURRENT_TEST_RXS}"
    schedule_method_for_current_split "${requested_method}"
  done
else
  for raw_split in "${SPLITS[@]}"; do
    split_profile="$(echo "${raw_split}" | xargs)"
    [ -z "${split_profile}" ] && continue
    apply_split_profile "${split_profile}"
    log_msg "[R010-QUEUE][${SPLIT_LABEL}] train_days=${CURRENT_TRAIN_DAYS} test_days=${CURRENT_TEST_DAYS} train_rxs=${CURRENT_TRAIN_RXS} test_rxs=${CURRENT_TEST_RXS}"
    for raw_method in "${METHODS[@]}"; do
      schedule_method_for_current_split "${raw_method}"
    done
  done
fi

if [ "${DRY_RUN}" != "1" ] && [ "${GPU_REFILL_QUEUE}" = "1" ]; then
  launch_refill_queues
fi

log_msg "[R010-QUEUE] finished status=${status} manifest=${MANIFEST}"
exit "${status}"
