#!/usr/bin/env bash
set -euo pipefail

# RIEI journal Table III full 12-row confirmation.
# Configuration is fixed from the completed row-1 parity matrix P02:
# no-momentum SGD, mean CE/MI/IE, no per-packet RMS, no feature-norm guard.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_riei_table3_confirm_sgd_mean_seed1337_20260714_190100}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/paper_reproduction/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-1}"

usage() {
  cat <<'EOF'
Options:
  --dry-run              Print and validate the 12-row matrix (default)
  --launch               Launch eight per-GPU sequential queues
  --gpu-ids CSV          Exactly eight GPU ids (default 0..7)
  --python PATH          Remote Python executable
  --wisig-pkl PATH       ManySig.pkl path
  --run-id ID            Unique run identifier
  --seed N               Confirmation seed (default 1337)
  --max-train-per-gpu N  Hard ceiling including existing jobs (default 2)
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --launch) DRY_RUN=0; shift ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; RUN_ROOT="${ROOT}/paper_reproduction/runs/${RUN_ID}"; LOG_ROOT="${ROOT}/paper_reproduction/logs/${RUN_ID}"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --max-train-per-gpu) MAX_TRAIN_PER_GPU="$2"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_IDS[@]}" -ne 8 ]]; then
  echo "ERROR: this confirmation requires exactly eight GPU ids" >&2
  exit 2
fi

# row|id|train receivers|test receiver|paper mean|paper SD
declare -a TABLE3_ROWS=(
  "1|rx1_1_rx7_7_to_rx1_19|1-1,7-7|1-19|77.88|2.23"
  "2|rx1_1_rx8_8_to_rx1_19|1-1,8-8|1-19|79.43|1.66"
  "3|rx1_1_rx14_7_to_rx1_19|1-1,14-7|1-19|66.09|0.67"
  "4|rx7_7_rx8_8_to_rx1_19|7-7,8-8|1-19|70.51|3.53"
  "5|rx7_7_rx14_7_to_rx1_19|7-7,14-7|1-19|77.35|1.53"
  "6|rx8_8_rx14_7_to_rx1_19|8-8,14-7|1-19|75.48|1.21"
  "7|rx1_1_rx1_19_to_rx14_7|1-1,1-19|14-7|71.91|2.08"
  "8|rx1_1_rx7_7_to_rx14_7|1-1,7-7|14-7|68.33|2.37"
  "9|rx1_1_rx8_8_to_rx14_7|1-1,8-8|14-7|73.54|1.27"
  "10|rx1_19_rx7_7_to_rx14_7|1-19,7-7|14-7|73.52|3.15"
  "11|rx1_19_rx8_8_to_rx14_7|1-19,8-8|14-7|72.05|2.71"
  "12|rx7_7_rx8_8_to_rx14_7|7-7,8-8|14-7|73.46|2.00"
)

format_cmd() { printf '%q ' "$@"; }
gpu_process_count() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

declare -A QUEUED=()
for gpu in "${GPU_IDS[@]}"; do QUEUED["${gpu}"]=0; done
for i in "${!TABLE3_ROWS[@]}"; do
  gpu="${GPU_IDS[$((i % 8))]}"
  QUEUED["${gpu}"]=$((QUEUED["${gpu}"] + 1))
done

echo "[RIEI-TABLE3-CONFIRM] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN} rows=${#TABLE3_ROWS[@]}"
echo "[RIEI-TABLE3-CONFIRM] config=optimizer:sgd,momentum:0,reduction:mean,rms:0,feature_norm:0,epochs:200,last5"
for gpu in "${GPU_IDS[@]}"; do
  current=0
  if [[ "${DRY_RUN}" != "1" ]]; then current="$(gpu_process_count "${gpu}")"; fi
  total=$((current + 1))
  echo "[CAPACITY] gpu=${gpu} current=${current} queued=${QUEUED["${gpu}"]} planned_peak=1 total_peak=${total} max=${MAX_TRAIN_PER_GPU}"
  if (( total > MAX_TRAIN_PER_GPU )); then
    echo "ERROR: capacity gate failed for GPU ${gpu}" >&2
    exit 3
  fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ -e "${RUN_ROOT}" || -e "${LOG_ROOT}" ]]; then
    echo "ERROR: unique run/log root already exists; refusing to overwrite" >&2
    exit 4
  fi
  [[ -x "${PYTHON_BIN}" ]] || { echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2; exit 5; }
  [[ -f "${WISIG_PKL}" ]] || { echo "ERROR: dataset not found: ${WISIG_PKL}" >&2; exit 6; }
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}/queues"
  printf 'row_id\tjob_id\tgpu\ttrain_rxs\ttest_rxs\tpaper_mean\tpaper_sd\toptimizer\treduction\trms_normalize\tfeature_norm\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'queue_id\tgpu\tpid\tqueue_log\n' > "${RUN_ROOT}/scheduler_pids.tsv"
  for gpu in "${GPU_IDS[@]}"; do printf '#!/usr/bin/env bash\nset -u\n' > "${LOG_ROOT}/queues/gpu_${gpu}.sh"; done
fi

queue_job() {
  local row_id="$1" combo="$2" train_rxs="$3" test_rxs="$4" paper_mean="$5" paper_sd="$6" gpu="$7"
  local job_id="riei_table3_row${row_id}_${combo}_seed${SEED}"
  local run_dir="${RUN_ROOT}/${job_id}" log_dir="${LOG_ROOT}/${job_id}"
  local -a cmd=(env
    "METHODS=riei_fd" "WISIG_PROTOCOL=riei_original" "GPU_IDS=${gpu}"
    "TRAIN_RXS=${train_rxs}" "TEST_RXS=${test_rxs}"
    "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}"
    "BASELINE_EPOCHS=200" "SEED=${SEED}" "SAT_EVAL=0"
    "RIEI_PAPER_EVAL_LAST_N=5" "RIEI_TEST_EVAL_INTERVAL=10"
    "RIEI_OPTIMIZER=sgd" "RIEI_SGD_MOMENTUM=0"
    "RIEI_CE_REDUCTION=mean" "RIEI_MI_REDUCTION=mean" "RIEI_IE_REDUCTION=mean"
    "RIEI_WISIG_RMS_NORMALIZE=0" "RIEI_LAMBDA_FEATURE_NORM=0"
    bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)

  echo "[JOB] row=${row_id} id=${job_id} gpu=${gpu} train=${train_rxs} test=${test_rxs} paper=${paper_mean}+/-${paper_sd}"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${row_id}" "${job_id}" "${gpu}" "${train_rxs}" "${test_rxs}" "${paper_mean}" "${paper_sd}" \
    sgd mean 0 0 "${run_dir}" "$(format_cmd "${cmd[@]}")" >> "${RUN_ROOT}/scheduler_manifest.tsv"
  local queue_file="${LOG_ROOT}/queues/gpu_${gpu}.sh"
  printf 'echo %q\n' "[QUEUE-JOB-START] row=${row_id} id=${job_id} gpu=${gpu}" >> "${queue_file}"
  format_cmd "${cmd[@]}" >> "${queue_file}"
  printf '\nstatus=$?\necho "[QUEUE-JOB-END] row=%s id=%s gpu=%s status=${status}"\n' "${row_id}" "${job_id}" "${gpu}" >> "${queue_file}"
}

for i in "${!TABLE3_ROWS[@]}"; do
  IFS='|' read -r row_id combo train_rxs test_rxs paper_mean paper_sd <<< "${TABLE3_ROWS[$i]}"
  queue_job "${row_id}" "${combo}" "${train_rxs}" "${test_rxs}" "${paper_mean}" "${paper_sd}" "${GPU_IDS[$((i % 8))]}"
done

if [[ "${DRY_RUN}" != "1" ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    queue_file="${LOG_ROOT}/queues/gpu_${gpu}.sh"
    queue_log="${LOG_ROOT}/gpu_${gpu}_queue.log"
    chmod +x "${queue_file}"
    nohup bash "${queue_file}" > "${queue_log}" 2>&1 < /dev/null &
    pid="$!"
    printf 'gpu_%s\t%s\t%s\t%s\n' "${gpu}" "${gpu}" "${pid}" "${queue_log}" >> "${RUN_ROOT}/scheduler_pids.tsv"
    echo "[QUEUE-LAUNCHED] gpu=${gpu} pid=${pid} log=${queue_log} jobs=${QUEUED["${gpu}"]}"
  done
fi

echo "[RIEI-TABLE3-CONFIRM] submitted dry_run=${DRY_RUN}"
