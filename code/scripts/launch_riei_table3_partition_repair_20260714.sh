#!/usr/bin/env bash
set -euo pipefail

# RIEI Table III confirmation after repairing the global WiSig partition:
# every tx/rx/eq group now reuses one receiver-combination invariant split.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_riei_table3_partition_repair_seed1337_20260714_220200}"
RUN_ROOT="${ROOT}/paper_reproduction/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/paper_reproduction/logs/${RUN_ID}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-1}"
RIEI_REDUCTION="${RIEI_REDUCTION:-mean}"

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
    *) echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
[[ "${#GPU_IDS[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPU ids are required" >&2; exit 2; }
[[ "${RIEI_REDUCTION}" == "mean" || "${RIEI_REDUCTION}" == "sum" ]] || {
  echo "ERROR: RIEI_REDUCTION must be mean or sum" >&2
  exit 2
}

# row|id|train receivers|test receiver|paper mean|paper SD
ROWS=(
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
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' '
}

declare -A QUEUED=()
for gpu in "${GPU_IDS[@]}"; do QUEUED["${gpu}"]=0; done
for i in "${!ROWS[@]}"; do gpu="${GPU_IDS[$((i % 8))]}"; QUEUED["${gpu}"]=$((QUEUED["${gpu}"] + 1)); done

echo "[RIEI-TABLE3-PARTITION-REPAIR] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN} rows=${#ROWS[@]}"
echo "[RIEI-TABLE3-PARTITION-REPAIR] split=stable_group_seed_shared_train_test_holdout metric=paper_last10 reduction=${RIEI_REDUCTION}"
for gpu in "${GPU_IDS[@]}"; do
  current=0
  [[ "${DRY_RUN}" == 1 ]] || current="$(gpu_process_count "${gpu}")"
  total=$((current + 1))
  echo "[CAPACITY] gpu=${gpu} current=${current} queued=${QUEUED["${gpu}"]} planned_peak=1 total_peak=${total} max=${MAX_TRAIN_PER_GPU}"
  (( total <= MAX_TRAIN_PER_GPU )) || { echo "ERROR: capacity gate failed for GPU ${gpu}" >&2; exit 3; }
done

if [[ "${DRY_RUN}" != 1 ]]; then
  [[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "ERROR: unique run/log root already exists" >&2; exit 4; }
  [[ -x "${PYTHON_BIN}" && -f "${WISIG_PKL}" ]] || { echo "ERROR: Python or dataset missing" >&2; exit 5; }
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}/queues"
  printf 'row_id\tjob_id\tgpu\ttrain_rxs\ttest_rxs\tpaper_mean\tpaper_sd\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'queue_id\tgpu\tpid\tqueue_log\n' > "${RUN_ROOT}/scheduler_pids.tsv"
  for gpu in "${GPU_IDS[@]}"; do printf '#!/usr/bin/env bash\nset -u\n' > "${LOG_ROOT}/queues/gpu_${gpu}.sh"; done
fi

for i in "${!ROWS[@]}"; do
  IFS='|' read -r row combo train_rxs test_rxs paper_mean paper_sd <<< "${ROWS[$i]}"
  gpu="${GPU_IDS[$((i % 8))]}"
  job_id="riei_table3_row${row}_${combo}_seed${SEED}"
  run_dir="${RUN_ROOT}/${job_id}"
  log_dir="${LOG_ROOT}/${job_id}"
  cmd=(env "METHODS=riei_fd" "WISIG_PROTOCOL=riei_original" "GPU_IDS=${gpu}"
    "TRAIN_RXS=${train_rxs}" "TEST_RXS=${test_rxs}" "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}" "BASELINE_EPOCHS=200" "SEED=${SEED}" "SAT_EVAL=0"
    "RIEI_PAPER_EVAL_LAST_N=10" "RIEI_TEST_EVAL_INTERVAL=10" "RIEI_OPTIMIZER=sgd" "RIEI_SGD_MOMENTUM=0"
    "RIEI_CE_REDUCTION=${RIEI_REDUCTION}" "RIEI_MI_REDUCTION=${RIEI_REDUCTION}" "RIEI_IE_REDUCTION=${RIEI_REDUCTION}"
    "RIEI_WISIG_RMS_NORMALIZE=0" "RIEI_LAMBDA_FEATURE_NORM=0"
    bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)
  echo "[JOB] row=${row} id=${job_id} gpu=${gpu} train=${train_rxs} test=${test_rxs} paper=${paper_mean}+/-${paper_sd}"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  [[ "${DRY_RUN}" != 1 ]] || continue
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${row}" "${job_id}" "${gpu}" "${train_rxs}" "${test_rxs}" "${paper_mean}" "${paper_sd}" "${run_dir}" "$(format_cmd "${cmd[@]}")" >> "${RUN_ROOT}/scheduler_manifest.tsv"
  queue="${LOG_ROOT}/queues/gpu_${gpu}.sh"
  printf 'echo %q\n' "[QUEUE-JOB-START] row=${row} id=${job_id} gpu=${gpu}" >> "${queue}"
  format_cmd "${cmd[@]}" >> "${queue}"
  printf '\nstatus=$?\necho "[QUEUE-JOB-END] row=%s id=%s gpu=%s status=${status}"\n' "${row}" "${job_id}" "${gpu}" >> "${queue}"
done

if [[ "${DRY_RUN}" != 1 ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    queue="${LOG_ROOT}/queues/gpu_${gpu}.sh"; queue_log="${LOG_ROOT}/gpu_${gpu}_queue.log"; chmod +x "${queue}"
    nohup bash "${queue}" > "${queue_log}" 2>&1 < /dev/null & pid="$!"
    printf 'gpu_%s\t%s\t%s\t%s\n' "${gpu}" "${gpu}" "${pid}" "${queue_log}" >> "${RUN_ROOT}/scheduler_pids.tsv"
    echo "[QUEUE-LAUNCHED] gpu=${gpu} pid=${pid} jobs=${QUEUED["${gpu}"]} log=${queue_log}"
  done
fi

echo "[RIEI-TABLE3-PARTITION-REPAIR] submitted dry_run=${DRY_RUN}"
