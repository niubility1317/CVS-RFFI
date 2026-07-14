#!/usr/bin/env bash
set -euo pipefail

# Controlled RIEI optimizer-detail probe. The paper does not disclose whether
# its SGD-style updates use momentum; this matrix changes only momentum 0->0.9.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_riei_momentum09_probe_seed1337_20260715_071500}"
RUN_ROOT="${ROOT}/paper_reproduction/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/paper_reproduction/logs/${RUN_ID}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-1}"

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
[[ "${#GPU_IDS[@]}" -eq 4 ]] || { echo "ERROR: exactly four GPU ids are required" >&2; exit 2; }

# row|id|train receivers|test receiver|paper mean|paper SD|momentum0 last10
ROWS=(
  "3|rx1_1_rx14_7_to_rx1_19|1-1,14-7|1-19|66.09|0.67|68.95"
  "4|rx7_7_rx8_8_to_rx1_19|7-7,8-8|1-19|70.51|3.53|79.32"
  "11|rx1_19_rx8_8_to_rx14_7|1-19,8-8|14-7|72.05|2.71|67.03"
  "12|rx7_7_rx8_8_to_rx14_7|7-7,8-8|14-7|73.46|2.00|66.83"
)

format_cmd() { printf '%q ' "$@"; }
gpu_process_count() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' '
}

echo "[RIEI-MOMENTUM09-PROBE] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN} jobs=4"
echo "[RIEI-MOMENTUM09-PROBE] rows=3,4,11,12 momentum=0.9 control_momentum=0 fed_variant=short_stem1d metric=paper_last10"
for gpu in "${GPU_IDS[@]}"; do
  current=0
  [[ "${DRY_RUN}" == 1 ]] || current="$(gpu_process_count "${gpu}")"
  total=$((current + 1))
  echo "[CAPACITY] gpu=${gpu} current=${current} queued=1 planned_peak=1 total_peak=${total} max=${MAX_TRAIN_PER_GPU}"
  (( total <= MAX_TRAIN_PER_GPU )) || { echo "ERROR: capacity gate failed for GPU ${gpu}" >&2; exit 3; }
done

if [[ "${DRY_RUN}" != 1 ]]; then
  [[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "ERROR: unique run/log root already exists" >&2; exit 4; }
  [[ -x "${PYTHON_BIN}" && -f "${WISIG_PKL}" ]] || { echo "ERROR: Python or dataset missing" >&2; exit 5; }
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}/queues"
  printf 'row_id\tjob_id\tgpu\ttrain_rxs\ttest_rxs\tpaper_mean\tpaper_sd\tcontrol_last10\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'queue_id\tgpu\tpid\tqueue_log\n' > "${RUN_ROOT}/scheduler_pids.tsv"
fi

for i in "${!ROWS[@]}"; do
  IFS='|' read -r row combo train_rxs test_rxs paper_mean paper_sd control_last10 <<< "${ROWS[$i]}"
  gpu="${GPU_IDS[$i]}"
  job_id="riei_momentum09_row${row}_${combo}_seed${SEED}"
  run_dir="${RUN_ROOT}/${job_id}"
  log_dir="${LOG_ROOT}/${job_id}"
  cmd=(env "METHODS=riei_fd" "WISIG_PROTOCOL=riei_original" "GPU_IDS=${gpu}"
    "TRAIN_RXS=${train_rxs}" "TEST_RXS=${test_rxs}" "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}" "BASELINE_EPOCHS=200" "SEED=${SEED}" "SAT_EVAL=0"
    "RIEI_PAPER_EVAL_LAST_N=10" "RIEI_TEST_EVAL_INTERVAL=10" "RIEI_OPTIMIZER=sgd" "RIEI_SGD_MOMENTUM=0.9"
    "RIEI_CE_REDUCTION=mean" "RIEI_MI_REDUCTION=mean" "RIEI_IE_REDUCTION=mean"
    "RIEI_WISIG_RMS_NORMALIZE=0" "RIEI_LAMBDA_FEATURE_NORM=0" "RIEI_FED_VARIANT=short_stem1d"
    bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)
  echo "[JOB] row=${row} id=${job_id} gpu=${gpu} train=${train_rxs} test=${test_rxs} paper=${paper_mean}+/-${paper_sd} control_last10=${control_last10}"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  [[ "${DRY_RUN}" != 1 ]] || continue
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${row}" "${job_id}" "${gpu}" "${train_rxs}" "${test_rxs}" "${paper_mean}" "${paper_sd}" "${control_last10}" "${run_dir}" "$(format_cmd "${cmd[@]}")" >> "${RUN_ROOT}/scheduler_manifest.tsv"
  queue="${LOG_ROOT}/queues/gpu_${gpu}.sh"
  printf '#!/usr/bin/env bash\nset -u\necho %q\n' "[QUEUE-JOB-START] row=${row} id=${job_id} gpu=${gpu}" > "${queue}"
  format_cmd "${cmd[@]}" >> "${queue}"
  printf '\nstatus=$?\necho "[QUEUE-JOB-END] row=%s id=%s gpu=%s status=${status}"\n' "${row}" "${job_id}" "${gpu}" >> "${queue}"
done

if [[ "${DRY_RUN}" != 1 ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    queue="${LOG_ROOT}/queues/gpu_${gpu}.sh"; queue_log="${LOG_ROOT}/gpu_${gpu}_queue.log"; chmod +x "${queue}"
    nohup bash "${queue}" > "${queue_log}" 2>&1 < /dev/null & pid="$!"
    printf 'gpu_%s\t%s\t%s\t%s\n' "${gpu}" "${gpu}" "${pid}" "${queue_log}" >> "${RUN_ROOT}/scheduler_pids.tsv"
    echo "[QUEUE-LAUNCHED] gpu=${gpu} pid=${pid} jobs=1 log=${queue_log}"
  done
fi

echo "[RIEI-MOMENTUM09-PROBE] submitted dry_run=${DRY_RUN}"
