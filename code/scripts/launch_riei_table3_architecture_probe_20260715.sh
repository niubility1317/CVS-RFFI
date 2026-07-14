#!/usr/bin/env bash
set -euo pipefail

# Controlled RIEI Table III architecture diagnostic. The paper names
# ResNet1D-18 but does not publish its stem/downsampling details.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_riei_archprobe_seed1337_20260715_030500}"
RUN_ROOT="${ROOT}/paper_reproduction/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/paper_reproduction/logs/${RUN_ID}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
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
[[ "${#GPU_IDS[@]}" -eq 8 ]] || { echo "ERROR: exactly eight GPU ids are required" >&2; exit 2; }

# row|combo|train receivers|test receiver|paper mean|paper SD
ROWS=(
  "3|rx1_1_rx14_7_to_rx1_19|1-1,14-7|1-19|66.09|0.67"
  "6|rx8_8_rx14_7_to_rx1_19|8-8,14-7|1-19|75.48|1.21"
  "10|rx1_19_rx7_7_to_rx14_7|1-19,7-7|14-7|73.52|3.15"
  "12|rx7_7_rx8_8_to_rx14_7|7-7,8-8|14-7|73.46|2.00"
)
VARIANTS=(imagenet1d short_stem1d)

format_cmd() { printf '%q ' "$@"; }
gpu_process_count() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' '
}

echo "[RIEI-ARCH-PROBE] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN} jobs=8"
echo "[RIEI-ARCH-PROBE] rows=3,6,10,12 variants=imagenet1d,short_stem1d metric=paper_last10 reduction=mean"
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
  printf 'row_id\tvariant\tjob_id\tgpu\ttrain_rxs\ttest_rxs\tpaper_mean\tpaper_sd\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'queue_id\tgpu\tpid\tqueue_log\n' > "${RUN_ROOT}/scheduler_pids.tsv"
fi

job_i=0
for row_spec in "${ROWS[@]}"; do
  IFS='|' read -r row combo train_rxs test_rxs paper_mean paper_sd <<< "${row_spec}"
  for variant in "${VARIANTS[@]}"; do
    gpu="${GPU_IDS[$job_i]}"
    job_id="riei_arch_${variant}_row${row}_${combo}_seed${SEED}"
    run_dir="${RUN_ROOT}/${job_id}"
    log_dir="${LOG_ROOT}/${job_id}"
    cmd=(env "METHODS=riei_fd" "WISIG_PROTOCOL=riei_original" "GPU_IDS=${gpu}"
      "TRAIN_RXS=${train_rxs}" "TEST_RXS=${test_rxs}" "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
      "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}" "BASELINE_EPOCHS=200" "SEED=${SEED}" "SAT_EVAL=0"
      "RIEI_PAPER_EVAL_LAST_N=10" "RIEI_TEST_EVAL_INTERVAL=10" "RIEI_OPTIMIZER=sgd" "RIEI_SGD_MOMENTUM=0"
      "RIEI_CE_REDUCTION=mean" "RIEI_MI_REDUCTION=mean" "RIEI_IE_REDUCTION=mean"
      "RIEI_WISIG_RMS_NORMALIZE=0" "RIEI_LAMBDA_FEATURE_NORM=0" "RIEI_FED_VARIANT=${variant}"
      bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)
    echo "[JOB] row=${row} variant=${variant} id=${job_id} gpu=${gpu} train=${train_rxs} test=${test_rxs} paper=${paper_mean}+/-${paper_sd}"
    echo "[CMD] $(format_cmd "${cmd[@]}")"
    if [[ "${DRY_RUN}" != 1 ]]; then
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${row}" "${variant}" "${job_id}" "${gpu}" "${train_rxs}" "${test_rxs}" "${paper_mean}" "${paper_sd}" "${run_dir}" "$(format_cmd "${cmd[@]}")" >> "${RUN_ROOT}/scheduler_manifest.tsv"
      queue="${LOG_ROOT}/queues/gpu_${gpu}.sh"
      printf '#!/usr/bin/env bash\nset -u\necho %q\n' "[QUEUE-JOB-START] row=${row} variant=${variant} id=${job_id} gpu=${gpu}" > "${queue}"
      format_cmd "${cmd[@]}" >> "${queue}"
      printf '\nstatus=$?\necho "[QUEUE-JOB-END] row=%s variant=%s id=%s gpu=%s status=${status}"\n' "${row}" "${variant}" "${job_id}" "${gpu}" >> "${queue}"
    fi
    job_i=$((job_i + 1))
  done
done

if [[ "${DRY_RUN}" != 1 ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    queue="${LOG_ROOT}/queues/gpu_${gpu}.sh"; queue_log="${LOG_ROOT}/gpu_${gpu}_queue.log"; chmod +x "${queue}"
    nohup bash "${queue}" > "${queue_log}" 2>&1 < /dev/null & pid="$!"
    printf 'gpu_%s\t%s\t%s\t%s\n' "${gpu}" "${gpu}" "${pid}" "${queue_log}" >> "${RUN_ROOT}/scheduler_pids.tsv"
    echo "[QUEUE-LAUNCHED] gpu=${gpu} pid=${pid} jobs=1 log=${queue_log}"
  done
fi

echo "[RIEI-ARCH-PROBE] submitted dry_run=${DRY_RUN}"
