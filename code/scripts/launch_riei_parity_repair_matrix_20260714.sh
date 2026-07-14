#!/usr/bin/env bash
set -euo pipefail

# RIEI Table III row-1 parity discovery matrix. One job is added per GPU.
# Target-domain interval measurements are diagnostic only; formal scoring is last5.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_riei_parity_repair_20260714_145800}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/paper_reproduction/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-1}"

usage() {
  cat <<'EOF'
Options:
  --dry-run              Print and validate the 8-job matrix (default)
  --launch               Launch one RIEI candidate per GPU
  --gpu-ids CSV          Exactly eight GPU ids (default 0..7)
  --python PATH          Remote Python executable
  --wisig-pkl PATH       ManySig.pkl path
  --run-id ID            Unique run identifier
  --seed N               Discovery seed (default 1337)
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
  echo "ERROR: this discovery matrix requires exactly eight GPU ids" >&2
  exit 2
fi

declare -a CANDIDATES=(
  "P01_sgd_sum_no_rms|sgd|sum|0|0"
  "P02_sgd_mean_no_rms|sgd|mean|0|0"
  "P03_adam_sum_no_rms|adam|sum|0|0"
  "P04_adam_mean_no_rms|adam|mean|0|0"
  "P05_adam_sum_no_rms_fn1e4|adam|sum|0|0.0001"
  "P06_sgd_sum_rms_control|sgd|sum|1|0"
  "P07_sgd_sum_no_rms_fn1e4|sgd|sum|0|0.0001"
  "P08_adam_sum_rms_fixopt_control|adam|sum|1|0.0001"
)

format_cmd() { printf '%q ' "$@"; }
gpu_process_count() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

echo "[RIEI-PARITY] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN} jobs=${#CANDIDATES[@]}"
echo "[RIEI-PARITY] row=rx1_1_rx7_7_to_rx1_19 paper_target=77.88+/-2.23 formal_metric=last5"
for gpu in "${GPU_IDS[@]}"; do
  current=0
  if [[ "${DRY_RUN}" != "1" ]]; then current="$(gpu_process_count "${gpu}")"; fi
  total=$((current + 1))
  echo "[CAPACITY] gpu=${gpu} current=${current} planned_peak=1 total_peak=${total} max=${MAX_TRAIN_PER_GPU}"
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
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
  printf 'job_id\tgpu\toptimizer\treduction\trms_normalize\tfeature_norm\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'job_id\tgpu\tpid\tlog\n' > "${RUN_ROOT}/scheduler_pids.tsv"
fi

for i in "${!CANDIDATES[@]}"; do
  IFS='|' read -r candidate optimizer reduction rms feature_norm <<< "${CANDIDATES[$i]}"
  gpu="${GPU_IDS[$i]}"
  job_id="riei_${candidate}_seed${SEED}"
  run_dir="${RUN_ROOT}/${job_id}"
  log_dir="${LOG_ROOT}/${job_id}"
  cmd=(env
    "METHODS=riei_fd" "WISIG_PROTOCOL=riei_original" "GPU_IDS=${gpu}"
    "TRAIN_RXS=1-1,7-7" "TEST_RXS=1-19"
    "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}"
    "BASELINE_EPOCHS=200" "SEED=${SEED}" "SAT_EVAL=0"
    "RIEI_PAPER_EVAL_LAST_N=5" "RIEI_TEST_EVAL_INTERVAL=10"
    "RIEI_OPTIMIZER=${optimizer}" "RIEI_SGD_MOMENTUM=0"
    "RIEI_CE_REDUCTION=${reduction}" "RIEI_MI_REDUCTION=${reduction}" "RIEI_IE_REDUCTION=${reduction}"
    "RIEI_WISIG_RMS_NORMALIZE=${rms}" "RIEI_LAMBDA_FEATURE_NORM=${feature_norm}"
    bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)

  echo "[JOB] id=${job_id} gpu=${gpu} optimizer=${optimizer} reduction=${reduction} rms=${rms} feature_norm=${feature_norm}"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  if [[ "${DRY_RUN}" == "1" ]]; then continue; fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${gpu}" "${optimizer}" "${reduction}" "${rms}" "${feature_norm}" "${run_dir}" "$(format_cmd "${cmd[@]}")" \
    >> "${RUN_ROOT}/scheduler_manifest.tsv"
  log_file="${LOG_ROOT}/${job_id}.launcher.log"
  nohup "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${gpu}" "${pid}" "${log_file}" >> "${RUN_ROOT}/scheduler_pids.tsv"
  echo "[LAUNCHED] id=${job_id} gpu=${gpu} pid=${pid} log=${log_file}"
done

echo "[RIEI-PARITY] submitted dry_run=${DRY_RUN}"
