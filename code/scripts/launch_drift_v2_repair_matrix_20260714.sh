#!/usr/bin/env bash
set -euo pipefail

# DRIFT v2 protocol-repair discovery matrix. One sequential job is added per GPU.
# This launcher must not be started until the preceding paper-reproduction queues exit.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_drift_v2_repair_20260714_141223}"
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
  --launch               Launch one DRIFT candidate per GPU
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

# id|batch|sample_strategy|rms_normalize|mse_reduction|mse_cap|lambda_mse|grad_clip
declare -a CANDIDATES=(
  "V201_strict_raw|256|random|0|sum|0|0.020|0"
  "V202_v2_cap3500|256|random|0|sum|3500|0.020|0"
  "V203_rms_control|256|random|1|sum|3500|0.020|0"
  "V204_front_control|256|front|0|sum|3500|0.020|0"
  "V205_batch64_control|64|random|0|sum|3500|0.020|0"
  "V206_mean_impl|256|random|0|mean|0|0.020|0"
  "V207_cap4000_lmse015|256|random|0|sum|4000|0.015|0"
  "V208_cap3500_clip5|256|random|0|sum|3500|0.020|5"
)

format_cmd() { printf '%q ' "$@"; }
gpu_process_count() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

echo "[DRIFT-V2] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN} jobs=${#CANDIDATES[@]}"
echo "[DRIFT-V2] paper_target=73.54 aggregation=five_seed_final_epoch discovery=single_seed"
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
  printf 'job_id\tgpu\tbatch\tsample_strategy\trms_normalize\tmse_reduction\tmse_cap\tlambda_mse\tgrad_clip\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'job_id\tgpu\tpid\tlog\n' > "${RUN_ROOT}/scheduler_pids.tsv"
fi

for i in "${!CANDIDATES[@]}"; do
  IFS='|' read -r candidate batch strategy rms reduction cap lambda_mse grad_clip <<< "${CANDIDATES[$i]}"
  gpu="${GPU_IDS[$i]}"
  job_id="drift_${candidate}_seed${SEED}"
  run_dir="${RUN_ROOT}/${job_id}"
  log_dir="${LOG_ROOT}/${job_id}"
  cmd=(env
    "METHODS=drift" "WISIG_PROTOCOL=drift_day1" "GPU_IDS=${gpu}"
    "TRAIN_RXS=1-1,14-7,7-7" "TEST_RXS=1-19,19-2,2-1,2-19,20-1,7-14,8-8"
    "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}"
    "BASELINE_EPOCHS=200" "SEED=${SEED}" "SAT_EVAL=0"
    "DRIFT_PAPER_EVAL_LAST_N=1" "DRIFT_BATCH_SIZE=${batch}"
    "DRIFT_PAPER_SAMPLE_STRATEGY=${strategy}" "DRIFT_WISIG_RMS_NORMALIZE=${rms}"
    "DRIFT_MSE_REDUCTION=${reduction}" "DRIFT_MSE_CAP=${cap}"
    "DRIFT_LAMBDA_MSE=${lambda_mse}" "DRIFT_LAMBDA_FEATURE_NORM=0"
    "DRIFT_GRAD_CLIP_NORM=${grad_clip}"
    bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)

  echo "[JOB] id=${job_id} gpu=${gpu} batch=${batch} strategy=${strategy} rms=${rms} reduction=${reduction} cap=${cap} lambda_mse=${lambda_mse} grad_clip=${grad_clip}"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  if [[ "${DRY_RUN}" == "1" ]]; then continue; fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${gpu}" "${batch}" "${strategy}" "${rms}" "${reduction}" "${cap}" "${lambda_mse}" "${grad_clip}" "${run_dir}" "$(format_cmd "${cmd[@]}")" \
    >> "${RUN_ROOT}/scheduler_manifest.tsv"
  log_file="${LOG_ROOT}/${job_id}.launcher.log"
  nohup "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${gpu}" "${pid}" "${log_file}" >> "${RUN_ROOT}/scheduler_pids.tsv"
  echo "[LAUNCHED] id=${job_id} gpu=${gpu} pid=${pid} log=${log_file}"
done

echo "[DRIFT-V2] submitted dry_run=${DRY_RUN}"
