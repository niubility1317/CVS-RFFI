#!/usr/bin/env bash
set -euo pipefail

# Five-seed final-epoch confirmation for the best DRIFT v2 discovery row (V206).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_drift_v2_confirm_v206_20260714_164900}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/paper_reproduction/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"
GPU_IDS_CSV="${GPU_IDS:-1,6,7,0,2}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-1}"
SEEDS_CSV="${SEEDS:-1337,2024,3407,4242,7777}"

usage() {
  cat <<'EOF'
Usage: launch_drift_v2_confirm_v206_20260714.sh [--dry-run|--launch]
       [--gpu-ids 1,6,7,0,2] [--max-train-per-gpu 2]
       [--seeds 1337,2024,3407,4242,7777] [--run-id RUN_ID]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --launch) DRY_RUN=0; shift ;;
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --max-train-per-gpu) MAX_TRAIN_PER_GPU="$2"; shift 2 ;;
    --seeds) SEEDS_CSV="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; RUN_ROOT="${ROOT}/paper_reproduction/runs/${RUN_ID}"; LOG_ROOT="${ROOT}/paper_reproduction/logs/${RUN_ID}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
if [[ "${#GPU_IDS[@]}" -ne 5 || "${#SEEDS[@]}" -ne 5 ]]; then
  echo "ERROR: confirmation requires exactly five GPU ids and five seeds" >&2
  exit 2
fi

format_cmd() { printf '%q ' "$@"; }
gpu_process_count() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

echo "[DRIFT-V2-CONFIRM] run_id=${RUN_ID} dry_run=${DRY_RUN} jobs=5"
echo "[DRIFT-V2-CONFIRM] candidate=V206_mean_impl paper_target=73.54 aggregation=five_seed_final_epoch"
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
  printf 'job_id\tgpu\tseed\tbatch\tsample_strategy\trms_normalize\tmse_reduction\tmse_cap\tlambda_mse\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'job_id\tgpu\tpid\tlog\n' > "${RUN_ROOT}/scheduler_pids.tsv"
fi

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPU_IDS[$i]}"
  job_id="drift_V206_mean_impl_seed${seed}"
  run_dir="${RUN_ROOT}/${job_id}"
  log_dir="${LOG_ROOT}/${job_id}"
  cmd=(env
    "METHODS=drift" "WISIG_PROTOCOL=drift_day1" "GPU_IDS=${gpu}"
    "TRAIN_RXS=1-1,14-7,7-7" "TEST_RXS=1-19,19-2,2-1,2-19,20-1,7-14,8-8"
    "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}"
    "BASELINE_EPOCHS=200" "SEED=${seed}" "SAT_EVAL=0"
    "DRIFT_PAPER_EVAL_LAST_N=1" "DRIFT_BATCH_SIZE=256"
    "DRIFT_PAPER_SAMPLE_STRATEGY=random" "DRIFT_WISIG_RMS_NORMALIZE=0"
    "DRIFT_MSE_REDUCTION=mean" "DRIFT_MSE_CAP=0"
    "DRIFT_LAMBDA_MSE=0.020" "DRIFT_LAMBDA_FEATURE_NORM=0"
    "DRIFT_GRAD_CLIP_NORM=0"
    bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)

  echo "[JOB] id=${job_id} gpu=${gpu} seed=${seed} batch=256 strategy=random rms=0 reduction=mean cap=0 lambda_mse=0.020"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  if [[ "${DRY_RUN}" == "1" ]]; then continue; fi

  printf '%s\t%s\t%s\t256\trandom\t0\tmean\t0\t0.020\t%s\t%s\n' \
    "${job_id}" "${gpu}" "${seed}" "${run_dir}" "$(format_cmd "${cmd[@]}")" \
    >> "${RUN_ROOT}/scheduler_manifest.tsv"
  log_file="${LOG_ROOT}/${job_id}.launcher.log"
  nohup "${cmd[@]}" > "${log_file}" 2>&1 < /dev/null &
  pid="$!"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${gpu}" "${pid}" "${log_file}" >> "${RUN_ROOT}/scheduler_pids.tsv"
  echo "[LAUNCHED] id=${job_id} gpu=${gpu} pid=${pid} log=${log_file}"
done

echo "[DRIFT-V2-CONFIRM] submitted dry_run=${DRY_RUN}"
