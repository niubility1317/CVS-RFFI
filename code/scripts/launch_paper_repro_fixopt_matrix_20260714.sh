#!/usr/bin/env bash
set -euo pipefail

# Repaired semantics plus historical fix_optimized guards.
# Stage 1 launches eight DRIFT Table I variants, one per GPU.
# Stage 2 runs the complete RIEI Table III 12-row matrix with feature-norm guard.
# Per-GPU queues keep peak added occupancy at one training process.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_fixopt_riei_drift_seed1337_20260714_105000}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/paper_reproduction/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-1}"

usage() {
  cat <<'EOF'
Options:
  --dry-run              Print the 20-job matrix without writing run artifacts (default)
  --launch               Launch 8 DRIFT variants then all 12 RIEI Table III rows
  --gpu-ids CSV          Exactly eight GPU ids (default 0..7)
  --python PATH          Remote Python executable
  --wisig-pkl PATH       ManySig.pkl path
  --run-id ID            Unique run identifier
  --seed N               Training seed (default 1337)
  --max-train-per-gpu N  Peak hard capacity ceiling including existing jobs (default 2)
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
  echo "ERROR: this matrix requires exactly eight GPU ids" >&2
  exit 2
fi

declare -a DRIFT_VARIANTS=(
  "D01_cap3000_mse020|3000|0.020|0"
  "D02_cap3500_mse020|3500|0.020|0"
  "D03_cap4000_mse020|4000|0.020|0"
  "D04_cap4500_mse020|4500|0.020|0"
  "D05_cap4000_mse015|4000|0.015|0"
  "D06_cap4500_mse015|4500|0.015|0"
  "D07_cap5000_mse015|5000|0.015|0"
  "D08_cap4000_mse020_fn1e5|4000|0.020|0.00001"
)

declare -a TABLE3_ROWS=(
  "rx1_1_rx7_7_to_rx1_19|1-1,7-7|1-19"
  "rx1_1_rx8_8_to_rx1_19|1-1,8-8|1-19"
  "rx1_1_rx14_7_to_rx1_19|1-1,14-7|1-19"
  "rx7_7_rx8_8_to_rx1_19|7-7,8-8|1-19"
  "rx7_7_rx14_7_to_rx1_19|7-7,14-7|1-19"
  "rx8_8_rx14_7_to_rx1_19|8-8,14-7|1-19"
  "rx1_1_rx1_19_to_rx14_7|1-1,1-19|14-7"
  "rx1_1_rx7_7_to_rx14_7|1-1,7-7|14-7"
  "rx1_1_rx8_8_to_rx14_7|1-1,8-8|14-7"
  "rx1_19_rx7_7_to_rx14_7|1-19,7-7|14-7"
  "rx1_19_rx8_8_to_rx14_7|1-19,8-8|14-7"
  "rx7_7_rx8_8_to_rx14_7|7-7,8-8|14-7"
)

format_cmd() { printf '%q ' "$@"; }
gpu_process_count() {
  nvidia-smi --id="$1" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

declare -A PLANNED=()
for gpu in "${GPU_IDS[@]}"; do PLANNED["${gpu}"]=1; done
for i in "${!TABLE3_ROWS[@]}"; do
  gpu="${GPU_IDS[$((i % 8))]}"
  PLANNED["${gpu}"]=$((PLANNED["${gpu}"] + 1))
done

echo "[FIXOPT] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN}"
echo "[FIXOPT] jobs=20 drift_variants=8 riei_table3=12"
echo "[FIXOPT] riei_guard=lambda_feature_norm=0.0001"
echo "[FIXOPT] drift_semantics=center_mode=batch,domain_sum,mse_reduction=sum"
for gpu in "${GPU_IDS[@]}"; do
  current=0
  if [[ "${DRY_RUN}" != "1" ]]; then current="$(gpu_process_count "${gpu}")"; fi
  total=$((current + 1))
  echo "[CAPACITY] gpu=${gpu} current=${current} queued=${PLANNED["${gpu}"]} planned_peak=1 total_peak=${total} max=${MAX_TRAIN_PER_GPU}"
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
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}" "${LOG_ROOT}/queues"
  printf 'job_id\tmethod\tprotocol\tvariant\tgpu\ttrain_rxs\ttest_rxs\trun_dir\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'queue_id\tgpu\tpid\tqueue_log\n' > "${RUN_ROOT}/scheduler_pids.tsv"
  for gpu in "${GPU_IDS[@]}"; do printf '#!/usr/bin/env bash\nset -u\n' > "${LOG_ROOT}/queues/gpu_${gpu}.sh"; done
fi

queue_job() {
  local job_id="$1" method="$2" protocol="$3" variant="$4" gpu="$5" train_rxs="$6" test_rxs="$7"
  local mse_cap="$8" lambda_mse="$9" drift_feature_norm="${10}" riei_feature_norm="${11}"
  local run_dir="${RUN_ROOT}/${job_id}" log_dir="${LOG_ROOT}/${job_id}"
  local -a cmd=(env
    "METHODS=${method}" "WISIG_PROTOCOL=${protocol}" "GPU_IDS=${gpu}"
    "TRAIN_RXS=${train_rxs}" "TEST_RXS=${test_rxs}"
    "RUN_ROOT=${run_dir}" "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}" "WISIG_PKL=${WISIG_PKL}"
    "BASELINE_EPOCHS=200" "SEED=${SEED}" "SAT_EVAL=0"
    "RIEI_LAMBDA_FEATURE_NORM=${riei_feature_norm}"
    "DRIFT_MSE_CAP=${mse_cap}" "DRIFT_LAMBDA_MSE=${lambda_mse}"
    "DRIFT_LAMBDA_FEATURE_NORM=${drift_feature_norm}")
  if [[ "${protocol}" == "drift_day1" ]]; then
    cmd+=("DRIFT_PAPER_EVAL_LAST_N=5")
  else
    cmd+=("RIEI_PAPER_EVAL_LAST_N=10")
  fi
  cmd+=(bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)

  echo "[JOB] id=${job_id} method=${method} variant=${variant} gpu=${gpu} train=${train_rxs} test=${test_rxs}"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${method}" "${protocol}" "${variant}" "${gpu}" "${train_rxs}" "${test_rxs}" "${run_dir}" "$(format_cmd "${cmd[@]}")" \
    >> "${RUN_ROOT}/scheduler_manifest.tsv"
  local queue_file="${LOG_ROOT}/queues/gpu_${gpu}.sh"
  printf 'echo %q\n' "[QUEUE-JOB-START] id=${job_id} gpu=${gpu}" >> "${queue_file}"
  format_cmd "${cmd[@]}" >> "${queue_file}"
  printf '\nstatus=$?\necho "[QUEUE-JOB-END] id=%s gpu=%s status=${status}"\n' "${job_id}" "${gpu}" >> "${queue_file}"
}

for i in "${!DRIFT_VARIANTS[@]}"; do
  IFS='|' read -r variant mse_cap lambda_mse feature_norm <<< "${DRIFT_VARIANTS[$i]}"
  queue_job "drift_${variant}_seed${SEED}" drift drift_day1 "${variant}" "${GPU_IDS[$i]}" \
    "1-1,14-7,7-7" "1-19,19-2,2-1,2-19,20-1,7-14,8-8" \
    "${mse_cap}" "${lambda_mse}" "${feature_norm}" 0
done

for i in "${!TABLE3_ROWS[@]}"; do
  IFS='|' read -r combo train_rxs test_rxs <<< "${TABLE3_ROWS[$i]}"
  queue_job "riei_${combo}_seed${SEED}" riei_fd riei_original RIEI_fixopt_fn1e4 "${GPU_IDS[$((i % 8))]}" \
    "${train_rxs}" "${test_rxs}" 0 0.02 0 0.0001
done

if [[ "${DRY_RUN}" != "1" ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    queue_file="${LOG_ROOT}/queues/gpu_${gpu}.sh"
    queue_log="${LOG_ROOT}/gpu_${gpu}_queue.log"
    chmod +x "${queue_file}"
    nohup bash "${queue_file}" > "${queue_log}" 2>&1 < /dev/null &
    pid="$!"
    printf 'gpu_%s\t%s\t%s\t%s\n' "${gpu}" "${gpu}" "${pid}" "${queue_log}" >> "${RUN_ROOT}/scheduler_pids.tsv"
    echo "[QUEUE-LAUNCHED] gpu=${gpu} pid=${pid} log=${queue_log} jobs=${PLANNED["${gpu}"]}"
  done
fi

echo "[FIXOPT] submitted dry_run=${DRY_RUN}"
