#!/usr/bin/env bash
set -euo pipefail

# Paper-faithful DRIFT Table I + RIEI Table III seed matrix.
# This launcher is intentionally isolated from CVS/Stage2/satellite experiments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ID="${RUN_ID:-paper_repro_original_riei_drift_seed1337_20260714_090706}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/paper_reproduction/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"
GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-1}"

usage() {
  cat <<'EOF'
Options:
  --dry-run              Print the 13-job matrix without writing run artifacts (default)
  --launch               Launch DRIFT Table I plus all 12 RIEI Table III rows
  --gpu-ids CSV          GPU ids for round-robin packing (default 0..7)
  --python PATH          Remote Python executable
  --wisig-pkl PATH       ManySig.pkl path
  --run-id ID            Unique run identifier
  --seed N               Training seed (default 1337)
  --max-train-per-gpu N  Hard capacity ceiling including existing jobs (default 2)
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
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
  echo "ERROR: at least one GPU id is required" >&2
  exit 2
fi

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

format_cmd() {
  printf '%q ' "$@"
}

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

declare -A PLANNED=()
for gpu in "${GPU_IDS[@]}"; do
  PLANNED["${gpu}"]=0
done
PLANNED["${GPU_IDS[0]}"]=$((PLANNED["${GPU_IDS[0]}"] + 1))
for i in "${!TABLE3_ROWS[@]}"; do
  gpu="${GPU_IDS[$(((i + 1) % ${#GPU_IDS[@]}))]}"
  PLANNED["${gpu}"]=$((PLANNED["${gpu}"] + 1))
done

echo "[PAPER-ORIGINAL] run_id=${RUN_ID} seed=${SEED} dry_run=${DRY_RUN}"
echo "[PAPER-ORIGINAL] run_root=${RUN_ROOT}"
echo "[PAPER-ORIGINAL] log_root=${LOG_ROOT}"
echo "[PAPER-ORIGINAL] jobs=13 drift=1 riei_table3=12"
for gpu in "${GPU_IDS[@]}"; do
  if [[ "${DRY_RUN}" == "1" ]]; then
    current=0
  else
    current="$(gpu_process_count "${gpu}")"
  fi
  total=$((current + PLANNED["${gpu}"]))
  echo "[CAPACITY] gpu=${gpu} current=${current} planned=${PLANNED["${gpu}"]} total=${total} max=${MAX_TRAIN_PER_GPU}"
  if (( total > MAX_TRAIN_PER_GPU )); then
    echo "ERROR: capacity gate failed for GPU ${gpu}" >&2
    exit 3
  fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
  if [[ -e "${RUN_ROOT}" || -e "${LOG_ROOT}" ]]; then
    echo "ERROR: unique run/log root already exists; refusing to overwrite: ${RUN_ROOT} ${LOG_ROOT}" >&2
    exit 4
  fi
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python is not executable: ${PYTHON_BIN}" >&2
    exit 5
  fi
  if [[ ! -f "${WISIG_PKL}" ]]; then
    echo "ERROR: dataset not found: ${WISIG_PKL}" >&2
    exit 6
  fi
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
  printf 'job_id\tmethod\tprotocol\tgpu\ttrain_rxs\ttest_rxs\trun_dir\tnohup_log\tcommand\n' > "${RUN_ROOT}/scheduler_manifest.tsv"
  printf 'job_id\tgpu\tpid\tnohup_log\n' > "${RUN_ROOT}/scheduler_pids.tsv"
fi

launch_job() {
  local job_id="$1"
  local method="$2"
  local protocol="$3"
  local gpu="$4"
  local train_rxs="$5"
  local test_rxs="$6"
  local run_dir="$7"
  local log_dir="$8"
  local nohup_log="${LOG_ROOT}/${job_id}_scheduler_nohup.out"
  local -a cmd=(
    env
    "METHODS=${method}"
    "WISIG_PROTOCOL=${protocol}"
    "GPU_IDS=${gpu}"
    "TRAIN_RXS=${train_rxs}"
    "TEST_RXS=${test_rxs}"
    "RUN_ROOT=${run_dir}"
    "LOG_ROOT=${log_dir}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "WISIG_PKL=${WISIG_PKL}"
    "BASELINE_EPOCHS=200"
    "SEED=${SEED}"
    "SAT_EVAL=0"
  )
  if [[ "${protocol}" == "drift_day1" ]]; then
    cmd+=("DRIFT_PAPER_EVAL_LAST_N=5")
  else
    cmd+=("RIEI_PAPER_EVAL_LAST_N=10")
  fi
  cmd+=(bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done)

  echo "[JOB] id=${job_id} method=${method} protocol=${protocol} gpu=${gpu} train=${train_rxs} test=${test_rxs}"
  echo "[CMD] $(format_cmd "${cmd[@]}")"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job_id}" "${method}" "${protocol}" "${gpu}" "${train_rxs}" "${test_rxs}" \
    "${run_dir}" "${nohup_log}" "$(format_cmd "${cmd[@]}")" >> "${RUN_ROOT}/scheduler_manifest.tsv"
  nohup "${cmd[@]}" > "${nohup_log}" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${gpu}" "${pid}" "${nohup_log}" >> "${RUN_ROOT}/scheduler_pids.tsv"
  echo "[LAUNCHED] id=${job_id} gpu=${gpu} pid=${pid} log=${nohup_log}"
}

launch_job \
  "drift_table1_seed${SEED}" "drift" "drift_day1" "${GPU_IDS[0]}" \
  "1-1,14-7,7-7" "1-19,19-2,2-1,2-19,20-1,7-14,8-8" \
  "${RUN_ROOT}/drift_table1" "${LOG_ROOT}/drift_table1"

for i in "${!TABLE3_ROWS[@]}"; do
  IFS='|' read -r combo_id train_rxs test_rxs <<< "${TABLE3_ROWS[$i]}"
  gpu="${GPU_IDS[$(((i + 1) % ${#GPU_IDS[@]}))]}"
  launch_job \
    "riei_${combo_id}_seed${SEED}" "riei_fd" "riei_original" "${gpu}" \
    "${train_rxs}" "${test_rxs}" \
    "${RUN_ROOT}/riei_table3/${combo_id}" "${LOG_ROOT}/riei_table3/${combo_id}"
done

echo "[PAPER-ORIGINAL] submitted dry_run=${DRY_RUN}"
