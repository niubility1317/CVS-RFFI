#!/usr/bin/env bash
set -euo pipefail

# Paper-faithful reproduction automation for DRIFT and RIEI-FD.
# This script is intentionally separate from optimizer launchers: it only runs
# the original reproduction protocols and does not enable search/stabilization
# variants.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

RUN_ID="${RUN_ID:-paper_repro_20260605_145347_riei_drift}"
RUN_ROOT="${RUN_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU_IDS_CSV="${GPU_IDS:-0,1}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-1}"
SUITES_CSV="${SUITES:-drift_day1,riei_table3}"
WITH_RIEI_DRIFT_DAY1="${WITH_RIEI_DRIFT_DAY1:-0}"
WITH_CEN_A31_COMPARISON="${WITH_CEN_A31_COMPARISON:-0}"

usage() {
  cat <<'EOF'
Options:
  --dry-run                    Print scheduler commands only (default)
  --launch                     Start suites through nohup after capacity checks
  --suites CSV                 Suites: drift_day1,riei_table3
  --gpu-ids CSV                GPU ids used by suites; first id for DRIFT-Day1, second id for RIEI Table III
  --with-riei-drift-day1       Also run RIEI-FD under the DRIFT Day1 scene as a cross-paper comparison
  --with-cen-a31-comparison    Also run CEN_A31 in DRIFT Day1 comparison scene; not part of paper reproduction
  --max-train-per-gpu N        Capacity gate, default 2
EOF
}

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --launch)
      DRY_RUN=0
      shift
      ;;
    --suites)
      SUITES_CSV="$2"
      shift 2
      ;;
    --gpu-ids)
      GPU_IDS_CSV="$2"
      shift 2
      ;;
    --with-riei-drift-day1)
      WITH_RIEI_DRIFT_DAY1=1
      shift
      ;;
    --with-cen-a31-comparison)
      WITH_CEN_A31_COMPARISON=1
      shift
      ;;
    --max-train-per-gpu)
      MAX_TRAIN_PER_GPU="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPU_IDS_ARRAY <<< "${GPU_IDS_CSV}"
IFS=',' read -r -a SUITES <<< "${SUITES_CSV}"

gpu_at() {
  local idx="$1"
  if (( idx < ${#GPU_IDS_ARRAY[@]} )); then
    echo "${GPU_IDS_ARRAY[$idx]}"
  else
    echo "${GPU_IDS_ARRAY[0]}"
  fi
}

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

contains_suite() {
  local target="$1"
  local suite
  for suite in "${SUITES[@]}"; do
    suite="$(echo "${suite}" | xargs)"
    [[ "${suite}" == "${target}" ]] && return 0
  done
  return 1
}

ensure_capacity() {
  local suite="$1"
  local gpu="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  local count
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\tBLOCKED_CAPACITY\tgpu=%s\tactive_count=%s\tmax=%s\n" \
      "${suite}" "${gpu}" "${count}" "${MAX_TRAIN_PER_GPU}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 1
  fi
  return 0
}

start_suite() {
  local suite="$1"
  local gpu="$2"
  local nohup_path="$3"
  shift 3
  local cmd=("$@")
  mkdir -p "${LOG_ROOT}" "${RUN_ROOT}"
  printf "%s\t%s\t%s\t%s\n" "${suite}" "${gpu}" "${nohup_path}" "$(print_cmd "${cmd[@]}")" >> "${RUN_ROOT}/scheduler_manifest.tsv"
  echo "[PAPER-REPRO] suite=${suite} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[PAPER-REPRO-CMD]'
  print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if ! ensure_capacity "${suite}" "${gpu}"; then
    return 0
  fi
  nohup "${cmd[@]}" > "${nohup_path}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\n" "${suite}" "${gpu}" "${pid}" "${nohup_path}" | tee -a "${RUN_ROOT}/scheduler_pids.tsv"
}

build_drift_methods() {
  local methods="drift"
  if [[ "${WITH_RIEI_DRIFT_DAY1}" == "1" ]]; then
    methods="${methods},riei_fd"
  fi
  if [[ "${WITH_CEN_A31_COMPARISON}" == "1" ]]; then
    methods="${methods},cvsrffi_cen_a31"
  fi
  echo "${methods}"
}

cd "${ROOT}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
: > "${RUN_ROOT}/scheduler_manifest.tsv"

echo "[PAPER-REPRO] root=${ROOT}"
echo "[PAPER-REPRO] run_id=${RUN_ID} suites=${SUITES_CSV} gpu_ids=${GPU_IDS_CSV} dry_run=${DRY_RUN}"
echo "[PAPER-REPRO] with_riei_drift_day1=${WITH_RIEI_DRIFT_DAY1} with_cen_a31_comparison=${WITH_CEN_A31_COMPARISON}"
echo "[PAPER-REPRO] run_root=${RUN_ROOT}"
echo "[PAPER-REPRO] log_root=${LOG_ROOT}"

if contains_suite "drift_day1"; then
  drift_gpu="$(gpu_at 0)"
  drift_methods="$(build_drift_methods)"
  start_suite \
    "drift_day1" \
    "${drift_gpu}" \
    "${LOG_ROOT}/drift_day1_scheduler_nohup.out" \
    env \
      "METHODS=${drift_methods}" \
      "WISIG_PROTOCOL=drift_day1" \
      "GPU_IDS=${drift_gpu}" \
      "RUN_ROOT=${RUN_ROOT}/drift_day1" \
      "LOG_ROOT=${LOG_ROOT}/drift_day1" \
      "PYTHON_BIN=${PYTHON_BIN}" \
      "WISIG_PKL=${WISIG_PKL}" \
      "BASELINE_EPOCHS=200" \
      "DRIFT_PAPER_EVAL_LAST_N=5" \
      "SAT_EVAL=0" \
      bash "${ROOT}/run_wisig_paper_scope_queue.sh" --no-skip-done
fi

if contains_suite "riei_table3"; then
  riei_gpu="$(gpu_at 1)"
  start_suite \
    "riei_table3" \
    "${riei_gpu}" \
    "${LOG_ROOT}/riei_table3_scheduler_nohup.out" \
    env \
      "GPU_IDS=${riei_gpu}" \
      "PYTHON_BIN=${PYTHON_BIN}" \
      "WISIG_PKL=${WISIG_PKL}" \
      bash "${ROOT}/run_riei_original_table3_queue.sh" \
        --launch \
        --gpu-ids "${riei_gpu}" \
        --python "${PYTHON_BIN}" \
        --run-root "${RUN_ROOT}/riei_table3" \
        --log-root "${LOG_ROOT}/riei_table3"
fi

echo "[PAPER-REPRO] scheduler finished dry_run=${DRY_RUN} manifest=${RUN_ROOT}/scheduler_manifest.tsv"
