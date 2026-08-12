#!/usr/bin/env bash
set -euo pipefail

RUN_ID="phase1_clic_target_metrics_20260812_v1"
PREDICTION_RUN_ID="phase1_clic_target_prediction_20260812_v1"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
EVALUATOR_ENTRY="${CODE_ROOT}/evaluate_phase1_clic_target_leo.py"
PREDICTION_RUN_ROOT="${PROJECT_ROOT}/runs/${PREDICTION_RUN_ID}"
PREDICTION_ROOT="${PREDICTION_RUN_ROOT}/predictions"
TRUTH_SIDECAR="${PREDICTION_RUN_ROOT}/sealed_target/truth_sidecar.json"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
METRICS_ROOT="${RUN_ROOT}/metrics"
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${EVALUATOR_ENTRY}" ]] || { echo "missing target evaluator entry" >&2; exit 2; }

metrics_command() {
  local fold="$1" arm="$2" candidate="F${fold}${arm}_CLIC12"
  METRICS_CMD=("${PYTHON}" -u "${EVALUATOR_ENTRY}"
    --seal-target-metrics
    --prediction "${PREDICTION_ROOT}/${candidate}.prediction.json"
    --truth-sidecar "${TRUTH_SIDECAR}"
    --output "${METRICS_ROOT}/${candidate}.metrics.json")
}

emit_command() {
  local fold="$1" arm="$2"; shift 2
  printf '[DRY-RUN] stage=CLIC_TARGET_METRICS fold=%s arm=%s' "${fold}" "${arm}"
  printf ' %q' "$@"
  printf '\n'
}

check_inputs() {
  local fold arm candidate
  [[ -f "${TRUTH_SIDECAR}" ]] || { echo "missing sealed target truth sidecar" >&2; return 2; }
  for fold in 1 2 3 4 5 6; do
    for arm in C G; do
      candidate="F${fold}${arm}_CLIC12"
      [[ -f "${PREDICTION_ROOT}/${candidate}.prediction.json" ]] || {
        echo "missing sealed target prediction: ${candidate}" >&2; return 2;
      }
    done
  done
}

run_fold() {
  local fold="$1" arm
  for arm in C G; do
    metrics_command "${fold}" "${arm}"
    PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      "${METRICS_CMD[@]}"
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for fold in 1 2 3 4 5 6; do
    for arm in C G; do
      metrics_command "${fold}" "${arm}"
      emit_command "${fold}" "${arm}" "${METRICS_CMD[@]}"
    done
  done
  exit 0
fi

[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || {
  echo "refusing to overwrite CLIC target metrics run/log root" >&2; exit 3;
}
check_inputs
mkdir -p "${METRICS_ROOT}" "${LOG_ROOT}"

declare -a pids folds logs
for fold in 1 2 3 4 5 6; do
  log_path="${LOG_ROOT}/F${fold}_target_metrics.out"
  run_fold "${fold}" >"${log_path}" 2>&1 &
  pids+=("$!"); folds+=("${fold}"); logs+=("${log_path}")
done

printf 'pid|fold|stage|log_path\n' >"${LOG_ROOT}/pids_target_metrics6.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|CLIC_TARGET_METRICS_C_THEN_G|%s\n' \
    "${pids[index]}" "${folds[index]}" "${logs[index]}" \
    >>"${LOG_ROOT}/pids_target_metrics6.tsv"
done

status=0
for index in "${!pids[@]}"; do
  wait "${pids[index]}" || status=1
done
exit "${status}"
