#!/usr/bin/env bash
set -euo pipefail

RUN_ID="phase1_clic_target_prediction_20260812_v1"
C_PREDICTOR_RUN_ID="phase1_clic_predictor_artifacts_20260812_v2"
G_PREDICTOR_RUN_ID="phase1_clic_g_bundles_20260812_v1"
TARGET_CACHE_RUN_ID="phase1_clic_target_confirmation_20260812_v2"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
EVALUATOR_ENTRY="${CODE_ROOT}/evaluate_phase1_clic_target_leo.py"
SEMANTICS_JSON="${CODE_ROOT}/configs/phase1_clic_target_test_semantics_20260812_v1.json"
C_PREDICTOR_ROOT="${PROJECT_ROOT}/runs/${C_PREDICTOR_RUN_ID}"
G_PREDICTOR_ROOT="${PROJECT_ROOT}/runs/${G_PREDICTOR_RUN_ID}"
CACHE_MANIFEST="${PROJECT_ROOT}/runs/${TARGET_CACHE_RUN_ID}/cache/cache_set.json"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
VALIDATION_ROOT="${RUN_ROOT}/validation"
SEALED_ROOT="${RUN_ROOT}/sealed_target"
PACKAGE_ROOT="${SEALED_ROOT}/iq_only_package"
PREDICTION_ROOT="${RUN_ROOT}/predictions"
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${EVALUATOR_ENTRY}" ]] || { echo "missing target evaluator entry" >&2; exit 2; }
[[ -f "${SEMANTICS_JSON}" ]] || { echo "missing target test-semantics JSON" >&2; exit 2; }

validation_command() {
  VALIDATION_CMD=("${PYTHON}" -u "${EVALUATOR_ENTRY}"
    --seal-target-validation
    --cache-set-manifest "${CACHE_MANIFEST}"
    --output-root "${VALIDATION_ROOT}"
    --test-semantics-json "${SEMANTICS_JSON}")
}

package_command() {
  local capsule_id="$1" split_id="$2"
  PACKAGE_CMD=("${PYTHON}" -u "${EVALUATOR_ENTRY}"
    --seal-target-package
    --cache-set-manifest "${CACHE_MANIFEST}"
    --output-root "${SEALED_ROOT}"
    --validator-receipt "${VALIDATION_ROOT}/validator_receipt.json"
    --expected-capsule-id "${capsule_id}"
    --expected-split-id "${split_id}"
    --expected-protocol-schema p2_min_v1)
}

prediction_command() {
  local fold="$1" arm="$2" predictor output
  if [[ "${arm}" == "C" ]]; then
    predictor="${C_PREDICTOR_ROOT}/F${fold}C_CLIC12/c_predictor_state.json"
  else
    predictor="${G_PREDICTOR_ROOT}/F${fold}G_CLIC12/g_deployment_bundle.zip"
  fi
  output="${PREDICTION_ROOT}/F${fold}${arm}_CLIC12.prediction.json"
  PREDICTION_CMD=("${PYTHON}" -u "${EVALUATOR_ENTRY}"
    --publish-target-prediction
    --predictor-state "${predictor}"
    --package "${PACKAGE_ROOT}"
    --output "${output}")
}

emit_command() {
  local stage="$1" fold="$2" arm="$3"; shift 3
  printf '[DRY-RUN] stage=%s fold=%s arm=%s' "${stage}" "${fold}" "${arm}"
  printf ' %q' "$@"
  printf '\n'
}

check_inputs() {
  local fold
  [[ -f "${CACHE_MANIFEST}" ]] || { echo "missing target confirmation cache manifest" >&2; return 2; }
  for fold in 1 2 3 4 5 6; do
    [[ -f "${C_PREDICTOR_ROOT}/F${fold}C_CLIC12/c_predictor_state.json" ]] || {
      echo "missing C predictor: F${fold}" >&2; return 2;
    }
    [[ -f "${G_PREDICTOR_ROOT}/F${fold}G_CLIC12/g_deployment_bundle.zip" ]] || {
      echo "missing G predictor: F${fold}" >&2; return 2;
    }
  done
}

run_fold() {
  local fold="$1" arm
  for arm in C G; do
    prediction_command "${fold}" "${arm}"
    PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
      "${PREDICTION_CMD[@]}"
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  validation_command
  emit_command TARGET_VALIDATION - - "${VALIDATION_CMD[@]}"
  package_command '<cache-derived-capsule-id>' '<cache-derived-split-id>'
  emit_command TARGET_PACKAGE - - "${PACKAGE_CMD[@]}"
  for fold in 1 2 3 4 5 6; do
    for arm in C G; do
      prediction_command "${fold}" "${arm}"
      emit_command TARGET_PREDICTION "${fold}" "${arm}" "${PREDICTION_CMD[@]}"
    done
  done
  exit 0
fi

[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || {
  echo "refusing to overwrite CLIC target prediction run/log root" >&2; exit 3;
}
check_inputs
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

validation_command
PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" "${VALIDATION_CMD[@]}" \
  >"${LOG_ROOT}/target_validation.out" 2>&1

mapfile -t derived_ids < <("${PYTHON}" -c \
  'import json,sys; x=json.load(open(sys.argv[1], encoding="utf-8")); print(x["capsule_id"]); print(x["split_id"])' \
  "${VALIDATION_ROOT}/validator_receipt.json")
[[ "${#derived_ids[@]}" == "2" && -n "${derived_ids[0]}" && -n "${derived_ids[1]}" ]] || {
  echo "target validation did not produce derived capsule/split IDs" >&2; exit 2;
}

package_command "${derived_ids[0]}" "${derived_ids[1]}"
PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" "${PACKAGE_CMD[@]}" \
  >"${LOG_ROOT}/target_package.out" 2>&1
[[ -f "${PACKAGE_ROOT}/manifest.json" && -f "${PACKAGE_ROOT}/received_iq.npz" ]] || {
  echo "target IQ-only package did not close" >&2; exit 2;
}
mkdir -p "${PREDICTION_ROOT}"

declare -a pids folds logs
for fold in 1 2 3 4 5 6; do
  log_path="${LOG_ROOT}/F${fold}_target_predictions.out"
  run_fold "${fold}" >"${log_path}" 2>&1 &
  pids+=("$!"); folds+=("${fold}"); logs+=("${log_path}")
done

printf 'pid|fold|stage|log_path\n' >"${LOG_ROOT}/pids_target_prediction6.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|CLIC_TARGET_C_THEN_G|%s\n' \
    "${pids[index]}" "${folds[index]}" "${logs[index]}" \
    >>"${LOG_ROOT}/pids_target_prediction6.tsv"
done

status=0
for index in "${!pids[@]}"; do
  wait "${pids[index]}" || status=1
done
exit "${status}"
