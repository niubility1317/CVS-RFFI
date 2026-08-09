#!/usr/bin/env bash
set -euo pipefail

# Second release-engineering repair for the frozen PAMR postfreeze pair stage.
# It consumes only the already-complete v2 NPZ exports and immutable training
# final checkpoints.  It performs no export, proxy score, fit, calibration,
# selection, checkpoint choice, or source/LEO data regeneration.
PAIR_ONLY_RUN_ID="${PAIR_ONLY_RUN_ID:-phase1_pamr12_20260809_v1_pair6_20260809_v1}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_RUN_ROOT="${TRAIN_RUN_ROOT:-${PROJECT_ROOT}/runs/phase1_pamr12_20260809_v1}"
FROZEN_V2_POSTFREEZE_ROOT="${PROJECT_ROOT}/runs/phase1_pamr12_20260809_v1_postfreeze_v2"
V2_POSTFREEZE_ROOT="${V2_POSTFREEZE_ROOT:-${FROZEN_V2_POSTFREEZE_ROOT}}"
PAIR_ROOT="${PAIR_ROOT:-${PROJECT_ROOT}/runs/${PAIR_ONLY_RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${PAIR_ONLY_RUN_ID}}"
PAIR_EVAL_SCRIPT="${CODE_ROOT}/scripts/eval_phase1_pamr_pair.py"
DRY_RUN="${DRY_RUN:-0}"

SOURCE_DAYS="2021_03_01,2021_03_08"
SOURCE_RXS="1-1,1-19,14-7,18-2,19-2,2-1"
SAT_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
SOURCE_SAT_SEED="7281718"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || {
  echo "DRY_RUN must be 0 or 1" >&2
  exit 2
}
[[ -f "${PAIR_EVAL_SCRIPT}" ]] || { echo "missing pair evaluator: ${PAIR_EVAL_SCRIPT}" >&2; exit 2; }
[[ "${V2_POSTFREEZE_ROOT}" == "${FROZEN_V2_POSTFREEZE_ROOT}" ]] || {
  echo "pair-only input must be the frozen v2 postfreeze root" >&2
  exit 3
}
[[ "${TRAIN_RUN_ROOT}" == "${PROJECT_ROOT}/runs/phase1_pamr12_20260809_v1" ]] || {
  echo "pair-only input must be the frozen PAMR12 training root" >&2
  exit 3
}
[[ "${PAIR_ROOT}" != "${V2_POSTFREEZE_ROOT}" && "${PAIR_ROOT}" != "${TRAIN_RUN_ROOT}" ]] || {
  echo "pair-only output root must differ from immutable inputs" >&2
  exit 3
}

# Frozen local4 source order, identical to the completed PAMR12 matrix.
FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20"
  "14-10,20-19,6-15,8-20"
  "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20"
  "14-10,14-7,20-15,20-19"
  "14-7,20-15,20-19,6-15"
)

print_command() {
  printf '[DRY-RUN][PAIR6] PYTHONFAULTHANDLER=1 PAMR_PAIR_NATIVE_DIAGNOSTIC=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=%q PYTHONPATH=%q' \
    "" "${CODE_ROOT}:${PROJECT_ROOT}"
  printf ' %q' "$@"
  printf '\n'
}

require_pair_inputs() {
  local fold="$1"
  local candidate
  for arm in C G; do
    candidate="F${fold}${arm}_PAMR12"
    for required in \
      "${V2_POSTFREEZE_ROOT}/${candidate}/clean_development.npz" \
      "${V2_POSTFREEZE_ROOT}/${candidate}/source_leo_final_only.npz" \
      "${TRAIN_RUN_ROOT}/${candidate}/final_ssdg.pth"; do
      [[ -f "${required}" ]] || { echo "missing frozen pair input: ${required}" >&2; return 2; }
    done
  done
}

run_pair() {
  local fold="$1"
  local fold_index=$((fold - 1))
  local train_tx="${FOLD_TRAIN_TX[fold_index]}"
  local c_candidate="F${fold}C_PAMR12"
  local g_candidate="F${fold}G_PAMR12"
  local output_json="${PAIR_ROOT}/F${fold}_C_vs_G_pair_metrics.json"
  local -a pair_command=(
    "${PYTHON}" -X faulthandler -u "${PAIR_EVAL_SCRIPT}"
    --c-clean-npz "${V2_POSTFREEZE_ROOT}/${c_candidate}/clean_development.npz"
    --g-clean-npz "${V2_POSTFREEZE_ROOT}/${g_candidate}/clean_development.npz"
    --c-leo-npz "${V2_POSTFREEZE_ROOT}/${c_candidate}/source_leo_final_only.npz"
    --g-leo-npz "${V2_POSTFREEZE_ROOT}/${g_candidate}/source_leo_final_only.npz"
    --c-final-checkpoint "${TRAIN_RUN_ROOT}/${c_candidate}/final_ssdg.pth"
    --g-final-checkpoint "${TRAIN_RUN_ROOT}/${g_candidate}/final_ssdg.pth"
    --candidate-pair "F${fold}_C_vs_G"
    --source-tx-ids "${train_tx}"
    --expected-scenarios "${SAT_SCENARIOS}"
    --expected-source-days "${SOURCE_DAYS}"
    --expected-source-rxs "${SOURCE_RXS}"
    --source-sat-seed "${SOURCE_SAT_SEED}"
    --expected-source-count 1600
    --expected-target-old-count 400
    --expected-proxy-count 400
    --native-thread-limit 1
    --native-diagnostic
    --output-metrics-json "${output_json}"
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    print_command "${pair_command[@]}"
    return 0
  fi
  env \
    PYTHONFAULTHANDLER=1 \
    PAMR_PAIR_NATIVE_DIAGNOSTIC=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES= \
    PYTHONPATH="${CODE_ROOT}:${PROJECT_ROOT}" \
    "${pair_command[@]}" >"${LOG_ROOT}/F${fold}_C_vs_G_pair.out" 2>&1
}

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -d "${V2_POSTFREEZE_ROOT}" ]] || { echo "missing frozen v2 postfreeze root" >&2; exit 2; }
  [[ -d "${TRAIN_RUN_ROOT}" ]] || { echo "missing frozen PAMR12 training root" >&2; exit 2; }
  [[ ! -e "${PAIR_ROOT}" ]] || { echo "refusing to overwrite pair-only root: ${PAIR_ROOT}" >&2; exit 3; }
  [[ ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite pair-only log root: ${LOG_ROOT}" >&2; exit 3; }
  for fold in 1 2 3 4 5 6; do
    require_pair_inputs "${fold}"
  done
  mkdir -p "${PAIR_ROOT}" "${LOG_ROOT}"
fi

# Deliberately serial CPU-only process isolation: each fold is an independent
# final-only reader.  The thread cap and faulthandler only expose/contain a
# native failure; they do not alter the frozen pair inputs or formulas.
if [[ "${DRY_RUN}" == "0" ]]; then
  printf 'fold|status|exit_code|log_path|metrics_json\n' >"${PAIR_ROOT}/pair_completion.tsv"
fi
overall_status=0
declare -A failure_counts=()
for fold in 1 2 3 4 5 6; do
  if run_pair "${fold}"; then
    if [[ "${DRY_RUN}" == "0" ]]; then
      printf '%s|SUCCEEDED|0|%s|%s\n' \
        "${fold}" "${LOG_ROOT}/F${fold}_C_vs_G_pair.out" \
        "${PAIR_ROOT}/F${fold}_C_vs_G_pair_metrics.json" >>"${PAIR_ROOT}/pair_completion.tsv"
    fi
    continue
  fi
  pair_exit="$?"
  overall_status=8
  if [[ "${DRY_RUN}" == "0" ]]; then
    printf '%s|FAILED|%s|%s|%s\n' \
      "${fold}" "${pair_exit}" "${LOG_ROOT}/F${fold}_C_vs_G_pair.out" \
      "${PAIR_ROOT}/F${fold}_C_vs_G_pair_metrics.json" >>"${PAIR_ROOT}/pair_completion.tsv"
  fi
  failure_key="exit_${pair_exit}"
  failure_counts["${failure_key}"]=$(( ${failure_counts["${failure_key}"]:-0} + 1 ))
  if [[ "${failure_counts["${failure_key}"]}" -ge 2 ]]; then
    echo "stopping pair-only dispatch after two distinct rows share ${failure_key}" >&2
    break
  fi
done
exit "${overall_status}"
