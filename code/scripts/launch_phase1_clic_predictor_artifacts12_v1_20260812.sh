#!/usr/bin/env bash
set -euo pipefail

# Seal the twelve source-authorized C/G predictor artifacts.  This stage never
# opens target IQ, truth, roles, or metrics.  Six CPU fold workers run C then G.
RUN_ID="phase1_clic_predictor_artifacts_20260812_v1"
TRAINING_RUN_ID="phase1_clic12_20260812_v5"
CLEAN_RUN_ID="phase1_clic_postfreeze_20260812_v4"
LEO_RUN_ID="phase1_clic_source_leo_20260812_v4"
PAIR_RUN_ID="phase1_clic_source_pair_20260812_v3"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
TRAINING_ROOT="${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}"
CLEAN_ROOT="${PROJECT_ROOT}/runs/${CLEAN_RUN_ID}"
LEO_ROOT="${PROJECT_ROOT}/runs/${LEO_RUN_ID}"
PAIR_ROOT="${PROJECT_ROOT}/runs/${PAIR_RUN_ID}"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
C_ENTRY_MODULE="cvsrffi.phase1_clic_target_leo"
G_ENTRY="${CODE_ROOT}/export_phase1_clic_deployment_bundle.py"
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${CODE_ROOT}/cvsrffi/phase1_clic_target_leo.py" ]] || { echo "missing C predictor entry" >&2; exit 2; }
[[ -f "${G_ENTRY}" ]] || { echo "missing G bundle entry" >&2; exit 2; }

check_inputs() {
  local fold arm candidate
  for fold in 1 2 3 4 5 6; do
    [[ -f "${PAIR_ROOT}/F${fold}_C_vs_G_pair.json" ]] || { echo "missing PAIR artifact: F${fold}" >&2; return 2; }
    for arm in C G; do
      candidate="F${fold}${arm}_CLIC12"
      [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" ]] || { echo "missing checkpoint: ${candidate}" >&2; return 2; }
      [[ -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || { echo "missing terminal: ${candidate}" >&2; return 2; }
      [[ -f "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz" ]] || { echo "missing clean NPZ: ${candidate}" >&2; return 2; }
      [[ -f "${LEO_ROOT}/${candidate}/source_leo.npz" ]] || { echo "missing source LEO NPZ: ${candidate}" >&2; return 2; }
      [[ -f "${LEO_ROOT}/${candidate}/source_leo.binding.json" ]] || { echo "missing source LEO binding: ${candidate}" >&2; return 2; }
    done
  done
}

c_command() {
  local fold="$1" output="$2" candidate="F${fold}C_CLIC12"
  C_CMD=("${PYTHON}" -u -m "${C_ENTRY_MODULE}"
    --seal-c-predictor-state
    --checkpoint "${TRAINING_ROOT}/${candidate}/final_ssdg.pth"
    --terminal-receipt-json "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json"
    --pair-artifact-json "${PAIR_ROOT}/F${fold}_C_vs_G_pair.json"
    --output "${output}"
    --fold-index "${fold}")
}

g_command() {
  local fold="$1" output="$2" candidate="F${fold}G_CLIC12"
  G_CMD=("${PYTHON}" -u "${G_ENTRY}"
    --checkpoint "${TRAINING_ROOT}/${candidate}/final_ssdg.pth"
    --terminal-receipt-json "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json"
    --clean-npz "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz"
    --leo-npz "${LEO_ROOT}/${candidate}/source_leo.npz"
    --leo-binding-json "${LEO_ROOT}/${candidate}/source_leo.binding.json"
    --output-bundle "${output}")
}

emit_command() {
  local stage="$1" fold="$2" arm="$3"; shift 3
  printf '[DRY-RUN] stage=%q fold=%q arm=%q' "${stage}" "${fold}" "${arm}"
  printf ' %q' "$@"
  printf '\n'
}

run_fold() {
  local fold="$1" c_dir g_dir
  c_dir="${RUN_ROOT}/F${fold}C_CLIC12"
  g_dir="${RUN_ROOT}/F${fold}G_CLIC12"
  mkdir -p "${c_dir}" "${g_dir}"
  c_command "${fold}" "${c_dir}/c_predictor_state.json"
  PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${C_CMD[@]}"
  g_command "${fold}" "${g_dir}/g_deployment_bundle.zip"
  PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${G_CMD[@]}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for fold in 1 2 3 4 5 6; do
    c_command "${fold}" "${RUN_ROOT}/F${fold}C_CLIC12/c_predictor_state.json"
    emit_command CLIC_C_DESCRIPTOR "${fold}" C "${C_CMD[@]}"
    g_command "${fold}" "${RUN_ROOT}/F${fold}G_CLIC12/g_deployment_bundle.zip"
    emit_command CLIC_G_BUNDLE "${fold}" G "${G_CMD[@]}"
  done
  exit 0
fi

[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite CLIC predictor artifact run/log root" >&2; exit 3; }
check_inputs
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

declare -a pids folds logs
for fold in 1 2 3 4 5 6; do
  log_path="${LOG_ROOT}/F${fold}_predictor_artifacts.out"
  run_fold "${fold}" >"${log_path}" 2>&1 &
  pids+=("$!"); folds+=("${fold}"); logs+=("${log_path}")
done

printf 'pid|fold|stage|log_path\n' >"${LOG_ROOT}/pids_predictor_artifacts6.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|CLIC_C_DESCRIPTOR_G_BUNDLE|%s\n' \
    "${pids[index]}" "${folds[index]}" "${logs[index]}" >>"${LOG_ROOT}/pids_predictor_artifacts6.tsv"
done

status=0
for index in "${!pids[@]}"; do
  wait "${pids[index]}" || status=1
done
exit "${status}"
