#!/usr/bin/env bash
set -euo pipefail

# Source-only CLIC postfreeze closure for the clean-v4 evidence chain.
# Each fold seals C/G common receipts and fixed400 proxy diagnostics, then
# reopens the complete clean + single-LEO chain to write one aggregate PAIR.
RUN_ID="phase1_clic_source_pair_20260812_v3"
TRAINING_RUN_ID="phase1_clic12_20260812_v5"
CLEAN_RUN_ID="phase1_clic_postfreeze_20260812_v4"
LEO_RUN_ID="phase1_clic_source_leo_20260812_v4"
POSTFREEZE_MATRIX_ID="phase1_clic_postfreeze_20260812_v4"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
TRAINING_ROOT="${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}"
CLEAN_ROOT="${PROJECT_ROOT}/runs/${CLEAN_RUN_ID}"
LEO_ROOT="${PROJECT_ROOT}/runs/${LEO_RUN_ID}"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
PAIR_ENTRY="${CODE_ROOT}/evaluate_phase1_clic_postfreeze_pair.py"
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${PAIR_ENTRY}" ]] || { echo "missing CLIC PAIR entry" >&2; exit 2; }

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20" "14-10,20-19,6-15,8-20" "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20" "14-10,14-7,20-15,20-19" "14-7,20-15,20-19,6-15"
)
EXPECTED_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"

check_inputs() {
  local fold arm candidate
  for fold in 1 2 3 4 5 6; do
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

common_command() {
  local fold="$1" arm="$2" output="$3" candidate="F${fold}${arm}_CLIC12"
  COMMON_CMD=("${PYTHON}" -u "${PAIR_ENTRY}"
    --export-common-training-receipt
    --checkpoint "${TRAINING_ROOT}/${candidate}/final_ssdg.pth"
    --terminal-receipt-json "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json"
    --output-common-receipt-json "${output}"
    --expected-arm "${arm}" --fold-index "${fold}" --training-run-root "${TRAINING_RUN_ID}")
}

proxy_command() {
  local fold="$1" arm="$2" output="$3" candidate="F${fold}${arm}_CLIC12"
  PROXY_CMD=("${PYTHON}" -u "${PAIR_ENTRY}"
    --export-proxy-diagnostic
    --clean-npz "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz"
    --output-proxy-diagnostic-json "${output}")
}

pair_command() {
  local fold="$1" output="$2" index=$((fold - 1))
  local c="F${fold}C_CLIC12" g="F${fold}G_CLIC12"
  PAIR_CMD=("${PYTHON}" -u "${PAIR_ENTRY}"
    --c-checkpoint "${TRAINING_ROOT}/${c}/final_ssdg.pth"
    --g-checkpoint "${TRAINING_ROOT}/${g}/final_ssdg.pth"
    --c-terminal-receipt-json "${TRAINING_ROOT}/${c}/phase1_clic_terminal_receipt.json"
    --g-terminal-receipt-json "${TRAINING_ROOT}/${g}/phase1_clic_terminal_receipt.json"
    --c-clean-npz "${CLEAN_ROOT}/${c}/source_clean_proxy.npz"
    --g-clean-npz "${CLEAN_ROOT}/${g}/source_clean_proxy.npz"
    --c-leo-npz "${LEO_ROOT}/${c}/source_leo.npz"
    --g-leo-npz "${LEO_ROOT}/${g}/source_leo.npz"
    --c-leo-binding-json "${LEO_ROOT}/${c}/source_leo.binding.json"
    --g-leo-binding-json "${LEO_ROOT}/${g}/source_leo.binding.json"
    --c-common-receipt-json "${RUN_ROOT}/${c}/common_training_receipt.json"
    --g-common-receipt-json "${RUN_ROOT}/${g}/common_training_receipt.json"
    --c-proxy-diagnostic-json "${RUN_ROOT}/${c}/proxy_diagnostic.json"
    --g-proxy-diagnostic-json "${RUN_ROOT}/${g}/proxy_diagnostic.json"
    --fold-index "${fold}" --training-run-root "${TRAINING_RUN_ID}"
    --source-tx-ids "${FOLD_TRAIN_TX[index]}"
    --postfreeze-matrix-id "${POSTFREEZE_MATRIX_ID}"
    --expected-scenarios "${EXPECTED_SCENARIOS}"
    --output-pair-json "${output}")
}

emit_command() {
  local stage="$1" fold="$2" arm="$3"; shift 3
  printf '[DRY-RUN] stage=%q fold=%q arm=%q' "${stage}" "${fold}" "${arm}"
  printf ' %q' "$@"
  printf '\n'
}

run_fold() {
  local fold="$1" arm candidate output_dir common_out proxy_out pair_out
  for arm in C G; do
    candidate="F${fold}${arm}_CLIC12"
    output_dir="${RUN_ROOT}/${candidate}"
    common_out="${output_dir}/common_training_receipt.json"
    proxy_out="${output_dir}/proxy_diagnostic.json"
    mkdir -p "${output_dir}"
    common_command "${fold}" "${arm}" "${common_out}"
    PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${COMMON_CMD[@]}"
    proxy_command "${fold}" "${arm}" "${proxy_out}"
    PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${PROXY_CMD[@]}"
  done
  pair_out="${RUN_ROOT}/F${fold}_C_vs_G_pair.json"
  pair_command "${fold}" "${pair_out}"
  PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${PAIR_CMD[@]}"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for fold in 1 2 3 4 5 6; do
    for arm in C G; do
      candidate="F${fold}${arm}_CLIC12"
      common_command "${fold}" "${arm}" "${RUN_ROOT}/${candidate}/common_training_receipt.json"
      emit_command CLIC_COMMON "${fold}" "${arm}" "${COMMON_CMD[@]}"
      proxy_command "${fold}" "${arm}" "${RUN_ROOT}/${candidate}/proxy_diagnostic.json"
      emit_command CLIC_PROXY "${fold}" "${arm}" "${PROXY_CMD[@]}"
    done
    pair_command "${fold}" "${RUN_ROOT}/F${fold}_C_vs_G_pair.json"
    emit_command CLIC_PAIR "${fold}" CG "${PAIR_CMD[@]}"
  done
  exit 0
fi

[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite source PAIR run/log root" >&2; exit 3; }
check_inputs
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

declare -a pids folds logs
for fold in 1 2 3 4 5 6; do
  log_path="${LOG_ROOT}/F${fold}_source_pair.out"
  run_fold "${fold}" >"${log_path}" 2>&1 &
  pids+=("$!"); folds+=("${fold}"); logs+=("${log_path}")
done

printf 'pid|fold|stage|log_path\n' >"${LOG_ROOT}/pids_source_pair6.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|CLIC_COMMON_PROXY_PAIR|%s\n' \
    "${pids[index]}" "${folds[index]}" "${logs[index]}" >>"${LOG_ROOT}/pids_source_pair6.tsv"
done

status=0
for index in "${!pids[@]}"; do
  wait "${pids[index]}" || status=1
done
exit "${status}"
