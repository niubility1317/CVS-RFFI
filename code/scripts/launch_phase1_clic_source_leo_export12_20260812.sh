#!/usr/bin/env bash
set -euo pipefail

# Final bounded repair: reuse the six immutable v3 received-IQ caches and run
# only the twelve C/G source-LEO forwards into a fresh non-overwriting root.
RUN_ID="${RUN_ID:-phase1_clic_source_leo_20260812_v4}"
CACHE_RUN_ID="${CACHE_RUN_ID:-phase1_clic_source_leo_20260812_v3}"
TRAINING_RUN_ID="${TRAINING_RUN_ID:-phase1_clic12_20260812_v5}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
EXPECTED_PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAINING_ROOT="${TRAINING_ROOT:-${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}}"
CACHE_ROOT="${CACHE_ROOT:-${PROJECT_ROOT}/runs/${CACHE_RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
LEO_EXPORTER="${CODE_ROOT}/export_phase1_clic_leo_features.py"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "${RUN_ID}" == "phase1_clic_source_leo_20260812_v4" ]] || { echo "source LEO export run ID drift" >&2; exit 2; }
[[ "${CACHE_RUN_ID}" == "phase1_clic_source_leo_20260812_v3" ]] || { echo "source LEO cache run ID drift" >&2; exit 2; }
[[ "${TRAINING_RUN_ID}" == "phase1_clic12_20260812_v5" ]] || { echo "training run ID drift" >&2; exit 2; }
[[ "${PROJECT_ROOT}" == "${EXPECTED_PROJECT_ROOT}" ]] || { echo "project root drift" >&2; exit 2; }
[[ -f "${LEO_EXPORTER}" ]] || { echo "missing source LEO exporter" >&2; exit 2; }
[[ "${TRAINING_ROOT}" == "${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}" ]] || { echo "training root drift" >&2; exit 2; }
[[ "${CACHE_ROOT}" == "${PROJECT_ROOT}/runs/${CACHE_RUN_ID}" ]] || { echo "cache root drift" >&2; exit 2; }
[[ "${RUN_ROOT}" == "${PROJECT_ROOT}/runs/${RUN_ID}" ]] || { echo "run root drift" >&2; exit 2; }
[[ "${LOG_ROOT}" == "${PROJECT_ROOT}/logs/${RUN_ID}" ]] || { echo "log root drift" >&2; exit 2; }

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20" "14-10,20-19,6-15,8-20" "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20" "14-10,14-7,20-15,20-19" "14-7,20-15,20-19,6-15"
)
GPU_MAP=(0 1 2 3 4 5)

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite source LEO export run/log root" >&2; exit 3; }
  for fold in 1 2 3 4 5 6; do
    cache_npz="${CACHE_ROOT}/F${fold}_SHARED/source_l_received_iq.npz"
    cache_receipt="${CACHE_ROOT}/F${fold}_SHARED/source_l_received_iq.receipt.json"
    [[ -f "${cache_npz}" && -f "${cache_receipt}" ]] || { echo "missing frozen v3 cache/receipt: F${fold}" >&2; exit 2; }
    for arm in C G; do
      candidate="F${fold}${arm}_CLIC12"
      [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" ]] || { echo "missing checkpoint: ${candidate}" >&2; exit 2; }
      [[ -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || { echo "missing terminal: ${candidate}" >&2; exit 2; }
    done
  done
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

declare -a pids folds arms gpus outputs logs
for fold in 1 2 3 4 5 6; do
  index=$((fold - 1))
  cache_npz="${CACHE_ROOT}/F${fold}_SHARED/source_l_received_iq.npz"
  cache_receipt="${CACHE_ROOT}/F${fold}_SHARED/source_l_received_iq.receipt.json"
  for arm in C G; do
    candidate="F${fold}${arm}_CLIC12"
    candidate_root="${TRAINING_ROOT}/${candidate}"
    output_dir="${RUN_ROOT}/${candidate}"
    output_npz="${output_dir}/source_leo.npz"
    binding_json="${output_dir}/source_leo.binding.json"
    log_path="${LOG_ROOT}/${candidate}_source_leo.out"
    command=("${PYTHON}" -u "${LEO_EXPORTER}"
      --ckpt "${candidate_root}/final_ssdg.pth"
      --terminal-receipt-json "${candidate_root}/phase1_clic_terminal_receipt.json"
      --existing-received-iq-npz "${cache_npz}"
      --existing-received-iq-receipt-json "${cache_receipt}"
      --cache-run-root "${CACHE_ROOT}" --require-sealed-source-leo-cache
      --out-npz "${output_npz}" --binding-json "${binding_json}"
      --training-run-root "${TRAINING_ROOT}" --postfreeze-output-root "${RUN_ROOT}"
      --candidate-id "${candidate}" --fold-index "${fold}" --arm "${arm}"
      --source-tx-ids "${FOLD_TRAIN_TX[index]}" --batch-size 64 --device cuda:0)
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[DRY-RUN] stage=CLIC_SOURCE_LEO_EXPORT candidate=%q cache_run=%q physical_gpu=%q' \
        "${candidate}" "${CACHE_RUN_ID}" "${GPU_MAP[index]}"
      printf ' %q' "${command[@]}"; printf '\n'
    else
      mkdir -p "${output_dir}"
      CUDA_VISIBLE_DEVICES="${GPU_MAP[index]}" PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1 &
      pids+=("$!"); folds+=("${fold}"); arms+=("${arm}"); gpus+=("${GPU_MAP[index]}")
      outputs+=("${output_npz}"); logs+=("${log_path}")
    fi
  done
done

[[ "${DRY_RUN}" == "1" ]] && exit 0
printf 'pid|fold|arm|physical_gpu|output_npz|log_path|stage\n' >"${LOG_ROOT}/pids_source_leo_export12.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s|CLIC_SOURCE_LEO_EXPORT\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${outputs[index]}" "${logs[index]}" >>"${LOG_ROOT}/pids_source_leo_export12.tsv"
done
status=0
for index in "${!pids[@]}"; do wait "${pids[index]}" || status=1; done
exit "${status}"
