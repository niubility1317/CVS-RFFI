#!/usr/bin/env bash
set -euo pipefail

# Second source-only postfreeze wave.  Build one immutable received-IQ cache
# per fold, then reuse those exact bytes for the corresponding C and G forward.
RUN_ID="${RUN_ID:-phase1_clic_source_leo_20260812_v1}"
TRAINING_RUN_ID="${TRAINING_RUN_ID:-phase1_clic12_20260812_v5}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
TRAINING_ROOT="${TRAINING_ROOT:-${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
CACHE_BUILDER="${CODE_ROOT}/build_phase1_clic_source_leo_iq.py"
LEO_EXPORTER="${CODE_ROOT}/export_phase1_clic_leo_features.py"
WISIG_SHA256="2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "${RUN_ID}" == "phase1_clic_source_leo_20260812_v1" ]] || { echo "source LEO run ID drift" >&2; exit 2; }
[[ "${TRAINING_RUN_ID}" == "phase1_clic12_20260812_v5" ]] || { echo "training run ID drift" >&2; exit 2; }
[[ -f "${CACHE_BUILDER}" && -f "${LEO_EXPORTER}" ]] || { echo "missing source LEO builder/exporter" >&2; exit 2; }

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20" "14-10,20-19,6-15,8-20" "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20" "14-10,14-7,20-15,20-19" "14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")
GPU_MAP=(0 1 2 3 4 5)

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "missing frozen WiSig dataset" >&2; exit 2; }
  [[ "$(sha256sum "${WISIG_PKL}" | awk '{print $1}')" == "${WISIG_SHA256}" ]] || { echo "WiSig SHA256 drift" >&2; exit 2; }
  [[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite source LEO run/log root" >&2; exit 3; }
  for fold in 1 2 3 4 5 6; do
    for arm in C G; do
      candidate="F${fold}${arm}_CLIC12"
      [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" ]] || { echo "missing checkpoint: ${candidate}" >&2; exit 2; }
      [[ -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || { echo "missing terminal: ${candidate}" >&2; exit 2; }
    done
  done
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

# Cache stage is deliberately serial: one cache per fold is created once.
for fold in 1 2 3 4 5 6; do
  index=$((fold - 1))
  fold_root="${RUN_ROOT}/F${fold}_SHARED"
  cache_npz="${fold_root}/source_l_received_iq.npz"
  cache_receipt="${fold_root}/source_l_received_iq.receipt.json"
  cache_log="${LOG_ROOT}/F${fold}_shared_cache.out"
  c_root="${TRAINING_ROOT}/F${fold}C_CLIC12"
  g_root="${TRAINING_ROOT}/F${fold}G_CLIC12"
  command=("${PYTHON}" -u "${CACHE_BUILDER}"
    --fold-index "${fold}"
    --c-ckpt "${c_root}/final_ssdg.pth"
    --c-terminal-receipt-json "${c_root}/phase1_clic_terminal_receipt.json"
    --g-ckpt "${g_root}/final_ssdg.pth"
    --g-terminal-receipt-json "${g_root}/phase1_clic_terminal_receipt.json"
    --wisig-pkl "${WISIG_PKL}" --expected-wisig-sha256 "${WISIG_SHA256}"
    --source-tx-ids "${FOLD_TRAIN_TX[index]}"
    --known-validation-tx-ids "${FOLD_KNOWN_VAL_TX[index]}"
    --proxy-unknown-tx-ids "${FOLD_PROXY_TX[index]}"
    --out-npz "${cache_npz}" --receipt-json "${cache_receipt}"
    --batch-size 256 --device cuda:0)
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] stage=CLIC_SOURCE_LEO_CACHE fold=%q physical_gpu=%q' "${fold}" "${GPU_MAP[index]}"
    printf ' %q' "${command[@]}"; printf '\n'
  else
    mkdir -p "${fold_root}"
    CUDA_VISIBLE_DEVICES="${GPU_MAP[index]}" PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${cache_log}" 2>&1
  fi
done

declare -a pids folds arms gpus outputs logs
for fold in 1 2 3 4 5 6; do
  index=$((fold - 1))
  cache_npz="${RUN_ROOT}/F${fold}_SHARED/source_l_received_iq.npz"
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
      --out-npz "${output_npz}" --binding-json "${binding_json}"
      --training-run-root "${TRAINING_ROOT}" --postfreeze-output-root "${RUN_ROOT}"
      --candidate-id "${candidate}" --fold-index "${fold}" --arm "${arm}"
      --source-tx-ids "${FOLD_TRAIN_TX[index]}" --batch-size 64 --device cuda:0)
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[DRY-RUN] stage=CLIC_SOURCE_LEO_EXPORT candidate=%q physical_gpu=%q' "${candidate}" "${GPU_MAP[index]}"
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
printf 'pid|fold|arm|physical_gpu|output_npz|log_path|stage\n' >"${LOG_ROOT}/pids_source_leo12.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s|CLIC_SOURCE_LEO_EXPORT\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${outputs[index]}" "${logs[index]}" >>"${LOG_ROOT}/pids_source_leo12.tsv"
done
status=0
for index in "${!pids[@]}"; do wait "${pids[index]}" || status=1; done
exit "${status}"
