#!/usr/bin/env bash
set -euo pipefail

# First executable wave of the immutable P1-CLIC postfreeze matrix.
# It exports source-L clean/source-V/fixed400 proxy features only.  No target,
# query, truth, role, source-LEO generation, scoring, or performance selection
# is reachable from this launcher.
RUN_ID="${RUN_ID:-phase1_clic_postfreeze_20260812_v1}"
TRAINING_RUN_ID="${TRAINING_RUN_ID:-phase1_clic12_20260812_v5}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
TRAINING_ROOT="${TRAINING_ROOT:-${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
EXPORTER="${CODE_ROOT}/export_phase1_clic_features.py"
WISIG_SHA256="2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done
[[ "${DRY_RUN}" == "0" || "${DRY_RUN}" == "1" ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
[[ "${RUN_ID}" == "phase1_clic_postfreeze_20260812_v1" ]] || { echo "postfreeze run ID drift" >&2; exit 2; }
[[ "${TRAINING_RUN_ID}" == "phase1_clic12_20260812_v5" ]] || { echo "training run ID drift" >&2; exit 2; }
[[ -f "${EXPORTER}" ]] || { echo "missing CLIC clean exporter: ${EXPORTER}" >&2; exit 2; }

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20" "14-10,20-19,6-15,8-20" "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20" "14-10,14-7,20-15,20-19" "14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "missing frozen WiSig dataset: ${WISIG_PKL}" >&2; exit 2; }
  actual_wisig_sha="$(sha256sum "${WISIG_PKL}" | awk '{print $1}')"
  [[ "${actual_wisig_sha}" == "${WISIG_SHA256}" ]] || { echo "WiSig SHA256 drift" >&2; exit 2; }
  [[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite postfreeze run/log root" >&2; exit 3; }
  for fold in 1 2 3 4 5 6; do
    for arm in C G; do
      candidate="F${fold}${arm}_CLIC12"
      [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" ]] || { echo "missing checkpoint: ${candidate}" >&2; exit 2; }
      [[ -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || { echo "missing terminal: ${candidate}" >&2; exit 2; }
    done
  done
  mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
fi

declare -a pids folds arms gpus outputs logs
launch_export() {
  local fold="$1"
  local arm="$2"
  local gpu="$3"
  local index=$((fold - 1))
  local candidate="F${fold}${arm}_CLIC12"
  local candidate_root="${TRAINING_ROOT}/${candidate}"
  local output_dir="${RUN_ROOT}/${candidate}"
  local output_npz="${output_dir}/source_clean_proxy.npz"
  local log_path="${LOG_ROOT}/${candidate}_clean_proxy.out"
  local -a command=("${PYTHON}" -u "${EXPORTER}"
    --ckpt "${candidate_root}/final_ssdg.pth"
    --terminal-receipt-json "${candidate_root}/phase1_clic_terminal_receipt.json"
    --wisig-pkl "${WISIG_PKL}"
    --out-npz "${output_npz}"
    --source-tx-ids "${FOLD_TRAIN_TX[index]}"
    --known-validation-tx-ids "${FOLD_KNOWN_VAL_TX[index]}"
    --proxy-unknown-tx-ids "${FOLD_PROXY_TX[index]}"
    --expected-wisig-sha256 "${WISIG_SHA256}"
    --batch-size 256 --device cuda:0)
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY-RUN] stage=CLIC_CLEAN_EXPORT candidate=%q physical_gpu=%q' "${candidate}" "${gpu}"
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${command[@]}" >"${log_path}" 2>&1 &
  pids+=("$!"); folds+=("${fold}"); arms+=("${arm}"); gpus+=("${gpu}")
  outputs+=("${output_npz}"); logs+=("${log_path}")
}

launch_export 1 C 0
launch_export 5 G 0
launch_export 1 G 1
launch_export 5 C 1
launch_export 2 C 2
launch_export 6 G 2
launch_export 2 G 3
launch_export 6 C 3
launch_export 3 C 4
launch_export 3 G 5
launch_export 4 C 6
launch_export 4 G 7

[[ "${DRY_RUN}" == "1" ]] && exit 0
printf 'pid|fold|arm|physical_gpu|output_npz|log_path|stage\n' >"${LOG_ROOT}/pids_source12.tsv"
for index in "${!pids[@]}"; do
  printf '%s|%s|%s|%s|%s|%s|CLIC_CLEAN_EXPORT\n' \
    "${pids[index]}" "${folds[index]}" "${arms[index]}" "${gpus[index]}" \
    "${outputs[index]}" "${logs[index]}" >>"${LOG_ROOT}/pids_source12.tsv"
done
status=0
for index in "${!pids[@]}"; do wait "${pids[index]}" || status=1; done
exit "${status}"
