#!/usr/bin/env bash
set -euo pipefail

# Re-export source-only clean/V/fixed400 evidence from the immutable v5 final
# checkpoints into the new manifest schema.  No target/query/truth/scorer path
# is reachable here.
RUN_ID="phase1_clic_postfreeze_20260812_v3"
TRAINING_RUN_ID="phase1_clic12_20260812_v5"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
WISIG_PKL="${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl"
TRAINING_ROOT="${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
EXPORTER="${CODE_ROOT}/export_phase1_clic_features.py"
WISIG_SHA256="2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20" "14-10,20-19,6-15,8-20" "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20" "14-10,14-7,20-15,20-19" "14-7,20-15,20-19,6-15"
)
FOLD_KNOWN_VAL_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")
GPU_MAP=(0 1 2 3 4 5 6 7 0 1 2 3)

make_command() {
  local fold="$1" arm="$2" index=$((fold - 1)) candidate="F${fold}${arm}_CLIC12"
  COMMAND=("${PYTHON}" -u "${EXPORTER}"
    --ckpt "${TRAINING_ROOT}/${candidate}/final_ssdg.pth"
    --terminal-receipt-json "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json"
    --wisig-pkl "${WISIG_PKL}" --out-npz "${RUN_ROOT}/${candidate}/source_clean_proxy.npz"
    --source-tx-ids "${FOLD_TRAIN_TX[index]}"
    --known-validation-tx-ids "${FOLD_KNOWN_VAL_TX[index]}"
    --proxy-unknown-tx-ids "${FOLD_PROXY_TX[index]}"
    --expected-wisig-sha256 "${WISIG_SHA256}" --batch-size 256 --device cuda:0)
}

if [[ "${DRY_RUN}" == "1" ]]; then
  row=0
  for fold in 1 2 3 4 5 6; do for arm in C G; do
    make_command "${fold}" "${arm}"
    printf '[DRY-RUN] stage=CLIC_CLEAN_EXPORT candidate=F%s%s_CLIC12 physical_gpu=%s' "${fold}" "${arm}" "${GPU_MAP[row]}"
    printf ' %q' "${COMMAND[@]}"; printf '\n'; row=$((row + 1))
  done; done
  exit 0
fi

[[ -f "${EXPORTER}" && -f "${WISIG_PKL}" ]] || { echo "missing source exporter/dataset" >&2; exit 2; }
[[ "$(sha256sum "${WISIG_PKL}" | awk '{print $1}')" == "${WISIG_SHA256}" ]] || { echo "WiSig SHA drift" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite clean v3 run/log root" >&2; exit 3; }
for fold in 1 2 3 4 5 6; do for arm in C G; do
  candidate="F${fold}${arm}_CLIC12"
  [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" && -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || { echo "missing frozen candidate ${candidate}" >&2; exit 2; }
done; done
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
declare -a pids candidates gpus logs
row=0
for fold in 1 2 3 4 5 6; do for arm in C G; do
  candidate="F${fold}${arm}_CLIC12"; gpu="${GPU_MAP[row]}"; row=$((row + 1))
  mkdir -p "${RUN_ROOT}/${candidate}"
  log="${LOG_ROOT}/${candidate}_clean_proxy.out"; make_command "${fold}" "${arm}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${CODE_ROOT}" "${COMMAND[@]}" >"${log}" 2>&1 &
  pids+=("$!"); candidates+=("${candidate}"); gpus+=("${gpu}"); logs+=("${log}")
done; done
printf 'pid|candidate|physical_gpu|stage|log_path\n' >"${LOG_ROOT}/pids_source12.tsv"
for i in "${!pids[@]}"; do printf '%s|%s|%s|CLIC_CLEAN_EXPORT|%s\n' "${pids[i]}" "${candidates[i]}" "${gpus[i]}" "${logs[i]}" >>"${LOG_ROOT}/pids_source12.tsv"; done
status=0; for pid in "${pids[@]}"; do wait "${pid}" || status=1; done; exit "${status}"
