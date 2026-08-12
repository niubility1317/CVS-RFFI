#!/usr/bin/env bash
set -euo pipefail

RUN_ID="phase1_clic_g_bundles_20260812_v3_safe_pack"
TRAINING_RUN_ID="phase1_clic12_20260812_v5"
CLEAN_RUN_ID="phase1_clic_postfreeze_20260812_v4"
LEO_RUN_ID="phase1_clic_source_leo_20260812_v4"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
TRAINING_ROOT="${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}"
CLEAN_ROOT="${PROJECT_ROOT}/runs/${CLEAN_RUN_ID}"
LEO_ROOT="${PROJECT_ROOT}/runs/${LEO_RUN_ID}"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
G_ENTRY="${CODE_ROOT}/export_phase1_clic_deployment_bundle.py"
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${G_ENTRY}" ]] || { echo "missing G bundle entry" >&2; exit 2; }

g_command() {
  local fold="$1" candidate="F${fold}G_CLIC12"
  G_CMD=("${PYTHON}" -u "${G_ENTRY}"
    --checkpoint "${TRAINING_ROOT}/${candidate}/final_ssdg.pth"
    --terminal-receipt-json "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json"
    --clean-npz "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz"
    --leo-npz "${LEO_ROOT}/${candidate}/source_leo.npz"
    --leo-binding-json "${LEO_ROOT}/${candidate}/source_leo.binding.json"
    --output-bundle "${RUN_ROOT}/${candidate}/g_deployment_bundle.zip")
}

emit_command() {
  local fold="$1"; shift
  printf '[DRY-RUN] stage=CLIC_G_BUNDLE_SAFE_PACK_SERIAL fold=%s arm=G' "${fold}"
  printf ' %q' "$@"
  printf '\n'
}

check_inputs() {
  local fold candidate
  for fold in 1 2 3 4 5 6; do
    candidate="F${fold}G_CLIC12"
    [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" ]] || { echo "missing checkpoint: ${candidate}" >&2; return 2; }
    [[ -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || { echo "missing terminal: ${candidate}" >&2; return 2; }
    [[ -f "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz" ]] || { echo "missing clean NPZ: ${candidate}" >&2; return 2; }
    [[ -f "${LEO_ROOT}/${candidate}/source_leo.npz" ]] || { echo "missing source LEO NPZ: ${candidate}" >&2; return 2; }
    [[ -f "${LEO_ROOT}/${candidate}/source_leo.binding.json" ]] || { echo "missing source LEO binding: ${candidate}" >&2; return 2; }
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for fold in 1 2 3 4 5 6; do
    g_command "${fold}"
    emit_command "${fold}" "${G_CMD[@]}"
  done
  exit 0
fi

[[ ! -e "${RUN_ROOT}" && ! -e "${LOG_ROOT}" ]] || { echo "refusing to overwrite safe-pack G bundle run/log root" >&2; exit 3; }
check_inputs
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
printf 'pid|fold|stage|log_path|exit_code\n' >"${LOG_ROOT}/pids_g_bundles6_safe_pack.tsv"

for fold in 1 2 3 4 5 6; do
  mkdir -p "${RUN_ROOT}/F${fold}G_CLIC12"
  log_path="${LOG_ROOT}/F${fold}G_CLIC12.out"
  g_command "${fold}"
  PYTHONPATH="${CODE_ROOT}" CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
    "${G_CMD[@]}" >"${log_path}" 2>&1 &
  worker_pid="$!"
  set +e
  wait "${worker_pid}"
  exit_code="$?"
  set -e
  printf '%s|%s|CLIC_G_BUNDLE_SAFE_PACK_SERIAL|%s|%s\n' \
    "${worker_pid}" "${fold}" "${log_path}" "${exit_code}" \
    >>"${LOG_ROOT}/pids_g_bundles6_safe_pack.tsv"
  [[ "${exit_code}" == "0" ]] || exit "${exit_code}"
done
