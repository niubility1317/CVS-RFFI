#!/usr/bin/env bash
set -euo pipefail

# Versioned, file-backed F1 one-shot technical smoke for ADV blind prediction.
# It is intended to be launched directly as `nohup bash <this-file>`; neither
# the wrapper nor its Python entry is transported through SSH stdin.
RUN_ID="phase1_adv3b02_target_prediction_20260816_v2"
SMOKE_ID=".smoke_phase1_adv3b02_target_prediction_20260816_v2_F1"
ADV_TRAINING_RUN_ID="phase1_adv3b02_clic6_20260816_v2"
CLEAN_RUN_ID="phase1_clic_postfreeze_20260812_v4"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SMOKE_ENTRY="${CODE_ROOT}/smoke_phase1_adv3b02_target_prediction_f1.py"
ADV_F1_ROOT="${PROJECT_ROOT}/runs/${ADV_TRAINING_RUN_ID}/F1_ADV3B02_CLIC"
CHECKPOINT="${ADV_F1_ROOT}/final_ssdg.pth"
COMPLETION="${ADV_F1_ROOT}/phase1_training_completion_receipt.json"
CLEAN_V4="${PROJECT_ROOT}/runs/${CLEAN_RUN_ID}/F1C_CLIC12/source_clean_proxy.npz"
SMOKE_RUN_ROOT="${PROJECT_ROOT}/runs/${SMOKE_ID}"
SMOKE_LOG_ROOT="${PROJECT_ROOT}/logs/${SMOKE_ID}"
FOLD_OUTPUT_ROOT="${SMOKE_RUN_ROOT}/F1_ADV3B02_CLIC"
TRAIN_CONFIG="${FOLD_OUTPUT_ROOT}/train_data_config.json"
RECEIPT="${FOLD_OUTPUT_ROOT}/technical_smoke_receipt.json"
LOG_PATH="${SMOKE_LOG_ROOT}/F1_ADV3B02_CLIC.out"
PID_PATH="${SMOKE_LOG_ROOT}/F1_ADV3B02_CLIC.pid"
DRY_RUN=0

for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${argument}" >&2; exit 2 ;;
  esac
done

smoke_command() {
  SMOKE_CMD=(
    "${PYTHON}" -u "${SMOKE_ENTRY}"
    --run-smoke
    --checkpoint "${CHECKPOINT}"
    --completion-receipt-json "${COMPLETION}"
    --clean-v4-npz "${CLEAN_V4}"
    --train-config-output "${TRAIN_CONFIG}"
    --receipt-output "${RECEIPT}"
  )
}

validate_command() {
  VALIDATE_CMD=(
    "${PYTHON}" -u "${SMOKE_ENTRY}"
    --validate-receipt
    --checkpoint "${CHECKPOINT}"
    --completion-receipt-json "${COMPLETION}"
    --train-config-output "${TRAIN_CONFIG}"
    --receipt-output "${RECEIPT}"
  )
}

if [[ "${DRY_RUN}" == "1" ]]; then
  smoke_command
  printf '[DRY-RUN] stage=ADV_TARGET_PREDICTION_F1_TECHNICAL_SMOKE run_id=%s fold=1 candidate=F1_ADV3B02_CLIC physical_gpu=0 source_only=1 synthetic_local_zero_iq=1 SMOKE_INVOCATION=1 FORMAL_INVOCATION=0 retry=NO' \
    "${RUN_ID}"
  printf ' %q' "${SMOKE_CMD[@]}"
  printf '\n'
  exit 0
fi

refuse_existing_roots() {
  if [[ -e "${SMOKE_RUN_ROOT}" || -e "${SMOKE_LOG_ROOT}" ]]; then
    echo "refusing to overwrite ADV target prediction v2 smoke run/log root" >&2
    exit 3
  fi
}

claim_exact_root() {
  local path="$1"
  if ! mkdir -- "${path}"; then
    echo "refusing to overwrite ADV target prediction v2 smoke run/log root" >&2
    exit 3
  fi
}

open_exclusive_fd() {
  local path="$1"
  if ! { set -o noclobber; exec {OPEN_FD}>"${path}"; }; then
    set +o noclobber
    echo "refusing to overwrite ADV target prediction v2 smoke log/PID evidence" >&2
    exit 3
  fi
  set +o noclobber
}

# Collision refusal intentionally precedes every input open.
refuse_existing_roots
[[ -x "${PYTHON}" ]] || { echo "missing frozen Python runtime: ${PYTHON}" >&2; exit 2; }
[[ -f "${SMOKE_ENTRY}" ]] || { echo "missing versioned F1 smoke entry" >&2; exit 2; }
for input in "${CHECKPOINT}" "${COMPLETION}" "${CLEAN_V4}"; do
  [[ -f "${input}" ]] || { echo "missing F1 smoke input: ${input}" >&2; exit 2; }
done
refuse_existing_roots

claim_exact_root "${SMOKE_RUN_ROOT}"
claim_exact_root "${SMOKE_LOG_ROOT}"
claim_exact_root "${FOLD_OUTPUT_ROOT}"

open_exclusive_fd "${PID_PATH}"
PID_FD="${OPEN_FD}"
printf '%s\n' "$$" >&"${PID_FD}"
exec {PID_FD}>&-

open_exclusive_fd "${LOG_PATH}"
LOG_FD="${OPEN_FD}"
smoke_command
validate_command
status=0
if {
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${CODE_ROOT}" \
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "${SMOKE_CMD[@]}" &&
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${CODE_ROOT}" \
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "${VALIDATE_CMD[@]}"
} >&"${LOG_FD}" 2>&1; then
  status=0
else
  status=$?
fi
exec {LOG_FD}>&-
[[ "${status}" == "0" ]] || exit "${status}"
[[ -f "${TRAIN_CONFIG}" && -f "${RECEIPT}" ]] || {
  echo "ADV target prediction v2 smoke did not close" >&2
  exit 2
}
