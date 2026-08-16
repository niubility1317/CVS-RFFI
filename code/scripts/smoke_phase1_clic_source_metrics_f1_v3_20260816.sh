#!/usr/bin/env bash
set -euo pipefail

# Structural-only v3 smoke: use the original formal inputs, write only beneath
# the pre-registered independent smoke root, and never score or inspect a
# performance result. The cache is built once and consumed sequentially by
# F1C then F1G. The v2 canonical-root and exclusive-root controls remain
# unchanged.
RUN_ID="phase1_clic_source_metrics_20260816_v3"
SMOKE_ROOT_NAME=".smoke_phase1_clic_source_metrics_20260816_v3_F1"
TRAINING_RUN_ID="phase1_clic12_20260812_v5"
CLEAN_RUN_ID="phase1_clic_postfreeze_20260812_v4"
SOURCE_PAIR_RUN_ID="phase1_clic_source_pair_20260812_v3"
CANONICAL_PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PROJECT_ROOT="${PROJECT_ROOT:-${CANONICAL_PROJECT_ROOT}}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
WISIG_PKL="${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl"
WISIG_SHA256="2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
TRAINING_ROOT="${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}"
CLEAN_ROOT="${PROJECT_ROOT}/runs/${CLEAN_RUN_ID}"
SOURCE_PAIR_ROOT="${PROJECT_ROOT}/runs/${SOURCE_PAIR_RUN_ID}"
SMOKE_ROOT="${PROJECT_ROOT}/runs/${SMOKE_ROOT_NAME}"
SOURCE_ROOT="${SMOKE_ROOT}/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${SMOKE_ROOT_NAME}"
CACHE_BUILDER="${CODE_ROOT}/build_phase1_clic_source_v_leo_iq.py"
FORWARD_ENTRY="${CODE_ROOT}/export_phase1_clic_source_v_leo_features.py"
DRY_RUN=0

FOLD=1
GPU=0
SOURCE_TX_IDS="20-15,20-19,6-15,8-20"
KNOWN_VALIDATION_TX_IDS="14-7"
PROXY_UNKNOWN_TX_IDS="14-10"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ "${PROJECT_ROOT}" == "${CANONICAL_PROJECT_ROOT}" ]] || {
  echo "technical smoke requires the frozen canonical project root" >&2
  exit 3
}

candidate_for() {
  local arm="$1"
  printf 'F1%s_CLIC12' "${arm}"
}

shared_cache_path() {
  printf '%s/F1_SHARED/source_validation_known_leo_weak.npz' "${SOURCE_ROOT}"
}

shared_receipt_path() {
  printf '%s/F1_SHARED/source_validation_known_leo_weak.receipt.json' "${SOURCE_ROOT}"
}

pair_path() {
  printf '%s/F1_C_vs_G_pair.json' "${SOURCE_PAIR_ROOT}"
}

cache_command() {
  local c g shared receipt
  c="$(candidate_for C)"
  g="$(candidate_for G)"
  shared="$(shared_cache_path)"
  receipt="$(shared_receipt_path)"
  CACHE_CMD=(
    "${PYTHON}" -u "${CACHE_BUILDER}"
    --fold-index "${FOLD}"
    --c-ckpt "${TRAINING_ROOT}/${c}/final_ssdg.pth"
    --c-terminal-receipt-json "${TRAINING_ROOT}/${c}/phase1_clic_terminal_receipt.json"
    --c-clean-npz "${CLEAN_ROOT}/${c}/source_clean_proxy.npz"
    --g-ckpt "${TRAINING_ROOT}/${g}/final_ssdg.pth"
    --g-terminal-receipt-json "${TRAINING_ROOT}/${g}/phase1_clic_terminal_receipt.json"
    --g-clean-npz "${CLEAN_ROOT}/${g}/source_clean_proxy.npz"
    --wisig-pkl "${WISIG_PKL}"
    --expected-wisig-sha256 "${WISIG_SHA256}"
    --source-tx-ids "${SOURCE_TX_IDS}"
    --known-validation-tx-ids "${KNOWN_VALIDATION_TX_IDS}"
    --proxy-unknown-tx-ids "${PROXY_UNKNOWN_TX_IDS}"
    --cache-run-root "${SOURCE_ROOT}"
    --out-npz "${shared}"
    --receipt-json "${receipt}"
    --batch-size 256 --device cuda:0
  )
}

forward_command() {
  local arm="$1" candidate shared receipt
  candidate="$(candidate_for "${arm}")"
  shared="$(shared_cache_path)"
  receipt="$(shared_receipt_path)"
  FORWARD_CMD=(
    "${PYTHON}" -u "${FORWARD_ENTRY}"
    --ckpt "${TRAINING_ROOT}/${candidate}/final_ssdg.pth"
    --terminal-receipt-json "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json"
    --clean-npz "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz"
    --source-v-received-iq-npz "${shared}"
    --source-v-received-iq-receipt-json "${receipt}"
    --pair-json "$(pair_path)"
    --formal-project-root "${PROJECT_ROOT}"
    --training-run-root "${TRAINING_ROOT}"
    --cache-run-root "${SOURCE_ROOT}"
    --output-root "${SOURCE_ROOT}"
    --out-npz "${SOURCE_ROOT}/${candidate}/source_validation_known_leo_weak_features.npz"
    --binding-json "${SOURCE_ROOT}/${candidate}/source_validation_known_leo_weak.binding.json"
    --candidate-id "${candidate}" --fold-index "${FOLD}" --arm "${arm}"
    --source-tx-ids "${SOURCE_TX_IDS}"
    --batch-size 256 --device cuda:0
    --technical-smoke
  )
}

emit_command() {
  local stage="$1" arm="$2"; shift 2
  printf '[DRY-RUN] stage=%s fold=1 arm=%s physical_gpu=%s source_only=1 retry=NO SMOKE_INVOCATION=1 FORMAL_INVOCATION=0' \
    "${stage}" "${arm}" "${GPU}"
  printf ' %q' "$@"
  printf '\n'
}

claim_exact_root() {
  local path="$1"
  if ! mkdir -- "${path}"; then
    echo "refusing to overwrite source metrics v3 smoke run/log root" >&2
    exit 3
  fi
}

open_exclusive_fd() {
  local path="$1"
  if ! { set -o noclobber; exec {OPEN_FD}>"${path}"; }; then
    set +o noclobber
    echo "refusing to overwrite source metrics v3 smoke log evidence" >&2
    exit 3
  fi
  set +o noclobber
}

launch_with_exclusive_log() {
  local log_path="$1" log_fd
  shift
  open_exclusive_fd "${log_path}"
  log_fd="${OPEN_FD}"
  (
    "$@" >&"${log_fd}" 2>&1
  ) &
  LAUNCHED_PID="$!"
  exec {log_fd}>&-
}

if [[ "${DRY_RUN}" == "1" ]]; then
  cache_command
  emit_command CLIC_SOURCE_V_CACHE CG "candidate_c=$(candidate_for C)" "candidate_g=$(candidate_for G)" \
    "shared_cache=$(shared_cache_path)" "shared_receipt=$(shared_receipt_path)" "${CACHE_CMD[@]}"
  for arm in C G; do
    forward_command "${arm}"
    emit_command CLIC_SOURCE_V_FORWARD "${arm}" "candidate=$(candidate_for "${arm}")" \
      "shared_cache=$(shared_cache_path)" "shared_receipt=$(shared_receipt_path)" "${FORWARD_CMD[@]}"
  done
  exit 0
fi

if [[ -e "${SMOKE_ROOT}" || -e "${LOG_ROOT}" ]]; then
  echo "refusing to overwrite source metrics v3 smoke run/log root" >&2
  exit 3
fi

[[ -f "${CACHE_BUILDER}" && -f "${FORWARD_ENTRY}" ]] || {
  echo "missing source metrics smoke entry" >&2
  exit 2
}
[[ -f "${WISIG_PKL}" ]] || { echo "missing frozen WiSig input" >&2; exit 2; }
[[ "$(sha256sum "${WISIG_PKL}" | awk '{print $1}')" == "${WISIG_SHA256}" ]] || {
  echo "frozen WiSig SHA256 drift" >&2
  exit 2
}

for arm in C G; do
  candidate="$(candidate_for "${arm}")"
  [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" ]] || {
    echo "missing checkpoint: ${candidate}" >&2
    exit 2
  }
  [[ -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || {
    echo "missing terminal: ${candidate}" >&2
    exit 2
  }
  [[ -f "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz" ]] || {
    echo "missing clean-v4 NPZ: ${candidate}" >&2
    exit 2
  }
done
[[ -f "$(pair_path)" ]] || { echo "missing source PAIR-v3 receipt: F1" >&2; exit 2; }

claim_exact_root "${SMOKE_ROOT}"
claim_exact_root "${LOG_ROOT}"
claim_exact_root "${SOURCE_ROOT}"
mkdir -p "${SOURCE_ROOT}/F1_SHARED" "${SOURCE_ROOT}/F1C_CLIC12" "${SOURCE_ROOT}/F1G_CLIC12"

cache_command
launch_with_exclusive_log "${LOG_ROOT}/F1_source_v_cache.out" \
  env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${CODE_ROOT}" "${CACHE_CMD[@]}"
wait "${LAUNCHED_PID}"

for arm in C G; do
  candidate="$(candidate_for "${arm}")"
  forward_command "${arm}"
  launch_with_exclusive_log "${LOG_ROOT}/${candidate}_source_v_forward.out" \
    env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${CODE_ROOT}" "${FORWARD_CMD[@]}"
  wait "${LAUNCHED_PID}"
done
