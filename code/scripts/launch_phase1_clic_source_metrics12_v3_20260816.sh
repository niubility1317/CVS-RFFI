#!/usr/bin/env bash
set -euo pipefail

# Immutable v3 formal source-only completion. This preserves the frozen
# six-cache/twelve-forward/six-score/one-aggregate matrix and the v2 root
# ownership controls; only the non-overwriting v3 run identity differs. No
# performance value controls dispatch, stopping, retry, selection, revival,
# or promotion.
RUN_ID="phase1_clic_source_metrics_20260816_v3"
TRAINING_RUN_ID="phase1_clic12_20260812_v5"
CLEAN_RUN_ID="phase1_clic_postfreeze_20260812_v4"
SOURCE_PAIR_RUN_ID="phase1_clic_source_pair_20260812_v3"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
WISIG_PKL="${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl"
WISIG_SHA256="2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f"
TRAINING_ROOT="${PROJECT_ROOT}/runs/${TRAINING_RUN_ID}"
CLEAN_ROOT="${PROJECT_ROOT}/runs/${CLEAN_RUN_ID}"
SOURCE_PAIR_ROOT="${PROJECT_ROOT}/runs/${SOURCE_PAIR_RUN_ID}"
RUN_ROOT="${PROJECT_ROOT}/runs/${RUN_ID}"
LOG_ROOT="${PROJECT_ROOT}/logs/${RUN_ID}"
CACHE_BUILDER="${CODE_ROOT}/build_phase1_clic_source_v_leo_iq.py"
FORWARD_ENTRY="${CODE_ROOT}/export_phase1_clic_source_v_leo_features.py"
METRICS_ENTRY="${CODE_ROOT}/evaluate_phase1_clic_source_metrics.py"
DRY_RUN=0

FOLD_TRAIN_TX=(
  "20-15,20-19,6-15,8-20" "14-10,20-19,6-15,8-20" "14-10,14-7,6-15,8-20"
  "14-10,14-7,20-15,8-20" "14-10,14-7,20-15,20-19" "14-7,20-15,20-19,6-15"
)
FOLD_VALIDATION_TX=("14-7" "20-15" "20-19" "6-15" "8-20" "14-10")
FOLD_PROXY_TX=("14-10" "14-7" "20-15" "20-19" "6-15" "8-20")
GPU_MAP=(0 1 2 3 4 5)

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${arg}" >&2; exit 2 ;;
  esac
done

candidate_for() {
  local fold="$1" arm="$2"
  printf 'F%s%s_CLIC12' "${fold}" "${arm}"
}

shared_cache_for() {
  local fold="$1"
  printf '%s/F%s_SHARED/source_validation_known_leo_weak.npz' "${RUN_ROOT}" "${fold}"
}

shared_receipt_for() {
  local fold="$1"
  printf '%s/F%s_SHARED/source_validation_known_leo_weak.receipt.json' "${RUN_ROOT}" "${fold}"
}

pair_for() {
  local fold="$1"
  printf '%s/F%s_C_vs_G_pair.json' "${SOURCE_PAIR_ROOT}" "${fold}"
}

cache_command() {
  local fold="$1" index=$((fold - 1)) c g shared receipt
  c="$(candidate_for "${fold}" C)"
  g="$(candidate_for "${fold}" G)"
  shared="$(shared_cache_for "${fold}")"
  receipt="$(shared_receipt_for "${fold}")"
  CACHE_CMD=(
    "${PYTHON}" -u "${CACHE_BUILDER}"
    --fold-index "${fold}"
    --c-ckpt "${TRAINING_ROOT}/${c}/final_ssdg.pth"
    --c-terminal-receipt-json "${TRAINING_ROOT}/${c}/phase1_clic_terminal_receipt.json"
    --c-clean-npz "${CLEAN_ROOT}/${c}/source_clean_proxy.npz"
    --g-ckpt "${TRAINING_ROOT}/${g}/final_ssdg.pth"
    --g-terminal-receipt-json "${TRAINING_ROOT}/${g}/phase1_clic_terminal_receipt.json"
    --g-clean-npz "${CLEAN_ROOT}/${g}/source_clean_proxy.npz"
    --wisig-pkl "${WISIG_PKL}"
    --expected-wisig-sha256 "${WISIG_SHA256}"
    --source-tx-ids "${FOLD_TRAIN_TX[index]}"
    --known-validation-tx-ids "${FOLD_VALIDATION_TX[index]}"
    --proxy-unknown-tx-ids "${FOLD_PROXY_TX[index]}"
    --cache-run-root "${RUN_ROOT}"
    --out-npz "${shared}"
    --receipt-json "${receipt}"
    --batch-size 256 --device cuda:0
  )
}

forward_command() {
  local fold="$1" arm="$2" index=$((fold - 1)) candidate shared receipt pair
  candidate="$(candidate_for "${fold}" "${arm}")"
  shared="$(shared_cache_for "${fold}")"
  receipt="$(shared_receipt_for "${fold}")"
  pair="$(pair_for "${fold}")"
  FORWARD_CMD=(
    "${PYTHON}" -u "${FORWARD_ENTRY}"
    --ckpt "${TRAINING_ROOT}/${candidate}/final_ssdg.pth"
    --terminal-receipt-json "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json"
    --clean-npz "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz"
    --source-v-received-iq-npz "${shared}"
    --source-v-received-iq-receipt-json "${receipt}"
    --pair-json "${pair}"
    --training-run-root "${TRAINING_ROOT}"
    --cache-run-root "${RUN_ROOT}"
    --output-root "${RUN_ROOT}"
    --out-npz "${RUN_ROOT}/${candidate}/source_validation_known_leo_weak_features.npz"
    --binding-json "${RUN_ROOT}/${candidate}/source_validation_known_leo_weak.binding.json"
    --candidate-id "${candidate}" --fold-index "${fold}" --arm "${arm}"
    --source-tx-ids "${FOLD_TRAIN_TX[index]}"
    --batch-size 256 --device cuda:0
  )
}

score_command() {
  local fold="$1" index=$((fold - 1)) c g shared receipt pair output
  c="$(candidate_for "${fold}" C)"
  g="$(candidate_for "${fold}" G)"
  shared="$(shared_cache_for "${fold}")"
  receipt="$(shared_receipt_for "${fold}")"
  pair="$(pair_for "${fold}")"
  output="${RUN_ROOT}/F${fold}_PAIR/source_metrics_pair.json"
  SCORE_CMD=(
    "${PYTHON}" -u "${METRICS_ENTRY}"
    --fold-index "${fold}"
    --training-run-root "${TRAINING_ROOT}"
    --clean-run-root "${CLEAN_ROOT}"
    --cache-run-root "${RUN_ROOT}"
    --output-root "${RUN_ROOT}"
    --output-metrics-json "${output}"
    --source-tx-ids "${FOLD_TRAIN_TX[index]}"
    --c-ckpt "${TRAINING_ROOT}/${c}/final_ssdg.pth"
    --c-terminal-receipt-json "${TRAINING_ROOT}/${c}/phase1_clic_terminal_receipt.json"
    --c-clean-npz "${CLEAN_ROOT}/${c}/source_clean_proxy.npz"
    --c-source-v-feature-npz "${RUN_ROOT}/${c}/source_validation_known_leo_weak_features.npz"
    --c-source-v-binding-json "${RUN_ROOT}/${c}/source_validation_known_leo_weak.binding.json"
    --g-ckpt "${TRAINING_ROOT}/${g}/final_ssdg.pth"
    --g-terminal-receipt-json "${TRAINING_ROOT}/${g}/phase1_clic_terminal_receipt.json"
    --g-clean-npz "${CLEAN_ROOT}/${g}/source_clean_proxy.npz"
    --g-source-v-feature-npz "${RUN_ROOT}/${g}/source_validation_known_leo_weak_features.npz"
    --g-source-v-binding-json "${RUN_ROOT}/${g}/source_validation_known_leo_weak.binding.json"
    --source-v-received-iq-npz "${shared}"
    --source-v-received-iq-receipt-json "${receipt}"
    --pair-json "${pair}"
  )
}

aggregate_command() {
  AGGREGATE_CMD=(
    "${PYTHON}" -u "${METRICS_ENTRY}" --aggregate-folds
    --output-root "${RUN_ROOT}"
    --output-metrics-json "${RUN_ROOT}/source_metrics_aggregate.json"
    --input-pair-metrics-json
  )
  local fold
  for fold in 1 2 3 4 5 6; do
    AGGREGATE_CMD+=("${RUN_ROOT}/F${fold}_PAIR/source_metrics_pair.json")
  done
}

emit_command() {
  local stage="$1" fold="$2" arm="$3" gpu="$4"; shift 4
  printf '[DRY-RUN] stage=%s fold=%s arm=%s physical_gpu=%s source_only=1 retry=NO' \
    "${stage}" "${fold}" "${arm}" "${gpu}"
  printf ' %q' "$@"
  printf '\n'
}

claim_exact_root() {
  local path="$1"
  if ! mkdir -- "${path}"; then
    echo "refusing to overwrite source metrics run/log root" >&2
    exit 3
  fi
}

open_exclusive_fd() {
  local path="$1"
  if ! { set -o noclobber; exec {OPEN_FD}>"${path}"; }; then
    set +o noclobber
    echo "refusing to overwrite source metrics PID/log evidence" >&2
    exit 3
  fi
  set +o noclobber
}

claim_log() {
  local log_path="$1"
  open_exclusive_fd "${log_path}"
  LOG_FDS["${log_path}"]="${OPEN_FD}"
}

launch_claimed_log() {
  local log_path="$1" log_fd
  shift
  log_fd="${LOG_FDS["${log_path}"]}"
  (
    "$@" >&"${log_fd}" 2>&1
  ) &
  LAUNCHED_PID="$!"
  exec {log_fd}>&-
  unset "LOG_FDS[${log_path}]"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  for fold in 1 2 3 4 5 6; do
    index=$((fold - 1))
    c="$(candidate_for "${fold}" C)"
    g="$(candidate_for "${fold}" G)"
    shared="$(shared_cache_for "${fold}")"
    receipt="$(shared_receipt_for "${fold}")"
    cache_command "${fold}"
    printf '[DRY-RUN] stage=CLIC_SOURCE_V_CACHE fold=%s arm=CG candidate_c=%s candidate_g=%s physical_gpu=%s shared_cache=%s shared_receipt=%s source_only=1 retry=NO' \
      "${fold}" "${c}" "${g}" "${GPU_MAP[index]}" "${shared}" "${receipt}"
    printf ' %q' "${CACHE_CMD[@]}"
    printf '\n'
    for arm in C G; do
      candidate="$(candidate_for "${fold}" "${arm}")"
      forward_command "${fold}" "${arm}"
      emit_command CLIC_SOURCE_V_FORWARD "${fold}" "${arm}" "${GPU_MAP[index]}" \
        "candidate=${candidate}" "shared_cache=${shared}" "shared_receipt=${receipt}" "pair=$(pair_for "${fold}")" \
        "${FORWARD_CMD[@]}"
    done
    score_command "${fold}"
    printf '[DRY-RUN] stage=CLIC_SOURCE_METRICS_PAIR fold=%s arm=CG physical_gpu=CPU candidate_c=%s candidate_g=%s shared_cache=%s shared_receipt=%s output=F%s_PAIR/source_metrics_pair.json source_only=1 retry=NO' \
      "${fold}" "${c}" "${g}" "${shared}" "${receipt}" "${fold}"
    printf ' %q' "${SCORE_CMD[@]}"
    printf '\n'
  done
  aggregate_command
  printf '[DRY-RUN] stage=CLIC_SOURCE_METRICS_AGGREGATE fold=ALL arm=CG physical_gpu=CPU source_only=1 retry=NO'
  printf ' %q' "${AGGREGATE_CMD[@]}"
  printf '\n'
  exit 0
fi

if [[ -e "${RUN_ROOT}" || -e "${LOG_ROOT}" ]]; then
  echo "refusing to overwrite source metrics run/log root" >&2
  exit 3
fi

[[ -f "${CACHE_BUILDER}" && -f "${FORWARD_ENTRY}" && -f "${METRICS_ENTRY}" ]] || {
  echo "missing source metrics entry" >&2
  exit 2
}
[[ -f "${WISIG_PKL}" ]] || { echo "missing frozen WiSig input" >&2; exit 2; }
[[ "$(sha256sum "${WISIG_PKL}" | awk '{print $1}')" == "${WISIG_SHA256}" ]] || {
  echo "frozen WiSig SHA256 drift" >&2
  exit 2
}

check_inputs() {
  local fold arm candidate
  for fold in 1 2 3 4 5 6; do
    [[ -f "$(pair_for "${fold}")" ]] || { echo "missing source PAIR-v3 receipt: F${fold}" >&2; return 2; }
    for arm in C G; do
      candidate="$(candidate_for "${fold}" "${arm}")"
      [[ -f "${TRAINING_ROOT}/${candidate}/final_ssdg.pth" ]] || { echo "missing checkpoint: ${candidate}" >&2; return 2; }
      [[ -f "${TRAINING_ROOT}/${candidate}/phase1_clic_terminal_receipt.json" ]] || { echo "missing terminal: ${candidate}" >&2; return 2; }
      [[ -f "${CLEAN_ROOT}/${candidate}/source_clean_proxy.npz" ]] || { echo "missing clean-v4 NPZ: ${candidate}" >&2; return 2; }
    done
  done
}

check_inputs
claim_exact_root "${RUN_ROOT}"
claim_exact_root "${LOG_ROOT}"
PID_FILE="${LOG_ROOT}/pids_source_metrics12.tsv"
open_exclusive_fd "${PID_FILE}"
PID_FD="${OPEN_FD}"
printf 'stage|fold|arm|physical_gpu|pid|output|log_path\n' >&"${PID_FD}"

declare -a PIDS=()
declare -A LOG_FDS=()
append_pid() {
  local stage="$1" fold="$2" arm="$3" gpu="$4" pid="$5" output="$6" log_path="$7"
  PIDS+=("${pid}")
  printf '%s|%s|%s|%s|%s|%s|%s\n' \
    "${stage}" "${fold}" "${arm}" "${gpu}" "${pid}" "${output}" "${log_path}" \
    >&"${PID_FD}"
}

for fold in 1 2 3 4 5 6; do
  mkdir -p "${RUN_ROOT}/F${fold}_SHARED"
  claim_log "${LOG_ROOT}/F${fold}_source_v_cache.out"
done
for fold in 1 2 3 4 5 6; do
  index=$((fold - 1))
  cache_command "${fold}"
  log_path="${LOG_ROOT}/F${fold}_source_v_cache.out"
  launch_claimed_log "${log_path}" env CUDA_VISIBLE_DEVICES="${GPU_MAP[index]}" PYTHONPATH="${CODE_ROOT}" \
    "${CACHE_CMD[@]}"
  append_pid CLIC_SOURCE_V_CACHE "${fold}" CG "${GPU_MAP[index]}" "${LAUNCHED_PID}" \
    "$(shared_cache_for "${fold}")" "${log_path}"
done
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" == "0" ]] || exit "${status}"

PIDS=()
for fold in 1 2 3 4 5 6; do
  mkdir -p "${RUN_ROOT}/$(candidate_for "${fold}" C)" "${RUN_ROOT}/$(candidate_for "${fold}" G)"
  for arm in C G; do
    candidate="$(candidate_for "${fold}" "${arm}")"
    claim_log "${LOG_ROOT}/${candidate}_source_v_forward.out"
  done
done
for fold in 1 2 3 4 5 6; do
  index=$((fold - 1))
  for arm in C G; do
    candidate="$(candidate_for "${fold}" "${arm}")"
    forward_command "${fold}" "${arm}"
    log_path="${LOG_ROOT}/${candidate}_source_v_forward.out"
    launch_claimed_log "${log_path}" env CUDA_VISIBLE_DEVICES="${GPU_MAP[index]}" PYTHONPATH="${CODE_ROOT}" \
      "${FORWARD_CMD[@]}"
    append_pid CLIC_SOURCE_V_FORWARD "${fold}" "${arm}" "${GPU_MAP[index]}" "${LAUNCHED_PID}" \
      "${RUN_ROOT}/${candidate}/source_validation_known_leo_weak_features.npz" "${log_path}"
  done
done
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" == "0" ]] || exit "${status}"

PIDS=()
for fold in 1 2 3 4 5 6; do
  mkdir -p "${RUN_ROOT}/F${fold}_PAIR"
  claim_log "${LOG_ROOT}/F${fold}_source_metrics_pair.out"
done
for fold in 1 2 3 4 5 6; do
  score_command "${fold}"
  log_path="${LOG_ROOT}/F${fold}_source_metrics_pair.out"
  launch_claimed_log "${log_path}" env CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_ROOT}" \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${SCORE_CMD[@]}"
  append_pid CLIC_SOURCE_METRICS_PAIR "${fold}" CG CPU "${LAUNCHED_PID}" \
    "${RUN_ROOT}/F${fold}_PAIR/source_metrics_pair.json" "${log_path}"
done
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" == "0" ]] || exit "${status}"

aggregate_command
log_path="${LOG_ROOT}/source_metrics_aggregate.out"
claim_log "${log_path}"
launch_claimed_log "${log_path}" env CUDA_VISIBLE_DEVICES="" PYTHONPATH="${CODE_ROOT}" \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "${AGGREGATE_CMD[@]}"
aggregate_pid="${LAUNCHED_PID}"
append_pid CLIC_SOURCE_METRICS_AGGREGATE ALL CG CPU "${aggregate_pid}" \
  "${RUN_ROOT}/source_metrics_aggregate.json" "${log_path}"
wait "${aggregate_pid}"
exec {PID_FD}>&-
