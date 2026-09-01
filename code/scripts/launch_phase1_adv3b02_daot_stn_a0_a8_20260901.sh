#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_daot_stn_a0_a8_s713104_20260901}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER="${WORKER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
BASE_CKPT="${BASE_CKPT:-}"
SEED="${SEED:-713104}"
DRY_RUN=0
ONLY=""

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=A[0-8]) ONLY="${arg#--only=}" ;;
    *) echo "[DAOT-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${WORKER}" ]] || { echo "[DAOT-ERROR] worker missing: ${WORKER}" >&2; exit 2; }
[[ -n "${BASE_CKPT}" ]] || { echo "[DAOT-ERROR] BASE_CKPT must bind the frozen ADV3B02/FastTrust-EFF source checkpoint" >&2; exit 2; }
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${BASE_CKPT}" ]] || { echo "[DAOT-ERROR] checkpoint missing: ${BASE_CKPT}" >&2; exit 2; }
  [[ ! -e "${RUNS_ROOT}" ]] || { echo "[DAOT-ERROR] refusing to overwrite run root: ${RUNS_ROOT}" >&2; exit 3; }
  mkdir -p "${RUNS_ROOT}/dispatcher_logs"
fi

ROWS=(A0 A1 A2 A3 A4 A5 A6 A7 A8)
GPUS=(0 1 2 3 4 5 6 7 0)
PIDS=()
NAMES=()

run_row() {
  local ablation="$1"
  local gpu="$2"
  local candidate="ADV3B02_DAOT_STN_${ablation}_S${SEED}"
  local args=(--only=M3)
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
  env \
    ROOT="${ROOT}" \
    CODE_ROOT="${CODE_ROOT}" \
    PYTHON="${PYTHON}" \
    CONTROL_PYTHON="${PYTHON}" \
    RUN_ID="${RUN_ID}" \
    RUNS_ROOT="${RUNS_ROOT}" \
    GPU="${gpu}" \
    SEED="${SEED}" \
    TOTAL_EPOCHS=200 \
    LABEL_EPOCHS=130 \
    PSEUDO_EPOCHS=70 \
    INIT_MODE=adv3b02_core90 \
    BASE_CKPT="${BASE_CKPT}" \
    CANDIDATE_ID_OVERRIDE="${candidate}" \
    MUSE_UNLABELED_BATCH_SIZE=256 \
    FASTTRUST_RC4=true \
    SAT_ANCHOR_SSL=false \
    RC4_USE_ANCHOR=true \
    RC4_USE_CALIBRATION=true \
    RC4_ENABLE_HARD=true \
    RC4_ENABLE_PARTIAL=true \
    RC4_ENABLE_PARTIAL_SET=true \
    RC4_ENABLE_PARTIAL_CONDITIONAL=true \
    RC4_ENABLE_NEGATIVE=false \
    RC4_PARTIAL_EFFECTIVE_BUDGET=0 \
    RC4_NEGATIVE_EFFECTIVE_BUDGET=0 \
    RC4_TOTAL_IDENTITY_EFFECTIVE_BUDGET=0.15 \
    RC4_USE_CALIBRATED_PARTIAL_THRESHOLD=true \
    RC4_CLASS_RX_CAP=true \
    ABLATION=NONE \
    DAOT_ABLATION="${ablation}" \
    bash "${WORKER}" "${args[@]}"
}

echo "[DAOT-MATRIX] run_id=${RUN_ID} seed=${SEED} rows=9 default_teacher=clean+medium+hard"
for index in "${!ROWS[@]}"; do
  ablation="${ROWS[$index]}"
  gpu="${GPUS[$index]}"
  [[ -n "${ONLY}" && "${ablation}" != "${ONLY}" ]] && continue
  candidate="ADV3B02_DAOT_STN_${ablation}_S${SEED}"
  echo "[DAOT-ROW] ablation=${ablation} gpu=${gpu} candidate=${candidate}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    run_row "${ablation}" "${gpu}"
  else
    (run_row "${ablation}" "${gpu}") > "${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 &
    PIDS+=("$!")
    NAMES+=("${candidate}")
  fi
done

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[DAOT-WORKER-COMPLETE] candidate=${NAMES[$index]}"
  else
    echo "[DAOT-WORKER-FAILED] candidate=${NAMES[$index]}" >&2
    failed=1
  fi
done
exit "${failed}"
