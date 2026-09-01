#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_daot_stn_a0_a7_manysig_s392005_20260901}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER="${WORKER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
BASE_CKPT="${BASE_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
SEED="${SEED:-392005}"
DRY_RUN=0
ONLY=""

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=A[0-7]) ONLY="${arg#--only=}" ;;
    *) echo "[DAOT-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

[[ -f "${WORKER}" ]] || { echo "[DAOT-ERROR] worker missing: ${WORKER}" >&2; exit 2; }
[[ -n "${BASE_CKPT}" ]] || { echo "[DAOT-ERROR] BASE_CKPT is required" >&2; exit 2; }
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${WISIG_PKL}" ]] || { echo "[DAOT-ERROR] dataset missing: ${WISIG_PKL}" >&2; exit 2; }
  [[ -f "${BASE_CKPT}" ]] || { echo "[DAOT-ERROR] checkpoint missing: ${BASE_CKPT}" >&2; exit 2; }
  [[ ! -e "${RUNS_ROOT}" ]] || { echo "[DAOT-ERROR] refusing to overwrite run root: ${RUNS_ROOT}" >&2; exit 3; }
  mkdir -p "${RUNS_ROOT}/dispatcher_logs"
fi

ROWS=(A0 A1 A2 A3 A4 A5 A6 A7)
GPUS=(0 1 2 3 4 5 6 7)
PIDS=()
NAMES=()

run_row() {
  local ablation="$1"
  local gpu="$2"
  local candidate="ADV3B02_DAOT_STN_${ablation}_MANYSIG_S${SEED}"
  local args=(--only=M3)
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
  env \
    ROOT="${ROOT}" \
    CODE_ROOT="${CODE_ROOT}" \
    PYTHON="${PYTHON}" \
    CONTROL_PYTHON="${PYTHON}" \
    RUN_ID="${RUN_ID}" \
    RUNS_ROOT="${RUNS_ROOT}" \
    WISIG_PKL="${WISIG_PKL}" \
    WISIG_EQUALIZED=1 \
    WISIG_TRAIN_DAYS=1,2,3 \
    WISIG_TEST_DAYS=0,1,2,3 \
    WISIG_TRAIN_RXS=1,3,4,6,8 \
    WISIG_TEST_RXS=0,2,5,7,9,10,11 \
    SPLIT_MODE=tx_rx_day_1_7_2 \
    LABELED_RATIO=0.07 \
    UNLABELED_RATIO=0.63 \
    SOURCE_VAL_RATIO=0.30 \
    SOURCE_CAL_RATIO=0 \
    SOURCE_SELECT_RATIO=0 \
    PHASE1_SOURCE_ROLE_PROTOCOL=legacy_l_u_v \
    ALLOW_SOURCE_TARGET_DAY_OVERLAP=true \
    TARGET_GROUP_LOADER=test_all_day_unseen_rx \
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

echo "[DAOT-MATRIX] run_id=${RUN_ID} seed=${SEED} rows=8 source_pool=90000 validation=single_v_27000 target_per_scenario=168000"
for index in "${!ROWS[@]}"; do
  ablation="${ROWS[$index]}"
  gpu="${GPUS[$index]}"
  [[ -n "${ONLY}" && "${ablation}" != "${ONLY}" ]] && continue
  candidate="ADV3B02_DAOT_STN_${ablation}_MANYSIG_S${SEED}"
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
