#!/usr/bin/env bash
set -uo pipefail

# PhyCon-CxRCM-SGC experiments on the N04 base checkpoint.
#
# Defaults:
#   - GPUs: 0,1,2,3
#   - Base checkpoint:
#     /home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth
#   - Output: runs/phycon_recon_sgc_n04
#   - Logs: logs/phycon_recon_sgc_n04
#
# Plans:
#   SMOKE   : parser/model/data-light dry-run style short checks
#   PHASE1  : train diffusion teacher variants
#   PHASE2  : distill consistency students from finished teachers
#   EVAL    : evaluate consistency frontends on satellite scenarios
#   JOINT   : joint fine-tune recon + SGC-V3 if an SGC ckpt is supplied
#   FULL    : PHASE1 -> PHASE2 -> EVAL, plus JOINT when --sgc-v3-ckpt is set
#
# Examples:
#   bash code/scripts/run_phycon_recon_sgc_n04_gpu0_3.sh --dry-run
#   bash code/scripts/run_phycon_recon_sgc_n04_gpu0_3.sh --plan SMOKE
#   bash code/scripts/run_phycon_recon_sgc_n04_gpu0_3.sh --plan FULL
#   bash code/scripts/run_phycon_recon_sgc_n04_gpu0_3.sh --plan JOINT --sgc-v3-ckpt runs/sgc_v3_n04/SGCV3-14_ipfa_blrc_mixed/best_sgc_v3.pth

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${CODE_ROOT}" || exit 1

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3}"
PLAN="${PLAN:-FULL}"
PYTHON_BIN="${PYTHON_BIN:-}"
BASE_CKPT="${BASE_CKPT:-/home/szu2070436088/2510044040/CV-SincNet/runs/b3b_asym_sat_baseline/N04_fishr002_cls010/latest_model.pth}"
SGC_V3_CKPT="${SGC_V3_CKPT:-}"
RUN_ROOT="${RUN_ROOT:-${WORKSPACE_ROOT}/runs/phycon_recon_sgc_n04}"
LOG_ROOT="${LOG_ROOT:-${WORKSPACE_ROOT}/logs/phycon_recon_sgc_n04}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-1}"
STREAM_LOGS="${STREAM_LOGS:-0}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

usage() {
  sed -n '1,32p' "$0"
  cat <<'EOF'

Options:
  --gpu-ids 0,1,2,3
  --plan FULL|SMOKE|PHASE1|PHASE2|EVAL|JOINT
  --base-ckpt PATH
  --sgc-v3-ckpt PATH
  --run-root PATH
  --log-root PATH
  --python PATH
  --no-stop-on-fail
  --stream-logs
  --dry-run
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS_CSV="$2"; shift 2 ;;
    --plan) PLAN="$2"; shift 2 ;;
    --base-ckpt) BASE_CKPT="$2"; shift 2 ;;
    --sgc-v3-ckpt) SGC_V3_CKPT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --no-stop-on-fail) STOP_ON_FAIL=0; shift ;;
    --stream-logs) STREAM_LOGS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [ -z "${PYTHON_BIN}" ]; then
  for candidate in python3 python python.exe py; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="${candidate}"
      break
    fi
  done
fi

if [ -z "${PYTHON_BIN}" ] || ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: no python executable found. Pass --python /path/to/python or set PYTHON_BIN." >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ ! -f "${BASE_CKPT}" ]; then
  echo "ERROR: base checkpoint not found: ${BASE_CKPT}" >&2
  exit 2
fi

if [ "${DRY_RUN}" != "1" ] && [ -n "${SGC_V3_CKPT}" ] && [ ! -f "${SGC_V3_CKPT}" ]; then
  echo "ERROR: SGC-V3 checkpoint not found: ${SGC_V3_CKPT}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"
if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty." >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${PLAN}_${STAMP}.log"
STATUS=0

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

run_cmd() {
  local gpu="$1"
  local exp="$2"
  shift 2
  local log_path="${LOG_ROOT}/${exp}_${STAMP}.log"
  log_msg "[RUN] gpu=${gpu} exp=${exp} log=${log_path}"
  log_msg "CMD=CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 PYTHONPATH=${CODE_ROOT} $*"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  if [ "${STREAM_LOGS}" = "1" ]; then
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_ROOT}" "$@" 2>&1 | sed -u "s/^/[${exp}|GPU${gpu}] /" | tee "${log_path}"
  else
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 PYTHONPATH="${CODE_ROOT}" "$@" > "${log_path}" 2>&1
  fi
}

run_bg() {
  local gpu="$1"
  local exp="$2"
  shift 2
  run_cmd "${gpu}" "${exp}" "$@" &
  PIDS+=("$!")
  TAGS+=("${exp}")
}

wait_all() {
  local i pid tag code
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    tag="${TAGS[$i]}"
    if wait "${pid}"; then
      log_msg "[FINISHED] exp=${tag} status=0"
    else
      code="$?"
      log_msg "[FAILED] exp=${tag} status=${code}"
      STATUS="${code}"
      if [ "${STOP_ON_FAIL}" = "1" ]; then
        return "${code}"
      fi
    fi
  done
  PIDS=()
  TAGS=()
  return 0
}

COMMON_DATA_ARGS=(
  --teacher_ckpt "${BASE_CKPT}"
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --wisig_train_ratio 0.2
  --wisig_out_len 256
  --batch_size 256
  --eval_batch_size 256
  --num_workers 4
  --prefetch_factor 2
  --seed 1337
  --device cuda:0
)

SAT_EVAL_ARGS=(
  --eval_sat_channel true
  --eval_sat_on test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
)

phase_smoke() {
  PIDS=(); TAGS=()
  run_bg "${GPU_LIST[0]}" R0_diff_dry "${PYTHON_BIN}" -u -m SGC.train_recon_diffusion \
    "${COMMON_DATA_ARGS[@]}" --output_dir "${RUN_ROOT}/R0_diff_dry" --dry_run
  run_bg "${GPU_LIST[1]:-${GPU_LIST[0]}}" R0_cm_dry "${PYTHON_BIN}" -u -m SGC.distill_recon_consistency \
    "${COMMON_DATA_ARGS[@]}" --output_dir "${RUN_ROOT}/R0_cm_dry" --dry_run
  run_bg "${GPU_LIST[2]:-${GPU_LIST[0]}}" R0_eval_dry "${PYTHON_BIN}" -u -m SGC.eval_recon_frontend \
    "${COMMON_DATA_ARGS[@]}" --output_dir "${RUN_ROOT}/R0_eval_dry" --dry_run
  run_bg "${GPU_LIST[3]:-${GPU_LIST[0]}}" R0_joint_dry "${PYTHON_BIN}" -u -m SGC.train_recon_sgc_joint \
    "${COMMON_DATA_ARGS[@]}" --output_dir "${RUN_ROOT}/R0_joint_dry" --dry_run
  wait_all
}

phase1_diffusion() {
  PIDS=(); TAGS=()
  run_bg "${GPU_LIST[0]}" R1_diff_pair_id_tf "${PYTHON_BIN}" -u -m SGC.train_recon_diffusion \
    "${COMMON_DATA_ARGS[@]}" \
    --config SGC/configs/recon_cxresdiff_020m.yaml \
    --output_dir "${RUN_ROOT}/R1_diff_pair_id_tf" \
    --epochs 80 \
    --lr_recon 2e-4 \
    --sat_train_scenario mixed_orbit \
    ${EXTRA_ARGS}
  run_bg "${GPU_LIST[1]:-${GPU_LIST[0]}}" R2_diff_phychan "${PYTHON_BIN}" -u -m SGC.train_recon_diffusion \
    "${COMMON_DATA_ARGS[@]}" \
    --config SGC/configs/recon_cxresdiff_020m_chan.yaml \
    --output_dir "${RUN_ROOT}/R2_diff_phychan" \
    --epochs 80 \
    --lr_recon 2e-4 \
    --sat_train_scenario mixed_orbit \
    --enable_channel_loss true \
    ${EXTRA_ARGS}
  wait_all
}

phase2_distill() {
  local teacher_main="${RUN_ROOT}/R1_diff_pair_id_tf/best_recon_diffusion.pth"
  local teacher_chan="${RUN_ROOT}/R2_diff_phychan/best_recon_diffusion.pth"
  if [ "${DRY_RUN}" != "1" ] && [ ! -f "${teacher_main}" ]; then
    echo "ERROR: missing diffusion teacher: ${teacher_main}" >&2
    return 2
  fi
  PIDS=(); TAGS=()
  run_bg "${GPU_LIST[0]}" R3_cm_from_pair_id_tf "${PYTHON_BIN}" -u -m SGC.distill_recon_consistency \
    "${COMMON_DATA_ARGS[@]}" \
    --config SGC/configs/recon_cxconsistency_020m.yaml \
    --diffusion_ckpt "${teacher_main}" \
    --output_dir "${RUN_ROOT}/R3_cm_from_pair_id_tf" \
    --epochs 50 \
    --lr_recon 1e-4 \
    --sat_train_scenario mixed_orbit \
    ${EXTRA_ARGS}
  if [ "${DRY_RUN}" = "1" ] || [ -f "${teacher_chan}" ]; then
    run_bg "${GPU_LIST[1]:-${GPU_LIST[0]}}" R4_cm_from_phychan "${PYTHON_BIN}" -u -m SGC.distill_recon_consistency \
      "${COMMON_DATA_ARGS[@]}" \
      --config SGC/configs/recon_cxconsistency_020m.yaml \
      --diffusion_ckpt "${teacher_chan}" \
      --output_dir "${RUN_ROOT}/R4_cm_from_phychan" \
      --epochs 50 \
      --lr_recon 1e-4 \
      --sat_train_scenario mixed_orbit \
      ${EXTRA_ARGS}
  else
    log_msg "[SKIP] R4_cm_from_phychan because ${teacher_chan} does not exist"
  fi
  wait_all
}

phase_eval() {
  local cm_main="${RUN_ROOT}/R3_cm_from_pair_id_tf/best_recon_consistency.pth"
  local cm_chan="${RUN_ROOT}/R4_cm_from_phychan/best_recon_consistency.pth"
  if [ "${DRY_RUN}" != "1" ] && [ ! -f "${cm_main}" ]; then
    echo "ERROR: missing consistency checkpoint: ${cm_main}" >&2
    return 2
  fi
  PIDS=(); TAGS=()
  run_bg "${GPU_LIST[0]}" E1_eval_cm_main_1step "${PYTHON_BIN}" -u -m SGC.eval_recon_frontend \
    "${COMMON_DATA_ARGS[@]}" "${SAT_EVAL_ARGS[@]}" \
    --recon_ckpt "${cm_main}" \
    --output_dir "${RUN_ROOT}/E1_eval_cm_main_1step" \
    --model_kind consistency \
    --steps 1 \
    --rho 0.15
  run_bg "${GPU_LIST[1]:-${GPU_LIST[0]}}" E2_eval_cm_main_2step "${PYTHON_BIN}" -u -m SGC.eval_recon_frontend \
    "${COMMON_DATA_ARGS[@]}" "${SAT_EVAL_ARGS[@]}" \
    --recon_ckpt "${cm_main}" \
    --output_dir "${RUN_ROOT}/E2_eval_cm_main_2step" \
    --model_kind consistency \
    --steps 2 \
    --rho 0.15
  run_bg "${GPU_LIST[2]:-${GPU_LIST[0]}}" E3_eval_cm_main_4step "${PYTHON_BIN}" -u -m SGC.eval_recon_frontend \
    "${COMMON_DATA_ARGS[@]}" "${SAT_EVAL_ARGS[@]}" \
    --recon_ckpt "${cm_main}" \
    --output_dir "${RUN_ROOT}/E3_eval_cm_main_4step" \
    --model_kind consistency \
    --steps 4 \
    --rho 0.15
  if [ "${DRY_RUN}" = "1" ] || [ -f "${cm_chan}" ]; then
    run_bg "${GPU_LIST[3]:-${GPU_LIST[0]}}" E4_eval_cm_phychan_2step "${PYTHON_BIN}" -u -m SGC.eval_recon_frontend \
      "${COMMON_DATA_ARGS[@]}" "${SAT_EVAL_ARGS[@]}" \
      --recon_ckpt "${cm_chan}" \
      --output_dir "${RUN_ROOT}/E4_eval_cm_phychan_2step" \
      --model_kind consistency \
      --steps 2 \
      --rho 0.15
  else
    log_msg "[SKIP] E4_eval_cm_phychan_2step because ${cm_chan} does not exist"
  fi
  wait_all
}

phase_joint() {
  local cm_main="${RUN_ROOT}/R3_cm_from_pair_id_tf/best_recon_consistency.pth"
  local sgc_args=()
  if [ "${DRY_RUN}" != "1" ] && [ ! -f "${cm_main}" ]; then
    echo "ERROR: missing consistency checkpoint: ${cm_main}" >&2
    return 2
  fi
  if [ -z "${SGC_V3_CKPT}" ]; then
    log_msg "[SKIP] JOINT because --sgc-v3-ckpt was not provided; script will initialize SGC-V3 if run explicitly with PLAN=JOINT and no ckpt."
  else
    sgc_args=(--sgc_v3_ckpt "${SGC_V3_CKPT}")
  fi
  PIDS=(); TAGS=()
  run_bg "${GPU_LIST[0]}" J1_recon_sgc_joint_main "${PYTHON_BIN}" -u -m SGC.train_recon_sgc_joint \
    "${COMMON_DATA_ARGS[@]}" \
    --config SGC/configs/recon_sgc_joint.yaml \
    --recon_ckpt "${cm_main}" \
    "${sgc_args[@]}" \
    --output_dir "${RUN_ROOT}/J1_recon_sgc_joint_main" \
    --epochs 20 \
    --lr_recon 2e-5 \
    --lr_sgc 5e-5 \
    ${EXTRA_ARGS}
  wait_all
}

log_msg "PhyCon-CxRCM-SGC N04 launcher"
log_msg "PLAN=${PLAN} GPU_IDS=${GPU_IDS_CSV} BASE_CKPT=${BASE_CKPT}"
log_msg "SGC_V3_CKPT=${SGC_V3_CKPT:-<none>}"
log_msg "RUN_ROOT=${RUN_ROOT}"
log_msg "LOG_ROOT=${LOG_ROOT}"
log_msg "DRY_RUN=${DRY_RUN} STOP_ON_FAIL=${STOP_ON_FAIL} STREAM_LOGS=${STREAM_LOGS}"

PLAN_UPPER="$(echo "${PLAN}" | tr '[:lower:]' '[:upper:]')"
case "${PLAN_UPPER}" in
  SMOKE)
    phase_smoke || STATUS=$?
    ;;
  PHASE1)
    phase1_diffusion || STATUS=$?
    ;;
  PHASE2)
    phase2_distill || STATUS=$?
    ;;
  EVAL)
    phase_eval || STATUS=$?
    ;;
  JOINT)
    phase_joint || STATUS=$?
    ;;
  FULL|ALL)
    phase1_diffusion || STATUS=$?
    if [ "${STATUS}" -eq 0 ]; then phase2_distill || STATUS=$?; fi
    if [ "${STATUS}" -eq 0 ]; then phase_eval || STATUS=$?; fi
    if [ "${STATUS}" -eq 0 ] && [ -n "${SGC_V3_CKPT}" ]; then phase_joint || STATUS=$?; fi
    ;;
  *)
    echo "ERROR: unknown plan ${PLAN}. Use FULL,SMOKE,PHASE1,PHASE2,EVAL,JOINT." >&2
    exit 2
    ;;
esac

log_msg "[DONE] status=${STATUS}"
exit "${STATUS}"
