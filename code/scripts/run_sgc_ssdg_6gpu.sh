#!/usr/bin/env bash
set -uo pipefail

# Parallel independent SGC + SSDG launcher.
#
# GPU 0-2 run standalone SGC experiments from the BEX02/N04 teacher checkpoint.
# GPU 3-5 run SSDG experiments from scratch. Their first 170 epochs mirror the
# BEX02_fishr002_mixed_e170 backbone configuration with labeled data ratio 0.1;
# pseudo-label variants then continue for 100 epochs on labeled + pseudo-labeled
# unlabeled data.
#
# Examples:
#   bash code/scripts/run_sgc_ssdg_6gpu.sh --dry-run
#   bash code/scripts/run_sgc_ssdg_6gpu.sh
#   BASE_CKPT=/path/to/latest_model.pth bash code/scripts/run_sgc_ssdg_6gpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-}"
BASE_CKPT="${BASE_CKPT:-/home/szu2070436088/2510044040/CV-SincNet/runs/best_base_explore/BEX02_fishr002_mixed_e170/latest_model.pth}"
RUN_ROOT="${RUN_ROOT:-${CODE_ROOT}/runs/sgc_ssdg_bex02}"
LOG_ROOT="${LOG_ROOT:-${CODE_ROOT}/logs/sgc_ssdg_bex02}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

usage() {
  sed -n '1,14p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --base-ckpt) BASE_CKPT="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --stop-on-fail) STOP_ON_FAIL=1; shift ;;
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
  echo "ERROR: SGC teacher checkpoint not found: ${BASE_CKPT}" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCHED_LOG="${LOG_ROOT}/scheduler_${STAMP}.log"

COMMON_DATA_ARGS=(
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --batch_size 256
  --eval_batch_size 256
  --num_workers 4
  --eval_max_batches 0
)

SAT_EVAL_ARGS=(
  --eval_sat_channel true
  --eval_sat_on test_unseen_day_seen_rx,test_seen_day_unseen_rx,test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
)

SSDG_MODEL_ARGS=(
  --lr 2e-4
  --label_smoothing 0.01
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --use_mixstyle true
  --mixstyle_p 0.18
  --mixstyle_strength 0.70
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
  --use_aug true
  --use_sat_consistency
  --sat_train_scenario mixed_orbit
  --sat_cons_start_epoch 20
  --lambda_domain 1.00
  --lambda_adv 0.45
  --lambda_orth 0.05
  --lambda_cons 0.08
  --lambda_group_ce 0.10
  --group_ce_min_domains 4
  --lambda_sat_cls 0.10
  --lambda_sat_cons 0.00
  --lambda_fishr 0.02
  --fishr_min_domains 4
)

PIDS=()
TAGS=()
STATUS=0

log_msg() {
  echo "$@" | tee -a "${SCHED_LOG}"
}

launch() {
  local gpu="$1"
  local exp="$2"
  shift 2
  local log_path="${LOG_ROOT}/${exp}_${STAMP}.log"
  log_msg "[LAUNCH] gpu=${gpu} exp=${exp} log=${log_path}"
  log_msg "CMD=CUDA_VISIBLE_DEVICES=${gpu} PYTHONUNBUFFERED=1 $*"
  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "$@" > "${log_path}" 2>&1 &
  PIDS+=("$!")
  TAGS+=("${exp}")
}

launch 0 S1_sgc_default_mixed \
  "${PYTHON_BIN}" -u train_sgc.py \
  --teacher_ckpt "${BASE_CKPT}" \
  --config SGC/configs/standalone_rsgc_v2.yaml \
  --stage stage_b \
  --sat_train_scenario mixed_orbit \
  --best_metric clean_val_tx \
  --output_dir "${RUN_ROOT}/S1_sgc_default_mixed" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

launch 1 S2_sgc_satworst_select \
  "${PYTHON_BIN}" -u train_sgc.py \
  --teacher_ckpt "${BASE_CKPT}" \
  --config SGC/configs/standalone_rsgc_v2.yaml \
  --stage stage_b \
  --sat_train_scenario mixed_orbit \
  --best_metric sat_worst_tx \
  --override stage_b.lambda_feat=0.6 \
  --override stage_b.lambda_budget=0.08 \
  --output_dir "${RUN_ROOT}/S2_sgc_satworst_select" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

launch 2 S3_sgc_no_residual_ctrl \
  "${PYTHON_BIN}" -u train_sgc.py \
  --teacher_ckpt "${BASE_CKPT}" \
  --config SGC/configs/standalone_rsgc_v2.yaml \
  --stage stage_b \
  --ablation_mode no_residual \
  --sat_train_scenario mixed_orbit \
  --best_metric clean_val_tx \
  --output_dir "${RUN_ROOT}/S3_sgc_no_residual_ctrl" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

launch 3 U0_ssdg_label_only_010 \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  --from_scratch true \
  --split_mode tx_rx_day_1_6_3 \
  --labeled_ratio 0.10 \
  --unlabeled_ratio 0.60 \
  --source_val_ratio 0.30 \
  --label_epochs 170 \
  --pseudo_epochs 0 \
  --use_unlabeled false \
  --output_dir "${RUN_ROOT}/U0_ssdg_label_only_010" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  "${SSDG_MODEL_ARGS[@]}" \
  --device cuda:0

launch 4 U1_ssdg_domain_temporal \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  --from_scratch true \
  --split_mode tx_rx_day_1_6_3 \
  --labeled_ratio 0.10 \
  --unlabeled_ratio 0.60 \
  --source_val_ratio 0.30 \
  --label_epochs 170 \
  --pseudo_epochs 100 \
  --pseudo_threshold_mode rx_day_quantile \
  --tau_min 0.80 \
  --tau_max 0.97 \
  --pseudo_domain_gate true \
  --pseudo_temporal_gate true \
  --pseudo_strong_agreement true \
  --lambda_u 1.0 \
  --output_dir "${RUN_ROOT}/U1_ssdg_domain_temporal" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  "${SSDG_MODEL_ARGS[@]}" \
  --device cuda:0

launch 5 U2_ssdg_domain_temporal_ema \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  --from_scratch true \
  --split_mode tx_rx_day_1_6_3 \
  --labeled_ratio 0.10 \
  --unlabeled_ratio 0.60 \
  --source_val_ratio 0.30 \
  --label_epochs 170 \
  --pseudo_epochs 100 \
  --pseudo_threshold_mode rx_day_quantile \
  --tau_min 0.82 \
  --tau_max 0.97 \
  --pseudo_domain_gate true \
  --pseudo_temporal_gate true \
  --pseudo_strong_agreement true \
  --use_ema_teacher true \
  --ema_decay 0.999 \
  --lambda_u 1.0 \
  --output_dir "${RUN_ROOT}/U2_ssdg_domain_temporal_ema" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  "${SSDG_MODEL_ARGS[@]}" \
  --device cuda:0

if [ "${DRY_RUN}" = "1" ]; then
  log_msg "[DRY-RUN] commands printed only."
  exit 0
fi

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
      break
    fi
  fi
done

log_msg "[DONE] status=${STATUS}"
exit "${STATUS}"
