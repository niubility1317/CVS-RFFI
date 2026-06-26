#!/usr/bin/env bash
set -uo pipefail

# SSDG two-GPU launcher for GPUs 4 and 5.
#
# Experiments:
#   U0_ssdg_label_only_010       : 0.1 labeled-only control, 170 epochs
#   U1_ssdg_domain_temporal      : domain + temporal + strong-agreement pseudo labels
#   U2_ssdg_domain_temporal_ema  : U1 plus EMA teacher
#   U3_ssdg_global_temporal      : global confidence threshold instead of rx/day quantile
#   U4_ssdg_domain_only          : no temporal gate ablation
#   U5_ssdg_temporal_only        : no domain gate ablation
#
# Data split inside source train_days x train_rxs:
#   labeled=0.10, unlabeled=0.60, source_val=0.30
#
# Examples:
#   bash code/scripts/run_ssdg_gpu45.sh --dry-run
#   bash code/scripts/run_ssdg_gpu45.sh
#   PYTHON_BIN=python3 RUN_ROOT=/path/to/runs LOG_ROOT=/path/to/logs bash code/scripts/run_ssdg_gpu45.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${CODE_ROOT}" || exit 1

PYTHON_BIN="${PYTHON_BIN:-}"
RUN_ROOT="${RUN_ROOT:-${CODE_ROOT}/runs/ssdg_gpu45_bex02}"
LOG_ROOT="${LOG_ROOT:-${CODE_ROOT}/logs/ssdg_gpu45_bex02}"
DRY_RUN="${DRY_RUN:-0}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

usage() {
  sed -n '1,15p' "$0"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --python) PYTHON_BIN="$2"; shift 2 ;;
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

BEX02_SSDG_ARGS=(
  --from_scratch true
  --split_mode tx_rx_day_1_6_3
  --labeled_ratio 0.10
  --unlabeled_ratio 0.60
  --source_val_ratio 0.30
  --label_epochs 170
  --lambda_u 1.0
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

wait_batch() {
  if [ "${#PIDS[@]}" -eq 0 ]; then
    return 0
  fi
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
        exit "${STATUS}"
      fi
    fi
  done
  PIDS=()
  TAGS=()
}

launch 4 U0_ssdg_label_only_010 \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  "${BEX02_SSDG_ARGS[@]}" \
  --pseudo_epochs 0 \
  --use_unlabeled false \
  --output_dir "${RUN_ROOT}/U0_ssdg_label_only_010" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

launch 5 U1_ssdg_domain_temporal \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  "${BEX02_SSDG_ARGS[@]}" \
  --pseudo_epochs 100 \
  --pseudo_threshold_mode rx_day_quantile \
  --pseudo_domain_gate true \
  --pseudo_temporal_gate true \
  --pseudo_strong_agreement true \
  --tau_min 0.80 \
  --tau_max 0.97 \
  --output_dir "${RUN_ROOT}/U1_ssdg_domain_temporal" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

wait_batch

launch 4 U2_ssdg_domain_temporal_ema \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  "${BEX02_SSDG_ARGS[@]}" \
  --pseudo_epochs 100 \
  --pseudo_threshold_mode rx_day_quantile \
  --pseudo_domain_gate true \
  --pseudo_temporal_gate true \
  --pseudo_strong_agreement true \
  --tau_min 0.82 \
  --tau_max 0.97 \
  --use_ema_teacher true \
  --ema_decay 0.999 \
  --output_dir "${RUN_ROOT}/U2_ssdg_domain_temporal_ema" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

launch 5 U3_ssdg_global_temporal \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  "${BEX02_SSDG_ARGS[@]}" \
  --pseudo_epochs 100 \
  --pseudo_threshold_mode global \
  --pseudo_domain_gate true \
  --pseudo_temporal_gate true \
  --pseudo_strong_agreement true \
  --tau_min 0.85 \
  --tau_max 0.97 \
  --output_dir "${RUN_ROOT}/U3_ssdg_global_temporal" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

wait_batch

launch 4 U4_ssdg_domain_only \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  "${BEX02_SSDG_ARGS[@]}" \
  --pseudo_epochs 100 \
  --pseudo_threshold_mode rx_day_quantile \
  --pseudo_domain_gate true \
  --tau_min 0.80 \
  --tau_max 0.97 \
  --pseudo_temporal_gate false \
  --pseudo_strong_agreement true \
  --output_dir "${RUN_ROOT}/U4_ssdg_domain_only" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

launch 5 U5_ssdg_temporal_only \
  "${PYTHON_BIN}" -u -m SSDG.train_ssdg \
  "${BEX02_SSDG_ARGS[@]}" \
  --pseudo_epochs 100 \
  --pseudo_threshold_mode rx_day_quantile \
  --pseudo_temporal_gate true \
  --tau_min 0.80 \
  --tau_max 0.97 \
  --pseudo_domain_gate false \
  --pseudo_strong_agreement true \
  --output_dir "${RUN_ROOT}/U5_ssdg_temporal_only" \
  "${COMMON_DATA_ARGS[@]}" \
  "${SAT_EVAL_ARGS[@]}" \
  --device cuda:0

wait_batch

if [ "${DRY_RUN}" = "1" ]; then
  log_msg "[DRY-RUN] commands printed only."
  exit 0
fi

log_msg "[DONE] status=${STATUS}"
exit "${STATUS}"
