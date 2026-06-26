#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen51_r04_hybrid_strong_leo_residual_20260624_1535}"
RUN_NAME="${RUN_NAME:-CEN51_R04_HYBRID_STRONG_LEO_RESIDUAL_R010}"
GPU_ID="${GPU_ID:-0}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

RUN_DIR="${RUNS_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.out"
LEGACY_STRONG_SCENARIOS="clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit"
LEO_RESIDUAL_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
HYBRID_TRAIN_SCENARIOS="${LEGACY_STRONG_SCENARIOS},${LEO_RESIDUAL_SCENARIOS}"
HYBRID_SCHEDULE="1@0.98:${HYBRID_TRAIN_SCENARIOS};115@0.82:${HYBRID_TRAIN_SCENARIOS}"

COMMON_CEN51_R04_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_pkl "${ROOT}/Dataset_WigSig/ManySig.pkl"
  --wisig_protocol cvs_day_rx
  --wisig_domain rx_day
  --wisig_equalized 1
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_val_ratio -1
  --wisig_guard_gap 8
  --wisig_split_strategy random
  --wisig_cap_strategy random
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --epochs 200
  --test_eval_policy interval_final
  --test_eval_start_epoch 1
  --test_eval_interval 10
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios "${LEO_RESIDUAL_SCENARIOS}"
  --sat_eval_max_batches -1
  --arch_family cvsincnet
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
  --use_aug
  --use_concat_sat_channel_aug
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
  --sat_train_scenario clear_leo
  --sat_train_scenarios "${HYBRID_TRAIN_SCENARIOS}"
  --sat_view_prob 1.00
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
  --lambda_group_ce 0.06
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.35
  --groupdro_tau 0.50
  --groupdro_cap 0.65
  --use_proto_memory
  --lambda_proto 0.015
  --proto_momentum 0.95
  --lambda_supcon_id 0.02
  --supcon_temp 0.12
  --lambda_fishr 0.005
  --fishr_min_domains 4
  --generalization_feature z_id
  --collapse_guard
  --collapse_guard_min_epoch 35
  --collapse_guard_best_margin 12.0
  --collapse_guard_max_skipped_delta 2
  --use_ema_ckpt
  --ema_decay 0.999
  --use_swad_ckpt
  --swad_start_epoch 90
  --swad_tolerance 0.8
)

R04_STRONGAUG_ARGS=(
  --primary_udu_weight 0.84
  --concat_sat_ce_weight 1.19
  --pa_orders 1,3,5
  --lambda_group_ce 0.088
  --group_ce_min_domains 4
  --group_ce_top_frac 0.20
  --groupdro_tau 0.37
  --groupdro_cap 0.48
  --lambda_proto 0.016
  --proto_momentum 0.970
  --lambda_supcon_id 0.022
  --lambda_fishr 0.002
  --fishr_min_domains 4
  --use_sat_consistency
  --lambda_sat_cons 0.006
  --sat_cons_start_epoch 118
  --swad_start_epoch 70
  --swad_tolerance 0.34
  --sat_view_schedule "${HYBRID_SCHEDULE}"
)

CMD=(
  env "CUDA_VISIBLE_DEVICES=${GPU_ID}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
  "${COMMON_CEN51_R04_ARGS[@]}"
  "${R04_STRONGAUG_ARGS[@]}"
  --run_name "${RUN_NAME}"
  --latest_save_path "${RUN_DIR}/latest_model.pth"
  --best_save_path "${RUN_DIR}/best_val_model.pth"
  --best_primary_save_path "${RUN_DIR}/best_primary_ood_model.pth"
  --best_unseen_day_unseen_rx_save_path "${RUN_DIR}/best_strict_udu_model.pth"
  --best_worst_rx_save_path "${RUN_DIR}/best_worst_rx_model.pth"
  --ema_save_path "${RUN_DIR}/ema_model.pth"
  --swa_save_path "${RUN_DIR}/swa_model.pth"
  --swad_save_path "${RUN_DIR}/swad_model.pth"
)

echo "[CEN51-R04-HYBRID] run_id=${RUN_ID} run_name=${RUN_NAME} gpu=${GPU_ID} dry_run=${DRY_RUN}"
printf '[CEN51-R04-HYBRID-CMD]'
print_cmd "${CMD[@]}"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

[[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
cd "${ROOT}"

if [[ -e "${RUN_DIR}" || -e "${LOG_PATH}" ]]; then
  mkdir -p "${LOG_ROOT}"
  printf "%s\t%s\t%s\t%s\n" "${RUN_NAME}" "BLOCKED_PATH_COLLISION" "${LOG_PATH}" "${RUN_DIR}" \
    | tee -a "${LOG_ROOT}/blocked.tsv"
  exit 0
fi

current_count="$(gpu_process_count "${GPU_ID}")"
if (( current_count >= MAX_TRAIN_PER_GPU )); then
  mkdir -p "${LOG_ROOT}"
  printf "%s\t%s\tgpu=%s active_count=%s max=%s\n" \
    "${RUN_NAME}" "BLOCKED_CAPACITY" "${GPU_ID}" "${current_count}" "${MAX_TRAIN_PER_GPU}" \
    | tee -a "${LOG_ROOT}/blocked.tsv"
  exit 0
fi

mkdir -p "${LOG_ROOT}" "${RUN_DIR}"
nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
pid="$!"
printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU_ID}" "${pid}" "${LOG_PATH}" "${RUN_DIR}" \
  | tee -a "${LOG_ROOT}/launch_pids.tsv"
