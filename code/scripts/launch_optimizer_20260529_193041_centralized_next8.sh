#!/usr/bin/env bash
set -euo pipefail

# Next centralized 8-run matrix after optimizer_20260529_143047 completed.
# Evidence anchors: CEN_A05 best strict UDU, CEN_A07 risk-adjusted/overall,
# CEN_C01 satellite floor, CEN_C03 clean/SWAD stability. Candidate IDs follow
# the lane sequence after completed CEN_C01-C04/A05-A08.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260529_193041_centralized_next8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --epochs 170
  --test_eval_policy every_epoch
  --test_eval_start_epoch 81
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
  --sat_eval_max_batches -1
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
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
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
)

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

launch() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${run_name}"
    "$@"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )

  echo "[CENTRALIZED-NEXT8] candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CENTRALIZED-NEXT8-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${ROOT}/train.py" ]]; then
    echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
    exit 2
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local count
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

cd "${ROOT}"

launch CEN_C09 \
  CEN_C09_a07_all5_swad_ce1p24_eval81_r010 \
  0 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.24 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --use_swad_ckpt \
  --swad_start_epoch 81 \
  --swad_interval 1 \
  --swad_tolerance 1.25
sleep 2

launch CEN_C10 \
  CEN_C10_c01_all3_satfloor_ce1p38_swad_r010 \
  1 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.38 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --use_swad_ckpt \
  --swad_start_epoch 81 \
  --swad_interval 1 \
  --swad_tolerance 1.50
sleep 2

launch CEN_C11 \
  CEN_C11_a05_strict_swad_ce1p42_udu72_r010 \
  2 \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.42 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.005 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --use_swad_ckpt \
  --swad_start_epoch 81 \
  --swad_interval 1 \
  --swad_tolerance 1.20
sleep 2

launch CEN_C12 \
  CEN_C12_c03_ema_swad_eval61_lowrain_r010 \
  3 \
  --test_eval_start_epoch 61 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.40 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.02 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --use_ema_ckpt \
  --ema_start_epoch 61 \
  --ema_decay 0.999 \
  --use_swad_ckpt \
  --swad_start_epoch 61 \
  --swad_interval 1 \
  --swad_tolerance 1.00
sleep 2

launch CEN_A13 \
  CEN_A13_a07_all5_groupdro_ce1p22_r010 \
  4 \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.22 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode smooth_dro_capped \
  --group_ce_top_frac 0.35 \
  --groupdro_tau 0.50 \
  --groupdro_cap 0.65
sleep 2

launch CEN_A14 \
  CEN_A14_lowrain_proto_supcon_ce1p36_r010 \
  5 \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.36 \
  --sat_train_scenarios low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --use_proto_memory \
  --lambda_proto 0.02 \
  --proto_momentum 0.95 \
  --lambda_supcon_id 0.03 \
  --supcon_temp 0.12 \
  --generalization_feature z_id
sleep 2

launch CEN_A15 \
  CEN_A15_all5_curric_rescue_ce1p30_r010 \
  6 \
  --primary_udu_weight 0.68 \
  --concat_sat_ce_weight 1.30 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_schedule '1@0.75:clear_leo,low_elev_leo;61@1.00:clear_leo,low_elev_leo,rain_leo;121@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2
sleep 2

launch CEN_A16 \
  CEN_A16_rxchain_microdose_ce1p25_r010 \
  7 \
  --primary_udu_weight 0.68 \
  --concat_sat_ce_weight 1.25 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.015 \
  --fishr_min_domains 4 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --aug_p_rx_chain 0.05 \
  --aug_rx_chain_envs 4 \
  --aug_rx_chain_p_lowpass 0.50 \
  --aug_rx_chain_p_multipath 0.50
