#!/usr/bin/env bash
set -euo pipefail

# Centralized next8 after CEN_C49-C52/A53-A56 completed.
# Evidence anchors: C51 is best clean/satellite/joint, A56 is best risk-adjusted,
# A54 collapsed satellite performance, and A55 collapsed RX8 under stronger rx-chain augmentation.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260530_203118_centralized_next8}"
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
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
)

MIXSTYLE_ARGS=(
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
)

CONCAT_SAT_ARGS=(
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
)

STACK_ARGS=(
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
  --generalization_feature z_id
)

DUAL_WORST_ARGS=(
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
  --lambda_group_ce 0.06
  --group_ce_mode dual_worst
  --group_ce_top_frac 0.30
  --groupdro_tau 0.45
  --groupdro_cap 0.62
  --use_proto_memory
  --lambda_proto 0.015
  --proto_momentum 0.95
  --lambda_supcon_id 0.02
  --supcon_temp 0.12
  --generalization_feature z_id
)

LATE_STABLE_ARGS=(
  --late_stable_start 110
  --late_stable_ramp_epochs 25
  --late_adv_min_scale 0.65
  --late_cons_min_scale 0.45
  --late_group_ce_min_scale 0.60
  --late_aug_min_scale 0.60
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

launch CEN_C57 \
  CEN_C57_c51_late_stable_swad_r010 \
  0 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  "${LATE_STABLE_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.26 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_fishr 0.006 \
  --fishr_min_domains 4 \
  --use_ema_ckpt \
  --ema_decay 0.999 \
  --use_swad_ckpt \
  --swad_start_epoch 90 \
  --swad_tolerance 1.2
sleep 2

launch CEN_C58 \
  CEN_C58_a56_jointfeat_lowfishr_r010 \
  1 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.26 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_proto 0.025 \
  --lambda_supcon_id 0.030 \
  --supcon_temp 0.10 \
  --generalization_feature id_feat_joint \
  --lambda_fishr 0.006 \
  --fishr_min_domains 4 \
  --use_swad_ckpt \
  --swad_start_epoch 95 \
  --swad_tolerance 1.2
sleep 2

launch CEN_C59 \
  CEN_C59_c51_iddomain_dsq_satfloor_r010 \
  2 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.30 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --sat_view_schedule '1@1.00:clear_leo,low_elev_leo,rain_leo;101@1.00:low_elev_leo,rain_leo,storm_mp' \
  --id_freq_stability_mode dsq \
  --domain_freq_stability_mode same \
  --freq_stability_channels 4 \
  --lambda_fishr 0.006 \
  --fishr_min_domains 4
sleep 2

launch CEN_C60 \
  CEN_C60_c51_pafeat_rx8_floor_r010 \
  3 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.24 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --domain_freq_stability_mode dsq \
  --freq_stability_channels 2 \
  --lambda_group_ce 0.08 \
  --group_ce_mode dual_worst \
  --group_ce_top_frac 0.25 \
  --groupdro_tau 0.40 \
  --groupdro_cap 0.70 \
  --use_proto_memory \
  --lambda_proto 0.015 \
  --proto_momentum 0.95 \
  --lambda_supcon_id 0.018 \
  --supcon_temp 0.12 \
  --generalization_feature id_feat_pa \
  --lambda_cls_pa 0.250 \
  --lambda_pa_joint_inv 0.100 \
  --lambda_pa_kl 0.040 \
  --lambda_fishr 0.006 \
  --fishr_min_domains 4
sleep 2

launch CEN_A61 \
  CEN_A61_c51_rxchain008_guard_r010 \
  4 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.26 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --aug_p_rx_chain 0.08 \
  --aug_rx_chain_envs 4 \
  --aug_rx_chain_p_lowpass 0.50 \
  --aug_rx_chain_p_multipath 0.50 \
  --lambda_fishr 0.006 \
  --fishr_min_domains 4
sleep 2

launch CEN_A62 \
  CEN_A62_a56_jointfeat_proto035_latestable_r010 \
  5 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${STACK_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.30 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --lambda_proto 0.035 \
  --lambda_supcon_id 0.040 \
  --supcon_temp 0.10 \
  --generalization_feature id_feat_joint \
  --lambda_fishr 0.008 \
  --fishr_min_domains 4 \
  "${LATE_STABLE_ARGS[@]}" \
  --use_swad_ckpt \
  --swad_start_epoch 90 \
  --swad_tolerance 1.0
sleep 2

launch CEN_A63 \
  CEN_A63_c51_late_storm_bridge_r010 \
  6 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  --primary_udu_weight 0.70 \
  --concat_sat_ce_weight 1.36 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 0.90 \
  --sat_view_schedule '1@0.80:clear_leo,low_elev_leo,rain_leo;121@0.95:low_elev_leo,rain_leo,storm_mp,mixed_orbit' \
  --lambda_fishr 0.006 \
  --fishr_min_domains 4 \
  --mixstyle_stop_epoch 150
sleep 2

launch CEN_A64 \
  CEN_A64_c51_phase_dsq_allguards_r010 \
  7 \
  "${MIXSTYLE_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.28 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit \
  --sat_view_prob 1.00 \
  --id_time_stability_mode phase_delta \
  --domain_time_stability_mode phase_delta \
  --time_stability_channels 4 \
  --lambda_proto 0.020 \
  --lambda_supcon_id 0.025 \
  --lambda_fishr 0.010 \
  --fishr_min_domains 4 \
  --use_swad_ckpt \
  --swad_start_epoch 95 \
  --swad_tolerance 1.3
