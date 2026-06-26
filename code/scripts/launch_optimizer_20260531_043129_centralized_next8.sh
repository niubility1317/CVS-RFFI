#!/usr/bin/env bash
set -euo pipefail

# Centralized next8 after CEN_C65-C68/A69-A72 completed.
# Evidence anchors: C71 is best clean/SAT/risk-adjusted, fixed A72 is best joint
# but collapses satellite floor, C68 is stable mid-pack, and C67/A70 show SAT CE
# alone is insufficient without clean/RX protection.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260531_043129_centralized_next8}"
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
  --seed 1337
)

CONCAT_SAT_ARGS=(
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
)

MIXSTYLE_LIGHT_ARGS=(
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.38
  --mixstyle_p 0.08
  --mixstyle_late_start 95
  --mixstyle_late_ramp_epochs 35
  --mixstyle_late_min_p 0.015
  --mixstyle_late_min_strength 0.10
)

PHASE_JOINT_ARGS=(
  --id_time_stability_mode phase_delta
  --domain_time_stability_mode off
  --time_stability_channels 2
  --use_proto_memory
  --lambda_proto 0.025
  --proto_momentum 0.95
  --lambda_supcon_id 0.030
  --supcon_temp 0.10
  --generalization_feature id_feat_joint
)

SMOOTH_DRO_ARGS=(
  --domain_freq_stability_mode off
  --lambda_group_ce 0.055
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.28
  --groupdro_tau 0.42
  --groupdro_cap 0.60
)

DUAL_WORST_ARGS=(
  --domain_freq_stability_mode off
  --lambda_group_ce 0.060
  --group_ce_mode dual_worst
  --group_ce_top_frac 0.30
  --groupdro_tau 0.38
  --groupdro_cap 0.56
)

LATE_STABLE_ARGS=(
  --late_stable_start 105
  --late_stable_ramp_epochs 30
  --late_adv_min_scale 0.70
  --late_cons_min_scale 0.50
  --late_group_ce_min_scale 0.70
  --late_aug_min_scale 0.70
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

  if [[ -n "${ONLY_CANDIDATE:-}" && "${candidate_id}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi

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
    --best_worst_rx_save_path "${run_dir}/best_worst_rx_model.pth"
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

launch CEN_C73 \
  CEN_C73_c71_rx8guard_phase_r010 \
  0 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.80 \
  --concat_sat_ce_weight 1.22 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.95 \
  --lambda_fishr 0.000 \
  --collapse_guard_best_margin 14.0 \
  --collapse_guard_max_skipped_delta 2 \
  --use_swad_ckpt \
  --swad_start_epoch 94 \
  --swad_tolerance 0.8
sleep 2

launch CEN_C74 \
  CEN_C74_c71_satfloor_rebalance_r010 \
  1 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --concat_sat_ce_weight 1.30 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 1.00 \
  --sat_view_schedule '1@0.95:clear_leo,low_elev_leo,rain_leo;120@0.85:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.000 \
  --use_swad_ckpt \
  --swad_start_epoch 92 \
  --swad_tolerance 0.9
sleep 2

launch CEN_C75 \
  CEN_C75_a72_nosatcons_concat_recover_r010 \
  2 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  --use_proto_memory \
  --lambda_proto 0.014 \
  --proto_momentum 0.95 \
  --lambda_supcon_id 0.014 \
  --supcon_temp 0.10 \
  --generalization_feature id_feat_joint \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.24 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.90 \
  --sat_view_schedule '1@0.90:clear_leo,low_elev_leo,rain_leo;130@0.75:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.000 \
  "${LATE_STABLE_ARGS[@]}" \
  --use_swad_ckpt \
  --swad_start_epoch 96 \
  --swad_tolerance 1.0
sleep 2

launch CEN_C76 \
  CEN_C76_c68_ema_phase_bridge_r010 \
  3 \
  "${CONCAT_SAT_ARGS[@]}" \
  "${DUAL_WORST_ARGS[@]}" \
  --id_time_stability_mode phase_delta \
  --domain_time_stability_mode off \
  --time_stability_channels 2 \
  --use_proto_memory \
  --lambda_proto 0.018 \
  --proto_momentum 0.95 \
  --lambda_supcon_id 0.018 \
  --supcon_temp 0.11 \
  --generalization_feature id_feat_joint \
  --primary_udu_weight 0.78 \
  --concat_sat_ce_weight 1.20 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.88 \
  --lambda_fishr 0.000 \
  "${LATE_STABLE_ARGS[@]}" \
  --collapse_guard_min_epoch 35 \
  --collapse_guard_best_margin 12.0 \
  --collapse_guard_max_skipped_delta 2 \
  --use_ema_ckpt \
  --ema_decay 0.999 \
  --use_swad_ckpt \
  --swad_start_epoch 88 \
  --swad_tolerance 0.8
sleep 2

launch CEN_A77 \
  CEN_A77_c71_rxchain_style_stress_r010 \
  4 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.76 \
  --concat_sat_ce_weight 1.22 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.90 \
  --aug_p_rx_chain 0.12 \
  --aug_rx_chain_envs 6 \
  --aug_rx_chain_p_lowpass 0.70 \
  --aug_rx_chain_p_multipath 0.70 \
  --lambda_fishr 0.000 \
  --use_swad_ckpt \
  --swad_start_epoch 94 \
  --swad_tolerance 1.0
sleep 2

launch CEN_A78 \
  CEN_A78_c71_microfishr_satguard_r010 \
  5 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.28 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.95 \
  --sat_view_schedule '1@0.90:clear_leo,low_elev_leo,rain_leo;115@0.80:clear_leo,low_elev_leo,rain_leo' \
  --lambda_fishr 0.003 \
  --fishr_min_domains 4 \
  --use_swad_ckpt \
  --swad_start_epoch 94 \
  --swad_tolerance 1.0
sleep 2

launch CEN_A79 \
  CEN_A79_a72_delayed_satcons_floor_r010 \
  6 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  --primary_udu_weight 0.72 \
  --use_sat_consistency \
  --lambda_sat_cls 0.10 \
  --lambda_sat_cons 0.004 \
  --sat_cons_start_epoch 125 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.30 \
  --sat_view_schedule '1@0.00:clear_leo,low_elev_leo,rain_leo;125@0.25:clear_leo,low_elev_leo,rain_leo;150@0.20:clear_leo,low_elev_leo,rain_leo' \
  --use_proto_memory \
  --proto_momentum 0.95 \
  --lambda_proto 0.010 \
  --lambda_supcon_id 0.010 \
  --supcon_temp 0.10 \
  --generalization_feature id_feat_joint \
  --lambda_fishr 0.000 \
  "${LATE_STABLE_ARGS[@]}" \
  --use_swad_ckpt \
  --swad_start_epoch 100 \
  --swad_tolerance 1.2
sleep 2

launch CEN_A80 \
  CEN_A80_c71_idfreq_dsq_micro_r010 \
  7 \
  "${MIXSTYLE_LIGHT_ARGS[@]}" \
  "${CONCAT_SAT_ARGS[@]}" \
  "${SMOOTH_DRO_ARGS[@]}" \
  "${PHASE_JOINT_ARGS[@]}" \
  --id_freq_stability_mode dsq \
  --domain_freq_stability_mode off \
  --freq_stability_channels 2 \
  --primary_udu_weight 0.74 \
  --concat_sat_ce_weight 1.22 \
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo \
  --sat_view_prob 0.90 \
  --lambda_fishr 0.000 \
  --use_swad_ckpt \
  --swad_start_epoch 94 \
  --swad_tolerance 1.0
