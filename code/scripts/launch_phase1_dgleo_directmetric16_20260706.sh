#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_dgleo_directmetric16_20260706}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
LAUNCH_SETTLE_SECONDS="${LAUNCH_SETTLE_SECONDS:-12}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATES="${ONLY_CANDIDATES:-}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATES="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

validate_source_wisig_pkl() {
  local pkl_path="$1"
  local lower
  lower="$(printf "%s" "${pkl_path}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    *manytx.pkl*|*manyrx.pkl*|*singleday.pkl*|*new_wisig*|*target*|*unknown*)
      echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO directmetric16: ${pkl_path}" >&2
      exit 4
      ;;
  esac
  if [[ "${lower}" != *manysig.pkl ]]; then
    echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO directmetric16: expected ManySig.pkl, got ${pkl_path}" >&2
    exit 4
  fi
}

validate_source_wisig_pkl "${WISIG_PKL}"

PHASE1_V2_FLAGS=(
  --phase1_v2_hard_gates true
  --endpoint_accept_policy_id endpoint_accept_v1
  --endpoint_threshold_source source_val_only
  --endpoint_calibration_split source_val
  --loss_gate_exported false
  --tail_safety_state_machine true
  --tail_stop_blocks_final true
  --tail_safety_warning_patience 2
  --tail_safety_rollback_patience 1
  --tail_safety_max_rollbacks 1
  --tail_safety_p95_target_deg 54
  --tail_safety_p99_target_deg 70
  --tail_safety_cvar_target_deg 56
  --tail_safety_proxy_vaccept_target 0.35
  --tail_safety_p99_expansion_block_final_delta 2.0
  --tail_safety_p99_expansion_block_best_delta 3.5
  --tail_safety_cvar_expansion_block_final_delta 4.0
  --tail_safety_cvar_expansion_block_best_delta 6.0
  --os_eff_min_budget 0.15
  --phase1_v2_os_eff_all_phases true
  --phase1_v2_guard_blocks_final true
  --u_tri_state_required true
  --u_direct_idle_blocks_promotion true
  --source_episode_density_gate true
  --source_episode_overflow_warn 0.90
  --source_episode_min_local_components 4
  --feasibility_gate true
  --feasibility_stage full
  --feasibility_relaxed_pass false
  --feasibility_local_pass false
)

candidate_enabled() {
  local cid="$1"
  [[ -z "${ONLY_CANDIDATES}" || ",${ONLY_CANDIDATES}," == *",${cid},"* ]]
}

gpu_active_count() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="${gpu}" '$1 == gpu && $3 == "C" { c++ } END { print c + 0 }'
}

wait_for_gpu_slot() {
  local gpu="$1"
  local active
  while true; do
    active="$(gpu_active_count "${gpu}")"
    if [[ "${active}" -lt "${MAX_ACTIVE_PER_GPU}" ]]; then
      return
    fi
    echo "[DM16-WAIT] gpu=${gpu} active=${active} max=${MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local spec="$1"
  local cid gpu seed group route clean_kl sat_kl zid_mse sat_cls sat_cons lambda_domain lambda_adv lambda_orth lambda_cons lambda_group_ce lambda_fishr lambda_ow lambda_zid lambda_source lambda_proxy lambda_direct lambda_soft proxy_start direct_start proxy_virtual dm_virtual proxy_bridge_w proxy_tail_w proxy_source_w proxy_lowden_w proxy_ratio_w dm_source_w dm_proxy_w dm_bridge_w dm_lowden_w dm_tail_w dm_overflow_w dm_ratio_w dm_core_w dm_satpair_w dm_p95 dm_p99 dm_tail_cvar lr
  IFS='|' read -r cid gpu seed group route clean_kl sat_kl zid_mse sat_cls sat_cons lambda_domain lambda_adv lambda_orth lambda_cons lambda_group_ce lambda_fishr lambda_ow lambda_zid lambda_source lambda_proxy lambda_direct lambda_soft proxy_start direct_start proxy_virtual dm_virtual proxy_bridge_w proxy_tail_w proxy_source_w proxy_lowden_w proxy_ratio_w dm_source_w dm_proxy_w dm_bridge_w dm_lowden_w dm_tail_w dm_overflow_w dm_ratio_w dm_core_w dm_satpair_w dm_p95 dm_p99 dm_tail_cvar lr <<< "${spec}"
  if ! candidate_enabled "${cid}"; then
    return
  fi

  local out_dir="${RUNS_ROOT}/${cid}"
  local log_path="${LOG_ROOT}/${cid}.out"
  local proxy_accept_w
  proxy_accept_w="$(awk -v p="${lambda_proxy}" 'BEGIN { printf "%.5f", (p > 0 ? 0.020 + p * 6.0 : 0.000) }')"

  echo "[DM16-CANDIDATE] id=${cid} group=${group} route=${route} algorithm=DGLEO_DIRECTMETRIC16 base=EPOC_CONCAT_SAT_ADV3B02_CORE90_SOFT_E200 phase1_dataset=ManySig_only source_only=1 dg_primary=1 leo_primary=1 concat_sa=1 domain_loss_on=1 adv_loss_on=1 sat_consistency_on=1 concat_sat_mode=full_2b_core_domain concat_sat_full_loss=1 concat_sat_ce_only=0 direct_metric_validation=1 direct_metric_loss_on=1 direct_metric_primary=proxy_vaccept,source_overflow,bridge_accept,low_density_accept,tail_overflow_accept,radius_inter,zid_quantiles phase1_v2_hard_gates=1 endpoint_accept_v1=1 tail_safety_state_machine=1 os_eff_min_budget=0.15 u_tri_state_required=1 feasibility_gate=1 final_export_fail_closed=1 real_unknown_classes_in_training=0 target_receiver_samples_in_training=0 target_unknown_training_count=0 manytx_in_training=0 proxy_unknown_real_tx_calibration=0 virtual_unknown_only=1 stage2_unknown_query_eval_only=1 stage2_success_claim=0 deployment_success_claim=0 gpu=${gpu}"
  CMD=(env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.10
    --unlabeled_ratio 0.70
    --source_val_ratio 0.20
    --baseline_ckpt "${TEACHER_CKPT}"
    --from_scratch false
    --teacher_ckpt "${TEACHER_CKPT}"
    --output_dir "${out_dir}"
    --epochs 200
    --label_epochs 150
    --pseudo_epochs 50
    --lr "${lr}"
    --weight_decay 0.00008
    --batch_size 128
    --eval_batch_size 256
    --best_metric joint_safe
    --enable_joint_safe_guard true
    --joint_guard_require_satellite true
    --joint_guard_min_strict_udu 80
    --joint_guard_min_receiver_floor 68
    --joint_guard_min_sat_floor 73
    --joint_guard_min_sat_strict_floor 66
    --one_epoch_drop_guard_pp 0.8
    --paic_guard_enabled true
    --paic_guard_sat_ce_delta 0.05
    --paic_guard_sat_cons_delta 0.02
    --paic_guard_domain_delta 0.02
    --paic_guard_grad_delta 2.0
    --paic_guard_reliable_drop 0.005
    --paic_guard_cooldown_epochs 1
    --paic_guard_sat_scale 0.62
    "${PHASE1_V2_FLAGS[@]}"
    --teacher_distill_start_epoch 1
    --teacher_distill_warmup_epochs 30
    --teacher_distill_temperature 2.5
    --lambda_teacher_clean_kl "${clean_kl}"
    --lambda_teacher_sat_kl "${sat_kl}"
    --lambda_teacher_zid_mse "${zid_mse}"
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_start_epoch 1
    --ow_feat_warmup_epochs 40
    --ow_feat_radius_deg 20
    --ow_feat_inter_margin_deg 58
    --ow_feat_sample_margin_deg 5
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight 0.08
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight 0.020
    --ow_feat_vacuum_width_deg 4
    --lambda_zid_compact "${lambda_zid}"
    --zid_compact_start_epoch 1
    --zid_compact_warmup_epochs 40
    --zid_compact_radius_deg 35
    --zid_compact_cvar_alpha 0.88
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.34
    --zid_compact_cvar_weight 0.36
    --lambda_source_episode "${lambda_source}"
    --source_episode_start_epoch 18
    --source_episode_warmup_epochs 45
    --source_episode_radius_cap_deg 18
    --source_episode_radius_mode min_three_sigma_core
    --source_episode_core_quantile 0.72
    --source_episode_min_sigma_deg 2
    --source_episode_mixup_weight 0.020
    --source_episode_mixup_hard_k 2
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs 45
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count "${proxy_virtual}"
    --proxy_unknown_virtual_mode hard
    --proxy_unknown_energy_margin 0.35
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_placeholder_weight 0.05
    --proxy_unknown_virtual_detach true
    --proxy_unknown_vacuum_weight 0.030
    --proxy_unknown_vacuum_width_deg 5
    --proxy_unknown_vacuum_hard_k 2
    --proxy_unknown_vacuum_radius_deg 34
    --proxy_unknown_core_quantile 0.70
    --proxy_unknown_accept_quantile 0.82
    --proxy_unknown_tail_quantile 0.90
    --proxy_unknown_overflow_quantile 0.97
    --proxy_unknown_component_radius_mode core_quantile
    --proxy_unknown_component_radius_quantile 0.70
    --proxy_unknown_vaccept_weight "${proxy_accept_w}"
    --proxy_unknown_core_accept_weight 0.035
    --proxy_unknown_component_gate_weight 0.060
    --proxy_unknown_tail_quarantine_weight "${proxy_tail_w}"
    --proxy_unknown_source_safe_weight "${proxy_source_w}"
    --proxy_unknown_bridge_accept_weight "${proxy_bridge_w}"
    --proxy_unknown_shell_outward_accept_weight 0.075
    --proxy_unknown_low_density_accept_weight "${proxy_lowden_w}"
    --proxy_unknown_energy_margin_quantile_weight 0.080
    --proxy_unknown_radius_budget_weight 0.090
    --proxy_unknown_radius_inter_ratio_weight "${proxy_ratio_w}"
    --proxy_unknown_vaccept_cvar_alpha 0.20
    --proxy_unknown_unknown_margin 0.10
    --proxy_unknown_known_margin 0.04
    --proxy_unknown_energy_softplus_temperature 0.04
    --proxy_unknown_accept_softplus_temperature 0.035
    --proxy_unknown_bridge_accept_target 0.18
    --proxy_unknown_shell_outward_accept_target 0.22
    --proxy_unknown_tail_accept_target 0.32
    --proxy_unknown_overflow_accept_target 0.18
    --proxy_unknown_energy_margin_q 0.08
    --proxy_unknown_energy_margin_target 0.10
    --proxy_unknown_radius_budget_deg 15
    --proxy_unknown_radius_max_budget_deg 24
    --proxy_unknown_radius_inter_ratio_target 0.78
    --proxy_unknown_density_temperature_deg 3
    --proxy_unknown_component_temperature_deg 3
    --proxy_unknown_component_margin_deg 4
    --proxy_unknown_component_margin_temperature_deg 3
    --proxy_unknown_shell_width_deg 4
    --lambda_direct_metric_accept "${lambda_direct}"
    --direct_metric_start_epoch "${direct_start}"
    --direct_metric_warmup_epochs 35
    --direct_metric_virtual_count "${dm_virtual}"
    --direct_metric_virtual_mode hard
    --direct_metric_virtual_detach true
    --direct_metric_core_quantile 0.70
    --direct_metric_accept_quantile 0.80
    --direct_metric_tail_quantile 0.90
    --direct_metric_overflow_quantile 0.97
    --direct_metric_zid_p50_target_deg 28
    --direct_metric_zid_p95_target_deg "${dm_p95}"
    --direct_metric_zid_p99_target_deg "${dm_p99}"
    --direct_metric_zid_tail_cvar_target_deg "${dm_tail_cvar}"
    --direct_metric_source_overflow_target 0.45
    --direct_metric_proxy_vaccept_target 0.35
    --direct_metric_bridge_accept_target 0.25
    --direct_metric_low_density_accept_target 0.10
    --direct_metric_tail_accept_target 0.35
    --direct_metric_overflow_accept_target 0.20
    --direct_metric_radius_inter_ratio_target 0.85
    --direct_metric_core_accept_target 0.82
    --direct_metric_sat_pair_target_deg 10
    --direct_metric_zid_quantile_weight 1.00
    --direct_metric_source_overflow_weight "${dm_source_w}"
    --direct_metric_proxy_vaccept_weight "${dm_proxy_w}"
    --direct_metric_bridge_accept_weight "${dm_bridge_w}"
    --direct_metric_low_density_accept_weight "${dm_lowden_w}"
    --direct_metric_tail_accept_weight "${dm_tail_w}"
    --direct_metric_overflow_accept_weight "${dm_overflow_w}"
    --direct_metric_radius_inter_ratio_weight "${dm_ratio_w}"
    --direct_metric_core_accept_weight "${dm_core_w}"
    --direct_metric_sat_pair_weight "${dm_satpair_w}"
    --direct_metric_quantile_temperature_deg 3
    --direct_metric_accept_temperature 0.04
    --direct_metric_component_temperature_deg 3
    --direct_metric_density_temperature_deg 3
    --direct_metric_component_margin_deg 4
    --direct_metric_source_margin_deg 2
    --direct_metric_shell_width_deg 4
    --direct_metric_accept_cvar_alpha 0.20
    --lambda_soft_unknown_mixup "${lambda_soft}"
    --soft_unknown_mixup_start_epoch "${proxy_start}"
    --soft_unknown_mixup_warmup_epochs 45
    --soft_unknown_mixup_count 24
    --soft_unknown_mixup_order 3
    --soft_unknown_mixup_alpha 0.45
    --soft_unknown_mixup_energy_margin 0.35
    --soft_unknown_mixup_ce_weight 0.10
    --soft_unknown_mixup_energy_weight 0.25
    --soft_unknown_mixup_vacuum_weight 0.020
    --soft_unknown_mixup_vacuum_width_deg 5
    --phase2_export_prototypes true
    --phase2_export_path "${out_dir}/phase1_source_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components 6
    --phase2_fuse_merge_angle_deg 1.7
    --phase2_fuse_radius_cap_deg 12
    --phase2_fuse_tail_abs_deg 15
    --phase2_fuse_accept_policy local_component
    --phase2_fuse_accept_radius_key p95
    --phase2_fuse_max_p95_increase_deg 0.6
    --phase2_fuse_keep_tail_sentinel true
    --phase2_fuse_tail_auto_accept false
    --phase2_fuse_global_ball_accept false
    --test_eval_policy interval_final
    --test_eval_start_epoch 1
    --test_eval_interval 10
    --test_eval_final_window 30
    --test_eval_final_interval 2
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --no_concat_sat_ce_only
    --concat_sat_ce_weight 1.0
    --concat_sat_start_epoch 1
    --sat_view_prob 1.0
    --sat_view_seed "${seed}"
    --sat_train_scenario leo_clear_weak
    --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_view_schedule "1@0.45:leo_clear_weak;31@0.72:leo_clear_weak,leo_low_elev_weak,leo_rain_weak;91@0.88:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --sat_cons_start_epoch 12
    --lambda_sat_cls "${sat_cls}"
    --lambda_sat_cons "${sat_cons}"
    --lambda_u 0.12
    --lambda_ent 0.008
    --lambda_domain "${lambda_domain}"
    --lambda_adv "${lambda_adv}"
    --lambda_orth "${lambda_orth}"
    --lambda_cons "${lambda_cons}"
    --lambda_group_ce "${lambda_group_ce}"
    --lambda_fishr "${lambda_fishr}"
    --tau_min 0.93
    --tau_max 0.985
    --pseudo_quantile 0.88
    --use_ema_teacher true
    --ema_decay 0.999
    --eval_sat_channel true
    --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_eval_max_batches -1
    --device cuda:0
    --seed "${seed}")
  printf "[DM16-CMD] "
  printf "%q " "${CMD[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${out_dir}" "${LOG_ROOT}"
  if [[ -s "${log_path}" ]]; then
    echo "[ERROR] refusing to overwrite existing log: ${log_path}" >&2
    exit 3
  fi
  nohup "${CMD[@]}" > "${log_path}" 2>&1 &
  echo "[DM16-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${log_path}"
  sleep "${LAUNCH_SETTLE_SECONDS}"
}

CANDIDATES=(
  "DGLEO_DM_P0A_CORETAIL_A|0|707101|P0A|core_tail_source_overflow|2.45|1.25|0.500|0.82|0.055|1.20|0.20|0.055|0.095|0.22|0.045|0.0052|0.060|0.020|0.0030|0.0070|0.00003|70|25|48|48|0.055|0.150|0.300|0.060|0.080|1.35|0.70|0.60|0.60|1.35|1.40|0.70|0.25|0.25|54|70|56|0.000046"
  "DGLEO_DM_P0A_CORETAIL_B|0|707102|P0A|core_tail_source_overflow_strict|2.50|1.28|0.500|0.82|0.058|1.20|0.20|0.055|0.095|0.22|0.045|0.0055|0.064|0.022|0.0032|0.0085|0.00003|65|18|56|56|0.060|0.170|0.340|0.065|0.090|1.60|0.75|0.65|0.65|1.55|1.65|0.85|0.28|0.30|52|68|54|0.000045"
  "DGLEO_DM_P0B_BRIDGE_A|1|707111|P0B|bridge_proxy_direct|2.35|1.20|0.470|0.80|0.055|1.25|0.22|0.060|0.100|0.24|0.050|0.0055|0.058|0.018|0.0035|0.0075|0.00003|60|25|56|64|0.110|0.120|0.280|0.075|0.090|0.90|1.20|1.60|0.80|0.90|0.90|0.80|0.30|0.25|54|70|56|0.000046"
  "DGLEO_DM_P0B_BRIDGE_B|1|707112|P0B|bridge_proxy_cvar_high|2.40|1.25|0.490|0.82|0.058|1.25|0.22|0.060|0.100|0.24|0.050|0.0058|0.060|0.018|0.0040|0.0095|0.00004|55|18|64|72|0.135|0.135|0.300|0.085|0.100|0.95|1.50|2.00|0.95|0.95|1.00|0.90|0.32|0.30|52|68|54|0.000045"
  "DGLEO_DM_P0D_RADIUS_A|2|707121|P0D|radius_inter_budget|2.35|1.20|0.470|0.80|0.055|1.25|0.20|0.060|0.100|0.24|0.050|0.0058|0.066|0.020|0.0032|0.0080|0.00003|65|25|56|56|0.080|0.145|0.310|0.065|0.130|0.95|0.90|0.90|0.70|1.10|1.15|1.60|0.28|0.25|52|68|54|0.000045"
  "DGLEO_DM_P0D_RADIUS_B|2|707122|P0D|radius_inter_tight_sat_safe|2.45|1.28|0.500|0.84|0.060|1.25|0.20|0.060|0.100|0.24|0.050|0.0060|0.070|0.022|0.0034|0.0090|0.00003|60|18|64|64|0.085|0.155|0.330|0.070|0.150|1.00|1.00|0.95|0.75|1.25|1.25|1.85|0.30|0.35|50|66|52|0.000044"
  "DGLEO_DM_P1C_SATPAIR_A|3|707131|P1C|sat_pair_tail_guard|2.45|1.30|0.520|0.86|0.065|1.20|0.20|0.055|0.095|0.22|0.045|0.0052|0.060|0.018|0.0030|0.0070|0.00003|70|25|48|48|0.060|0.140|0.300|0.060|0.080|1.00|0.80|0.70|0.70|1.00|1.05|0.75|0.25|0.75|54|70|56|0.000046"
  "DGLEO_DM_P1C_SATPAIR_B|3|707132|P1C|sat_pair_stronger_floor|2.55|1.38|0.540|0.88|0.070|1.20|0.20|0.055|0.095|0.22|0.045|0.0054|0.062|0.018|0.0032|0.0080|0.00003|65|18|56|56|0.065|0.150|0.320|0.065|0.090|1.10|0.85|0.75|0.75|1.10|1.15|0.85|0.28|1.00|52|68|54|0.000045"
  "DGLEO_DM_P0C_BAL_A|4|707141|P0C|balanced_direct_metric|2.45|1.25|0.500|0.84|0.060|1.35|0.20|0.070|0.105|0.26|0.055|0.0060|0.064|0.020|0.0035|0.0080|0.00003|60|22|56|64|0.095|0.150|0.320|0.075|0.110|1.20|1.10|1.20|0.90|1.20|1.25|1.15|0.30|0.50|52|68|54|0.000045"
  "DGLEO_DM_P0C_BAL_B|4|707142|P0C|balanced_kdhi_direct_metric|2.55|1.32|0.530|0.86|0.065|1.35|0.20|0.070|0.105|0.26|0.055|0.0062|0.066|0.020|0.0038|0.0095|0.00004|55|18|64|72|0.105|0.160|0.340|0.080|0.120|1.30|1.25|1.35|1.00|1.30|1.35|1.25|0.32|0.60|50|66|52|0.000044"
  "DGLEO_DM_P1A_LATE_A|5|707151|P1A|late_pseudo_tail_guard|2.45|1.25|0.500|0.82|0.055|1.25|0.20|0.060|0.100|0.24|0.050|0.0055|0.062|0.020|0.0033|0.0075|0.00003|80|45|56|64|0.085|0.150|0.320|0.070|0.100|1.25|1.00|1.10|0.85|1.25|1.30|1.05|0.32|0.35|52|68|54|0.000045"
  "DGLEO_DM_P1A_LATE_B|5|707152|P1A|late_strong_final_stability|2.50|1.30|0.520|0.84|0.060|1.25|0.20|0.060|0.100|0.24|0.050|0.0058|0.064|0.022|0.0036|0.0090|0.00004|80|40|64|72|0.095|0.165|0.350|0.080|0.115|1.40|1.15|1.25|0.95|1.40|1.45|1.20|0.34|0.45|50|66|52|0.000044"
  "DGLEO_DM_P1B_FLOOR_A|6|707161|P1B|receiver_floor_protected|2.45|1.30|0.520|0.86|0.065|1.45|0.18|0.075|0.110|0.28|0.060|0.0055|0.060|0.018|0.0030|0.0070|0.00003|70|25|48|56|0.070|0.135|0.300|0.060|0.085|1.00|0.90|0.85|0.75|1.05|1.10|0.85|0.25|0.60|54|70|56|0.000046"
  "DGLEO_DM_P1B_FLOOR_B|6|707162|P1B|receiver_floor_sat_stronger|2.55|1.40|0.540|0.88|0.070|1.55|0.18|0.080|0.115|0.30|0.065|0.0058|0.062|0.018|0.0032|0.0080|0.00003|65|22|56|64|0.075|0.145|0.320|0.065|0.095|1.10|1.00|0.95|0.80|1.15|1.20|0.95|0.28|0.75|52|68|54|0.000045"
  "DGLEO_DM_P0E_STRONG_A|7|707171|P0E|strong_direct_reject_upper|2.50|1.30|0.520|0.84|0.060|1.35|0.20|0.070|0.105|0.26|0.055|0.0065|0.070|0.024|0.0045|0.0110|0.00005|50|15|72|80|0.145|0.190|0.380|0.100|0.150|1.70|1.60|2.10|1.20|1.60|1.70|1.70|0.36|0.70|48|64|50|0.000043"
  "DGLEO_DM_P0E_STRONG_B|7|707172|P0E|strong_old_protected_direct|2.65|1.42|0.560|0.88|0.070|1.35|0.18|0.070|0.105|0.26|0.055|0.0062|0.068|0.022|0.0042|0.0100|0.00005|55|18|72|80|0.130|0.175|0.360|0.095|0.140|1.55|1.45|1.90|1.10|1.50|1.60|1.55|0.40|0.85|50|66|52|0.000043"
)

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[DM16] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CANDIDATES[@]} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} teacher=ADV3B02_CORE90_SOFT_E200 base=EPOC_CONCAT_SAT_DIRECT_METRIC phase1_dataset=ManySig_only source_only=1 dg_primary=1 leo_primary=1 domain_loss_on=1 direct_metric_validation=1 concat_sat_mode=full_2b_core_domain concat_sat_ce_only=0 phase1_v2_hard_gates=1 endpoint_accept_v1=1 tail_safety_state_machine=1 os_eff_min_budget=0.15 u_tri_state_required=1 feasibility_gate=1 final_export_fail_closed=1 stage2_success_claim=0 deployment_success_claim=0 only=${ONLY_CANDIDATES:-ALL}"

for spec in "${CANDIDATES[@]}"; do
  launch_candidate "${spec}"
done

echo "[DM16-SUBMIT-COMPLETE] run_id=${RUN_ID}"
