#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_dgleo_v2full32_20260707}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-4}"
LAUNCH_SETTLE_SECONDS="${LAUNCH_SETTLE_SECONDS:-8}"
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
      echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO v2full32: ${pkl_path}" >&2
      exit 4
      ;;
  esac
  if [[ "${lower}" != *manysig.pkl ]]; then
    echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO v2full32: expected ManySig.pkl, got ${pkl_path}" >&2
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
  --feasibility_stage audit
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
    echo "[V2FULL32-WAIT] gpu=${gpu} active=${active} max=${MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local spec="$1"
  local cid gpu seed group strength route u_dom u_adv u_sat u_dm u_q u_start u_min dm_lambda dm_source dm_proxy dm_bridge dm_lowden dm_tail dm_overflow dm_ratio dm_satpair sat_cls sat_cons clean_kl sat_kl zid_mse domain_w adv_w source_w proxy_w q_target q_accept_q dm_p95 dm_p99 dm_tail_cvar lr
  IFS='|' read -r cid gpu seed group strength route u_dom u_adv u_sat u_dm u_q u_start u_min dm_lambda dm_source dm_proxy dm_bridge dm_lowden dm_tail dm_overflow dm_ratio dm_satpair sat_cls sat_cons clean_kl sat_kl zid_mse domain_w adv_w source_w proxy_w q_target q_accept_q dm_p95 dm_p99 dm_tail_cvar lr <<< "${spec}"
  if ! candidate_enabled "${cid}"; then
    return
  fi

  local out_dir="${RUNS_ROOT}/${cid}"
  local log_path="${LOG_ROOT}/${cid}.out"
  local proxy_vaccept_w dm_on source_on proxy_on u_domain_on u_sat_on u_dm_on u_q_on
  proxy_vaccept_w="$(awk -v p="${proxy_w}" 'BEGIN { printf "%.5f", (p > 0 ? 0.035 + p * 8.5 : 0.000) }')"
  dm_on="$(awk -v v="${dm_lambda}" 'BEGIN { print ((v + 0) > 0.0) ? 1 : 0 }')"
  source_on="$(awk -v v="${source_w}" 'BEGIN { print ((v + 0) > 0.0) ? 1 : 0 }')"
  proxy_on="$(awk -v v="${proxy_w}" 'BEGIN { print ((v + 0) > 0.0) ? 1 : 0 }')"
  u_domain_on="$(awk -v d="${u_dom}" -v a="${u_adv}" 'BEGIN { print (((d + 0) + (a + 0)) > 0.0) ? 1 : 0 }')"
  u_sat_on="$(awk -v v="${u_sat}" 'BEGIN { print ((v + 0) > 0.0) ? 1 : 0 }')"
  u_dm_on="$(awk -v v="${u_dm}" 'BEGIN { print ((v + 0) > 0.0) ? 1 : 0 }')"
  u_q_on="$(awk -v v="${u_q}" 'BEGIN { print ((v + 0) > 0.0) ? 1 : 0 }')"

  echo "[V2FULL32-CANDIDATE] id=${cid} group=${group} strength=${strength} route=${route} algorithm=DGLEO_V2FULL32 base=EPOC_CONCAT_SAT_OSFIX_V2 phase1_dataset=ManySig_only source_only=1 dg_primary=1 leo_primary=1 concat_sa=1 concat_sat_mode=full_2b_core_domain concat_sat_full_loss=1 concat_sat_ce_only=0 direct_open_set_metric_loss=${dm_on} source_episode_loss=${source_on} proxy_unknown_loss=${proxy_on} direct_metric_primary=proxy_vaccept,source_overflow,bridge_accept,low_density_accept,tail_overflow_accept,radius_inter,zid_quantiles unlabeled_domain_supervision=${u_domain_on} unlabeled_satellite_consistency=${u_sat_on} unlabeled_direct_metric_accept=${u_dm_on} unlabeled_quarantine_accept=${u_q_on} trusted_core_ambiguous_tail_outside_reject=1 domain_loss_on=1 adv_loss_on=1 phase1_v2_hard_gates=1 endpoint_accept_v1=1 tail_safety_state_machine=1 os_eff_min_budget=0.15 u_tri_state_required=1 feasibility_gate=1 feasibility_stage=audit final_export_fail_closed=1 real_unknown_classes_in_training=0 target_receiver_samples_in_training=0 target_unknown_training_count=0 manytx_in_training=0 proxy_unknown_real_tx_calibration=0 virtual_unknown_only=1 stage2_unknown_query_eval_only=1 stage2_success_claim=0 deployment_success_claim=0 gpu=${gpu}"

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
    --run_id "${RUN_ID}"
    --candidate_id "${cid}"
    --base_candidate "ADV3B02_CORE90_SOFT_E200"
    --epochs 200
    --label_epochs 145
    --pseudo_epochs 55
    --lr "${lr}"
    --weight_decay 0.00008
    --batch_size 112
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
    --lambda_domain "${domain_w}"
    --lambda_adv "${adv_w}"
    --lambda_orth 0.060
    --lambda_cons 0.100
    --lambda_group_ce 0.250
    --lambda_fishr 0.055
    --lambda_open_world_feat 0.0065
    --ow_feat_start_epoch 1
    --ow_feat_warmup_epochs 40
    --ow_feat_radius_deg 19
    --ow_feat_inter_margin_deg 60
    --ow_feat_sample_margin_deg 5
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight 0.10
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight 0.024
    --ow_feat_vacuum_width_deg 4
    --lambda_zid_compact 0.070
    --zid_compact_start_epoch 1
    --zid_compact_warmup_epochs 40
    --zid_compact_radius_deg 34
    --zid_compact_cvar_alpha 0.88
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.34
    --zid_compact_cvar_weight 0.36
    --lambda_source_episode "${source_w}"
    --source_episode_start_epoch 18
    --source_episode_warmup_epochs 45
    --source_episode_radius_cap_deg 18
    --source_episode_radius_mode min_three_sigma_core
    --source_episode_core_quantile 0.72
    --source_episode_min_sigma_deg 2
    --source_episode_mixup_weight 0.020
    --source_episode_mixup_hard_k 2
    --lambda_proxy_unknown "${proxy_w}"
    --proxy_unknown_start_epoch 56
    --proxy_unknown_warmup_epochs 45
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count 80
    --proxy_unknown_virtual_mode hard
    --proxy_unknown_energy_margin 0.35
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_placeholder_weight 0.05
    --proxy_unknown_virtual_detach true
    --proxy_unknown_vacuum_weight 0.034
    --proxy_unknown_vacuum_width_deg 5
    --proxy_unknown_vacuum_hard_k 2
    --proxy_unknown_vacuum_radius_deg 32
    --proxy_unknown_core_quantile 0.70
    --proxy_unknown_accept_quantile 0.80
    --proxy_unknown_tail_quantile 0.90
    --proxy_unknown_overflow_quantile 0.97
    --proxy_unknown_component_radius_mode core_quantile
    --proxy_unknown_component_radius_quantile 0.70
    --proxy_unknown_vaccept_weight "${proxy_vaccept_w}"
    --proxy_unknown_core_accept_weight 0.040
    --proxy_unknown_component_gate_weight 0.070
    --proxy_unknown_tail_quarantine_weight 0.180
    --proxy_unknown_source_safe_weight 0.360
    --proxy_unknown_bridge_accept_weight 0.140
    --proxy_unknown_shell_outward_accept_weight 0.100
    --proxy_unknown_low_density_accept_weight 0.110
    --proxy_unknown_energy_margin_quantile_weight 0.090
    --proxy_unknown_radius_budget_weight 0.120
    --proxy_unknown_radius_inter_ratio_weight 0.145
    --proxy_unknown_vaccept_cvar_alpha 0.16
    --proxy_unknown_unknown_margin 0.10
    --proxy_unknown_known_margin 0.04
    --proxy_unknown_energy_softplus_temperature 0.04
    --proxy_unknown_accept_softplus_temperature 0.035
    --proxy_unknown_bridge_accept_target 0.14
    --proxy_unknown_shell_outward_accept_target 0.18
    --proxy_unknown_tail_accept_target 0.28
    --proxy_unknown_overflow_accept_target 0.14
    --proxy_unknown_energy_margin_q 0.08
    --proxy_unknown_energy_margin_target 0.10
    --proxy_unknown_radius_budget_deg 14
    --proxy_unknown_radius_max_budget_deg 22
    --proxy_unknown_radius_inter_ratio_target 0.74
    --proxy_unknown_density_temperature_deg 3
    --proxy_unknown_component_temperature_deg 3
    --proxy_unknown_component_margin_deg 4
    --proxy_unknown_component_margin_temperature_deg 3
    --proxy_unknown_shell_width_deg 4
    --lambda_direct_metric_accept "${dm_lambda}"
    --direct_metric_start_epoch 28
    --direct_metric_warmup_epochs 32
    --direct_metric_virtual_count 88
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
    --direct_metric_source_overflow_target 0.40
    --direct_metric_proxy_vaccept_target 0.28
    --direct_metric_bridge_accept_target 0.18
    --direct_metric_low_density_accept_target 0.08
    --direct_metric_tail_accept_target 0.28
    --direct_metric_overflow_accept_target 0.14
    --direct_metric_radius_inter_ratio_target 0.76
    --direct_metric_core_accept_target 0.82
    --direct_metric_sat_pair_target_deg 9
    --direct_metric_zid_quantile_weight 1.10
    --direct_metric_source_overflow_weight "${dm_source}"
    --direct_metric_proxy_vaccept_weight "${dm_proxy}"
    --direct_metric_bridge_accept_weight "${dm_bridge}"
    --direct_metric_low_density_accept_weight "${dm_lowden}"
    --direct_metric_tail_accept_weight "${dm_tail}"
    --direct_metric_overflow_accept_weight "${dm_overflow}"
    --direct_metric_radius_inter_ratio_weight "${dm_ratio}"
    --direct_metric_core_accept_weight 0.34
    --direct_metric_sat_pair_weight "${dm_satpair}"
    --direct_metric_quantile_temperature_deg 3
    --direct_metric_accept_temperature 0.04
    --direct_metric_component_temperature_deg 3
    --direct_metric_density_temperature_deg 3
    --direct_metric_component_margin_deg 4
    --direct_metric_source_margin_deg 2
    --direct_metric_shell_width_deg 4
    --direct_metric_accept_cvar_alpha 0.16
    --lambda_u 0.12
    --lambda_ent 0.008
    --lambda_u_domain "${u_dom}"
    --lambda_u_adv "${u_adv}"
    --lambda_u_sat_cons "${u_sat}"
    --lambda_u_direct_metric_accept "${u_dm}"
    --lambda_u_quarantine_accept "${u_q}"
    --u_domain_start_epoch 1
    --u_sat_cons_start_epoch 1
    --u_direct_metric_start_epoch "${u_start}"
    --u_direct_metric_min_selected "${u_min}"
    --u_direct_metric_use_sat_pair true
    --u_direct_metric_valid_domain_only true
    --u_quarantine_start_epoch "${u_start}"
    --u_quarantine_valid_domain_only true
    --u_quarantine_include_sat_view true
    --u_quarantine_min_count 4
    --u_quarantine_core_quantile 0.70
    --u_quarantine_accept_quantile "${q_accept_q}"
    --u_quarantine_accept_target "${q_target}"
    --u_quarantine_cvar_alpha 0.20
    --u_quarantine_accept_temperature 0.04
    --u_sat_zid_cons_weight 0.30
    --tau_min 0.93
    --tau_max 0.985
    --pseudo_quantile 0.88
    --use_ema_teacher true
    --ema_decay 0.999
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --no_concat_sat_ce_only
    --concat_sat_ce_weight 1.0
    --concat_sat_start_epoch 1
    --sat_view_prob 1.0
    --sat_view_seed "${seed}"
    --sat_train_scenario leo_clear_weak
    --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_view_schedule "1@0.45:leo_clear_weak;31@0.72:leo_clear_weak,leo_low_elev_weak,leo_rain_weak;91@0.90:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --sat_cons_start_epoch 12
    --lambda_sat_cls "${sat_cls}"
    --lambda_sat_cons "${sat_cons}"
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
    --eval_sat_channel true
    --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_eval_max_batches -1
    --device cuda:0
    --seed "${seed}")

  printf "[V2FULL32-CMD] "
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
  echo "[V2FULL32-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${log_path}"
  sleep "${LAUNCH_SETTLE_SECONDS}"
}

CANDIDATES=(
  "DGLEO_V2FULL32_FULL_WEAK|0|707901|G0_FULL_LADDER|weak|full_stack_weak|0.16|0.06|0.26|0.0044|0.0022|118|24|0.0070|1.30|1.15|1.15|0.88|1.25|1.30|1.15|0.60|0.86|0.068|2.60|1.42|0.560|1.32|0.18|0.026|0.0038|0.18|0.80|50|66|52|0.000045"
  "DGLEO_V2FULL32_FULL_STABLE|0|707902|G0_FULL_LADDER|stable|full_stack_stable|0.18|0.08|0.30|0.0050|0.0030|110|24|0.0080|1.55|1.45|1.45|1.10|1.55|1.65|1.45|0.85|0.88|0.074|2.70|1.50|0.580|1.40|0.20|0.034|0.0050|0.14|0.76|49|64|50|0.000043"
  "DGLEO_V2FULL32_FULL_AGGR|0|707903|G0_FULL_LADDER|aggressive|full_stack_aggressive|0.22|0.11|0.34|0.0064|0.0046|100|20|0.0102|2.15|2.10|2.15|1.65|2.25|2.35|2.10|1.20|0.92|0.086|2.82|1.62|0.620|1.55|0.24|0.052|0.0072|0.10|0.72|47|61|48|0.000040"
  "DGLEO_V2FULL32_FULL_AGGR_SAFE|0|707904|G0_FULL_LADDER|aggressive_safe|full_stack_aggressive_floor_guard|0.20|0.09|0.38|0.0058|0.0040|104|24|0.0092|1.90|1.80|1.85|1.40|2.00|2.10|1.88|1.18|0.94|0.090|2.90|1.72|0.650|1.48|0.20|0.044|0.0062|0.12|0.74|48|63|49|0.000041"
  "DGLEO_V2FULL32_DM_OFF|1|707911|G1_DIRECT_LOSS_ABLATION|ablation|direct_metric_off_endpoint_eval|0.18|0.08|0.30|0.0000|0.0000|120|24|0.0000|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.00|0.88|0.074|2.70|1.50|0.580|1.40|0.20|0.034|0.0050|0.14|0.76|54|70|56|0.000043"
  "DGLEO_V2FULL32_DM_RELAXED|1|707912|G1_DIRECT_LOSS_ABLATION|explore|direct_metric_relaxed_targets|0.18|0.08|0.30|0.0046|0.0030|112|24|0.0066|1.20|1.10|1.10|0.85|1.20|1.25|1.12|0.70|0.88|0.074|2.72|1.54|0.590|1.40|0.20|0.034|0.0050|0.14|0.76|54|70|56|0.000043"
  "DGLEO_V2FULL32_DM_PROXY_ALIGNED|1|707913|G1_DIRECT_LOSS_ABLATION|explore|direct_metric_proxy_aligned|0.20|0.09|0.34|0.0058|0.0038|104|24|0.0088|1.55|1.95|1.60|1.25|1.70|1.82|1.65|1.00|0.92|0.084|2.82|1.64|0.630|1.50|0.24|0.038|0.0064|0.12|0.74|49|63|50|0.000041"
  "DGLEO_V2FULL32_DM_HARD|1|707914|G1_DIRECT_LOSS_ABLATION|stress|direct_metric_hard_targets|0.22|0.11|0.36|0.0068|0.0046|96|20|0.0108|2.20|2.10|2.25|1.75|2.35|2.45|2.22|1.25|0.94|0.090|2.86|1.70|0.650|1.58|0.26|0.046|0.0074|0.09|0.70|46|60|48|0.000039"
  "DGLEO_V2FULL32_SOURCE_OFF|2|707921|G2_KNOWN_GEOMETRY|ablation|source_episode_off_density_eval|0.18|0.08|0.30|0.0052|0.0032|112|24|0.0082|1.70|1.50|1.50|1.15|1.65|1.75|1.60|0.92|0.88|0.076|2.72|1.54|0.590|1.42|0.22|0.0000|0.0054|0.13|0.76|49|64|50|0.000043"
  "DGLEO_V2FULL32_SOURCE_FOCUS|2|707922|G2_KNOWN_GEOMETRY|isolation|source_density_focus|0.16|0.07|0.28|0.0036|0.0024|116|28|0.0062|2.25|0.90|1.05|0.82|1.20|1.35|1.45|0.65|0.86|0.070|2.62|1.46|0.570|1.38|0.20|0.060|0.0026|0.16|0.80|51|67|53|0.000044"
  "DGLEO_V2FULL32_SOURCE_STRICT|2|707923|G2_KNOWN_GEOMETRY|stress|source_density_strict|0.20|0.09|0.34|0.0058|0.0038|102|24|0.0092|2.40|1.60|1.70|1.35|1.95|2.15|2.05|1.02|0.92|0.086|2.84|1.66|0.640|1.52|0.24|0.062|0.0058|0.11|0.72|47|61|48|0.000040"
  "DGLEO_V2FULL32_RECEIVER_LOCAL_SAFE|2|707924|G2_KNOWN_GEOMETRY|guarded|receiver_local_component_safe|0.20|0.10|0.40|0.0056|0.0040|104|28|0.0088|2.10|1.55|1.65|1.25|1.85|2.00|1.85|1.12|0.96|0.094|2.96|1.78|0.680|1.56|0.26|0.052|0.0054|0.12|0.74|48|62|49|0.000041"
  "DGLEO_V2FULL32_PROXY_OFF|3|707931|G3_PROXY_BRIDGE|ablation|proxy_unknown_off_endpoint_eval|0.18|0.08|0.30|0.0052|0.0032|112|24|0.0084|1.65|1.60|1.62|1.22|1.70|1.80|1.65|0.92|0.88|0.076|2.70|1.52|0.590|1.42|0.22|0.038|0.0000|0.13|0.76|49|64|50|0.000043"
  "DGLEO_V2FULL32_PROXY_VACCEPT|3|707932|G3_PROXY_BRIDGE|explore|proxy_vaccept_focus|0.18|0.08|0.32|0.0058|0.0038|104|24|0.0090|1.55|2.30|1.60|1.20|1.72|1.82|1.70|1.02|0.90|0.080|2.78|1.60|0.620|1.48|0.24|0.040|0.0080|0.10|0.72|48|62|49|0.000041"
  "DGLEO_V2FULL32_BRIDGE_LOW_DENSITY|3|707933|G3_PROXY_BRIDGE|stress|bridge_low_density_focus|0.20|0.10|0.34|0.0062|0.0042|100|24|0.0098|1.80|1.80|2.70|2.20|2.05|2.15|2.05|1.18|0.94|0.090|2.90|1.72|0.660|1.54|0.26|0.046|0.0068|0.10|0.72|47|61|48|0.000040"
  "DGLEO_V2FULL32_SHELL_TAIL_PROXY|3|707934|G3_PROXY_BRIDGE|stress|shell_tail_proxy_focus|0.20|0.09|0.36|0.0060|0.0042|102|24|0.0096|1.82|2.10|2.10|1.70|2.45|2.55|2.25|1.20|0.96|0.094|2.96|1.78|0.680|1.52|0.24|0.048|0.0074|0.10|0.72|47|61|48|0.000040"
  "DGLEO_V2FULL32_U_OFF|4|707941|G4_U_TRISTATE|ablation|u_branch_off_tri_state_eval|0.0000|0.0000|0.0000|0.0000|0.0000|140|32|0.0080|1.55|1.45|1.50|1.12|1.60|1.72|1.55|0.90|0.88|0.076|2.72|1.54|0.590|1.42|0.22|0.038|0.0054|0.13|0.76|49|64|50|0.000043"
  "DGLEO_V2FULL32_U_DOMAIN_SAT|4|707942|G4_U_TRISTATE|ablation|u_domain_sat_only|0.22|0.11|0.42|0.0000|0.0000|130|32|0.0080|1.55|1.45|1.50|1.12|1.60|1.72|1.55|0.95|0.92|0.086|2.86|1.68|0.650|1.50|0.24|0.038|0.0054|0.13|0.76|49|64|50|0.000042"
  "DGLEO_V2FULL32_U_DIRECT_QUAR|4|707943|G4_U_TRISTATE|isolation|u_direct_quarantine_only|0.04|0.02|0.12|0.0064|0.0052|100|24|0.0090|1.65|1.60|1.68|1.28|1.75|1.88|1.72|0.98|0.88|0.078|2.76|1.58|0.600|1.40|0.20|0.040|0.0058|0.10|0.72|48|62|49|0.000041"
  "DGLEO_V2FULL32_U_TRISTATE_FULL|4|707944|G4_U_TRISTATE|explore|u_tristate_full|0.22|0.11|0.44|0.0066|0.0056|98|24|0.0098|1.85|1.80|1.90|1.45|2.05|2.18|1.98|1.25|0.96|0.096|2.98|1.80|0.680|1.54|0.24|0.050|0.0068|0.10|0.72|47|61|48|0.000040"
  "DGLEO_V2FULL32_SAT_WEAK|5|707951|G5_SAT_DG_STRESS|weak|sat_concat_weak|0.16|0.06|0.30|0.0048|0.0028|116|28|0.0076|1.45|1.35|1.38|1.05|1.52|1.62|1.45|0.85|0.82|0.066|2.58|1.40|0.550|1.36|0.18|0.034|0.0048|0.15|0.78|50|65|51|0.000044"
  "DGLEO_V2FULL32_SAT_STRONG|5|707952|G5_SAT_DG_STRESS|explore|sat_concat_strong|0.18|0.07|0.48|0.0054|0.0038|106|28|0.0086|1.60|1.55|1.58|1.20|1.75|1.88|1.70|1.45|0.98|0.102|2.98|1.82|0.700|1.46|0.20|0.044|0.0060|0.12|0.74|48|62|49|0.000041"
  "DGLEO_V2FULL32_SAT_DOMAIN_ADV|5|707953|G5_SAT_DG_STRESS|stress|sat_domain_adv_strong|0.24|0.16|0.48|0.0058|0.0042|102|28|0.0090|1.70|1.65|1.75|1.35|1.88|2.00|1.85|1.50|0.98|0.104|3.02|1.88|0.720|1.62|0.34|0.046|0.0062|0.11|0.72|48|62|49|0.000040"
  "DGLEO_V2FULL32_SAT_OPEN_PAIR|5|707954|G5_SAT_DG_STRESS|stress|sat_pair_open_set_high|0.20|0.08|0.52|0.0062|0.0046|100|24|0.0098|1.88|1.85|1.92|1.48|2.08|2.20|2.00|1.90|1.02|0.112|3.10|1.98|0.750|1.52|0.22|0.052|0.0072|0.10|0.72|47|61|48|0.000039"
  "DGLEO_V2FULL32_BUDGET_CLOSED|6|707961|G6_GRADIENT_BUDGET|ablation|closed_kd_sat_dominant|0.14|0.05|0.34|0.0032|0.0016|122|32|0.0048|0.90|0.80|0.85|0.65|0.90|0.95|0.85|0.75|0.92|0.088|3.00|1.86|0.700|1.30|0.14|0.020|0.0024|0.18|0.82|54|70|56|0.000045"
  "DGLEO_V2FULL32_BUDGET_BALANCED|6|707962|G6_GRADIENT_BUDGET|stable|os_budget_balanced|0.18|0.08|0.36|0.0054|0.0036|108|24|0.0088|1.70|1.65|1.68|1.28|1.82|1.94|1.78|1.10|0.92|0.086|2.82|1.66|0.630|1.46|0.22|0.042|0.0060|0.12|0.74|48|62|49|0.000042"
  "DGLEO_V2FULL32_BUDGET_OS_HIGH|6|707963|G6_GRADIENT_BUDGET|stress|os_budget_high|0.22|0.11|0.34|0.0068|0.0050|98|20|0.0110|2.35|2.20|2.30|1.78|2.55|2.70|2.45|1.35|0.88|0.074|2.70|1.50|0.580|1.58|0.28|0.066|0.0086|0.08|0.70|46|59|47|0.000038"
  "DGLEO_V2FULL32_BUDGET_OS_HIGH_SAFE|6|707964|G6_GRADIENT_BUDGET|guarded|os_budget_high_kd_sat_protect|0.20|0.09|0.46|0.0062|0.0046|102|24|0.0100|2.05|2.00|2.08|1.58|2.28|2.42|2.20|1.45|1.00|0.108|3.10|1.96|0.740|1.50|0.22|0.056|0.0074|0.10|0.72|47|61|48|0.000039"
  "DGLEO_V2FULL32_EXPORT_LOCAL|7|707971|G7_EXPORT_GATE|explore|local_component_export_strict|0.18|0.08|0.34|0.0056|0.0038|106|24|0.0090|1.90|1.70|1.75|1.35|2.05|2.18|2.00|1.10|0.92|0.086|2.84|1.66|0.640|1.50|0.24|0.052|0.0064|0.12|0.74|48|62|49|0.000041"
  "DGLEO_V2FULL32_EXPORT_TAIL_GATE|7|707972|G7_EXPORT_GATE|explore|tail_gate_delta_stable|0.18|0.08|0.36|0.0058|0.0040|104|24|0.0092|1.80|1.80|1.88|1.42|2.20|2.35|2.10|1.20|0.94|0.092|2.92|1.76|0.680|1.48|0.22|0.050|0.0068|0.11|0.72|47|61|48|0.000040"
  "DGLEO_V2FULL32_EXPORT_FEASIBILITY|7|707973|G7_EXPORT_GATE|stress|feasibility_stress_strict_targets|0.22|0.11|0.38|0.0068|0.0052|96|20|0.0110|2.25|2.10|2.22|1.72|2.55|2.70|2.45|1.42|0.96|0.100|3.00|1.88|0.720|1.62|0.28|0.064|0.0084|0.08|0.70|46|59|47|0.000038"
  "DGLEO_V2FULL32_EXPORT_PROMOTE_SAFE|7|707974|G7_EXPORT_GATE|guarded|promotion_safe_joint_export|0.20|0.09|0.48|0.0062|0.0048|102|24|0.0100|2.00|1.95|2.02|1.55|2.35|2.48|2.25|1.50|1.02|0.112|3.12|2.02|0.760|1.54|0.22|0.056|0.0074|0.10|0.72|47|61|48|0.000039"
)

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[V2FULL32] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CANDIDATES[@]} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} four_per_gpu=1 teacher=ADV3B02_CORE90_SOFT_E200 base=EPOC_CONCAT_SAT_OSFIX_V2 phase1_dataset=ManySig_only source_only=1 dg_primary=1 leo_primary=1 domain_loss_on=1 adv_loss_on=1 direct_open_set_metric_loss=1 unlabeled_domain_supervision=1 unlabeled_satellite_consistency=1 unlabeled_direct_metric_accept=1 unlabeled_quarantine_accept=1 trusted_core_ambiguous_tail_outside_reject=1 concat_sat_mode=full_2b_core_domain concat_sat_ce_only=0 phase1_v2_hard_gates=1 endpoint_accept_v1=1 tail_safety_state_machine=1 os_eff_min_budget=0.15 u_tri_state_required=1 feasibility_gate=1 feasibility_stage=audit final_export_fail_closed=1 stage2_success_claim=0 deployment_success_claim=0 only=${ONLY_CANDIDATES:-ALL}"

for spec in "${CANDIDATES[@]}"; do
  launch_candidate "${spec}"
done

echo "[V2FULL32-SUBMIT-COMPLETE] run_id=${RUN_ID}"
