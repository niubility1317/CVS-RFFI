#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_dgleo_osfix16_20260707}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
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
      echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO osfix16: ${pkl_path}" >&2
      exit 4
      ;;
  esac
  if [[ "${lower}" != *manysig.pkl ]]; then
    echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO osfix16: expected ManySig.pkl, got ${pkl_path}" >&2
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
    echo "[OSFIX16-WAIT] gpu=${gpu} active=${active} max=${MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local spec="$1"
  local cid gpu seed group route u_dom u_adv u_sat u_dm u_q u_dm_start u_min dm_lambda dm_source dm_proxy dm_bridge dm_lowden dm_tail dm_overflow dm_ratio dm_satpair sat_cls sat_cons clean_kl sat_kl zid_mse domain_w adv_w source_w proxy_w q_target q_accept_q dm_p95 dm_p99 dm_tail_cvar lr
  IFS='|' read -r cid gpu seed group route u_dom u_adv u_sat u_dm u_q u_dm_start u_min dm_lambda dm_source dm_proxy dm_bridge dm_lowden dm_tail dm_overflow dm_ratio dm_satpair sat_cls sat_cons clean_kl sat_kl zid_mse domain_w adv_w source_w proxy_w q_target q_accept_q dm_p95 dm_p99 dm_tail_cvar lr <<< "${spec}"
  if ! candidate_enabled "${cid}"; then
    return
  fi

  local out_dir="${RUNS_ROOT}/${cid}"
  local log_path="${LOG_ROOT}/${cid}.out"
  local proxy_vaccept_w
  proxy_vaccept_w="$(awk -v p="${proxy_w}" 'BEGIN { printf "%.5f", (p > 0 ? 0.030 + p * 8.0 : 0.000) }')"

  echo "[OSFIX16-CANDIDATE] id=${cid} group=${group} route=${route} algorithm=DGLEO_OSFIX16 base=EPOC_CONCAT_SAT_DIRECT_METRIC_UOPT phase1_dataset=ManySig_only source_only=1 dg_primary=1 leo_primary=1 concat_sa=1 concat_sat_mode=full_2b_core_domain concat_sat_full_loss=1 concat_sat_ce_only=0 direct_open_set_metric_loss=1 direct_metric_primary=proxy_vaccept,source_overflow,bridge_accept,low_density_accept,tail_overflow_accept,radius_inter,zid_quantiles unlabeled_domain_supervision=1 unlabeled_satellite_consistency=1 unlabeled_direct_metric_accept=1 unlabeled_quarantine_accept=1 domain_loss_on=1 adv_loss_on=1 phase1_v2_hard_gates=1 endpoint_accept_v1=1 tail_safety_state_machine=1 os_eff_min_budget=0.15 u_tri_state_required=1 feasibility_gate=1 final_export_fail_closed=1 real_unknown_classes_in_training=0 target_receiver_samples_in_training=0 target_unknown_training_count=0 manytx_in_training=0 proxy_unknown_real_tx_calibration=0 virtual_unknown_only=1 stage2_unknown_query_eval_only=1 stage2_success_claim=0 deployment_success_claim=0 gpu=${gpu}"

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
    --lambda_open_world_feat 0.0060
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
    --lambda_zid_compact 0.064
    --zid_compact_start_epoch 1
    --zid_compact_warmup_epochs 40
    --zid_compact_radius_deg 35
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
    --proxy_unknown_start_epoch 60
    --proxy_unknown_warmup_epochs 45
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count 72
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
    --proxy_unknown_vaccept_weight "${proxy_vaccept_w}"
    --proxy_unknown_core_accept_weight 0.035
    --proxy_unknown_component_gate_weight 0.065
    --proxy_unknown_tail_quarantine_weight 0.170
    --proxy_unknown_source_safe_weight 0.340
    --proxy_unknown_bridge_accept_weight 0.130
    --proxy_unknown_shell_outward_accept_weight 0.090
    --proxy_unknown_low_density_accept_weight 0.095
    --proxy_unknown_energy_margin_quantile_weight 0.085
    --proxy_unknown_radius_budget_weight 0.105
    --proxy_unknown_radius_inter_ratio_weight 0.130
    --proxy_unknown_vaccept_cvar_alpha 0.18
    --proxy_unknown_unknown_margin 0.10
    --proxy_unknown_known_margin 0.04
    --proxy_unknown_energy_softplus_temperature 0.04
    --proxy_unknown_accept_softplus_temperature 0.035
    --proxy_unknown_bridge_accept_target 0.16
    --proxy_unknown_shell_outward_accept_target 0.20
    --proxy_unknown_tail_accept_target 0.30
    --proxy_unknown_overflow_accept_target 0.16
    --proxy_unknown_energy_margin_q 0.08
    --proxy_unknown_energy_margin_target 0.10
    --proxy_unknown_radius_budget_deg 14
    --proxy_unknown_radius_max_budget_deg 22
    --proxy_unknown_radius_inter_ratio_target 0.76
    --proxy_unknown_density_temperature_deg 3
    --proxy_unknown_component_temperature_deg 3
    --proxy_unknown_component_margin_deg 4
    --proxy_unknown_component_margin_temperature_deg 3
    --proxy_unknown_shell_width_deg 4
    --lambda_direct_metric_accept "${dm_lambda}"
    --direct_metric_start_epoch 30
    --direct_metric_warmup_epochs 35
    --direct_metric_virtual_count 80
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
    --direct_metric_radius_inter_ratio_target 0.78
    --direct_metric_core_accept_target 0.82
    --direct_metric_sat_pair_target_deg 9
    --direct_metric_zid_quantile_weight 1.05
    --direct_metric_source_overflow_weight "${dm_source}"
    --direct_metric_proxy_vaccept_weight "${dm_proxy}"
    --direct_metric_bridge_accept_weight "${dm_bridge}"
    --direct_metric_low_density_accept_weight "${dm_lowden}"
    --direct_metric_tail_accept_weight "${dm_tail}"
    --direct_metric_overflow_accept_weight "${dm_overflow}"
    --direct_metric_radius_inter_ratio_weight "${dm_ratio}"
    --direct_metric_core_accept_weight 0.32
    --direct_metric_sat_pair_weight "${dm_satpair}"
    --direct_metric_quantile_temperature_deg 3
    --direct_metric_accept_temperature 0.04
    --direct_metric_component_temperature_deg 3
    --direct_metric_density_temperature_deg 3
    --direct_metric_component_margin_deg 4
    --direct_metric_source_margin_deg 2
    --direct_metric_shell_width_deg 4
    --direct_metric_accept_cvar_alpha 0.18
    --lambda_u 0.12
    --lambda_ent 0.008
    --lambda_u_domain "${u_dom}"
    --lambda_u_adv "${u_adv}"
    --lambda_u_sat_cons "${u_sat}"
    --lambda_u_direct_metric_accept "${u_dm}"
    --lambda_u_quarantine_accept "${u_q}"
    --u_domain_start_epoch 1
    --u_sat_cons_start_epoch 1
    --u_direct_metric_start_epoch "${u_dm_start}"
    --u_direct_metric_min_selected "${u_min}"
    --u_direct_metric_use_sat_pair true
    --u_direct_metric_valid_domain_only true
    --u_quarantine_start_epoch 95
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

  printf "[OSFIX16-CMD] "
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
  echo "[OSFIX16-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${log_path}"
  sleep "${LAUNCH_SETTLE_SECONDS}"
}

CANDIDATES=(
  "DGLEO_OSFIX_CORE_A|0|707801|P0_CORE|uopt_core_safe_plus_quarantine|0.16|0.06|0.24|0.0042|0.0022|120|24|0.0068|1.25|1.05|1.05|0.80|1.15|1.20|1.05|0.55|0.84|0.062|2.55|1.36|0.540|1.30|0.18|0.022|0.0032|0.18|0.80|50|66|52|0.000045"
  "DGLEO_OSFIX_CORE_B|0|707802|P0_CORE|directmetric_radius_safe_plus_quarantine|0.18|0.07|0.24|0.0046|0.0025|115|24|0.0072|1.35|1.15|1.15|0.85|1.25|1.30|1.15|0.60|0.84|0.064|2.50|1.32|0.520|1.32|0.20|0.024|0.0035|0.16|0.78|50|66|52|0.000044"
  "DGLEO_OSFIX_DENSITY_A|1|707811|P0_DENSITY|source_overflow_density_gate|0.18|0.08|0.24|0.0048|0.0024|110|24|0.0076|1.75|1.15|1.20|0.95|1.25|1.45|1.25|0.60|0.84|0.064|2.50|1.32|0.520|1.34|0.20|0.030|0.0038|0.16|0.78|49|64|51|0.000044"
  "DGLEO_OSFIX_DENSITY_B|1|707812|P0_DENSITY|source_overflow_density_strict|0.18|0.08|0.24|0.0050|0.0028|105|24|0.0080|1.95|1.20|1.25|1.00|1.30|1.60|1.35|0.62|0.84|0.064|2.55|1.36|0.540|1.34|0.20|0.034|0.0042|0.15|0.76|48|63|50|0.000043"
  "DGLEO_OSFIX_PROXY_A|2|707821|P0_PROXY|proxy_vaccept_cvar_safe|0.18|0.08|0.26|0.0050|0.0024|110|24|0.0078|1.35|1.75|1.20|0.95|1.25|1.35|1.25|0.70|0.86|0.068|2.60|1.42|0.560|1.34|0.20|0.026|0.0046|0.16|0.78|50|65|51|0.000043"
  "DGLEO_OSFIX_PROXY_B|2|707822|P0_PROXY|proxy_vaccept_cvar_strict|0.20|0.09|0.26|0.0054|0.0028|105|24|0.0082|1.45|2.05|1.25|1.00|1.35|1.45|1.35|0.72|0.86|0.068|2.65|1.46|0.570|1.36|0.22|0.030|0.0050|0.14|0.76|49|64|50|0.000042"
  "DGLEO_OSFIX_BRIDGE_A|3|707831|P0_BRIDGE|bridge_low_density_shell|0.18|0.08|0.24|0.0048|0.0026|110|24|0.0078|1.35|1.20|1.85|1.35|1.25|1.35|1.30|0.65|0.84|0.064|2.50|1.32|0.520|1.35|0.22|0.026|0.0042|0.16|0.78|50|65|51|0.000043"
  "DGLEO_OSFIX_BRIDGE_B|3|707832|P0_BRIDGE|bridge_low_density_strict|0.20|0.09|0.24|0.0052|0.0030|105|24|0.0082|1.45|1.25|2.15|1.55|1.35|1.45|1.45|0.68|0.84|0.064|2.55|1.36|0.540|1.38|0.22|0.030|0.0048|0.14|0.76|49|64|50|0.000042"
  "DGLEO_OSFIX_TAIL_A|4|707841|P0_TAIL|tail_overflow_radius_clamp|0.18|0.08|0.24|0.0050|0.0026|110|24|0.0080|1.45|1.25|1.30|1.05|1.80|1.85|1.75|0.70|0.84|0.064|2.55|1.36|0.540|1.36|0.20|0.028|0.0042|0.16|0.78|48|63|50|0.000043"
  "DGLEO_OSFIX_TAIL_B|4|707842|P0_TAIL|tail_overflow_radius_strict|0.20|0.09|0.24|0.0054|0.0030|105|24|0.0085|1.55|1.30|1.35|1.10|2.10|2.15|2.00|0.72|0.84|0.064|2.60|1.40|0.550|1.38|0.22|0.032|0.0048|0.14|0.76|47|62|49|0.000042"
  "DGLEO_OSFIX_SATOPEN_A|5|707851|P0_SATOPEN|sat_pair_open_set_floor|0.16|0.06|0.36|0.0048|0.0024|115|24|0.0075|1.30|1.15|1.15|0.90|1.25|1.30|1.20|1.15|0.90|0.078|2.70|1.52|0.580|1.32|0.18|0.024|0.0038|0.16|0.80|50|65|51|0.000043"
  "DGLEO_OSFIX_SATOPEN_B|5|707852|P0_SATOPEN|sat_pair_open_set_strict|0.18|0.07|0.40|0.0052|0.0028|110|24|0.0080|1.40|1.20|1.25|1.00|1.35|1.45|1.30|1.35|0.92|0.082|2.80|1.58|0.600|1.35|0.18|0.028|0.0042|0.14|0.78|49|64|50|0.000042"
  "DGLEO_OSFIX_UQ_A|6|707861|P1_UQ|unlabeled_quarantine_safe|0.18|0.08|0.28|0.0046|0.0035|120|24|0.0072|1.35|1.20|1.25|1.00|1.30|1.40|1.25|0.75|0.86|0.068|2.60|1.42|0.560|1.35|0.20|0.024|0.0040|0.14|0.78|50|65|51|0.000043"
  "DGLEO_OSFIX_UQ_B|6|707862|P1_UQ|unlabeled_quarantine_strict|0.20|0.10|0.30|0.0050|0.0045|115|24|0.0078|1.45|1.30|1.35|1.10|1.45|1.55|1.40|0.80|0.88|0.072|2.70|1.50|0.580|1.40|0.22|0.026|0.0048|0.12|0.76|49|64|50|0.000042"
  "DGLEO_OSFIX_JOINT_A|7|707871|P1_JOINT|joint_balanced_osfix|0.18|0.08|0.30|0.0052|0.0032|110|24|0.0080|1.55|1.55|1.55|1.20|1.55|1.65|1.55|0.95|0.88|0.074|2.70|1.50|0.580|1.40|0.20|0.030|0.0050|0.14|0.76|49|64|50|0.000042"
  "DGLEO_OSFIX_JOINT_B|7|707872|P1_JOINT|joint_strong_osfix|0.20|0.10|0.34|0.0058|0.0040|105|24|0.0086|1.75|1.75|1.80|1.35|1.85|1.95|1.80|1.10|0.90|0.080|2.80|1.58|0.600|1.45|0.22|0.036|0.0058|0.12|0.74|48|63|49|0.000041"
)

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[OSFIX16] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CANDIDATES[@]} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} two_per_gpu=1 teacher=ADV3B02_CORE90_SOFT_E200 base=EPOC_CONCAT_SAT_DIRECT_METRIC_UOPT phase1_dataset=ManySig_only source_only=1 dg_primary=1 leo_primary=1 direct_open_set_metric_loss=1 unlabeled_quarantine_accept=1 concat_sat_mode=full_2b_core_domain concat_sat_ce_only=0 phase1_v2_hard_gates=1 endpoint_accept_v1=1 tail_safety_state_machine=1 os_eff_min_budget=0.15 u_tri_state_required=1 feasibility_gate=1 final_export_fail_closed=1 stage2_success_claim=0 deployment_success_claim=0 only=${ONLY_CANDIDATES:-ALL}"

for spec in "${CANDIDATES[@]}"; do
  launch_candidate "${spec}"
done

echo "[OSFIX16-SUBMIT-COMPLETE] run_id=${RUN_ID}"
