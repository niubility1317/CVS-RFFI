#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_dgleo_joint16_20260706}"
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
      echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO joint16: ${pkl_path}" >&2
      exit 4
      ;;
  esac
  if [[ "${lower}" != *manysig.pkl ]]; then
    echo "[ERROR] refusing non-source Phase1 WISIG_PKL for DGLEO joint16: expected ManySig.pkl, got ${pkl_path}" >&2
    exit 4
  fi
}

validate_source_wisig_pkl "${WISIG_PKL}"

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
    echo "[DGLEO-WAIT] gpu=${gpu} active=${active} max=${MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local spec="$1"
  local cid gpu seed group route clean_kl sat_kl zid_mse sat_cls sat_cons lambda_domain lambda_adv lambda_orth lambda_cons lambda_group_ce lambda_fishr lambda_ow lambda_zid lambda_source lambda_proxy lambda_soft proxy_start virtual_count bridge_w tail_w source_safe_w low_density_w radius_ratio_w source_radius phase2_radius lr
  IFS='|' read -r cid gpu seed group route clean_kl sat_kl zid_mse sat_cls sat_cons lambda_domain lambda_adv lambda_orth lambda_cons lambda_group_ce lambda_fishr lambda_ow lambda_zid lambda_source lambda_proxy lambda_soft proxy_start virtual_count bridge_w tail_w source_safe_w low_density_w radius_ratio_w source_radius phase2_radius lr <<< "${spec}"
  if ! candidate_enabled "${cid}"; then
    return
  fi

  local out_dir="${RUNS_ROOT}/${cid}"
  local log_path="${LOG_ROOT}/${cid}.out"
  local proxy_accept_w
  proxy_accept_w="$(awk -v p="${lambda_proxy}" 'BEGIN { printf "%.5f", (p > 0 ? 0.020 + p * 6.0 : 0.000) }')"

  echo "[DGLEO-CANDIDATE] id=${cid} group=${group} route=${route} algorithm=DGLEO_JOINT16 base=EPOC_CONCAT_SAT_ADV3B02_CORE90_SOFT_E200 phase1_dataset=ManySig_only dg_primary=1 leo_primary=1 domain_loss_on=1 sat_consistency_on=1 concat_sat_mode=full_2b_core_domain concat_sat_full_loss=1 concat_sat_ce_only=0 open_boundary_constrained=1 real_unknown_classes_in_training=0 target_receiver_samples_in_training=0 target_unknown_training_count=0 manytx_in_training=0 proxy_unknown_real_tx_calibration=0 virtual_unknown_only=1 stage2_unknown_query_eval_only=1 qknn8_same_row_eval_required=1 stage2_success_claim=0 deployment_success_claim=0 gpu=${gpu}"
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
    --teacher_distill_start_epoch 1
    --teacher_distill_warmup_epochs 30
    --teacher_distill_temperature 2.5
    --lambda_teacher_clean_kl "${clean_kl}"
    --lambda_teacher_sat_kl "${sat_kl}"
    --lambda_teacher_zid_mse "${zid_mse}"
    --use_phase2_ground_prototypes true
    --use_feature_masks true
    --use_txrx_geometry_losses true
    --phase1_distribution_audit_only true
    --use_proto_memory true
    --lambda_proto 0.0150
    --proto_domain_align_weight 0.24
    --proto_margin 0.30
    --proto_push_weight 0.20
    --proto_min_count 2
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_start_epoch 10
    --ow_feat_warmup_epochs 42
    --ow_feat_radius_deg 9
    --ow_feat_inter_margin_deg 120
    --ow_feat_sample_margin_deg 8
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight "${tail_w}"
    --ow_feat_vacuum_weight 0.18
    --ow_feat_vacuum_width_deg 8
    --ow_feat_vacuum_hard_k 5
    --lambda_zid_compact "${lambda_zid}"
    --zid_compact_start_epoch 4
    --zid_compact_warmup_epochs 44
    --zid_compact_radius_deg 20
    --zid_compact_cvar_alpha 0.90
    --zid_compact_domain_aware true
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs 50
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count "${virtual_count}"
    --proxy_unknown_virtual_mode legacy_hard
    --proxy_unknown_energy_margin 1.45
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_placeholder_weight 0.06
    --proxy_unknown_virtual_detach true
    --proxy_unknown_vacuum_weight 0.08
    --proxy_unknown_vacuum_width_deg 8
    --proxy_unknown_vacuum_hard_k 4
    --proxy_unknown_vacuum_radius_deg 24
    --proxy_unknown_component_radius_mode core_quantile
    --proxy_unknown_component_radius_quantile 0.58
    --proxy_unknown_vaccept_weight "${proxy_accept_w}"
    --proxy_unknown_core_accept_weight "${proxy_accept_w}"
    --proxy_unknown_component_gate_weight "${proxy_accept_w}"
    --proxy_unknown_tail_quarantine_weight "${tail_w}"
    --proxy_unknown_source_safe_weight "${source_safe_w}"
    --proxy_unknown_bridge_accept_weight "${bridge_w}"
    --proxy_unknown_shell_outward_accept_weight "${bridge_w}"
    --proxy_unknown_low_density_accept_weight "${low_density_w}"
    --proxy_unknown_energy_margin_quantile_weight "${bridge_w}"
    --proxy_unknown_radius_budget_weight "${radius_ratio_w}"
    --proxy_unknown_radius_inter_ratio_weight "${radius_ratio_w}"
    --proxy_unknown_vaccept_cvar_alpha 0.10
    --proxy_unknown_unknown_margin 0.10
    --proxy_unknown_known_margin 0.08
    --proxy_unknown_energy_softplus_temperature 0.040
    --proxy_unknown_accept_softplus_temperature 0.040
    --proxy_unknown_bridge_accept_target 0.00
    --proxy_unknown_shell_outward_accept_target 0.02
    --proxy_unknown_tail_accept_target 0.02
    --proxy_unknown_overflow_accept_target 0.12
    --proxy_unknown_energy_margin_q 0.05
    --proxy_unknown_energy_margin_target 0.22
    --proxy_unknown_radius_budget_deg 5
    --proxy_unknown_radius_max_budget_deg 8
    --proxy_unknown_radius_inter_ratio_target 0.08
    --proxy_unknown_density_temperature_deg 2.4
    --proxy_unknown_component_temperature_deg 2.4
    --proxy_unknown_component_margin_deg 4.8
    --proxy_unknown_component_margin_temperature_deg 2.4
    --proxy_unknown_shell_width_deg 5.0
    --lambda_soft_unknown_mixup "${lambda_soft}"
    --soft_unknown_mixup_start_epoch "${proxy_start}"
    --soft_unknown_mixup_warmup_epochs 50
    --soft_unknown_mixup_count 16
    --soft_unknown_mixup_order 4
    --soft_unknown_mixup_alpha 0.70
    --soft_unknown_mixup_energy_margin 1.45
    --soft_unknown_mixup_energy_weight 0.18
    --soft_unknown_mixup_ce_weight 0.00
    --soft_unknown_mixup_vacuum_weight 0.00
    --soft_unknown_mixup_vacuum_width_deg 8
    --soft_unknown_mixup_vacuum_hard_k 4
    --lambda_source_episode "${lambda_source}"
    --source_episode_start_epoch 5
    --source_episode_warmup_epochs 44
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg "${source_radius}"
    --source_episode_radius_mode min_three_sigma_core
    --source_episode_core_quantile 0.68
    --source_episode_min_sigma_deg 2.0
    --source_episode_mixup_weight 0.54
    --source_episode_mixup_hard_k 5
    --tail_quarantine true
    --lambda_tail_cvar "${tail_w}"
    --lambda_overflow_cap "${source_safe_w}"
    --phase2_export_prototypes true
    --phase2_export_path "${out_dir}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components 6
    --phase2_fuse_merge_angle_deg 1.7
    --phase2_fuse_radius_cap_deg "${phase2_radius}"
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
    --sat_view_schedule "1@0.42:leo_clear_weak;31@0.70:leo_clear_weak,leo_low_elev_weak,leo_rain_weak;91@0.86:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
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
  printf "[DGLEO-CMD] "
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
  echo "[DGLEO-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${log_path}"
  sleep "${LAUNCH_SETTLE_SECONDS}"
}

CANDIDATES=(
  "DGLEO_J1_BASE_A|0|707001|J1|dg_leo_base_sat_cons|2.20|1.00|0.420|0.74|0.040|1.20|0.20|0.050|0.090|0.24|0.045|0.0048|0.056|0.014|0.0000|0.00000|120|16|0.000|0.040|0.180|0.040|0.040|20|13|0.000050"
  "DGLEO_J1_BASE_B|0|707002|J1|dg_leo_base_fishr_floor|2.10|1.05|0.400|0.78|0.048|1.28|0.18|0.060|0.100|0.28|0.060|0.0052|0.058|0.015|0.0000|0.00000|120|16|0.000|0.045|0.200|0.040|0.045|19|13|0.000049"
  "DGLEO_J2_DOMAIN_A|1|707011|J2|limited_grl_domain_split|2.20|1.05|0.430|0.76|0.045|1.45|0.24|0.075|0.105|0.24|0.050|0.0050|0.056|0.014|0.0000|0.00000|120|16|0.000|0.040|0.190|0.040|0.045|20|13|0.000049"
  "DGLEO_J2_DOMAIN_B|1|707012|J2|limited_grl_domain_stronger|2.15|1.10|0.420|0.76|0.050|1.55|0.28|0.085|0.115|0.24|0.055|0.0050|0.055|0.014|0.0000|0.00000|120|16|0.000|0.040|0.190|0.040|0.045|20|13|0.000048"
  "DGLEO_J7_KD_A|2|707021|J7|old_leo_teacher_protect|2.45|1.25|0.500|0.82|0.055|1.20|0.20|0.055|0.095|0.22|0.045|0.0050|0.056|0.014|0.0010|0.00001|90|24|0.020|0.060|0.220|0.050|0.050|20|13|0.000047"
  "DGLEO_J7_KD_B|2|707022|J7|old_leo_teacher_highsat|2.35|1.35|0.480|0.86|0.060|1.20|0.18|0.055|0.095|0.22|0.045|0.0050|0.054|0.014|0.0010|0.00001|90|24|0.020|0.060|0.220|0.050|0.050|20|13|0.000047"
  "DGLEO_J3_BRIDGE_A|3|707031|J3|bridge_cvar_medium|2.20|1.10|0.430|0.78|0.048|1.25|0.22|0.060|0.100|0.24|0.050|0.0055|0.058|0.016|0.0020|0.00002|70|32|0.075|0.095|0.230|0.060|0.075|19|12|0.000047"
  "DGLEO_J3_BRIDGE_B|3|707032|J3|bridge_cvar_high|2.15|1.12|0.420|0.78|0.050|1.25|0.22|0.060|0.100|0.24|0.050|0.0058|0.060|0.017|0.0025|0.00002|65|40|0.100|0.110|0.240|0.060|0.085|18|12|0.000046"
  "DGLEO_J4_PROXY_A|4|707041|J4|proxy_unknown_leo_medium|2.20|1.10|0.430|0.78|0.050|1.25|0.22|0.060|0.100|0.24|0.050|0.0055|0.058|0.016|0.0035|0.00003|60|48|0.050|0.095|0.240|0.065|0.075|19|12|0.000046"
  "DGLEO_J4_PROXY_B|4|707042|J4|proxy_unknown_leo_dense|2.15|1.12|0.420|0.78|0.052|1.25|0.22|0.060|0.100|0.24|0.050|0.0058|0.060|0.017|0.0045|0.00005|55|64|0.060|0.105|0.250|0.070|0.080|18|12|0.000045"
  "DGLEO_J5_RADIUS_A|5|707051|J5|local_radius_budget_medium|2.20|1.10|0.430|0.78|0.050|1.25|0.22|0.060|0.100|0.24|0.050|0.0058|0.065|0.018|0.0020|0.00002|70|32|0.055|0.105|0.250|0.060|0.105|18|11|0.000046"
  "DGLEO_J5_RADIUS_B|5|707052|J5|local_radius_budget_tight|2.15|1.12|0.420|0.78|0.052|1.25|0.22|0.060|0.100|0.24|0.050|0.0062|0.070|0.020|0.0025|0.00002|65|40|0.060|0.120|0.260|0.065|0.120|17|10|0.000045"
  "DGLEO_J10_BALANCED_A|6|707061|J10|balanced_dg_leo_boundary|2.35|1.20|0.460|0.82|0.055|1.35|0.22|0.070|0.105|0.26|0.055|0.0060|0.064|0.018|0.0030|0.00003|60|48|0.070|0.110|0.260|0.065|0.090|18|12|0.000046"
  "DGLEO_J10_BALANCED_B|6|707062|J10|balanced_dg_leo_boundary_kdhi|2.45|1.25|0.500|0.84|0.060|1.35|0.20|0.070|0.105|0.26|0.055|0.0060|0.062|0.018|0.0030|0.00003|60|48|0.070|0.110|0.270|0.065|0.090|18|12|0.000046"
  "DGLEO_J11_STRONG_A|7|707071|J11|strong_reject_upper_bound|2.35|1.20|0.460|0.82|0.055|1.35|0.22|0.070|0.105|0.26|0.055|0.0065|0.070|0.020|0.0055|0.00005|45|64|0.120|0.150|0.300|0.080|0.120|17|10|0.000044"
  "DGLEO_J11_STRONG_B|7|707072|J11|strong_reject_old_protected|2.55|1.35|0.520|0.84|0.060|1.30|0.20|0.070|0.105|0.26|0.055|0.0062|0.068|0.020|0.0048|0.00005|50|64|0.110|0.145|0.300|0.080|0.115|17|10|0.000044"
)

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[DGLEO-JOINT16] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CANDIDATES[@]} max_active_per_gpu=${MAX_ACTIVE_PER_GPU} teacher=ADV3B02_CORE90_SOFT_E200 base=EPOC_CONCAT_SAT phase1_dataset=ManySig_only source_only=1 dg_primary=1 leo_primary=1 domain_loss_on=1 concat_sat_mode=full_2b_core_domain concat_sat_ce_only=0 stage2_success_claim=0 deployment_success_claim=0 only=${ONLY_CANDIDATES:-ALL}"

for spec in "${CANDIDATES[@]}"; do
  launch_candidate "${spec}"
done

echo "[DGLEO-SUBMIT-COMPLETE] run_id=${RUN_ID}"
