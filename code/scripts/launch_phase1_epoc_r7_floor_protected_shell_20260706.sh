#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_epoc_r7_floor_protected_shell_20260706}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
LAUNCH_SETTLE_SECONDS="${LAUNCH_SETTLE_SECONDS:-20}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATES="${ONLY_CANDIDATES:-}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATES="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

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
    echo "[EPOC-R7-WAIT] gpu=${gpu} active=${active} max=${MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

maybe_launch_candidate() {
  local cid="$1"
  if ! candidate_enabled "${cid}"; then
    return
  fi
  launch_candidate "$@"
}

launch_candidate() {
  local cid="$1"
  local gpu="$2"
  local seed="$3"
  local lambda_clean_kl="$4"
  local lambda_sat_kl="$5"
  local lambda_zid_mse="$6"
  local lambda_proxy="$7"
  local lambda_soft_mix="$8"
  local lambda_ow="$9"
  local lambda_source_episode="${10}"
  local lambda_zid_compact="${11}"
  local proxy_virtual_count="${12}"
  local proxy_energy_margin="${13}"
  local proxy_vac="${14}"
  local ow_radius="${15}"
  local ow_inter="${16}"
  local source_radius_cap="${17}"
  local phase2_radius_cap="${18}"
  local proxy_start="${19}"
  local lr="${20}"

  local out_dir="${RUNS_ROOT}/${cid}"
  local log_path="${LOG_ROOT}/${cid}.out"
  echo "[EPOC-R7-CANDIDATE] id=${cid} route=source_only_floor_protected_feature_shell base=ADV3B02_CORE90_SOFT_E200 target_visibility=source_only_ground_training_no_target_receiver target_receiver_samples_in_training=0 target_unknown_training_count=0 real_unknown_classes_in_training=0 source_heldout_proxy_unknown=1 virtual_unknown_shell=1 manytx_in_training=0 old_floor_first=1"
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
    --label_epochs 140
    --pseudo_epochs 60
    --lr "${lr}"
    --weight_decay 0.00008
    --batch_size 128
    --eval_batch_size 256
    --best_metric joint_safe
    --enable_joint_safe_guard true
    --joint_guard_require_satellite true
    --one_epoch_drop_guard_pp 1.2
    --paic_guard_enabled true
    --paic_guard_sat_ce_delta 0.07
    --paic_guard_grad_delta 2.5
    --paic_guard_reliable_drop 0.008
    --paic_guard_cooldown_epochs 1
    --paic_guard_sat_scale 0.68
    --teacher_distill_start_epoch 1
    --teacher_distill_warmup_epochs 24
    --teacher_distill_temperature 2.5
    --lambda_teacher_clean_kl "${lambda_clean_kl}"
    --lambda_teacher_sat_kl "${lambda_sat_kl}"
    --lambda_teacher_zid_mse "${lambda_zid_mse}"
    --use_phase2_ground_prototypes true
    --use_feature_masks true
    --use_txrx_geometry_losses true
    --phase1_distribution_audit_only true
    --use_proto_memory true
    --lambda_proto 0.0080
    --proto_domain_align_weight 0.16
    --proto_margin 0.30
    --proto_push_weight 0.14
    --proto_min_count 2
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_start_epoch 18
    --ow_feat_warmup_epochs 34
    --ow_feat_radius_deg "${ow_radius}"
    --ow_feat_inter_margin_deg "${ow_inter}"
    --ow_feat_sample_margin_deg 9
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight 0.18
    --ow_feat_vacuum_weight 0.22
    --ow_feat_vacuum_width_deg 9
    --ow_feat_vacuum_hard_k 4
    --lambda_zid_compact "${lambda_zid_compact}"
    --zid_compact_start_epoch 8
    --zid_compact_warmup_epochs 30
    --zid_compact_radius_deg 28
    --zid_compact_cvar_alpha 0.92
    --zid_compact_domain_aware true
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs 40
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count "${proxy_virtual_count}"
    --proxy_unknown_virtual_mode legacy_hard
    --proxy_unknown_energy_margin "${proxy_energy_margin}"
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_placeholder_weight 0.22
    --proxy_unknown_virtual_detach true
    --proxy_unknown_vacuum_weight "${proxy_vac}"
    --proxy_unknown_vacuum_width_deg 9
    --proxy_unknown_vacuum_hard_k 4
    --proxy_unknown_vacuum_radius_deg 34
    --proxy_unknown_component_radius_mode core_quantile
    --proxy_unknown_component_radius_quantile 0.70
    --proxy_unknown_vaccept_weight 0.06
    --proxy_unknown_core_accept_weight 0.05
    --proxy_unknown_component_gate_weight 0.05
    --proxy_unknown_tail_quarantine_weight 0.18
    --proxy_unknown_source_safe_weight 0.24
    --proxy_unknown_bridge_accept_weight 0.06
    --proxy_unknown_shell_outward_accept_weight 0.10
    --proxy_unknown_low_density_accept_weight 0.10
    --proxy_unknown_energy_margin_quantile_weight 0.18
    --proxy_unknown_radius_budget_weight 0.14
    --proxy_unknown_radius_inter_ratio_weight 0.22
    --proxy_unknown_vaccept_cvar_alpha 0.20
    --proxy_unknown_unknown_margin 0.14
    --proxy_unknown_known_margin 0.10
    --proxy_unknown_energy_softplus_temperature 0.035
    --proxy_unknown_accept_softplus_temperature 0.035
    --proxy_unknown_bridge_accept_target 0.02
    --proxy_unknown_shell_outward_accept_target 0.04
    --proxy_unknown_tail_accept_target 0.08
    --proxy_unknown_overflow_accept_target 0.12
    --proxy_unknown_energy_margin_q 0.05
    --proxy_unknown_energy_margin_target 0.28
    --proxy_unknown_radius_budget_deg 5
    --proxy_unknown_radius_max_budget_deg 8
    --proxy_unknown_radius_inter_ratio_target 0.10
    --proxy_unknown_density_temperature_deg 2.2
    --proxy_unknown_component_temperature_deg 2.2
    --proxy_unknown_component_margin_deg 6.0
    --proxy_unknown_component_margin_temperature_deg 2.2
    --proxy_unknown_shell_width_deg 5.5
    --lambda_soft_unknown_mixup "${lambda_soft_mix}"
    --soft_unknown_mixup_start_epoch "${proxy_start}"
    --soft_unknown_mixup_warmup_epochs 40
    --soft_unknown_mixup_count 24
    --soft_unknown_mixup_order 4
    --soft_unknown_mixup_alpha 0.72
    --soft_unknown_mixup_energy_margin "${proxy_energy_margin}"
    --soft_unknown_mixup_energy_weight 0.70
    --soft_unknown_mixup_ce_weight 0.05
    --soft_unknown_mixup_vacuum_weight 0.02
    --soft_unknown_mixup_vacuum_width_deg 8
    --soft_unknown_mixup_vacuum_hard_k 4
    --lambda_source_episode "${lambda_source_episode}"
    --source_episode_start_epoch 10
    --source_episode_warmup_epochs 30
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg "${source_radius_cap}"
    --source_episode_radius_mode min_three_sigma_core
    --source_episode_core_quantile 0.70
    --source_episode_min_sigma_deg 2.5
    --source_episode_mixup_weight 0.82
    --source_episode_mixup_hard_k 5
    --phase2_export_prototypes true
    --phase2_export_path "${out_dir}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components 6
    --phase2_fuse_merge_angle_deg 1.8
    --phase2_fuse_radius_cap_deg "${phase2_radius_cap}"
    --phase2_fuse_tail_abs_deg 18
    --phase2_fuse_accept_policy local_component
    --phase2_fuse_accept_radius_key p95
    --phase2_fuse_max_p95_increase_deg 1.0
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
    --concat_sat_ce_only
    --sat_train_scenario leo_clear_weak
    --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_view_schedule "1@0.38:leo_clear_weak;31@0.68:leo_clear_weak,leo_low_elev_weak,leo_rain_weak;91@0.86:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --sat_cons_start_epoch 20
    --lambda_sat_cls 0.66
    --lambda_sat_cons 0.030
    --lambda_u 0.12
    --lambda_ent 0.008
    --lambda_domain 1.0
    --lambda_adv 0.24
    --lambda_orth 0.04
    --lambda_cons 0.080
    --lambda_group_ce 0.20
    --lambda_fishr 0.040
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
  printf "[EPOC-R7-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${out_dir}" "${LOG_ROOT}"
  if [[ -s "${log_path}" ]]; then
    echo "[ERROR] refusing to overwrite existing log: ${log_path}" >&2
    exit 3
  fi
  "${CMD[@]}" > "${log_path}" 2>&1 &
  echo "[EPOC-R7-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${log_path}"
  sleep "${LAUNCH_SETTLE_SECONDS}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[EPOC-R7] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=2 teacher=ADV3B02_CORE90_SOFT_E200 route=source_only_floor_protected_feature_shell real_unknown_classes_in_training=0 target_receiver_samples_in_training=0 manytx_in_training=0 only=${ONLY_CANDIDATES:-ALL}"

maybe_launch_candidate "EPOC_R7_FLOOR_LOCKED_SHELL" 2 706701 1.75 0.58 0.320 0.0030 0.00000 0.0026 0.0078 0.042 32 1.25 0.20 12 104 24 14 55 0.000060
maybe_launch_candidate "EPOC_R7_BALANCED_LOW_DENSITY" 3 706711 1.55 0.64 0.280 0.0045 0.00002 0.0030 0.0068 0.036 48 1.35 0.26 14 100 26 16 60 0.000065

echo "[EPOC-R7-SUBMIT-COMPLETE] run_id=${RUN_ID}"
