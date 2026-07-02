#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adg_v2_gpu8_20260702}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-1}"
LAUNCH_STABILIZE_SEC="${LAUNCH_STABILIZE_SEC:-35}"
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
    if [[ "${active}" -lt "${STAGE2_MAX_ACTIVE_PER_GPU}" ]]; then
      return
    fi
    echo "[ADG8-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 90
  done
}

set_candidate_defaults() {
  epochs=200
  label_epochs=130
  base_candidate="adv3b02_core90_soft_adg_v2"
  mechanism="B02 anchor with ADG telemetry"
  zid_start=8
  ow_start=12
  source_start=20
  soft_start=25
  proxy_start=45
  warmup=25

  lambda_ow=0.0024
  ow_tail=0.14
  ow_vacuum=0.40
  lambda_zid=0.032
  lambda_proxy=0.0045
  proxy_vacuum=0.55
  proxy_virtual_count=48
  proxy_virtual_mode="hard"
  proxy_core_q=0.90
  proxy_accept_q=0.85
  proxy_tail_q=0.92
  proxy_overflow_q=0.97
  proxy_vaccept_w=1.00
  proxy_core_w=0.45
  proxy_gate_w=0.65
  proxy_tail_w=0.20
  proxy_source_w=0.20
  proxy_cvar_alpha=0.30
  proxy_unknown_margin=0.08
  proxy_known_margin=0.05
  proxy_tau_e=0.04
  proxy_accept_tau=0.04
  proxy_comp_temp=3.0
  proxy_comp_margin=4.0
  proxy_comp_margin_temp=3.0
  proxy_shell_width=4.0

  adg_bridge_w=0.0
  adg_shell_out_w=0.0
  adg_low_density_w=0.0
  adg_energy_q_w=0.0
  adg_radius_w=0.0
  adg_ratio_w=0.0
  adg_bridge_target=0.20
  adg_shell_target=0.25
  adg_tail_target=0.45
  adg_overflow_target=0.25
  adg_energy_q=0.10
  adg_energy_target=0.08
  adg_radius_budget=10.0
  adg_radius_max_budget=15.0
  adg_ratio_target=0.25
  adg_density_temp=3.0

  lambda_soft_mix=0.0045
  soft_count=24
  soft_order=3
  soft_ce=0.60
  soft_vacuum=0.35
  lambda_source=0.0035
  source_mix=0.75
  source_radius=33
  fuse_components=6
  fuse_radius=15.0
  sat_start=80
  lambda_sat_cls=0.68
  lambda_sat_cons=0
  sat_schedule="1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
}

apply_candidate_variant() {
  local cid="$1"
  case "${cid}" in
    ADG8G0_B02_ANCHOR_E200)
      mechanism="B02 anchor rerun with ADG metrics, no new ADG side weight" ;;
    ADG8G1_BRIDGE_CVAR_E200)
      mechanism="direct bridge governance on same-class/inter-class bridge accept"
      adg_bridge_w=0.0030
      adg_bridge_target=0.15
      adg_energy_target=0.10 ;;
    ADG8G2_SHELL_LOW_DENS_E200)
      mechanism="shell/outward plus low-density accept governance"
      adg_shell_out_w=0.0025
      adg_low_density_w=0.0025
      adg_shell_target=0.20
      adg_density_temp=2.5
      proxy_shell_width=6.0
      proxy_gate_w=0.75 ;;
    ADG8G3_ENERGY_Q10_E200)
      mechanism="bottom energy-margin quantile governance for hard unknown"
      adg_energy_q_w=0.0035
      adg_energy_q=0.10
      adg_energy_target=0.10
      proxy_unknown_margin=0.10
      proxy_accept_q=0.82 ;;
    ADG8G4_RADIUS_RATIO_E200)
      mechanism="component radius and radius-to-inter-class-ratio budget"
      adg_radius_w=0.0015
      adg_ratio_w=0.0015
      adg_radius_budget=9.0
      adg_radius_max_budget=14.0
      adg_ratio_target=0.22
      proxy_core_w=0.55 ;;
    ADG8G5_TAIL_OVERFLOW_E200)
      mechanism="core/tail/outside quarantine through accept-aware tail and overflow"
      proxy_tail_w=0.35
      proxy_source_w=0.35
      proxy_accept_q=0.82
      proxy_tail_q=0.90
      proxy_overflow_q=0.95
      adg_tail_target=0.35
      adg_overflow_target=0.18
      lambda_source=0.0040
      source_mix=1.00
      source_radius=31 ;;
    ADG8G6_CONSERVATIVE_ALL_E200)
      mechanism="conservative all-term ADG stack"
      adg_bridge_w=0.0020
      adg_shell_out_w=0.0015
      adg_low_density_w=0.0015
      adg_energy_q_w=0.0020
      adg_radius_w=0.0008
      adg_ratio_w=0.0008
      adg_bridge_target=0.18
      adg_shell_target=0.22
      adg_tail_target=0.40
      adg_overflow_target=0.22
      adg_energy_target=0.09
      adg_radius_budget=10.0
      adg_ratio_target=0.24
      proxy_accept_q=0.82 ;;
    ADG8G7_STRONG_ALL_SAT_E200)
      mechanism="strong ADG stack with satellite-stress guard"
      adg_bridge_w=0.0040
      adg_shell_out_w=0.0030
      adg_low_density_w=0.0030
      adg_energy_q_w=0.0040
      adg_radius_w=0.0012
      adg_ratio_w=0.0012
      adg_bridge_target=0.12
      adg_shell_target=0.18
      adg_tail_target=0.35
      adg_overflow_target=0.18
      adg_energy_target=0.11
      adg_radius_budget=9.0
      adg_radius_max_budget=14.0
      adg_ratio_target=0.22
      proxy_tail_w=0.30
      proxy_source_w=0.30
      proxy_accept_q=0.82
      lambda_zid=0.036
      lambda_ow=0.0028
      ow_vacuum=0.45
      sat_start=70
      lambda_sat_cls=0.74
      sat_schedule="1@0.35:leo_clear_weak;31@0.65:leo_low_elev_weak,leo_rain_weak;81@0.85:leo_clear_weak,leo_low_elev_weak,leo_rain_weak" ;;
    *)
      echo "[ERROR] unknown candidate variant: ${cid}" >&2
      exit 2 ;;
  esac
}

build_command() {
  local cid="$1"
  local gpu="$2"
  local seed="$3"
  local pseudo_epochs=$((epochs - label_epochs))
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.10
    --unlabeled_ratio 0.70
    --source_val_ratio 0.20
    --output_dir "${RUNS_ROOT}/${cid}"
    --run_id "${RUN_ID}"
    --candidate_id "${cid}"
    --base_candidate "${base_candidate}"
    --epochs "${epochs}"
    --label_epochs "${label_epochs}"
    --pseudo_epochs "${pseudo_epochs}"
    --from_scratch true
    --best_metric joint_safe
    --enable_joint_safe_guard true
    --one_epoch_drop_guard_pp 2.0
    --paic_guard_enabled true
    --paic_guard_sat_ce_delta 0.12
    --paic_guard_grad_delta 3.0
    --paic_guard_reliable_drop 0.01
    --paic_guard_cooldown_epochs 1
    --paic_guard_sat_scale 0.75
    --use_phase2_ground_prototypes true
    --use_feature_masks true
    --use_txrx_geometry_losses true
    --use_tx_rx_balanced_sampler false
    --phase1_distribution_audit_only true
    --lambda_tx_proto 0
    --lambda_rx_proto 0
    --lambda_mask_aux 0
    --lambda_tx_supcon_masked 0
    --lambda_rx_supcon_masked 0
    --lambda_txrx_rect 0
    --use_proto_memory true
    --lambda_proto 0.0032
    --proto_domain_align_weight 0.10
    --proto_margin 0.15
    --proto_push_weight 0.10
    --proto_min_count 2
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_start_epoch "${ow_start}"
    --ow_feat_warmup_epochs "${warmup}"
    --ow_feat_radius_deg 12
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_domain_align_weight 0
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight "${ow_tail}"
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight "${ow_vacuum}"
    --ow_feat_vacuum_width_deg 6
    --ow_feat_vacuum_hard_k 3
    --lambda_zid_compact "${lambda_zid}"
    --zid_compact_start_epoch "${zid_start}"
    --zid_compact_warmup_epochs "${warmup}"
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.35
    --zid_compact_cvar_weight 0.35
    --zid_compact_cvar_alpha 0.95
    --zid_compact_radius_deg 40
    --zid_compact_domain_aware true
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs "${warmup}"
    --proxy_unknown_holdout_tx_per_batch 3
    --proxy_unknown_virtual_count "${proxy_virtual_count}"
    --proxy_unknown_virtual_mode "${proxy_virtual_mode}"
    --proxy_unknown_energy_margin 0.0
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_placeholder_weight 0.0
    --proxy_unknown_virtual_detach false
    --proxy_unknown_vacuum_weight "${proxy_vacuum}"
    --proxy_unknown_vacuum_width_deg 5
    --proxy_unknown_vacuum_hard_k 3
    --proxy_unknown_vacuum_radius_deg 40
    --proxy_unknown_core_quantile "${proxy_core_q}"
    --proxy_unknown_accept_quantile "${proxy_accept_q}"
    --proxy_unknown_tail_quantile "${proxy_tail_q}"
    --proxy_unknown_overflow_quantile "${proxy_overflow_q}"
    --proxy_unknown_vaccept_weight "${proxy_vaccept_w}"
    --proxy_unknown_core_accept_weight "${proxy_core_w}"
    --proxy_unknown_component_gate_weight "${proxy_gate_w}"
    --proxy_unknown_tail_quarantine_weight "${proxy_tail_w}"
    --proxy_unknown_source_safe_weight "${proxy_source_w}"
    --proxy_unknown_bridge_accept_weight "${adg_bridge_w}"
    --proxy_unknown_shell_outward_accept_weight "${adg_shell_out_w}"
    --proxy_unknown_low_density_accept_weight "${adg_low_density_w}"
    --proxy_unknown_energy_margin_quantile_weight "${adg_energy_q_w}"
    --proxy_unknown_radius_budget_weight "${adg_radius_w}"
    --proxy_unknown_radius_inter_ratio_weight "${adg_ratio_w}"
    --proxy_unknown_vaccept_cvar_alpha "${proxy_cvar_alpha}"
    --proxy_unknown_unknown_margin "${proxy_unknown_margin}"
    --proxy_unknown_known_margin "${proxy_known_margin}"
    --proxy_unknown_energy_softplus_temperature "${proxy_tau_e}"
    --proxy_unknown_accept_softplus_temperature "${proxy_accept_tau}"
    --proxy_unknown_bridge_accept_target "${adg_bridge_target}"
    --proxy_unknown_shell_outward_accept_target "${adg_shell_target}"
    --proxy_unknown_tail_accept_target "${adg_tail_target}"
    --proxy_unknown_overflow_accept_target "${adg_overflow_target}"
    --proxy_unknown_energy_margin_q "${adg_energy_q}"
    --proxy_unknown_energy_margin_target "${adg_energy_target}"
    --proxy_unknown_radius_budget_deg "${adg_radius_budget}"
    --proxy_unknown_radius_max_budget_deg "${adg_radius_max_budget}"
    --proxy_unknown_radius_inter_ratio_target "${adg_ratio_target}"
    --proxy_unknown_density_temperature_deg "${adg_density_temp}"
    --proxy_unknown_component_temperature_deg "${proxy_comp_temp}"
    --proxy_unknown_component_margin_deg "${proxy_comp_margin}"
    --proxy_unknown_component_margin_temperature_deg "${proxy_comp_margin_temp}"
    --proxy_unknown_shell_width_deg "${proxy_shell_width}"
    --lambda_soft_unknown_mixup "${lambda_soft_mix}"
    --soft_unknown_mixup_start_epoch "${soft_start}"
    --soft_unknown_mixup_warmup_epochs "${warmup}"
    --soft_unknown_mixup_count "${soft_count}"
    --soft_unknown_mixup_order "${soft_order}"
    --soft_unknown_mixup_alpha 0.5
    --soft_unknown_mixup_energy_margin 1.0
    --soft_unknown_mixup_ce_weight "${soft_ce}"
    --soft_unknown_mixup_energy_weight 1.0
    --soft_unknown_mixup_vacuum_weight "${soft_vacuum}"
    --soft_unknown_mixup_vacuum_width_deg 6
    --soft_unknown_mixup_vacuum_hard_k 3
    --soft_unknown_mixup_detach false
    --lambda_source_episode "${lambda_source}"
    --source_episode_start_epoch "${source_start}"
    --source_episode_warmup_epochs "${warmup}"
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg "${source_radius}"
    --source_episode_mixup_weight "${source_mix}"
    --source_episode_mixup_hard_k 3
    --phase2_export_prototypes true
    --phase2_export_path "${RUNS_ROOT}/${cid}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components "${fuse_components}"
    --phase2_fuse_merge_angle_deg 2.5
    --phase2_fuse_radius_cap_deg "${fuse_radius}"
    --phase2_fuse_tail_abs_deg 24
    --phase2_fuse_accept_policy local_component
    --phase2_fuse_accept_radius_key p95
    --phase2_fuse_max_p95_increase_deg 2.0
    --phase2_fuse_keep_tail_sentinel true
    --phase2_fuse_global_ball_accept false
    --test_eval_policy interval_final
    --test_eval_start_epoch 1
    --test_eval_interval 10
    --test_eval_final_window 20
    --test_eval_final_interval 2
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --sat_train_scenario leo_clear_weak
    --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_view_schedule "${sat_schedule}"
    --sat_cons_start_epoch "${sat_start}"
    --lambda_sat_cls "${lambda_sat_cls}"
    --lambda_sat_cons "${lambda_sat_cons}"
    --lambda_u 0.16
    --lambda_ent 0.01
    --lambda_domain 1
    --lambda_adv 0.35
    --lambda_group_ce 0.16
    --lambda_fishr 0.04
    --tau_min 0.92
    --tau_max 0.97
    --pseudo_quantile 0.86
    --use_ema_teacher true
    --eval_sat_channel true
    --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_eval_max_batches -1
    --device cuda:0
    --seed "${seed}")
}

run_candidate() {
  local cid="$1"
  local gpu="$2"
  local seed="$3"
  if ! candidate_enabled "${cid}"; then
    echo "[ADG8-SKIP] id=${cid} gpu=${gpu} reason=only-filter"
    return 0
  fi
  set_candidate_defaults
  apply_candidate_variant "${cid}"
  build_command "${cid}" "${gpu}" "${seed}"
  echo "[ADG8-CANDIDATE] id=${cid} gpu=${gpu} seed=${seed} epochs=${epochs} mechanism=${mechanism}"
  printf "[ADG8-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}" "${LOG_ROOT}/status"
  if [[ -e "${RUNS_ROOT}/${cid}" || -e "${LOG_ROOT}/${cid}.out" ]]; then
    echo "[ERROR] refusing to overwrite existing run/log for ${cid}" >&2
    return 3
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${RUNS_ROOT}/${cid}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  local pid=$!
  echo "${pid}" > "${LOG_ROOT}/${cid}.pid"
  echo "running pid=${pid} gpu=${gpu} started=$(date -Is)" > "${LOG_ROOT}/status/${cid}.status"
  echo "[ADG8-LAUNCHED] id=${cid} pid=${pid} gpu=${gpu} log=${LOG_ROOT}/${cid}.out"
  sleep "${LAUNCH_STABILIZE_SEC}"

  set +e
  wait "${pid}"
  local status=$?
  set -e
  echo "exit=${status} finished=$(date -Is)" >> "${LOG_ROOT}/status/${cid}.status"
  echo "[ADG8-FINISHED] id=${cid} pid=${pid} gpu=${gpu} exit=${status}"
  return "${status}"
}

echo "[ADG8-RUN] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=8 gpus=0-7 one_per_gpu=1 cap_per_gpu=${STAGE2_MAX_ACTIVE_PER_GPU}"

if [[ "${DRY_RUN}" == "1" ]]; then
  run_candidate ADG8G0_B02_ANCHOR_E200 0 493000
  run_candidate ADG8G1_BRIDGE_CVAR_E200 1 493001
  run_candidate ADG8G2_SHELL_LOW_DENS_E200 2 493002
  run_candidate ADG8G3_ENERGY_Q10_E200 3 493003
  run_candidate ADG8G4_RADIUS_RATIO_E200 4 493004
  run_candidate ADG8G5_TAIL_OVERFLOW_E200 5 493005
  run_candidate ADG8G6_CONSERVATIVE_ALL_E200 6 493006
  run_candidate ADG8G7_STRONG_ALL_SAT_E200 7 493007
else
  run_candidate ADG8G0_B02_ANCHOR_E200 0 493000 || true &
  run_candidate ADG8G1_BRIDGE_CVAR_E200 1 493001 || true &
  run_candidate ADG8G2_SHELL_LOW_DENS_E200 2 493002 || true &
  run_candidate ADG8G3_ENERGY_Q10_E200 3 493003 || true &
  run_candidate ADG8G4_RADIUS_RATIO_E200 4 493004 || true &
  run_candidate ADG8G5_TAIL_OVERFLOW_E200 5 493005 || true &
  run_candidate ADG8G6_CONSERVATIVE_ALL_E200 6 493006 || true &
  run_candidate ADG8G7_STRONG_ALL_SAT_E200 7 493007 || true &
  wait
fi

echo "[ADG8-DONE] run_id=${RUN_ID}"
