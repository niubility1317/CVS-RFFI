#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3_vaccept_gpu0_20260701}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"
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
    echo "[ADV3-VACCEPT-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local cid="$1"
  local seed="$2"
  local base="$3"
  local proxy_start="$4"
  local soft_start="$5"
  local lambda_ow="$6"
  local ow_vacuum="$7"
  local lambda_zid="$8"
  local lambda_proxy="$9"
  local proxy_vacuum="${10}"
  local proxy_virtual_count="${11}"
  local proxy_core_q="${12}"
  local proxy_accept_q="${13}"
  local proxy_vaccept_w="${14}"
  local proxy_core_w="${15}"
  local proxy_gate_w="${16}"
  local proxy_tail_w="${17}"
  local proxy_source_w="${18}"
  local lambda_soft_mix="${19}"
  local soft_count="${20}"
  local soft_ce="${21}"
  local soft_vacuum="${22}"
  local lambda_source="${23}"
  local source_mix="${24}"
  local source_radius="${25}"

  if ! candidate_enabled "${cid}"; then
    return
  fi

  echo "[ADV3-VACCEPT-CANDIDATE] id=${cid} gpu=0 epochs=200 base=${base} protocol=Phase1-source-only hard_unknown=1"
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.10
    --unlabeled_ratio 0.70
    --source_val_ratio 0.20
    --output_dir "${RUNS_ROOT}/${cid}"
    --run_id "${RUN_ID}"
    --candidate_id "${cid}"
    --base_candidate "${base}"
    --epochs 200
    --label_epochs 130
    --pseudo_epochs 70
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
    --ow_feat_start_epoch 12
    --ow_feat_warmup_epochs 30
    --ow_feat_radius_deg 12
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_domain_align_weight 0
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight 0.14
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight "${ow_vacuum}"
    --ow_feat_vacuum_width_deg 6
    --ow_feat_vacuum_hard_k 3
    --lambda_zid_compact "${lambda_zid}"
    --zid_compact_start_epoch 8
    --zid_compact_warmup_epochs 30
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.35
    --zid_compact_cvar_weight 0.35
    --zid_compact_cvar_alpha 0.95
    --zid_compact_radius_deg 40
    --zid_compact_domain_aware true
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs 20
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count "${proxy_virtual_count}"
    --proxy_unknown_virtual_mode hard
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
    --proxy_unknown_tail_quantile 0.92
    --proxy_unknown_overflow_quantile 0.97
    --proxy_unknown_vaccept_weight "${proxy_vaccept_w}"
    --proxy_unknown_core_accept_weight "${proxy_core_w}"
    --proxy_unknown_component_gate_weight "${proxy_gate_w}"
    --proxy_unknown_tail_quarantine_weight "${proxy_tail_w}"
    --proxy_unknown_source_safe_weight "${proxy_source_w}"
    --proxy_unknown_vaccept_cvar_alpha 0.25
    --proxy_unknown_unknown_margin 0.08
    --proxy_unknown_known_margin 0.05
    --proxy_unknown_energy_softplus_temperature 0.04
    --proxy_unknown_component_temperature_deg 3
    --proxy_unknown_component_margin_deg 4
    --proxy_unknown_component_margin_temperature_deg 3
    --proxy_unknown_shell_width_deg 4
    --lambda_soft_unknown_mixup "${lambda_soft_mix}"
    --soft_unknown_mixup_start_epoch "${soft_start}"
    --soft_unknown_mixup_warmup_epochs 20
    --soft_unknown_mixup_count "${soft_count}"
    --soft_unknown_mixup_order 3
    --soft_unknown_mixup_alpha 0.5
    --soft_unknown_mixup_energy_margin 1.0
    --soft_unknown_mixup_ce_weight "${soft_ce}"
    --soft_unknown_mixup_energy_weight 1.0
    --soft_unknown_mixup_vacuum_weight "${soft_vacuum}"
    --soft_unknown_mixup_vacuum_width_deg 6
    --soft_unknown_mixup_vacuum_hard_k 3
    --soft_unknown_mixup_detach false
    --lambda_source_episode "${lambda_source}"
    --source_episode_start_epoch 20
    --source_episode_warmup_epochs 30
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg "${source_radius}"
    --source_episode_mixup_weight "${source_mix}"
    --source_episode_mixup_hard_k 3
    --phase2_export_prototypes true
    --phase2_export_path "${RUNS_ROOT}/${cid}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components 6
    --phase2_fuse_merge_angle_deg 2.5
    --phase2_fuse_radius_cap_deg 15
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
    --sat_view_schedule "1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --sat_cons_start_epoch 80
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0
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
  printf "[ADV3-VACCEPT-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot 0
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
  if [[ -e "${RUNS_ROOT}/${cid}" || -e "${LOG_ROOT}/${cid}.out" ]]; then
    echo "[ERROR] refusing to overwrite existing run/log for ${cid}" >&2
    exit 3
  fi
  mkdir -p "${RUNS_ROOT}/${cid}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  echo "[ADV3-VACCEPT-LAUNCHED] id=${cid} pid=$! gpu=0 log=${LOG_ROOT}/${cid}.out"
  sleep 8
}

echo "[ADV3-VACCEPT-RUN] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=2 gpu=0 cap_per_gpu=${STAGE2_MAX_ACTIVE_PER_GPU}"
launch_candidate ADV3_GPU0_E200_ENERGY_HARDUNK 391501 E200_ENERGY_HEAVY 35 25 0.0024 0.40 0.032 0.0050 0.65 36 0.80 0.80 1.00 0.40 0.60 0.20 0.20 0.0050 24 0.50 0.35 0.0035 0.75 33
launch_candidate ADV3_GPU0_E200_MIXUP_HARDVETO 391502 E200_MIXUP_DOMINANT 60 20 0.0022 0.30 0.028 0.0040 0.45 48 0.90 0.85 1.00 0.35 0.75 0.20 0.20 0.0045 32 0.70 0.30 0.0030 1.00 34
echo "[ADV3-VACCEPT-DONE] run_id=${RUN_ID}"
