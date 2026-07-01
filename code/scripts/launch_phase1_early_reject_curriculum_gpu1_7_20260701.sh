#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_early_reject_curriculum_gpu1_7_20260701_1643}"
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

print_gpu_baseline() {
  local gpu
  for gpu in 1 2 3 4 5 6 7; do
    echo "[EARLY-REJECT-GPU-BASELINE] gpu=${gpu} active=$(gpu_active_count "${gpu}") cap=${STAGE2_MAX_ACTIVE_PER_GPU}"
  done
}

wait_for_gpu_slot() {
  local gpu="$1"
  local active
  while true; do
    active="$(gpu_active_count "${gpu}")"
    if [[ "${active}" -lt "${STAGE2_MAX_ACTIVE_PER_GPU}" ]]; then
      return
    fi
    echo "[EARLY-REJECT-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local cid="$1"
  local gpu="$2"
  local seed="$3"
  local epochs="$4"
  local label_epochs="$5"
  local zid_start="$6"
  local ow_start="$7"
  local source_start="$8"
  local soft_start="$9"
  local proxy_start="${10}"
  local lambda_ow="${11}"
  local ow_tail="${12}"
  local ow_vacuum="${13}"
  local lambda_zid="${14}"
  local lambda_proxy="${15}"
  local proxy_vacuum="${16}"
  local lambda_soft_mix="${17}"
  local soft_count="${18}"
  local soft_ce="${19}"
  local soft_vacuum="${20}"
  local lambda_source="${21}"
  local source_mix="${22}"
  local source_radius="${23}"
  local fuse_components="${24}"
  local fuse_radius="${25}"
  local warmup="${26}"

  if ! candidate_enabled "${cid}"; then
    return
  fi

  local pseudo_epochs=$((epochs - label_epochs))
  echo "[EARLY-REJECT-CANDIDATE] id=${cid} gpu=${gpu} epochs=${epochs} label_epochs=${label_epochs} pseudo_epochs=${pseudo_epochs} protocol=Safe-SSDG-CVS-R01 target_visibility=source_only_ground_training_no_target_receiver mixup_order=3 soft_start=${soft_start} proxy_start=${proxy_start}"
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.10
    --unlabeled_ratio 0.70
    --source_val_ratio 0.20
    --output_dir "${RUNS_ROOT}/${cid}"
    --run_id "${RUN_ID}"
    --candidate_id "${cid}"
    --base_candidate early_reject_curriculum_v1
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
    --ow_feat_domain_align_weight 0.000
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight "${ow_tail}"
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight "${ow_vacuum}"
    --ow_feat_vacuum_width_deg 6
    --ow_feat_vacuum_hard_k 3
    --ow_feat_soft_gate false
    --ow_feat_gate_floor 0.25
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
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count 16
    --proxy_unknown_energy_margin 1.0
    --proxy_unknown_placeholder_weight 0.5
    --proxy_unknown_virtual_detach true
    --proxy_unknown_vacuum_weight "${proxy_vacuum}"
    --proxy_unknown_vacuum_width_deg 5
    --proxy_unknown_vacuum_hard_k 3
    --proxy_unknown_vacuum_radius_deg 40
    --lambda_soft_unknown_mixup "${lambda_soft_mix}"
    --soft_unknown_mixup_start_epoch "${soft_start}"
    --soft_unknown_mixup_warmup_epochs "${warmup}"
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
  printf "[EARLY-REJECT-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
  if [[ -e "${RUNS_ROOT}/${cid}" || -e "${LOG_ROOT}/${cid}.out" ]]; then
    echo "[ERROR] refusing to overwrite existing run/log for ${cid}" >&2
    exit 3
  fi
  mkdir -p "${RUNS_ROOT}/${cid}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  echo "[EARLY-REJECT-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${LOG_ROOT}/${cid}.out"
  sleep 15
}

echo "[EARLY-REJECT-RUN] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=14 gpus=1-7 cap_per_gpu=${STAGE2_MAX_ACTIVE_PER_GPU}"
print_gpu_baseline

launch_candidate E120_EARLY_LITE       1 381101 120  80   5  10  15  20  40 0.0018 0.10 0.20 0.024 0.0015 0.25 0.0025 16 0.90 0.20 0.0020 0.60 36 5 16.5 25
launch_candidate E200_EARLY_MAIN       2 381201 200 130   8  12  20  25  60 0.0024 0.14 0.35 0.030 0.0025 0.45 0.0040 24 0.80 0.35 0.0030 0.75 34 6 15.5 30
launch_candidate E300_EARLY_CONSOL     3 381301 300 180  10  15  25  30  75 0.0024 0.14 0.35 0.030 0.0025 0.45 0.0040 24 0.80 0.30 0.0030 0.75 34 6 15.5 35
launch_candidate E200_MID_CURRIC       4 381401 200 130  20  30  45  60  80 0.0030 0.18 0.50 0.038 0.0035 0.60 0.0060 24 0.70 0.45 0.0045 1.00 32 6 15.0 30
launch_candidate E200_CE_HEAVY_MIX     5 381501 200 130   8  12  20  20  65 0.0022 0.14 0.25 0.028 0.0015 0.25 0.0045 32 1.20 0.15 0.0030 0.75 34 6 15.5 30
launch_candidate E220_VACUUM_STRONG    6 381601 220 140  10  15  25  35  55 0.0030 0.18 0.55 0.038 0.0035 0.65 0.0060 24 0.60 0.60 0.0045 1.00 32 6 15.0 30
launch_candidate E200_MIXUP_DOMINANT   7 381701 200 130   8  12  20  20  80 0.0022 0.14 0.30 0.028 0.0015 0.25 0.0045 32 1.00 0.30 0.0030 1.00 34 6 15.5 30

launch_candidate E160_EARLY_MAIN       1 381102 160 100   8  12  20  25  50 0.0024 0.14 0.35 0.030 0.0025 0.45 0.0040 16 0.80 0.30 0.0030 0.75 34 6 15.5 30
launch_candidate E240_EARLY_MAIN       2 381202 240 150  10  15  25  30  70 0.0024 0.14 0.35 0.030 0.0025 0.45 0.0040 24 0.80 0.35 0.0030 0.75 34 6 15.5 35
launch_candidate E200_VERYEARLY_STRONG 3 381302 200 120   1   5  10  15  35 0.0030 0.18 0.50 0.038 0.0035 0.60 0.0060 24 0.70 0.45 0.0045 1.00 32 6 15.0 30
launch_candidate E200_LATE_CONTROL     4 381402 200 150  80 100 120 140 150 0.0024 0.14 0.35 0.030 0.0025 0.45 0.0040 16 0.80 0.35 0.0030 0.75 34 6 15.5 15
launch_candidate E200_ENERGY_HEAVY     5 381502 200 130   8  12  20  25  35 0.0024 0.14 0.40 0.032 0.0045 0.65 0.0050 24 0.50 0.35 0.0035 0.75 33 6 15.0 30
launch_candidate E220_3SIGMA_STRONG    6 381602 220 140   8  12  15  25  65 0.0030 0.18 0.50 0.038 0.0035 0.60 0.0060 24 0.80 0.35 0.0045 1.20 31 6 15.0 30
launch_candidate E200_PROXY_DOMINANT   7 381702 200 130   8  12  20  25  30 0.0024 0.14 0.40 0.032 0.0045 0.70 0.0050 16 0.50 0.25 0.0035 0.50 33 6 15.0 30

echo "[EARLY-REJECT-DONE] run_id=${RUN_ID}"
