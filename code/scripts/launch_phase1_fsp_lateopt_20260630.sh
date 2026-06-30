#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_fsp_lateopt_20260630}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"
FSP_LAUNCH_SETTLE_SECONDS="${FSP_LAUNCH_SETTLE_SECONDS:-20}"
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

maybe_launch_candidate() {
  local cid="$1"
  if ! candidate_enabled "${cid}"; then
    return
  fi
  launch_candidate "$@"
}

gpu_active_count() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo 0
    return
  fi
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="${gpu}" '$1 == gpu && $4 == "C" { c++ } END { print c + 0 }'
}

wait_for_gpu_slot() {
  local gpu="$1"
  local active
  while true; do
    active="$(gpu_active_count "${gpu}")"
    if [[ "${active}" -lt "${STAGE2_MAX_ACTIVE_PER_GPU}" ]]; then
      return
    fi
    echo "[PHASE1-FSP-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local cid="$1"
  local gpu="$2"
  local seed="$3"
  local epochs="$4"
  local label_epochs="$5"
  local pseudo_epochs="$6"
  local feature_start="$7"
  local feature_warmup="$8"
  local lambda_proto="$9"
  local lambda_ow="${10}"
  local ow_tail="${11}"
  local lambda_zid="${12}"
  local zid_radius="${13}"
  local zid_cvar_alpha="${14}"
  local lambda_proxy="${15}"
  local proxy_start="${16}"
  local lambda_source_episode="${17}"
  local source_radius_cap="${18}"
  local fuse_max_components="${19}"
  local fuse_merge_angle="${20}"
  local fuse_radius_cap="${21}"
  local lambda_sat_cls="${22}"

  echo "[PHASE1-FSP-CANDIDATE] id=${cid} base=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 feature_stage_start=${feature_start} protocol=Safe-SSDG-CVS-R01 target_visibility=source_only_ground_training_no_target_receiver"
  CMD=(env PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.10
    --unlabeled_ratio 0.70
    --source_val_ratio 0.20
    --output_dir "${RUNS_ROOT}/${cid}"
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
    --lambda_proto "${lambda_proto}"
    --proto_domain_align_weight 0.10
    --proto_margin 0.15
    --proto_push_weight 0.10
    --proto_min_count 2
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_start_epoch "${feature_start}"
    --ow_feat_warmup_epochs "${feature_warmup}"
    --ow_feat_radius_deg 12
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_domain_align_weight 0.000
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight "${ow_tail}"
    --ow_feat_cvar_alpha 0.95
    --ow_feat_soft_gate false
    --ow_feat_gate_floor 0.25
    --lambda_zid_compact "${lambda_zid}"
    --zid_compact_start_epoch "${feature_start}"
    --zid_compact_warmup_epochs "${feature_warmup}"
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.35
    --zid_compact_cvar_weight 0.35
    --zid_compact_cvar_alpha "${zid_cvar_alpha}"
    --zid_compact_radius_deg "${zid_radius}"
    --zid_compact_domain_aware true
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs "${feature_warmup}"
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count 16
    --proxy_unknown_energy_margin 1.0
    --proxy_unknown_placeholder_weight 0.5
    --proxy_unknown_virtual_detach true
    --lambda_source_episode "${lambda_source_episode}"
    --source_episode_start_epoch "${feature_start}"
    --source_episode_warmup_epochs "${feature_warmup}"
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg "${source_radius_cap}"
    --phase2_export_prototypes true
    --phase2_export_path "${RUNS_ROOT}/${cid}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components "${fuse_max_components}"
    --phase2_fuse_merge_angle_deg "${fuse_merge_angle}"
    --phase2_fuse_radius_cap_deg "${fuse_radius_cap}"
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
    --lambda_sat_cls "${lambda_sat_cls}"
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
  printf "[PHASE1-FSP-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${RUNS_ROOT}/${cid}" "${LOG_ROOT}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  echo "[PHASE1-FSP-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${LOG_ROOT}/${cid}.out"
  sleep "${FSP_LAUNCH_SETTLE_SECONDS}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[PHASE1-FSP] run_id=${RUN_ID} dry_run=${DRY_RUN} base=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 candidates=16 mode=late_feature_space_polish two_per_gpu_with_gpu0 only=${ONLY_CANDIDATES:-ALL}"

# Late FSP: protect original performance by starting geometry/unknown losses only near the end.
maybe_launch_candidate "FSP_G0A_LATE40_MILD_E240" 0 362901 240 200 40 201 10 0.0030 0.0010 0.08 0.010 42 0.90 0.004 211 0.0025 34 4 3.0 18 0.64
maybe_launch_candidate "FSP_G0B_LATE60_MID_E260" 0 362911 260 200 60 201 15 0.0030 0.0012 0.10 0.014 40 0.90 0.005 221 0.0030 36 4 3.0 18 0.64
maybe_launch_candidate "FSP_C0_LATE40_MILD_E240" 1 363001 240 200 40 201 10 0.0030 0.0010 0.08 0.010 42 0.90 0.004 211 0.0025 34 4 3.0 18 0.64
maybe_launch_candidate "FSP_C1_LATE40_MID_E240" 2 363011 240 200 40 201 10 0.0030 0.0012 0.10 0.014 40 0.90 0.005 211 0.0030 34 4 3.0 18 0.64
maybe_launch_candidate "FSP_C2_LATE40_CVAR95_E240" 3 363021 240 200 40 201 10 0.0032 0.0012 0.10 0.014 42 0.95 0.005 211 0.0030 34 4 3.0 18 0.64
maybe_launch_candidate "FSP_C3_LATE40_PROXYLOW_E240" 4 363031 240 200 40 201 10 0.0030 0.0010 0.08 0.012 42 0.90 0.003 216 0.0025 36 4 3.0 18 0.64
maybe_launch_candidate "FSP_C4_LATE40_SATSAFE_E240" 5 363041 240 200 40 201 10 0.0030 0.0008 0.06 0.010 44 0.90 0.004 216 0.0020 36 4 3.0 18 0.66
maybe_launch_candidate "FSP_C5_LATE40_TAILMID_E240" 6 363051 240 200 40 201 10 0.0032 0.0014 0.12 0.016 40 0.90 0.004 211 0.0035 34 4 3.0 18 0.64
maybe_launch_candidate "FSP_C6_LATE40_LOCALSTRICT_E240" 7 363061 240 200 40 201 10 0.0030 0.0010 0.08 0.012 42 0.90 0.004 211 0.0025 34 5 2.5 16 0.64

maybe_launch_candidate "FSP_C7_LATE60_MILD_E260" 1 363071 260 200 60 201 15 0.0030 0.0010 0.08 0.010 42 0.90 0.004 221 0.0025 36 4 3.0 18 0.64
maybe_launch_candidate "FSP_C8_LATE60_MID_E260" 2 363081 260 200 60 201 15 0.0030 0.0012 0.10 0.014 40 0.90 0.005 221 0.0030 36 4 3.0 18 0.64
maybe_launch_candidate "FSP_C9_LATE60_CVAR95_E260" 3 363091 260 200 60 201 15 0.0032 0.0012 0.10 0.014 42 0.95 0.005 221 0.0030 36 4 3.0 18 0.64
maybe_launch_candidate "FSP_C10_LATE60_PROXYLOW_E260" 4 363101 260 200 60 201 15 0.0030 0.0010 0.08 0.012 42 0.90 0.003 231 0.0025 38 4 3.0 18 0.64
maybe_launch_candidate "FSP_C11_LATE60_SATSAFE_E260" 5 363111 260 200 60 201 15 0.0030 0.0008 0.06 0.010 44 0.90 0.004 231 0.0020 38 4 3.0 18 0.66
maybe_launch_candidate "FSP_C12_LATE60_TAILMID_E260" 6 363121 260 200 60 201 15 0.0032 0.0014 0.12 0.016 40 0.90 0.004 221 0.0035 36 4 3.0 18 0.64
maybe_launch_candidate "FSP_C13_LATE60_LOCALSTRICT_E260" 7 363131 260 200 60 201 15 0.0030 0.0010 0.08 0.012 42 0.90 0.004 221 0.0025 36 5 2.5 16 0.64

echo "[PHASE1-FSP-SUBMIT-COMPLETE] run_id=${RUN_ID}"
