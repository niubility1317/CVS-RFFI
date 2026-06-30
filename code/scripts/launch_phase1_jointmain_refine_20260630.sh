#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_jointmain_refine_20260630}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

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
    echo "[PHASE1-JOINTREFINE-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
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
  local lambda_proto="$7"
  local lambda_ow="$8"
  local proto_domain_align="$9"
  local ow_domain_align="${10}"
  local tail_weight="${11}"
  local lambda_source_episode="${12}"
  local source_episode_radius_cap="${13}"
  local fuse_max_components="${14}"
  local fuse_merge_angle="${15}"
  local fuse_radius_cap="${16}"
  local fuse_tail_abs="${17}"
  local lambda_sat_cls="${18}"

  echo "[PHASE1-JOINTREFINE-CANDIDATE] id=${cid} base=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 protocol=Safe-SSDG-CVS-R01 target_visibility=source_only_ground_training_no_target_receiver"
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
    --proto_domain_align_weight "${proto_domain_align}"
    --proto_margin 0.15
    --proto_push_weight 0.10
    --proto_min_count 2
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_radius_deg 12
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_domain_align_weight "${ow_domain_align}"
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight "${tail_weight}"
    --ow_feat_cvar_alpha 0.95
    --ow_feat_soft_gate false
    --ow_feat_gate_floor 0.25
    --lambda_source_episode "${lambda_source_episode}"
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg "${source_episode_radius_cap}"
    --phase2_export_prototypes true
    --phase2_export_path "${RUNS_ROOT}/${cid}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components "${fuse_max_components}"
    --phase2_fuse_merge_angle_deg "${fuse_merge_angle}"
    --phase2_fuse_radius_cap_deg "${fuse_radius_cap}"
    --phase2_fuse_tail_abs_deg "${fuse_tail_abs}"
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
  printf "[PHASE1-JOINTREFINE-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${RUNS_ROOT}/${cid}" "${LOG_ROOT}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  echo "[PHASE1-JOINTREFINE-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${LOG_ROOT}/${cid}.out"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[PHASE1-JOINTREFINE] run_id=${RUN_ID} dry_run=${DRY_RUN} base=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 candidates=14 mode=joint_mainline_refine two_per_gpu_no_gpu0"

# C8 lineage: preserve strict-UDU strength while reducing over-fusion and tail acceptance.
launch_candidate "JREF_C0_C8_M3_SC30_E220" 1 362003 220 205 15 0.0030 0.0012 0.08 0.000 0.10 0.0030 30 4 3 18 24 0.62
launch_candidate "JREF_C1_C8_M4_SC30_E220" 2 362013 220 205 15 0.0030 0.0014 0.10 0.000 0.12 0.0035 30 4 4 20 26 0.62
launch_candidate "JREF_C2_C8_M3_SC32_E240" 3 362023 240 220 20 0.0032 0.0015 0.10 0.005 0.12 0.0040 32 4 3 18 24 0.62
launch_candidate "JREF_C3_C8_M4_SC34_E240" 4 362033 240 220 20 0.0035 0.0016 0.12 0.005 0.14 0.0040 34 4 4 20 26 0.62

# C10 lineage: keep receiver-floor benefit, but reduce tail pressure and protect satellite floor.
launch_candidate "JREF_C4_C10_FLOOR_M3_E220" 5 362043 220 205 15 0.0040 0.0020 0.16 0.005 0.18 0.0030 30 4 3 18 24 0.64
launch_candidate "JREF_C5_C10_FLOOR_M4_E220" 6 362053 220 205 15 0.0040 0.0022 0.18 0.005 0.20 0.0035 32 4 4 20 26 0.64
launch_candidate "JREF_C6_C10_SATSAFE_E240" 7 362063 240 220 20 0.0035 0.0018 0.14 0.000 0.16 0.0040 32 4 3 18 24 0.66

# C3 lineage: preserve satellite floor, with stricter fusion and moderate source episode.
launch_candidate "JREF_C7_C3_SAT_M3_E220" 1 362103 220 205 15 0.0040 0.0022 0.16 0.005 0.14 0.0035 30 4 3 18 24 0.64
launch_candidate "JREF_C8_C3_SAT_M4_E240" 2 362113 240 220 20 0.0040 0.0024 0.18 0.005 0.16 0.0040 32 4 4 20 26 0.64

# Multi-component probes: force a chance to retain real domain modes instead of collapsing every TX to one component.
launch_candidate "JREF_C9_MULTICOMP_M2_E220" 3 362123 220 205 15 0.0035 0.0016 0.12 0.005 0.12 0.0040 30 4 2 16 22 0.62
launch_candidate "JREF_C10_MULTICOMP_M25_E220" 4 362133 220 205 15 0.0035 0.0018 0.12 0.005 0.14 0.0040 32 4 2.5 16 22 0.62

# Tail-control probes: stronger compactness without the C10 satellite drop.
launch_candidate "JREF_C11_TAILSOFT_E220" 5 362143 220 205 15 0.0035 0.0018 0.12 0.000 0.18 0.0045 30 4 3 18 24 0.64
launch_candidate "JREF_C12_TAILMID_E240" 6 362153 240 220 20 0.0038 0.0020 0.14 0.005 0.20 0.0045 32 4 3 18 24 0.64

# Conservative long-run: shortest path for beating SHORT195_S3 without over-pushing tails.
launch_candidate "JREF_C13_CONSERVE_E240" 7 362163 240 220 20 0.0030 0.0012 0.08 0.000 0.10 0.0035 32 4 3 18 24 0.64

echo "[PHASE1-JOINTREFINE-SUBMIT-COMPLETE] run_id=${RUN_ID}"
