#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_epoc_adv3b02_distill_20260705}"
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
    echo "[EPOC-DISTILL-WAIT] gpu=${gpu} active=${active} max=${MAX_ACTIVE_PER_GPU}"
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
  local epochs="$4"
  local label_epochs="$5"
  local lambda_clean_kl="$6"
  local lambda_sat_kl="$7"
  local lambda_zid_mse="$8"
  local lambda_proxy="$9"
  local lambda_soft_mix="${10}"
  local lambda_ow="${11}"
  local lambda_source_episode="${12}"
  local sat_prob_a="${13}"
  local sat_prob_b="${14}"
  local proxy_start="${15}"
  local proxy_vac="${16}"
  local soft_vac="${17}"
  local lr="${18}"

  local out_dir="${RUNS_ROOT}/${cid}"
  local log_path="${LOG_ROOT}/${cid}.out"
  echo "[EPOC-DISTILL-CANDIDATE] id=${cid} base=ADV3B02_CORE90_SOFT_E200 target_visibility=source_only_ground_training_no_target_receiver real_unknown_classes_in_training=0 source_heldout_proxy_only=1"
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
    --epochs "${epochs}"
    --label_epochs "${label_epochs}"
    --pseudo_epochs "$((epochs - label_epochs))"
    --lr "${lr}"
    --weight_decay 0.00008
    --batch_size 128
    --eval_batch_size 256
    --best_metric joint_safe
    --enable_joint_safe_guard true
    --joint_guard_require_satellite true
    --one_epoch_drop_guard_pp 3.0
    --paic_guard_enabled true
    --paic_guard_sat_ce_delta 0.16
    --paic_guard_grad_delta 4.5
    --paic_guard_reliable_drop 0.02
    --paic_guard_cooldown_epochs 1
    --paic_guard_sat_scale 0.80
    --teacher_distill_start_epoch 1
    --teacher_distill_warmup_epochs 20
    --teacher_distill_temperature 2.5
    --lambda_teacher_clean_kl "${lambda_clean_kl}"
    --lambda_teacher_sat_kl "${lambda_sat_kl}"
    --lambda_teacher_zid_mse "${lambda_zid_mse}"
    --use_phase2_ground_prototypes true
    --use_feature_masks true
    --use_txrx_geometry_losses true
    --phase1_distribution_audit_only true
    --use_proto_memory true
    --lambda_proto 0.0030
    --proto_domain_align_weight 0.10
    --proto_margin 0.15
    --proto_push_weight 0.10
    --proto_min_count 2
    --lambda_open_world_feat "${lambda_ow}"
    --ow_feat_start_epoch 30
    --ow_feat_warmup_epochs 20
    --ow_feat_radius_deg 13
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight 0.08
    --ow_feat_vacuum_weight 0.16
    --ow_feat_vacuum_width_deg 5
    --ow_feat_vacuum_hard_k 3
    --lambda_zid_compact 0.012
    --zid_compact_start_epoch 25
    --zid_compact_warmup_epochs 25
    --zid_compact_radius_deg 42
    --zid_compact_cvar_alpha 0.92
    --zid_compact_domain_aware true
    --lambda_proxy_unknown "${lambda_proxy}"
    --proxy_unknown_start_epoch "${proxy_start}"
    --proxy_unknown_warmup_epochs 25
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count 24
    --proxy_unknown_virtual_mode mixed
    --proxy_unknown_energy_margin 1.0
    --proxy_unknown_placeholder_weight 0.5
    --proxy_unknown_virtual_detach true
    --proxy_unknown_vacuum_weight "${proxy_vac}"
    --proxy_unknown_vacuum_width_deg 6
    --proxy_unknown_vacuum_hard_k 4
    --proxy_unknown_vacuum_radius_deg 40
    --proxy_unknown_component_radius_mode core_quantile
    --proxy_unknown_component_radius_quantile 0.75
    --proxy_unknown_vaccept_weight 0.06
    --proxy_unknown_low_density_accept_weight 0.04
    --proxy_unknown_radius_inter_ratio_weight 0.04
    --lambda_soft_unknown_mixup "${lambda_soft_mix}"
    --soft_unknown_mixup_start_epoch "${proxy_start}"
    --soft_unknown_mixup_warmup_epochs 25
    --soft_unknown_mixup_count 24
    --soft_unknown_mixup_order 3
    --soft_unknown_mixup_alpha 0.55
    --soft_unknown_mixup_energy_weight 1.0
    --soft_unknown_mixup_ce_weight 0.5
    --soft_unknown_mixup_vacuum_weight "${soft_vac}"
    --soft_unknown_mixup_vacuum_width_deg 7
    --soft_unknown_mixup_vacuum_hard_k 3
    --lambda_source_episode "${lambda_source_episode}"
    --source_episode_start_epoch 35
    --source_episode_warmup_epochs 20
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg 36
    --source_episode_radius_mode min_three_sigma_core
    --source_episode_mixup_weight 0.25
    --source_episode_mixup_hard_k 3
    --phase2_export_prototypes true
    --phase2_export_path "${out_dir}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components 5
    --phase2_fuse_merge_angle_deg 3.0
    --phase2_fuse_radius_cap_deg 18
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
    --sat_view_schedule "1@${sat_prob_a}:leo_clear_weak;41@${sat_prob_b}:leo_low_elev_weak,leo_rain_weak;91@0.85:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
    --sat_cons_start_epoch 35
    --lambda_sat_cls 0.58
    --lambda_sat_cons 0.03
    --lambda_u 0.18
    --lambda_ent 0.01
    --lambda_domain 1.0
    --lambda_adv 0.35
    --lambda_orth 0.04
    --lambda_cons 0.08
    --lambda_group_ce 0.15
    --lambda_fishr 0.04
    --tau_min 0.90
    --tau_max 0.97
    --pseudo_quantile 0.84
    --use_ema_teacher true
    --ema_decay 0.999
    --eval_sat_channel true
    --eval_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_eval_max_batches -1
    --device cuda:0
    --seed "${seed}")
  printf "[EPOC-DISTILL-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
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
  echo "[EPOC-DISTILL-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${log_path}"
  sleep "${LAUNCH_SETTLE_SECONDS}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[EPOC-DISTILL] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=8 teacher=ADV3B02_CORE90_SOFT_E200 real_unknown_classes_in_training=0 only=${ONLY_CANDIDATES:-ALL}"

maybe_launch_candidate "EPOC_DISTILL_A_MILD" 0 705101 160 110 0.30 0.12 0.025 0.0025 0.0010 0.0008 0.0015 0.25 0.55 35 0.25 0.16 0.00018
maybe_launch_candidate "EPOC_DISTILL_B_KDHI" 1 705111 170 115 0.55 0.20 0.050 0.0025 0.0012 0.0008 0.0015 0.30 0.65 35 0.25 0.16 0.00016
maybe_launch_candidate "EPOC_DISTILL_C_OPENHI" 2 705121 180 120 0.35 0.18 0.035 0.0045 0.0025 0.0014 0.0025 0.35 0.70 30 0.45 0.30 0.00016
maybe_launch_candidate "EPOC_DISTILL_D_SATHI" 3 705131 180 120 0.35 0.28 0.035 0.0035 0.0018 0.0010 0.0020 0.45 0.80 30 0.36 0.24 0.00016
maybe_launch_candidate "EPOC_DISTILL_E_SOFTMIX" 4 705141 190 125 0.40 0.22 0.040 0.0035 0.0035 0.0010 0.0025 0.35 0.72 30 0.35 0.45 0.00015
maybe_launch_candidate "EPOC_DISTILL_F_RELAXED" 5 705151 190 125 0.25 0.16 0.025 0.0055 0.0030 0.0016 0.0030 0.45 0.80 25 0.55 0.42 0.00014
maybe_launch_candidate "EPOC_DISTILL_G_BALANCED" 6 705161 200 130 0.45 0.24 0.050 0.0040 0.0024 0.0012 0.0024 0.40 0.76 30 0.42 0.34 0.00014
maybe_launch_candidate "EPOC_DISTILL_H_AGGRESSIVE" 7 705171 200 130 0.30 0.25 0.040 0.0065 0.0040 0.0018 0.0035 0.50 0.85 25 0.65 0.55 0.00013

echo "[EPOC-DISTILL-SUBMIT-COMPLETE] run_id=${RUN_ID}"
