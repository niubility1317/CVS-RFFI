#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_short195s3_zidbridge_20260630_0030}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"
SCREEN_EPOCHS="${SCREEN_EPOCHS:-160}"
SCREEN_LABEL_EPOCHS="${SCREEN_LABEL_EPOCHS:-150}"
SCREEN_PSEUDO_EPOCHS="${SCREEN_PSEUDO_EPOCHS:-10}"
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
    echo "[PHASE1-ZIDBRIDGE-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
    sleep 60
  done
}

launch_candidate() {
  local cid="$1"
  local gpu="$2"
  local seed="$3"
  local use_proto="$4"
  local lambda_proto="$5"
  local lambda_ow="$6"
  local proto_domain_align="$7"
  local ow_domain_align="$8"
  local epochs="$9"
  local label_epochs="${10}"
  local pseudo_epochs="${11}"

  echo "[PHASE1-ZIDBRIDGE-CANDIDATE] id=${cid} base=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 protocol=Safe-SSDG-CVS-R01 target_visibility=source_only_ground_training_no_target_receiver"
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
    --use_proto_memory "${use_proto}"
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
    --phase2_export_prototypes true
    --phase2_export_path "${RUNS_ROOT}/${cid}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
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
    --lambda_sat_cls 0.6
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
  printf "[PHASE1-ZIDBRIDGE-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${RUNS_ROOT}/${cid}" "${LOG_ROOT}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  echo "[PHASE1-ZIDBRIDGE-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${LOG_ROOT}/${cid}.out"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[PHASE1-ZIDBRIDGE] run_id=${RUN_ID} dry_run=${DRY_RUN} base=PHASE1_GPU0_JOINTSAFE36_SOFTPSEUDO_190X10_SHORT195_S3 candidates=4 screen_epochs=${SCREEN_EPOCHS}/${SCREEN_LABEL_EPOCHS}+${SCREEN_PSEUDO_EPOCHS}"
launch_candidate "PHASE1_SHORT195S3_ZIDBRIDGE_C0_EXPORT_E${SCREEN_EPOCHS}" 1 360703 false 0.000 0.000 0.25 0.00 "${SCREEN_EPOCHS}" "${SCREEN_LABEL_EPOCHS}" "${SCREEN_PSEUDO_EPOCHS}"
launch_candidate "PHASE1_SHORT195S3_ZIDBRIDGE_C1_PROTO_LOW_E${SCREEN_EPOCHS}" 2 360713 true 0.004 0.000 0.25 0.00 "${SCREEN_EPOCHS}" "${SCREEN_LABEL_EPOCHS}" "${SCREEN_PSEUDO_EPOCHS}"
launch_candidate "PHASE1_SHORT195S3_ZIDBRIDGE_C2_OWFEAT_LOW_E${SCREEN_EPOCHS}" 3 360723 false 0.000 0.004 0.25 0.01 "${SCREEN_EPOCHS}" "${SCREEN_LABEL_EPOCHS}" "${SCREEN_PSEUDO_EPOCHS}"
launch_candidate "PHASE1_SHORT195S3_ZIDBRIDGE_C3_PROTO_OWFEAT_LOW_E${SCREEN_EPOCHS}" 4 360733 true 0.004 0.004 0.25 0.01 "${SCREEN_EPOCHS}" "${SCREEN_LABEL_EPOCHS}" "${SCREEN_PSEUDO_EPOCHS}"
echo "[PHASE1-ZIDBRIDGE-SUBMIT-COMPLETE] run_id=${RUN_ID}"
