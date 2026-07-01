#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_accept_domain_verify_20260701_130328}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
STAGE2_MAX_ACTIVE_PER_GPU="${STAGE2_MAX_ACTIVE_PER_GPU:-2}"
FSP_LAUNCH_SETTLE_SECONDS="${FSP_LAUNCH_SETTLE_SECONDS:-30}"
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
    echo "[ACCEPT-DOMAIN-VERIFY-WAIT] gpu=${gpu} active=${active} max=${STAGE2_MAX_ACTIVE_PER_GPU}"
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
  local ow_vacuum_weight="${23}"
  local ow_vacuum_width="${24}"
  local proxy_vacuum_weight="${25}"
  local proxy_vacuum_width="${26}"
  local vacuum_hard_k="${27}"

  if [[ "${gpu}" == "0" ]]; then
    echo "[ERROR] GPU0 is excluded for this validation run: ${cid}" >&2
    exit 4
  fi

  echo "[ACCEPT-DOMAIN-VERIFY-CANDIDATE] id=${cid} gpu=${gpu} cap_per_gpu=${STAGE2_MAX_ACTIVE_PER_GPU} protocol=Safe-SSDG-CVS-R01 target_visibility=source_only_ground_training_no_target_receiver"
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
    --ow_feat_vacuum_weight "${ow_vacuum_weight}"
    --ow_feat_vacuum_width_deg "${ow_vacuum_width}"
    --ow_feat_vacuum_hard_k "${vacuum_hard_k}"
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
    --proxy_unknown_vacuum_weight "${proxy_vacuum_weight}"
    --proxy_unknown_vacuum_width_deg "${proxy_vacuum_width}"
    --proxy_unknown_vacuum_hard_k "${vacuum_hard_k}"
    --proxy_unknown_vacuum_radius_deg 40
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
  printf "[ACCEPT-DOMAIN-VERIFY-CMD] "; printf "%q " "${CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi
  wait_for_gpu_slot "${gpu}"
  mkdir -p "${LOG_ROOT}"
  if [[ -e "${RUNS_ROOT}/${cid}" || -e "${LOG_ROOT}/${cid}.out" ]]; then
    echo "[ERROR] refusing to overwrite existing run/log for ${cid}" >&2
    exit 3
  fi
  mkdir -p "${RUNS_ROOT}/${cid}"
  "${CMD[@]}" > "${LOG_ROOT}/${cid}.out" 2>&1 &
  echo "[ACCEPT-DOMAIN-VERIFY-LAUNCHED] id=${cid} pid=$! gpu=${gpu} log=${LOG_ROOT}/${cid}.out"
  sleep "${FSP_LAUNCH_SETTLE_SECONDS}"
}

CANDIDATES=(
  "ADV2_R17_CORESTRICT_E260|1|364601|260|200|60|201|15|0.0030|0.0011|0.09|0.014|40|0.95|0.0045|231|0.0030|34|4|3.0|17|0.64|0.24|5|0.50|5|3"
  "ADV2_R17_PROXYHI_E260|1|364602|260|200|60|201|15|0.0030|0.0011|0.09|0.014|40|0.95|0.0055|231|0.0030|34|4|3.0|17|0.64|0.22|5|0.65|6|3"
  "ADV2_R20_SAT70_E260|2|364611|260|200|60|201|15|0.0030|0.0009|0.07|0.012|42|0.90|0.0040|231|0.0025|36|4|3.0|18|0.70|0.14|4|0.35|4|2"
  "ADV2_R20_VACMID_E260|2|364612|260|200|60|201|15|0.0030|0.0009|0.07|0.012|42|0.90|0.0040|231|0.0025|36|4|3.0|18|0.70|0.18|5|0.45|5|2"
  "ADV2_R28_PROXYLOW_E260|3|364621|260|200|60|201|15|0.0030|0.0009|0.07|0.012|42|0.90|0.0040|231|0.0025|36|4|3.0|18|0.72|0.14|4|0.35|4|2"
  "ADV2_R28_FUSE6_E260|3|364622|260|200|60|201|15|0.0030|0.0009|0.07|0.012|42|0.90|0.0040|231|0.0025|34|6|2.0|15|0.72|0.16|4|0.38|4|2"
  "ADV2_T13_CONSERVE_E260|4|364631|260|200|60|201|15|0.0030|0.0008|0.06|0.010|44|0.90|0.0040|231|0.0020|38|4|3.0|18|0.68|0.12|4|0.30|4|2"
  "ADV2_T13_TAILGUARD_E260|4|364632|260|200|60|201|15|0.0030|0.0010|0.08|0.012|42|0.90|0.0040|231|0.0025|36|4|3.0|18|0.68|0.16|5|0.35|5|2"
  "ADV2_SRCLOW_R17_E260|5|364641|260|200|60|201|15|0.0030|0.0010|0.08|0.014|40|0.95|0.0045|231|0.0020|38|4|3.0|17|0.64|0.22|5|0.45|5|3"
  "ADV2_SOURCECAP32_R20_E260|5|364642|260|200|60|201|15|0.0030|0.0009|0.07|0.012|42|0.90|0.0040|231|0.0030|32|4|3.0|18|0.70|0.16|5|0.40|5|2"
  "ADV2_FUSE6_R17_E260|6|364651|260|200|60|201|15|0.0030|0.0011|0.09|0.014|40|0.95|0.0045|231|0.0030|34|6|2.0|15|0.64|0.24|5|0.50|5|3"
  "ADV2_FUSE5_R20_E260|6|364652|260|200|60|201|15|0.0030|0.0009|0.07|0.012|42|0.90|0.0040|231|0.0025|36|5|2.5|16|0.70|0.14|4|0.35|4|2"
  "ADV2_TAILCV_R17_E260|7|364661|260|200|60|201|15|0.0032|0.0014|0.12|0.016|40|0.95|0.0045|231|0.0035|32|4|3.0|17|0.64|0.30|6|0.60|6|3"
  "ADV2_TAILCV_R20_E260|7|364662|260|200|60|201|15|0.0030|0.0012|0.10|0.014|42|0.95|0.0040|231|0.0030|34|4|3.0|18|0.70|0.24|5|0.50|5|2"
)

queue_gpu() {
  local target_gpu="$1"
  local row cid gpu seed epochs label_epochs pseudo_epochs feature_start feature_warmup
  local lambda_proto lambda_ow ow_tail lambda_zid zid_radius zid_cvar_alpha
  local lambda_proxy proxy_start lambda_source_episode source_radius_cap
  local fuse_max_components fuse_merge_angle fuse_radius_cap lambda_sat_cls
  local ow_vacuum_weight ow_vacuum_width proxy_vacuum_weight proxy_vacuum_width vacuum_hard_k

  for row in "${CANDIDATES[@]}"; do
    IFS='|' read -r cid gpu seed epochs label_epochs pseudo_epochs feature_start feature_warmup \
      lambda_proto lambda_ow ow_tail lambda_zid zid_radius zid_cvar_alpha \
      lambda_proxy proxy_start lambda_source_episode source_radius_cap \
      fuse_max_components fuse_merge_angle fuse_radius_cap lambda_sat_cls \
      ow_vacuum_weight ow_vacuum_width proxy_vacuum_weight proxy_vacuum_width vacuum_hard_k <<< "${row}"
    if [[ "${gpu}" != "${target_gpu}" ]] || ! candidate_enabled "${cid}"; then
      continue
    fi
    launch_candidate "${cid}" "${gpu}" "${seed}" "${epochs}" "${label_epochs}" "${pseudo_epochs}" \
      "${feature_start}" "${feature_warmup}" "${lambda_proto}" "${lambda_ow}" "${ow_tail}" \
      "${lambda_zid}" "${zid_radius}" "${zid_cvar_alpha}" "${lambda_proxy}" "${proxy_start}" \
      "${lambda_source_episode}" "${source_radius_cap}" "${fuse_max_components}" "${fuse_merge_angle}" \
      "${fuse_radius_cap}" "${lambda_sat_cls}" "${ow_vacuum_weight}" "${ow_vacuum_width}" \
      "${proxy_vacuum_weight}" "${proxy_vacuum_width}" "${vacuum_hard_k}"
  done
}

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}"
fi

echo "[ACCEPT-DOMAIN-VERIFY] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=14 gpu_scope=1-7 excluded_gpu=0 cap_per_gpu=${STAGE2_MAX_ACTIVE_PER_GPU} only=${ONLY_CANDIDATES:-ALL}"

if [[ "${DRY_RUN}" == "1" ]]; then
  for gpu in 1 2 3 4 5 6 7; do
    queue_gpu "${gpu}"
  done
else
  pids=()
  for gpu in 1 2 3 4 5 6 7; do
    queue_gpu "${gpu}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
fi

echo "[ACCEPT-DOMAIN-VERIFY-SUBMIT-COMPLETE] run_id=${RUN_ID}"
