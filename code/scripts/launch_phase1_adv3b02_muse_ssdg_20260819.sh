#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_muse_ssdg_20260819}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU="${GPU:-0}"
SEED=392002
DRY_RUN=0
ONLY_CANDIDATES="M0,M1,M2,M3"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATES="${arg#--only=}" ;;
    *) echo "[MUSE-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[MUSE-ERROR] unsafe RUN_ID: ${RUN_ID}" >&2
  exit 2
fi

candidate_selected() {
  local level="$1"
  [[ ",${ONLY_CANDIDATES}," == *",${level},"* ]]
}

validate_only() {
  local raw="${ONLY_CANDIDATES//,/ }"
  local level
  [[ -n "${raw// /}" ]] || { echo "[MUSE-ERROR] --only must not be empty" >&2; return 2; }
  for level in ${raw}; do
    case "${level}" in
      M0|M1|M2|M3) ;;
      *) echo "[MUSE-ERROR] unknown candidate: ${level}" >&2; return 2 ;;
    esac
  done
}

capability_label() {
  case "$1" in
    M0) echo "ADV3B02_CONTROL" ;;
    M1) echo "BASE" ;;
    M2) echo "BASE_FUSION_HML" ;;
    M3) echo "BASE_FUSION_HML_SATELLITE_CROSSRX_PROTO" ;;
  esac
}

build_train_command() {
  local level="$1"
  local candidate_root="$2"
  TRAIN_CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --labeled_ratio 0.07
    --unlabeled_ratio 0.63
    --source_val_ratio 0.30
    --source_cal_ratio 0.15
    --source_select_ratio 0.15
    --phase1_source_role_protocol l_s_u_s_v_cal_v_select
    --output_dir "${candidate_root}"
    --run_id "${RUN_ID}"
    --candidate_id "${level}"
    --base_candidate ADV3B02_CORE90_SOFT_E200
    --epochs 200
    --label_epochs 130
    --pseudo_epochs 70
    --from_scratch true
    --phase1_source_val_selection_only true
    --checkpoint_selection final_only
    --best_metric source_val_sat_hmean
    --use_muse_ssdg true
    --muse_level "${level}"
    --muse_epoch_basis unlabeled_loader
    --muse_final_epoch 200
    --use_unlabeled true
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
    --lambda_open_world_feat 0.0024
    --ow_feat_start_epoch 12
    --ow_feat_warmup_epochs 25
    --ow_feat_radius_deg 12
    --ow_feat_inter_margin_deg 55
    --ow_feat_sample_margin_deg 5
    --ow_feat_domain_align_weight 0
    --ow_feat_min_classes 2
    --ow_feat_min_samples_per_class 1
    --ow_feat_tail_mode robust_3sigma
    --ow_feat_tail_weight 0.14
    --ow_feat_cvar_alpha 0.95
    --ow_feat_vacuum_weight 0.40
    --ow_feat_vacuum_width_deg 6
    --ow_feat_vacuum_hard_k 3
    --lambda_zid_compact 0.032
    --zid_compact_start_epoch 8
    --zid_compact_warmup_epochs 25
    --zid_compact_supcon_weight 0.30
    --zid_compact_radius_weight 0.35
    --zid_compact_cvar_weight 0.35
    --zid_compact_cvar_alpha 0.95
    --zid_compact_radius_deg 40
    --zid_compact_domain_aware true
    --lambda_proxy_unknown 0.0045
    --proxy_unknown_start_epoch 45
    --proxy_unknown_warmup_epochs 25
    --proxy_unknown_holdout_tx_per_batch 1
    --proxy_unknown_virtual_count 48
    --proxy_unknown_virtual_mode hard
    --proxy_unknown_energy_margin 0.0
    --proxy_unknown_energy_temperature 1.0
    --proxy_unknown_placeholder_weight 0.0
    --proxy_unknown_virtual_detach false
    --proxy_unknown_vacuum_weight 0.55
    --proxy_unknown_vacuum_width_deg 5
    --proxy_unknown_vacuum_hard_k 3
    --proxy_unknown_vacuum_radius_deg 40
    --proxy_unknown_core_quantile 0.90
    --proxy_unknown_accept_quantile 0.85
    --proxy_unknown_tail_quantile 0.92
    --proxy_unknown_overflow_quantile 0.97
    --proxy_unknown_vaccept_weight 1.00
    --proxy_unknown_core_accept_weight 0.45
    --proxy_unknown_component_gate_weight 0.65
    --proxy_unknown_tail_quarantine_weight 0.20
    --proxy_unknown_source_safe_weight 0.20
    --proxy_unknown_vaccept_cvar_alpha 0.30
    --proxy_unknown_unknown_margin 0.08
    --proxy_unknown_known_margin 0.05
    --proxy_unknown_energy_softplus_temperature 0.04
    --proxy_unknown_component_temperature_deg 3.0
    --proxy_unknown_component_margin_deg 4.0
    --proxy_unknown_component_margin_temperature_deg 3.0
    --proxy_unknown_shell_width_deg 4.0
    --lambda_soft_unknown_mixup 0.0045
    --soft_unknown_mixup_start_epoch 25
    --soft_unknown_mixup_warmup_epochs 25
    --soft_unknown_mixup_count 24
    --soft_unknown_mixup_order 3
    --soft_unknown_mixup_alpha 0.5
    --soft_unknown_mixup_energy_margin 1.0
    --soft_unknown_mixup_ce_weight 0.60
    --soft_unknown_mixup_energy_weight 1.0
    --soft_unknown_mixup_vacuum_weight 0.35
    --soft_unknown_mixup_vacuum_width_deg 6
    --soft_unknown_mixup_vacuum_hard_k 3
    --soft_unknown_mixup_detach false
    --lambda_source_episode 0.0035
    --source_episode_start_epoch 20
    --source_episode_warmup_epochs 25
    --source_episode_min_domains 2
    --source_episode_radius_cap_deg 33
    --source_episode_mixup_weight 0.75
    --source_episode_mixup_hard_k 3
    --phase2_export_prototypes true
    --phase2_export_path "${candidate_root}/phase2_zid_prototypes.pt"
    --phase2_export_feature_key z_id
    --phase2_export_split train
    --phase2_fuse_prototypes true
    --phase2_fuse_max_components 6
    --phase2_fuse_merge_angle_deg 2.5
    --phase2_fuse_radius_cap_deg 15.0
    --phase2_fuse_tail_abs_deg 24
    --phase2_fuse_accept_policy local_component
    --phase2_fuse_accept_radius_key p95
    --phase2_fuse_max_p95_increase_deg 2.0
    --phase2_fuse_keep_tail_sentinel true
    --phase2_fuse_global_ball_accept false
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --sat_training_mode concat_masked
    --sat_train_scenario leo_clear_weak
    --sat_train_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
    --sat_view_schedule '1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak'
    --sat_cons_start_epoch 80
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0
    --lambda_zid_channel_invariance 0
    --zid_channel_pair_weight 1.0
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
    --seed "${SEED}"
  )
}

build_eval_command() {
  local scenario="$1"
  local candidate_root="$2"
  local eval_scenario="${scenario}"
  if [[ "${scenario}" == "clean" ]]; then
    # The real evaluator always materializes clean fields alongside a scenario.
    eval_scenario="leo_clear_weak"
  fi
  EVAL_CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${ROOT}/code/scripts/eval_ssdg_sat_per_rx.py"
    --ckpt "${candidate_root}/final_ssdg.pth"
    --output_json "${candidate_root}/metrics_${scenario}.json"
    --eval_on unseen_rx
    --scenarios "${eval_scenario}"
    --device cuda:0
    --max_batches -1
    --sat_seed "${SEED}"
  )
}

write_config() {
  local level="$1"
  local capabilities="$2"
  local candidate_root="$3"
  printf '{\n  "run_id": "%s",\n  "candidate": "%s",\n  "base_candidate": "ADV3B02_CORE90_SOFT_E200",\n  "capabilities": "%s",\n  "seed": %d,\n  "epochs": 200,\n  "ratios": {"L_s": 0.07, "U_s": 0.63, "V_cal": 0.15, "V_select": 0.15},\n  "checkpoint_selection": "final_only"\n}\n' \
    "${RUN_ID}" "${level}" "${capabilities}" "${SEED}" > "${candidate_root}/config.json"
}

run_candidate() {
  local level="$1"
  local capabilities
  local candidate_root="${RUNS_ROOT}/${level}"
  local scenario
  local status
  capabilities="$(capability_label "${level}")"
  build_train_command "${level}" "${candidate_root}"

  echo "[MUSE-CANDIDATE] candidate=${level} capabilities=${capabilities} output=${candidate_root} seed=${SEED} epochs=200"
  printf '[MUSE-TRAIN-CMD] '; printf '%q ' "${TRAIN_CMD[@]}"; printf '\n'
  for scenario in clean leo_clear_weak leo_low_elev_weak leo_rain_weak; do
    build_eval_command "${scenario}" "${candidate_root}"
    printf '[MUSE-EVAL-CMD] scenario=%s log=%s ' "${scenario}" "${candidate_root}/eval_${scenario}.log"
    printf '%q ' "${EVAL_CMD[@]}"
    printf '\n'
  done
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ -e "${candidate_root}" ]]; then
    echo "[MUSE-ERROR] refusing to overwrite existing candidate root: ${candidate_root}" >&2
    return 3
  fi
  mkdir -p "${candidate_root}"
  write_config "${level}" "${capabilities}" "${candidate_root}"

  if ! "${TRAIN_CMD[@]}" > "${candidate_root}/train.log" 2>&1; then
    printf 'TRAIN_FAILED\n' > "${candidate_root}/status.txt"
    echo "[MUSE-ERROR] candidate=${level} training failed; outputs preserved at ${candidate_root}" >&2
    return 4
  fi
  if [[ ! -s "${candidate_root}/final_ssdg.pth" ]]; then
    printf 'FINAL_CHECKPOINT_MISSING\n' > "${candidate_root}/status.txt"
    echo "[MUSE-ERROR] candidate=${level} missing non-empty final_ssdg.pth" >&2
    return 5
  fi

  for scenario in clean leo_clear_weak leo_low_elev_weak leo_rain_weak; do
    build_eval_command "${scenario}" "${candidate_root}"
    if ! "${EVAL_CMD[@]}" > "${candidate_root}/eval_${scenario}.log" 2>&1; then
      status="EVAL_FAILED_${scenario^^}"
      printf '%s\n' "${status}" > "${candidate_root}/status.txt"
      echo "[MUSE-ERROR] candidate=${level} status=${status}; training outputs preserved" >&2
      return 6
    fi
    if [[ ! -s "${candidate_root}/eval_${scenario}.log" || ! -s "${candidate_root}/metrics_${scenario}.json" ]]; then
      status="EVAL_FAILED_${scenario^^}"
      printf '%s\n' "${status}" > "${candidate_root}/status.txt"
      echo "[MUSE-ERROR] candidate=${level} status=${status}; empty evaluation artifact" >&2
      return 7
    fi
  done

  for scenario in clean leo_clear_weak leo_low_elev_weak leo_rain_weak; do
    [[ -s "${candidate_root}/eval_${scenario}.log" ]]
    [[ -s "${candidate_root}/metrics_${scenario}.json" ]]
  done
  printf 'ARTIFACTS_COMPLETE\n' > "${candidate_root}/status.txt"
  echo "[MUSE-COMPLETE] candidate=${level} status=ARTIFACTS_COMPLETE root=${candidate_root}"
}

validate_only
echo "[MUSE-RUN] run_id=${RUN_ID} root=${RUNS_ROOT} dry_run=${DRY_RUN} gpu=${GPU} seed=${SEED} ratios=0.07/0.63/0.15/0.15 roles=L_s/U_s/V_cal/V_select checkpoint_selection=final_only"
for level in M0 M1 M2 M3; do
  if candidate_selected "${level}"; then
    run_candidate "${level}"
  fi
done
