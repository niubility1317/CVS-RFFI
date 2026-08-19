#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_advb02_crra_leo_weak_20260819_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU="${GPU:-0}"
SEED="${SEED:-392034}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"

readonly CANDIDATE="ADVB02_CRRA_S_LEO_WEAK_E200"
readonly LEO_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
readonly CORE90_LEO_SCHEDULE="1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[CRRA-LEO-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

validate_source_wisig_pkl() {
  local pkl_path="$1"
  local lower
  lower="$(printf "%s" "${pkl_path}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    *manytx.pkl*|*manyrx.pkl*|*singleday.pkl*|*new_wisig*|*target*|*unknown*)
      echo "[CRRA-LEO-ERROR] refusing non-source Phase1 WISIG_PKL: ${pkl_path}" >&2
      exit 4
      ;;
  esac
  if [[ "${lower}" != *manysig.pkl ]]; then
    echo "[CRRA-LEO-ERROR] refusing non-source Phase1 WISIG_PKL: expected ManySig.pkl, got ${pkl_path}" >&2
    exit 4
  fi
}

gpu_active_count() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="${gpu}" '$1 == gpu && $3 == "C" { c++ } END { print c + 0 }'
}

build_train_command() {
  local out_dir="$1"
  TRAIN_CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --phase1_source_role_protocol l_s_u_s_v_cal_v_select
    --labeled_ratio 0.07
    --unlabeled_ratio 0.63
    --source_cal_ratio 0.15
    --source_select_ratio 0.15
    --source_val_ratio 0.30
    --output_dir "${out_dir}"
    --run_id "${RUN_ID}"
    --candidate_id "${CANDIDATE}"
    --base_candidate ADV3B02_CORE90_SOFT_E200
    --from_scratch true
    --epochs 200
    --label_epochs 130
    --pseudo_epochs 70
    --batch_size 128
    --eval_batch_size 256
    --num_workers 4
    --prefetch_factor 2
    --lr 0.0002
    --weight_decay 0.0001
    --model_size M
    --model_variant lite_d
    --branch_ablation no_dac
    --domain_branch_ablation no_stats
    --id_feature_key feat_joint
    --domain_enhancer rcn_stats
    --domain_enhancer_strength 0.35
    --use_mixstyle true
    --mixstyle_p 0.18
    --mixstyle_alpha 0.10
    --mixstyle_eps 1e-6
    --mixstyle_layers time_down,t1
    --mixstyle_use_domain_label true
    --mixstyle_mix same_tx_crossdomain
    --mixstyle_strength 0.70
    --mixstyle_fallback skip
    --mixstyle_late_start 110
    --mixstyle_late_ramp_epochs 40
    --mixstyle_late_min_p 0.05
    --mixstyle_late_min_strength 0.32
    --phase1_source_val_selection_only true
    --checkpoint_selection final_only
    --best_metric source_val_sat_hmean
    --test_eval_policy interval_final
    --test_eval_start_epoch 999999
    --test_eval_interval 0
    --test_eval_final_window 0
    --test_eval_final_interval 0
    --enable_joint_safe_guard false
    --paic_guard_enabled false
    --phase1_v2_hard_gates false
    --tail_safety_state_machine false
    --phase2_export_prototypes false
    --use_feature_masks true
    --use_txrx_geometry_losses true
    --use_tx_rx_balanced_sampler false
    --phase1_distribution_audit_only true
    --use_proto_memory true
    --lambda_tx_proto 0
    --lambda_rx_proto 0
    --lambda_mask_aux 0
    --lambda_tx_supcon_masked 0
    --lambda_rx_supcon_masked 0
    --lambda_txrx_rect 0
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
    --lambda_domain 1.0
    --lambda_adv 0.35
    --lambda_orth 0.05
    --lambda_cons 0.08
    --lambda_group_ce 0.16
    --lambda_fishr 0.04
    --lambda_u 0.16
    --lambda_ent 0.01
    --tau_min 0.92
    --tau_max 0.97
    --pseudo_quantile 0.86
    --use_unlabeled true
    --use_ema_teacher true
    --ema_decay 0.999
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --sat_training_mode concat_masked
    --concat_sat_ce_weight 1.0
    --concat_sat_start_epoch 1
    --sat_train_scenario leo_clear_weak
    --sat_train_scenarios "${LEO_SCENARIOS}"
    --sat_view_schedule "${CORE90_LEO_SCHEDULE}"
    --sat_view_prob 1.0
    --sat_view_seed "${SEED}"
    --sat_cons_start_epoch 17
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0.05
    --use_crra
    --crra_scenario leo_weak
    --crra_rank 8
    --crra_alpha_max 0.25
    --crra_shrinkage 0.10
    --crra_condition_dim 32
    --crra_nuisance_dim 9
    --crra_start_epoch 17
    --crra_ramp_epochs 30
    --crra_s3_lr_scale 0.25
    --crra_support_tau 1.0
    --crra_target_adapter false
    --lambda_crra_pair 0.05
    --lambda_crra_sat_kl 0.0
    --lambda_crra_sat_shell 0.0
    --lambda_crra_energy 0.001
    --lambda_crra_gate_l1 0.001
    --lambda_crra_nuisance 0.02
    --lambda_crra_condition_tx_adv 0.02
    --eval_sat_channel true
    --eval_sat_scenarios "${LEO_SCENARIOS}"
    --eval_sat_on main
    --sat_eval_max_batches -1
    --sat_seed 2027
    --device cuda:0
    --seed "${SEED}")
}

build_eval_command() {
  local checkpoint_path="$1"
  local eval_dir="$2"
  EVAL_CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU}"
    "${PYTHON}" -u "${ROOT}/tools/eval_cvs_checkpoint_sat_channel.py"
    --ckpt "${checkpoint_path}"
    --expect_run_name "${RUN_ID}"
    --device cuda:0
    --eval_batch_size 256
    --num_workers 4
    --prefetch_factor 2
    --eval_sat_channel
    --eval_sat_scenarios "${LEO_SCENARIOS}"
    --eval_sat_on main
    --sat_seed 2027
    --eval_crra_telemetry
    --output_json "${eval_dir}/final_eval.json"
    --output_txt "${eval_dir}/final_eval.txt")
}

run() {
  local out_dir="${RUNS_ROOT}/${CANDIDATE}"
  local train_log="${LOG_ROOT}/${CANDIDATE}.out"
  local eval_log="${LOG_ROOT}/${CANDIDATE}.final_eval.out"
  local status_path="${LOG_ROOT}/${CANDIDATE}.status"
  local checkpoint_path="${out_dir}/final_ssdg.pth"
  local eval_dir="${out_dir}/independent_final_eval"

  build_train_command "${out_dir}"
  build_eval_command "${checkpoint_path}" "${eval_dir}"
  echo "[CRRA-LEO-RUN] run_id=${RUN_ID} candidate=${CANDIDATE} gpu=${GPU} seed=${SEED} ratios=0.07/0.63/0.15/0.15 roles=L_s/U_s/V_cal/V_select channel=leo_weak only source_only=1 target_receiver_samples_in_training=0 target_unknown_training_count=0 crra_variant=CRRA-S sat_training_mode=concat_masked core90_schedule=E1-40_p0.30,E41-90_p0.60,E91-200_p0.80"
  printf "[CRRA-LEO-TRAIN-CMD] "; printf "%q " "${TRAIN_CMD[@]}"; printf "\n"
  printf "[CRRA-LEO-EVAL-CMD] "; printf "%q " "${EVAL_CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ -e "${out_dir}" || -e "${train_log}" || -e "${eval_log}" ]]; then
    echo "[CRRA-LEO-ERROR] refusing to overwrite candidate output or log for ${CANDIDATE}" >&2
    return 3
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[CRRA-LEO-ERROR] nvidia-smi is required for bounded GPU preflight" >&2
    return 5
  fi
  local active
  active="$(gpu_active_count "${GPU}")"
  if [[ "${active}" -ge "${MAX_ACTIVE_PER_GPU}" ]]; then
    echo "[CRRA-LEO-ERROR] gpu=${GPU} active=${active} cap=${MAX_ACTIVE_PER_GPU}" >&2
    return 5
  fi

  mkdir -p "${out_dir}" "${LOG_ROOT}"
  echo "running gpu=${GPU} active_before=${active} started=$(date -Is)" > "${status_path}"
  set +e
  "${TRAIN_CMD[@]}" > "${train_log}" 2>&1
  local train_status=$?
  set -e
  echo "train_exit=${train_status} finished=$(date -Is)" >> "${status_path}"

  local eval_status=0
  if [[ -f "${checkpoint_path}" ]]; then
    mkdir -p "${eval_dir}"
    echo "[CRRA-LEO-TEST] checkpoint=${checkpoint_path} scenarios=${LEO_SCENARIOS} telemetry=1"
    set +e
    "${EVAL_CMD[@]}" > "${eval_log}" 2>&1
    eval_status=$?
    set -e
    echo "eval_exit=${eval_status} finished=$(date -Is)" >> "${status_path}"
    echo "[CRRA-LEO-TEST-FINISHED] checkpoint=${checkpoint_path} exit=${eval_status} log=${eval_log} json=${eval_dir}/final_eval.json"
  else
    eval_status=6
    echo "eval_exit=${eval_status} reason=final_ssdg_missing finished=$(date -Is)" >> "${status_path}"
    echo "[CRRA-LEO-ERROR] final checkpoint missing; independent test was not runnable" >&2
  fi

  echo "[CRRA-LEO-FINISHED] candidate=${CANDIDATE} train_exit=${train_status} eval_exit=${eval_status}"
  if [[ "${train_status}" -ne 0 ]]; then
    return "${train_status}"
  fi
  return "${eval_status}"
}

validate_source_wisig_pkl "${WISIG_PKL}"
run
