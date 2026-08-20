#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_advb02_ntrs_leo_weak_20260820_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU="${GPU:-1}"
SEED="${SEED:-392034}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
NTRS_PROFILE="${NTRS_PROFILE:-full}"
REPEAT="${REPEAT:-r1}"
BASELINE_CKPT="${BASELINE_CKPT:-${ROOT}/runs/phase1_advb02_ntrs_v2_recovery_20260820_d1_bypass/ADVB02_NTRS_V2_D1_BYPASS_E200/final_ssdg.pth}"
A2_GATE_FILE="${A2_GATE_FILE:-}"
A3_GATE_FILE="${A3_GATE_FILE:-}"

readonly LEO_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
readonly CORE90_LEO_SCHEDULE="1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"

USE_NTRS=1
CANDIDATE="ADVB02_NTRS_LEO_WEAK_E200"
NTRS_ALPHA_MAX="0.20"
NTRS_VARIANT="v1"
NTRS_IDENTITY_BYPASS="false"
NTRS_CORE_LR_MODE="v1"
NTRS_CORE_LR_RATIO="0.02"
NTRS_Q_TRAINABLE="true"
NTRS_USE_SUPPORT_GATE="false"
NTRS_ADAPTER_ONLY="false"
FROM_SCRATCH="true"
USE_EMA_TEACHER="true"
TEACHER_CKPT=""
L_TEACHER_CLEAN_KL="0"
L_TEACHER_SAT_KL="0"
L_TEACHER_ZID_MSE="0"
L_NTRS_SAT_KL="0.01"
L_NTRS_ROBUST_CE="0"
L_NTRS_MARGIN="0.03"
L_NTRS_RELATION="0.02"
L_NTRS_CLASS_CONDITIONAL="0.01"
L_NTRS_RECEIVER="0.02"
L_NTRS_DAY="0.02"
L_NTRS_CHANNEL="0.02"
L_NTRS_COND_DECORR="0.01"
L_NTRS_SHARED_RX="0.01"
L_NTRS_CONTEXT_TX_ADV="0.02"
L_NTRS_MIN_CORRECTION="0.001"
L_NTRS_ALPHA="0.001"
L_NTRS_SUBSPACE="0.02"
L_NTRS_CORRECTABILITY="0.02"
L_NTRS_SCORE_STABILITY="0.01"
L_NTRS_CLASS_ATTRACTION="0.01"
L_NTRS_CLEAN_ZERO="0"
L_NTRS_SAT_RELATIVE="0"

configure_adapter_profile() {
  NTRS_VARIANT="v3_adapter"
  NTRS_IDENTITY_BYPASS="false"
  NTRS_CORE_LR_MODE="baseline"
  NTRS_CORE_LR_RATIO="0.02"
  NTRS_Q_TRAINABLE="true"
  NTRS_USE_SUPPORT_GATE="false"
  NTRS_ADAPTER_ONLY="true"
  FROM_SCRATCH="false"
  USE_EMA_TEACHER="false"
  NTRS_ALPHA_MAX="0.02"
  L_NTRS_SAT_KL="0"
  L_NTRS_ROBUST_CE="1.0"
  L_NTRS_MARGIN="0"
  L_NTRS_RELATION="0"
  L_NTRS_CLASS_CONDITIONAL="0"
  L_NTRS_RECEIVER="0"
  L_NTRS_DAY="0"
  L_NTRS_CHANNEL="0"
  L_NTRS_COND_DECORR="0"
  L_NTRS_SHARED_RX="0"
  L_NTRS_CONTEXT_TX_ADV="0"
  L_NTRS_MIN_CORRECTION="0"
  L_NTRS_ALPHA="0"
  L_NTRS_SUBSPACE="0"
  L_NTRS_CORRECTABILITY="0"
  L_NTRS_SCORE_STABILITY="0"
  L_NTRS_CLASS_ATTRACTION="0"
  L_NTRS_CLEAN_ZERO="1.0"
  L_NTRS_SAT_RELATIVE="0.10"
}

case "${NTRS_PROFILE}" in
  full)
    ;;
  control)
    USE_NTRS=0
    CANDIDATE="ADVB02_CORE90_LEO_WEAK_CONTROL_E200"
    ;;
  a0_control)
    USE_NTRS=0
    CANDIDATE="ADVB02_NTRS_A0_CONTROL_${REPEAT}_E200"
    ;;
  a0_bypass)
    NTRS_VARIANT="v2_min"
    NTRS_IDENTITY_BYPASS="true"
    NTRS_CORE_LR_MODE="baseline"
    CANDIDATE="ADVB02_NTRS_A0_BYPASS_${REPEAT}_E200"
    L_NTRS_SAT_KL="0"
    L_NTRS_ROBUST_CE="0"
    L_NTRS_MARGIN="0"
    L_NTRS_RELATION="0"
    L_NTRS_CLASS_CONDITIONAL="0"
    L_NTRS_RECEIVER="0"
    L_NTRS_DAY="0"
    L_NTRS_CHANNEL="0"
    L_NTRS_COND_DECORR="0"
    L_NTRS_SHARED_RX="0"
    L_NTRS_CONTEXT_TX_ADV="0"
    L_NTRS_MIN_CORRECTION="0"
    L_NTRS_ALPHA="0"
    L_NTRS_SUBSPACE="0"
    L_NTRS_CORRECTABILITY="0"
    L_NTRS_SCORE_STABILITY="0"
    L_NTRS_CLASS_ATTRACTION="0"
    ;;
  a1_random_q)
    configure_adapter_profile
    NTRS_Q_TRAINABLE="false"
    CANDIDATE="ADVB02_NTRS_A1_RANDOM_Q_${REPEAT}_E200"
    ;;
  a1_trainable_q)
    configure_adapter_profile
    CANDIDATE="ADVB02_NTRS_A1_TRAINABLE_Q_${REPEAT}_E200"
    ;;
  a2_teacher_margin)
    configure_adapter_profile
    NTRS_ALPHA_MAX="0.05"
    L_NTRS_SAT_KL="0.01"
    L_NTRS_MARGIN="0.03"
    CANDIDATE="ADVB02_NTRS_A2_TEACHER_MARGIN_${REPEAT}_E200"
    ;;
  a3_support_gate)
    configure_adapter_profile
    NTRS_ALPHA_MAX="0.05"
    NTRS_USE_SUPPORT_GATE="true"
    L_NTRS_SAT_KL="0.01"
    L_NTRS_MARGIN="0.03"
    CANDIDATE="ADVB02_NTRS_A3_SUPPORT_GATE_${REPEAT}_E200"
    ;;
  a4_joint_core)
    configure_adapter_profile
    NTRS_ALPHA_MAX="0.05"
    NTRS_USE_SUPPORT_GATE="true"
    NTRS_ADAPTER_ONLY="false"
    NTRS_CORE_LR_MODE="adapter_joint"
    NTRS_CORE_LR_RATIO="0.02"
    L_NTRS_SAT_KL="0.01"
    L_NTRS_MARGIN="0.03"
    TEACHER_CKPT="${BASELINE_CKPT}"
    L_TEACHER_CLEAN_KL="1.0"
    L_TEACHER_ZID_MSE="1.0"
    CANDIDATE="ADVB02_NTRS_A4_JOINT_CORE_${REPEAT}_E200"
    ;;
  no_identity_structure)
    CANDIDATE="ADVB02_NTRS_NO_IDSTRUCT_LEO_WEAK_E200"
    L_NTRS_SAT_KL="0"
    L_NTRS_MARGIN="0"
    L_NTRS_RELATION="0"
    L_NTRS_CLASS_CONDITIONAL="0"
    ;;
  no_nuisance_factorization)
    CANDIDATE="ADVB02_NTRS_NO_NUISANCE_LEO_WEAK_E200"
    L_NTRS_RECEIVER="0"
    L_NTRS_DAY="0"
    L_NTRS_CHANNEL="0"
    L_NTRS_COND_DECORR="0"
    L_NTRS_SHARED_RX="0"
    L_NTRS_CONTEXT_TX_ADV="0"
    ;;
  no_embed_residual)
    CANDIDATE="ADVB02_NTRS_NO_EMBEDRES_LEO_WEAK_E200"
    NTRS_ALPHA_MAX="0"
    L_NTRS_MIN_CORRECTION="0"
    L_NTRS_ALPHA="0"
    L_NTRS_SUBSPACE="0"
    ;;
  no_safety_losses)
    CANDIDATE="ADVB02_NTRS_NO_SAFETY_LEO_WEAK_E200"
    L_NTRS_CORRECTABILITY="0"
    L_NTRS_SCORE_STABILITY="0"
    L_NTRS_CLASS_ATTRACTION="0"
    ;;
  v2_identity_bypass|v2_identity_bypass_v1_lr)
    NTRS_VARIANT="v2_min"
    NTRS_IDENTITY_BYPASS="true"
    NTRS_CORE_LR_MODE="baseline"
    CANDIDATE="ADVB02_NTRS_V2_D1_BYPASS_E200"
    if [[ "${NTRS_PROFILE}" == "v2_identity_bypass_v1_lr" ]]; then
      NTRS_CORE_LR_MODE="v1"
      CANDIDATE="ADVB02_NTRS_V2_D2_BYPASS_V1LR_E200"
    fi
    L_NTRS_SAT_KL="0"
    L_NTRS_MARGIN="0"
    L_NTRS_RELATION="0"
    L_NTRS_CLASS_CONDITIONAL="0"
    L_NTRS_RECEIVER="0"
    L_NTRS_DAY="0"
    L_NTRS_CHANNEL="0"
    L_NTRS_COND_DECORR="0"
    L_NTRS_SHARED_RX="0"
    L_NTRS_CONTEXT_TX_ADV="0"
    L_NTRS_MIN_CORRECTION="0"
    L_NTRS_ALPHA="0"
    L_NTRS_SUBSPACE="0"
    L_NTRS_CORRECTABILITY="0"
    L_NTRS_SCORE_STABILITY="0"
    L_NTRS_CLASS_ATTRACTION="0"
    ;;
  v1_fair_core_lr)
    CANDIDATE="ADVB02_NTRS_V1_D3_FAIRLR_E200"
    NTRS_CORE_LR_MODE="baseline"
    ;;
  v2_min_shared_head)
    CANDIDATE="ADVB02_NTRS_V2_MIN_SHARED_E200"
    NTRS_VARIANT="v2_min"
    NTRS_CORE_LR_MODE="baseline"
    L_NTRS_ROBUST_CE="1.0"
    L_NTRS_RELATION="0"
    L_NTRS_CLASS_CONDITIONAL="0"
    L_NTRS_RECEIVER="0"
    L_NTRS_DAY="0"
    L_NTRS_CHANNEL="0"
    L_NTRS_COND_DECORR="0"
    L_NTRS_SHARED_RX="0"
    L_NTRS_CONTEXT_TX_ADV="0"
    L_NTRS_ALPHA="0"
    L_NTRS_SUBSPACE="0"
    L_NTRS_CORRECTABILITY="0"
    L_NTRS_SCORE_STABILITY="0"
    L_NTRS_CLASS_ATTRACTION="0"
    ;;
  *)
    echo "[NTRS-LEO-ERROR] unknown NTRS_PROFILE: ${NTRS_PROFILE}" >&2
    exit 2
    ;;
esac
readonly CANDIDATE

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[NTRS-LEO-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

validate_source_wisig_pkl() {
  local pkl_path="$1"
  local lower
  lower="$(printf "%s" "${pkl_path}" | tr '[:upper:]' '[:lower:]')"
  case "${lower}" in
    *manytx.pkl*|*manyrx.pkl*|*singleday.pkl*|*new_wisig*|*target*|*unknown*)
      echo "[NTRS-LEO-ERROR] refusing non-source Phase1 WISIG_PKL: ${pkl_path}" >&2
      exit 4
      ;;
  esac
  if [[ "${lower}" != *manysig.pkl ]]; then
    echo "[NTRS-LEO-ERROR] refusing non-source Phase1 WISIG_PKL: expected ManySig.pkl, got ${pkl_path}" >&2
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
    --baseline_ckpt "${BASELINE_CKPT}"
    --from_scratch "${FROM_SCRATCH}"
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
    --use_ema_teacher "${USE_EMA_TEACHER}"
    --ema_decay 0.999
    --teacher_ckpt "${TEACHER_CKPT}"
    --lambda_teacher_clean_kl "${L_TEACHER_CLEAN_KL}"
    --lambda_teacher_sat_kl "${L_TEACHER_SAT_KL}"
    --lambda_teacher_zid_mse "${L_TEACHER_ZID_MSE}"
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
    --lambda_sat_cons 0.0
    --eval_sat_channel true
    --eval_sat_scenarios "${LEO_SCENARIOS}"
    --eval_sat_on main
    --sat_eval_max_batches -1
    --sat_seed 2027
    --device cuda:0
    --seed "${SEED}")
  if [[ "${USE_NTRS}" == "1" ]]; then
    TRAIN_CMD+=(
      --use_ntrs
      --ntrs_rank 8
      --ntrs_alpha_max "${NTRS_ALPHA_MAX}"
      --ntrs_q_dim 32
      --ntrs_fast_dim 24
      --ntrs_slow_dim 24
      --ntrs_metadata_dim 9
      --ntrs_slow_ema_decay 0.95
      --ntrs_support_tau 1.0
      --ntrs_energy_threshold 0.10
      --ntrs_unknown_rescue false
      --ntrs_target_adapter false
      --ntrs_variant "${NTRS_VARIANT}"
      --ntrs_identity_bypass "${NTRS_IDENTITY_BYPASS}"
      --ntrs_q_trainable "${NTRS_Q_TRAINABLE}"
      --ntrs_use_support_gate "${NTRS_USE_SUPPORT_GATE}"
      --ntrs_adapter_only "${NTRS_ADAPTER_ONLY}"
      --ntrs_core_lr_mode "${NTRS_CORE_LR_MODE}"
      --ntrs_core_lr_ratio "${NTRS_CORE_LR_RATIO}"
      --ntrs_margin_epsilon 0.05
      --ntrs_correctability_epsilon 0.01
      --ntrs_class_attraction_max_cosine 0.50
      --lambda_ntrs_sat_kl "${L_NTRS_SAT_KL}"
      --lambda_ntrs_robust_ce "${L_NTRS_ROBUST_CE}"
      --lambda_ntrs_margin "${L_NTRS_MARGIN}"
      --lambda_ntrs_relation "${L_NTRS_RELATION}"
      --lambda_ntrs_class_conditional "${L_NTRS_CLASS_CONDITIONAL}"
      --lambda_ntrs_receiver "${L_NTRS_RECEIVER}"
      --lambda_ntrs_day "${L_NTRS_DAY}"
      --lambda_ntrs_channel "${L_NTRS_CHANNEL}"
      --lambda_ntrs_cond_decorr "${L_NTRS_COND_DECORR}"
      --lambda_ntrs_shared_rx "${L_NTRS_SHARED_RX}"
      --lambda_ntrs_context_tx_adv "${L_NTRS_CONTEXT_TX_ADV}"
      --lambda_ntrs_min_correction "${L_NTRS_MIN_CORRECTION}"
      --lambda_ntrs_alpha "${L_NTRS_ALPHA}"
      --lambda_ntrs_subspace "${L_NTRS_SUBSPACE}"
      --lambda_ntrs_correctability "${L_NTRS_CORRECTABILITY}"
      --lambda_ntrs_score_stability "${L_NTRS_SCORE_STABILITY}"
      --lambda_ntrs_class_attraction "${L_NTRS_CLASS_ATTRACTION}")
    TRAIN_CMD+=(
      --lambda_ntrs_clean_zero "${L_NTRS_CLEAN_ZERO}"
      --lambda_ntrs_sat_relative "${L_NTRS_SAT_RELATIVE}")
  else
    TRAIN_CMD+=(--no_use_ntrs)
  fi
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
    --sat_seed 2027)
  if [[ "${USE_NTRS}" == "1" ]]; then
    EVAL_CMD+=(--eval_ntrs_telemetry)
  fi
  EVAL_CMD+=(
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
  echo "[NTRS-LEO-RUN] run_id=${RUN_ID} candidate=${CANDIDATE} profile=${NTRS_PROFILE} repeat=${REPEAT} gpu=${GPU} seed=${SEED} ratios=0.07/0.63/0.15/0.15 roles=L_s/U_s/V_cal/V_select channel=leo_weak only source_only=1 target_receiver_samples_in_training=0 target_unknown_training_count=0 ntrs_variant=${NTRS_VARIANT} ntrs_identity_bypass=${NTRS_IDENTITY_BYPASS} ntrs_q_trainable=${NTRS_Q_TRAINABLE} ntrs_adapter_only=${NTRS_ADAPTER_ONLY} ntrs_support_gate=${NTRS_USE_SUPPORT_GATE} ntrs_core_lr_mode=${NTRS_CORE_LR_MODE} ntrs_core_lr_ratio=${NTRS_CORE_LR_RATIO} from_scratch=${FROM_SCRATCH} baseline_ckpt=${BASELINE_CKPT} sat_training_mode=concat_masked core90_schedule=E1-40_p0.30,E41-90_p0.60,E91-200_p0.80"
  printf "[NTRS-LEO-TRAIN-CMD] "; printf "%q " "${TRAIN_CMD[@]}"; printf "\n"
  printf "[NTRS-LEO-EVAL-CMD] "; printf "%q " "${EVAL_CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ "${FROM_SCRATCH}" == "false" && ! -f "${BASELINE_CKPT}" ]]; then
    echo "[NTRS-LEO-ERROR] required mature checkpoint missing: ${BASELINE_CKPT}" >&2
    return 6
  fi
  if [[ "${NTRS_PROFILE}" == "a3_support_gate" && ( -z "${A2_GATE_FILE}" || ! -f "${A2_GATE_FILE}" ) ]]; then
    echo "[NTRS-LEO-ERROR] A3 requires an A2 promotion marker via A2_GATE_FILE" >&2
    return 6
  fi
  if [[ "${NTRS_PROFILE}" == "a4_joint_core" && ( -z "${A3_GATE_FILE}" || ! -f "${A3_GATE_FILE}" ) ]]; then
    echo "[NTRS-LEO-ERROR] A4 requires an A3 promotion marker via A3_GATE_FILE" >&2
    return 6
  fi

  if [[ -e "${out_dir}" || -e "${train_log}" || -e "${eval_log}" ]]; then
    echo "[NTRS-LEO-ERROR] refusing to overwrite candidate output or log for ${CANDIDATE}" >&2
    return 3
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[NTRS-LEO-ERROR] nvidia-smi is required for bounded GPU preflight" >&2
    return 5
  fi
  local active
  active="$(gpu_active_count "${GPU}")"
  if [[ "${active}" -ge "${MAX_ACTIVE_PER_GPU}" ]]; then
    echo "[NTRS-LEO-ERROR] gpu=${GPU} active=${active} cap=${MAX_ACTIVE_PER_GPU}" >&2
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
    echo "[NTRS-LEO-TEST] checkpoint=${checkpoint_path} scenarios=${LEO_SCENARIOS} telemetry=1"
    set +e
    "${EVAL_CMD[@]}" > "${eval_log}" 2>&1
    eval_status=$?
    set -e
    echo "eval_exit=${eval_status} finished=$(date -Is)" >> "${status_path}"
    echo "[NTRS-LEO-TEST-FINISHED] checkpoint=${checkpoint_path} exit=${eval_status} log=${eval_log} json=${eval_dir}/final_eval.json"
  else
    eval_status=6
    echo "eval_exit=${eval_status} reason=final_ssdg_missing finished=$(date -Is)" >> "${status_path}"
    echo "[NTRS-LEO-ERROR] final checkpoint missing; independent test was not runnable" >&2
  fi

  echo "[NTRS-LEO-FINISHED] candidate=${CANDIDATE} train_exit=${train_status} eval_exit=${eval_status}"
  if [[ "${train_status}" -ne 0 ]]; then
    return "${train_status}"
  fi
  return "${eval_status}"
}

validate_source_wisig_pkl "${WISIG_PKL}"
if [[ "${SEED}" != "392034" ]]; then
  echo "[NTRS-LEO-ERROR] seed must be 392034, got ${SEED}" >&2
  exit 4
fi
run
