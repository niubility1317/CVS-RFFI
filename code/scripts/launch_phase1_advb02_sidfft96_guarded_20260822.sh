#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_advb02_sidfft96_guarded_20260822_v1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
BASELINE_CKPT="${BASELINE_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
SEED="${SEED:-392002}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
GPU_S0="${GPU_S0:-0}"
GPU_S3G="${GPU_S3G:-1}"
DRY_RUN="${DRY_RUN:-0}"
ONLY="${ONLY:-S0,S3G}"

readonly MATRIX="S0,S3G"
readonly LEO_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
readonly CORE90_LEO_SCHEDULE="1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
SID_MASK_PATH="${SID_MASK_PATH:-${ROOT}/runs/phase1_advb02_sidfft96_leo_weak_20260821_v1/P0_SPECTRAL_AUDIT/sid_mask.npz}"

for argument in "$@"; do
  case "${argument}" in
    --dry-run)
      DRY_RUN=1
      ;;
    --only=*)
      ONLY="${argument#--only=}"
      ;;
    *)
      echo "[SID-FFT96-ERROR] unknown argument: ${argument}" >&2
      exit 2
      ;;
  esac
done

validate_source_wisig_pkl() {
  local lower
  lower="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
  if [[ "${lower}" != *manysig.pkl ]]; then
    echo "[SID-FFT96-ERROR] refusing non-source Phase1 WISIG_PKL: expected ManySig.pkl, got $1" >&2
    exit 4
  fi
}

selected() {
  case ",${ONLY}," in
    *",$1,"*) return 0 ;;
    *) return 1 ;;
  esac
}

validate_selection() {
  local item
  IFS=',' read -r -a items <<< "${ONLY}"
  for item in "${items[@]}"; do
    case "${item}" in
      S0|S3G) ;;
      *)
        echo "[SID-FFT96-ERROR] unknown matrix row: ${item}" >&2
        exit 2
        ;;
    esac
  done
}

gpu_active_count() {
  local gpu="$1"
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="${gpu}" '$1 == gpu && $3 == "C" { count++ } END { print count + 0 }'
}

preflight_gpu() {
  local gpu="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[SID-FFT96-ERROR] nvidia-smi is required for bounded GPU preflight" >&2
    return 5
  fi
  local active
  active="$(gpu_active_count "${gpu}")"
  if [[ "${active}" -ge "${MAX_ACTIVE_PER_GPU}" ]]; then
    echo "[SID-FFT96-ERROR] gpu=${gpu} active=${active} cap=${MAX_ACTIVE_PER_GPU}" >&2
    return 5
  fi
}

print_command() {
  local label="$1"
  shift
  printf '[SID-FFT96-%s-CMD] ' "${label}"
  printf '%q ' "$@"
  printf '\n'
}

build_p0_command() {
  P0_CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${GPU_P0}"
    "${PYTHON}" -u "${ROOT}/code/scripts/audit_phase1_spectral_identifiability.py"
    --output_dir "${P0_DIR}"
    --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2
    --phase1_source_role_protocol l_s_u_s_v_cal_v_select
    --labeled_ratio 0.07
    --unlabeled_ratio 0.63
    --source_cal_ratio 0.15
    --source_select_ratio 0.15
    --source_val_ratio 0.30
    --wisig_out_len 256
    --fft_bins 256
    --num_bands 64
    --keep_fraction 0.50
    --dc_notch 1
    --max_batches 0
    --eval_batch_size 256
    --num_workers 4
    --sat_train_scenarios "${LEO_SCENARIOS}"
    --sat_view_schedule "${CORE90_LEO_SCHEDULE}"
    --device cuda:0
    --seed "${SEED}")
}

build_eval_command() {
  local checkpoint_path="$1"
  local eval_dir="$2"
  local gpu="$3"
  EVAL_CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${gpu}"
    "${PYTHON}" -u "${ROOT}/tools/eval_cvs_checkpoint_sat_channel.py"
    --ckpt "${checkpoint_path}"
    --device cuda:0
    --eval_batch_size 256
    --num_workers 4
    --prefetch_factor 2
    --eval_sat_channel
    --eval_sat_scenarios "${LEO_SCENARIOS}"
    --eval_sat_on main
    --sat_eval_max_batches 0
    --sat_seed 2027
    --output_json "${eval_dir}/final_eval.json"
    --output_txt "${eval_dir}/final_eval.txt")
}

build_train_command() {
  local candidate="$1"
  local mode="$2"
  local gpu="$3"
  local out_dir="$4"
  TRAIN_CMD=(env
    "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "CUDA_VISIBLE_DEVICES=${gpu}"
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
    --candidate_id "${candidate}"
    --base_candidate ADV3B02_CORE90_SOFT_E200
    --baseline_ckpt "${BASELINE_CKPT}"
    --from_scratch false
    --epochs 200
    --label_epochs 130
    --pseudo_epochs 70
    --batch_size 128
    --eval_batch_size 256
    --num_workers 4
    --prefetch_factor 2
    --lr 0.00002
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
    --checkpoint_selection source_validation_only
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
    --phase1_distribution_audit_only true
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
    --lambda_zid_compact 0.032
    --zid_compact_start_epoch 8
    --zid_compact_warmup_epochs 25
    --zid_compact_radius_deg 40
    --lambda_proxy_unknown 0.0045
    --proxy_unknown_start_epoch 45
    --proxy_unknown_warmup_epochs 25
    --lambda_soft_unknown_mixup 0.0045
    --soft_unknown_mixup_start_epoch 25
    --soft_unknown_mixup_warmup_epochs 25
    --lambda_source_episode 0.0035
    --source_episode_start_epoch 20
    --source_episode_warmup_epochs 25
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
    --use_ema_teacher false
    --use_sat_consistency
    --use_concat_sat_channel_aug
    --concat_sat_ce_only
    --sat_training_mode concat_ce_only
    --concat_sat_ce_weight 1.0
    --concat_sat_start_epoch 1
    --sat_train_scenarios "${LEO_SCENARIOS}"
    --sat_view_schedule "${CORE90_LEO_SCHEDULE}"
    --sat_view_prob 1.0
    --sat_view_seed "${SEED}"
    --sat_cons_start_epoch 80
    --lambda_sat_cls 0.68
    --lambda_sat_cons 0.0
    --eval_sat_channel true
    --eval_sat_scenarios "${LEO_SCENARIOS}"
    --eval_sat_on main
    --sat_eval_max_batches -1
    --sid_fft96_mode "${mode}"
    --sid_mask_path "${SID_MASK_PATH}"
    --sid_residual_scale 1.0
    --sid_max_residual_ratio 0.10
    --sid_adapter_only true
    --sid_guarded_training true
    --lambda_sid_identity_anchor 0.05
    --max_grad_norm 1.0
    --device cuda:0
    --seed "${SEED}")
}

run_p0() {
  local log_path="${LOG_ROOT}/P0_SPECTRAL_AUDIT.out"
  build_p0_command
  print_command P0 "${P0_CMD[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${P0_DIR}" || -e "${log_path}" ]]; then
    echo "[SID-FFT96-ERROR] refusing to overwrite P0 output or log" >&2
    return 3
  fi
  preflight_gpu "${GPU_P0}"
  mkdir -p "${LOG_ROOT}"
  "${P0_CMD[@]}" > "${log_path}" 2>&1
  test -f "${SID_MASK_PATH}"
}

run_s0() {
  local out_dir="${RUNS_ROOT}/S0_FROZEN_CORE90"
  local log_path="${LOG_ROOT}/S0_FROZEN_CORE90.out"
  build_eval_command "${BASELINE_CKPT}" "${out_dir}" "${GPU_S0}"
  print_command S0 "${EVAL_CMD[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${out_dir}" || -e "${log_path}" ]]; then
    echo "[SID-FFT96-ERROR] refusing to overwrite S0 output or log" >&2
    return 3
  fi
  test -f "${BASELINE_CKPT}"
  preflight_gpu "${GPU_S0}"
  mkdir -p "${LOG_ROOT}"
  "${EVAL_CMD[@]}" > "${log_path}" 2>&1
}

run_training_row() {
  local row="$1"
  local candidate="$2"
  local mode="$3"
  local gpu="$4"
  local out_dir="${RUNS_ROOT}/${candidate}"
  local train_log="${LOG_ROOT}/${candidate}.out"
  local eval_log="${LOG_ROOT}/${candidate}.final_eval.out"
  local eval_dir="${out_dir}/independent_final_eval"
  local checkpoint_path="${out_dir}/final_ssdg.pth"
  build_train_command "${candidate}" "${mode}" "${gpu}" "${out_dir}"
  build_eval_command "${checkpoint_path}" "${eval_dir}" "${gpu}"
  print_command "${row}-TRAIN" "${TRAIN_CMD[@]}"
  print_command "${row}-EVAL" "${EVAL_CMD[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${out_dir}" || -e "${train_log}" || -e "${eval_log}" ]]; then
    echo "[SID-FFT96-ERROR] refusing to overwrite ${row} output or log" >&2
    return 3
  fi
  test -f "${BASELINE_CKPT}"
  test -f "${SID_MASK_PATH}"
  preflight_gpu "${gpu}"
  mkdir -p "${LOG_ROOT}"
  "${TRAIN_CMD[@]}" > "${train_log}" 2>&1
  test -f "${checkpoint_path}"
  "${EVAL_CMD[@]}" > "${eval_log}" 2>&1
}

validate_source_wisig_pkl "${WISIG_PKL}"
validate_selection
if [[ "${SEED}" != "392002" ]]; then
  echo "[SID-FFT96-ERROR] seed must be 392002, got ${SEED}" >&2
  exit 4
fi

echo "[SID-FFT96-MATRIX] matrix=${MATRIX} selected=${ONLY} run_id=${RUN_ID} seed=${SEED} ratios=0.07/0.63/0.15/0.15 roles=L_s/U_s/V_cal/V_select scenarios=${LEO_SCENARIOS}"
selected S0 && run_s0
selected S3G && run_training_row S3G S3G_SIDFFT96_GUARDED sid "${GPU_S3G}"
