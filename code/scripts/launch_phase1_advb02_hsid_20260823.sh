#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_advb02_hsid_minimal_s392002_20260823_v1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
BASELINE_CKPT="${BASELINE_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
SEED="${SEED:-392002}"
GPU_0="${GPU_0:-0}"
GPU_1="${GPU_1:-1}"
MAX_ACTIVE_PER_GPU="${MAX_ACTIVE_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
P0_ONLY="${P0_ONLY:-0}"
ONLY="${ONLY:-S0,R3,X0,F0,X2}"

readonly MATRIX="S0,R3,X0,F0,X2"
readonly LEO_SCENARIOS="leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
readonly CORE90_LEO_SCHEDULE="1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak"
P0_DIR="${RUNS_ROOT}/P0_SPECTRAL_AUDIT"
LEGACY_MASK="${P0_DIR}/sid_mask.npz"
HIERARCHICAL_MASK="${P0_DIR}/sid_mask_hierarchical.npz"

for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY_RUN=1 ;;
    --prepare-p0) P0_ONLY=1 ;;
    --only=*) ONLY="${argument#--only=}" ;;
    *) echo "[HSID-ERROR] unknown argument: ${argument}" >&2; exit 2 ;;
  esac
done

selected() {
  case ",${ONLY}," in *",$1,"*) return 0 ;; *) return 1 ;; esac
}

validate_selection() {
  local item
  IFS=',' read -r -a items <<< "${ONLY}"
  for item in "${items[@]}"; do
    case "${item}" in S0|R3|X0|F0|X2) ;; *) echo "[HSID-ERROR] unknown matrix row: ${item}" >&2; exit 2 ;; esac
  done
}

gpu_active_count() {
  nvidia-smi pmon -c 1 2>/dev/null | awk -v gpu="$1" '$1 == gpu && $3 == "C" { count++ } END { print count + 0 }'
}

preflight_gpu() {
  local gpu="$1" active
  command -v nvidia-smi >/dev/null 2>&1 || { echo "[HSID-ERROR] nvidia-smi unavailable" >&2; return 5; }
  active="$(gpu_active_count "${gpu}")"
  if [[ "${active}" -ge "${MAX_ACTIVE_PER_GPU}" ]]; then
    echo "[HSID-ERROR] gpu=${gpu} active=${active} cap=${MAX_ACTIVE_PER_GPU}" >&2
    return 5
  fi
}

print_command() {
  local label="$1"; shift
  printf '[HSID-%s-CMD] ' "${label}"
  printf '%q ' "$@"
  printf '\n'
}

build_p0_command() {
  P0_CMD=(env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU_0}"
    "${PYTHON}" -u "${ROOT}/code/scripts/audit_phase1_spectral_identifiability.py"
    --output_dir "${P0_DIR}" --wisig_pkl "${WISIG_PKL}"
    --split_mode tx_rx_day_1_7_2 --phase1_source_role_protocol l_s_u_s_v_cal_v_select
    --labeled_ratio 0.07 --unlabeled_ratio 0.63 --source_cal_ratio 0.15 --source_select_ratio 0.15 --source_val_ratio 0.30
    --wisig_out_len 256 --fft_bins 256 --num_bands 64 --keep_fraction 0.50 --dc_notch 1 --max_batches 0
    --bootstrap_repeats 64 --bootstrap_keep_fraction 0.30
    --eval_batch_size 256 --num_workers 4 --sat_train_scenarios "${LEO_SCENARIOS}"
    --sat_view_schedule "${CORE90_LEO_SCHEDULE}" --device cuda:0 --seed "${SEED}")
}

build_smoke_command() {
  local output_json="$1"
  SMOKE_CMD=(env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU_0}"
    "${PYTHON}" -u "${ROOT}/code/scripts/smoke_phase1_advb02_sidfft96.py"
    --checkpoint "${BASELINE_CKPT}" --wisig_pkl "${WISIG_PKL}"
    --sid_mask_path "${HIERARCHICAL_MASK}" --sid_architecture hsid
    --output_json "${output_json}" --device cuda:0 --batch_size 4 --seed "${SEED}")
}

build_eval_command() {
  local checkpoint="$1" output="$2" gpu="$3"
  EVAL_CMD=(env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}"
    "${PYTHON}" -u "${ROOT}/tools/eval_cvs_checkpoint_sat_channel.py"
    --ckpt "${checkpoint}" --device cuda:0 --eval_batch_size 256 --num_workers 4 --prefetch_factor 2
    --eval_sat_channel --eval_sat_scenarios "${LEO_SCENARIOS}" --eval_sat_on main --sat_eval_max_batches 0 --sat_seed 2027
    --output_json "${output}/final_eval.json" --output_txt "${output}/final_eval.txt")
}

build_train_command() {
  local candidate="$1" fusion_mode="$2" mask_path="$3" best_metric="$4" cross_rx="$5" receiver_cvar="$6" interaction="$7" margin_safety="$8" gpu="$9" output_dir="${10}"
  TRAIN_CMD=(env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}"
    "${PYTHON}" -u "${ROOT}/code/SSDG/train_ssdg.py"
    --wisig_pkl "${WISIG_PKL}" --split_mode tx_rx_day_1_7_2
    --phase1_source_role_protocol l_s_u_s_v_cal_v_select
    --labeled_ratio 0.07 --unlabeled_ratio 0.63 --source_cal_ratio 0.15 --source_select_ratio 0.15 --source_val_ratio 0.30
    --output_dir "${output_dir}" --run_id "${RUN_ID}" --candidate_id "${candidate}"
    --base_candidate ADV3B02_CORE90_SOFT_E200 --baseline_ckpt "${BASELINE_CKPT}" --from_scratch false
    --epochs 200 --label_epochs 130 --pseudo_epochs 70 --batch_size 128 --eval_batch_size 256 --num_workers 4 --prefetch_factor 2
    --lr 0.00002 --weight_decay 0.0001 --model_size M --model_variant lite_d --branch_ablation no_dac
    --domain_branch_ablation no_stats --id_feature_key feat_joint --domain_enhancer rcn_stats --domain_enhancer_strength 0.35
    --use_mixstyle true --mixstyle_p 0.18 --mixstyle_alpha 0.10 --mixstyle_eps 1e-6 --mixstyle_layers time_down,t1
    --mixstyle_use_domain_label true --mixstyle_mix same_tx_crossdomain --mixstyle_strength 0.70 --mixstyle_fallback skip
    --mixstyle_late_start 110 --mixstyle_late_ramp_epochs 40 --mixstyle_late_min_p 0.05 --mixstyle_late_min_strength 0.32
    --phase1_source_val_selection_only true --checkpoint_selection source_validation_only --best_metric "${best_metric}"
    --source_val_heavy_eval_start_epoch 1 --source_val_heavy_eval_interval 10 --source_val_heavy_eval_final_window 20 --source_val_heavy_eval_final_interval 2
    --test_eval_policy interval_final --test_eval_start_epoch 999999 --test_eval_interval 0 --test_eval_final_window 0 --test_eval_final_interval 0
    --enable_joint_safe_guard false --paic_guard_enabled false --phase1_v2_hard_gates false --tail_safety_state_machine false
    --phase2_export_prototypes false --phase1_distribution_audit_only true
    --lambda_domain 1.0 --lambda_adv 0.35 --lambda_orth 0.05 --lambda_cons 0.08 --lambda_group_ce 0.16 --lambda_fishr 0.04
    --lambda_u 0.16 --lambda_ent 0.01 --tau_min 0.92 --tau_max 0.97 --pseudo_quantile 0.86 --use_unlabeled true --use_ema_teacher false
    --use_sat_consistency --use_concat_sat_channel_aug --concat_sat_ce_only --sat_training_mode concat_ce_only
    --concat_sat_ce_weight 1.0 --concat_sat_start_epoch 1 --sat_train_scenarios "${LEO_SCENARIOS}"
    --sat_view_schedule "${CORE90_LEO_SCHEDULE}" --sat_view_prob 1.0 --sat_view_seed "${SEED}"
    --sat_cons_start_epoch 80 --lambda_sat_cls 0.68 --lambda_sat_cons 0.0
    --eval_sat_channel true --eval_sat_scenarios "${LEO_SCENARIOS}" --eval_sat_on main --sat_eval_max_batches -1
    --sid_fft96_mode sid --sid_mask_path "${mask_path}" --sid_architecture hsid --sid_fusion_mode "${fusion_mode}"
    --sid_spectral_dim 48 --sid_fusion_alpha_max 0.20 --sid_residual_scale 0.0 --sid_max_residual_ratio 0.0
    --sid_adapter_only true --sid_guarded_training true --lambda_sid_identity_anchor 0.0
    --lambda_hsid_cross_rx "${cross_rx}" --lambda_hsid_receiver_cvar "${receiver_cvar}"
    --lambda_hsid_interaction "${interaction}" --lambda_hsid_margin_safety "${margin_safety}" --hsid_harm_margin 0.50
    --max_grad_norm 1.0 --device cuda:0 --seed "${SEED}")
}

run_p0() {
  local log="${LOG_ROOT}/P0_SPECTRAL_AUDIT.out"
  build_p0_command; print_command P0 "${P0_CMD[@]}"
  [[ "${DRY_RUN}" == 1 ]] && return 0
  [[ ! -e "${P0_DIR}" && ! -e "${log}" ]] || { echo "[HSID-ERROR] refusing to overwrite P0" >&2; return 3; }
  preflight_gpu "${GPU_0}"; mkdir -p "${LOG_ROOT}"; "${P0_CMD[@]}" >"${log}" 2>&1
  test -f "${LEGACY_MASK}"; test -f "${HIERARCHICAL_MASK}"
}

run_s0() {
  local output="${RUNS_ROOT}/S0_CORE90" log="${LOG_ROOT}/S0_CORE90.out"
  build_eval_command "${BASELINE_CKPT}" "${output}" "${GPU_0}"; print_command S0 "${EVAL_CMD[@]}"
  [[ "${DRY_RUN}" == 1 ]] && return 0
  [[ ! -e "${output}" && ! -e "${log}" ]] || { echo "[HSID-ERROR] refusing to overwrite S0" >&2; return 3; }
  test -f "${BASELINE_CKPT}"; preflight_gpu "${GPU_0}"; mkdir -p "${LOG_ROOT}"; "${EVAL_CMD[@]}" >"${log}" 2>&1
}

run_smoke() {
  local smoke_dir="${RUNS_ROOT}/SMOKE" json="${RUNS_ROOT}/SMOKE/smoke.json" log="${LOG_ROOT}/SMOKE.out"
  build_smoke_command "${json}"; print_command SMOKE "${SMOKE_CMD[@]}"
  [[ "${DRY_RUN}" == 1 ]] && return 0
  [[ ! -e "${smoke_dir}" && ! -e "${log}" ]] || { echo "[HSID-ERROR] refusing to overwrite smoke" >&2; return 3; }
  test -f "${BASELINE_CKPT}"; test -f "${HIERARCHICAL_MASK}"; preflight_gpu "${GPU_0}"; mkdir -p "${LOG_ROOT}"
  "${SMOKE_CMD[@]}" >"${log}" 2>&1
  "${PYTHON}" -c 'import json,sys; p=json.load(open(sys.argv[1],encoding="utf-8")); assert p["status"]=="VERIFIED"; assert p["query_input_count"]==0; assert p["target_input_count"]==0; assert p["raw_trainable_parameters"]==0; assert p["primary_raw_logit_max_abs"]==0.0; assert p["all_outputs_finite"] is True' "${json}"
}

run_training_row() {
  local row="$1" candidate="$2" fusion="$3" mask="$4" best="$5" cross="$6" cvar="$7" interaction="$8" safety="$9" gpu="${10}"
  local output="${RUNS_ROOT}/${candidate}" train_log="${LOG_ROOT}/${candidate}.out" eval_log="${LOG_ROOT}/${candidate}.final_eval.out"
  local checkpoint="${output}/final_ssdg.pth" eval_dir="${output}/independent_final_eval"
  build_train_command "${candidate}" "${fusion}" "${mask}" "${best}" "${cross}" "${cvar}" "${interaction}" "${safety}" "${gpu}" "${output}"
  build_eval_command "${checkpoint}" "${eval_dir}" "${gpu}"
  EVAL_CMD+=(--hsid_predictions_npz "${eval_dir}/hsid_predictions.npz")
  print_command "${row}-TRAIN" "${TRAIN_CMD[@]}"; print_command "${row}-EVAL" "${EVAL_CMD[@]}"
  [[ "${DRY_RUN}" == 1 ]] && return 0
  [[ ! -e "${output}" && ! -e "${train_log}" && ! -e "${eval_log}" ]] || { echo "[HSID-ERROR] refusing to overwrite ${row}" >&2; return 3; }
  test -f "${BASELINE_CKPT}"; test -f "${mask}"; preflight_gpu "${gpu}"; mkdir -p "${LOG_ROOT}"
  "${TRAIN_CMD[@]}" >"${train_log}" 2>&1
  test -f "${checkpoint}"; "${EVAL_CMD[@]}" >"${eval_log}" 2>&1
}

validate_selection
[[ "${WISIG_PKL,,}" == *manysig.pkl ]] || { echo "[HSID-ERROR] source ManySig.pkl is required" >&2; exit 4; }
[[ "${SEED}" == 392002 ]] || { echo "[HSID-ERROR] seed must be 392002" >&2; exit 4; }
echo "[HSID-MATRIX] matrix=${MATRIX} selected=${ONLY} run_id=${RUN_ID} seed=${SEED} roles=L_s/U_s/V_cal/V_select"

if [[ "${P0_ONLY}" == 1 ]]; then
  run_p0
  exit 0
fi
if selected R3 || selected X0 || selected F0 || selected X2; then
  run_smoke
fi
selected S0 && run_s0

pids=()
selected R3 && { run_training_row R3 R3_SPEC_PROTO spec "${LEGACY_MASK}" source_val_sat_hmean 0 0 0 0 "${GPU_0}" & pids+=("$!"); }
selected X0 && { run_training_row X0 X0_HIER_PROTO spec "${HIERARCHICAL_MASK}" source_hsid 0 0 0 0 "${GPU_1}" & pids+=("$!"); }
for pid in "${pids[@]}"; do wait "${pid}"; done

pids=()
selected F0 && { run_training_row F0 F0_HIER_FUSION fused "${HIERARCHICAL_MASK}" source_hsid 0 0 0 0.10 "${GPU_0}" & pids+=("$!"); }
selected X2 && { run_training_row X2 X2_RX_ROBUST fused "${HIERARCHICAL_MASK}" source_hsid 0.05 0.10 0.02 0.10 "${GPU_1}" & pids+=("$!"); }
for pid in "${pids[@]}"; do wait "${pid}"; done

echo "[HSID-COMPLETE] run_id=${RUN_ID} selected=${ONLY}"
