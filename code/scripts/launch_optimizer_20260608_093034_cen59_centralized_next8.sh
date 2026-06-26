#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-optimizer_20260608_093034_cen59_centralized_next8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' | wc -l | tr -d ' '
}

print_cmd() { printf '%q ' "$@"; printf '\n'; }

should_skip() {
  local candidate_id="$1" run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}

declare -A LAUNCHED_BY_GPU=()

run_cmd() {
  local candidate_id="$1" run_name="$2" gpu="$3" run_dir="$4" log_path="$5"
  shift 5
  local cmd=("$@")

  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  echo "[CEN59-CANDIDATE] lane=centralized candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CEN59-CMD]'; print_cmd "${cmd[@]}"
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "NON_LAUNCH_DIAGNOSTIC_PATH_COLLISION" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi
  local current_count local_count
  current_count="$(gpu_process_count "${gpu}")"
  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s local_count=%s max=%s\n" "${candidate_id}" "${run_name}" "DEFERRED_RETRY_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi
  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  LAUNCHED_BY_GPU["${gpu}"]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

COMMON_CEN_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --epochs 200
  --test_eval_policy interval_final
  --test_eval_start_epoch 1
  --test_eval_interval 10
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches -1
  --arch_family cvsincnet
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
  --use_aug
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.70
  --mixstyle_p 0.18
  --mixstyle_late_start 110
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.05
  --mixstyle_late_min_strength 0.32
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
  --lambda_group_ce 0.06
  --group_ce_mode smooth_dro_capped
  --group_ce_top_frac 0.35
  --groupdro_tau 0.50
  --groupdro_cap 0.65
  --use_proto_memory
  --lambda_proto 0.015
  --proto_momentum 0.95
  --lambda_supcon_id 0.02
  --supcon_temp 0.12
  --lambda_fishr 0.005
  --fishr_min_domains 4
  --generalization_feature z_id
  --collapse_guard
  --collapse_guard_min_epoch 35
  --collapse_guard_best_margin 12.0
  --collapse_guard_max_skipped_delta 2
  --use_ema_ckpt
  --ema_decay 0.999
  --use_swad_ckpt
  --swad_start_epoch 90
  --swad_tolerance 0.8
)

launch_cen_train() {
  local candidate_id="$1" run_name="$2" gpu="$3"
  shift 3
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_CEN_ARGS[@]}"
    --run_name "${run_name}"
    "$@"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --best_worst_rx_save_path "${run_dir}/best_worst_rx_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi

if [[ "${DRY_RUN}" == "1" && ! -d "${ROOT}" ]]; then
  :
else
  cd "${ROOT}"
fi
echo "[CEN59] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_cen_train CEN59_R01 CEN59_R01_r01_rxfloor73_anchor_repair_cons_r010 0 \
  --primary_udu_weight 0.908 --concat_sat_ce_weight 1.18 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.36 --mixstyle_late_start 126 --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.032 --group_ce_mode dual_worst --lambda_group_ce 0.086 --group_ce_min_domains 4 --group_ce_top_frac 0.10 --groupdro_tau 0.24 --groupdro_cap 0.33 --lambda_proto 0.032 --proto_momentum 0.982 --lambda_supcon_id 0.042 --lambda_fishr 0.000 --fishr_min_domains 4 --use_sat_consistency --lambda_sat_cons 0.005 --sat_cons_start_epoch 118 --swad_start_epoch 46 --swad_tolerance 0.18 --collapse_guard_best_margin 5.0 --sat_view_schedule '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;118@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'

launch_cen_train CEN59_R02 CEN59_R02_a06_sat_rx_balance_cons_r010 1 \
  --primary_udu_weight 0.905 --concat_sat_ce_weight 1.20 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.36 --group_ce_mode dual_worst --lambda_group_ce 0.086 --group_ce_min_domains 4 --group_ce_top_frac 0.10 --groupdro_tau 0.24 --groupdro_cap 0.34 --lambda_proto 0.034 --proto_momentum 0.983 --lambda_supcon_id 0.042 --lambda_fishr 0.000 --fishr_min_domains 4 --mixstyle_late_min_p 0.003 --mixstyle_late_min_strength 0.040 --use_sat_consistency --lambda_sat_cons 0.006 --sat_cons_start_epoch 118 --swad_start_epoch 46 --swad_tolerance 0.19 --collapse_guard_best_margin 5.1 --sat_view_schedule '1@0.99:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;118@0.91:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'

launch_cen_train CEN59_R03 CEN59_R03_a07_strict_rx8_guard_cons_r010 2 \
  --primary_udu_weight 0.912 --concat_sat_ce_weight 1.17 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.36 --mixstyle_late_start 124 --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.035 --group_ce_mode dual_worst --lambda_group_ce 0.088 --group_ce_min_domains 4 --group_ce_top_frac 0.10 --groupdro_tau 0.24 --groupdro_cap 0.33 --lambda_proto 0.032 --proto_momentum 0.982 --lambda_supcon_id 0.044 --lambda_fishr 0.000 --fishr_min_domains 4 --use_sat_consistency --lambda_sat_cons 0.004 --sat_cons_start_epoch 126 --swad_start_epoch 42 --swad_tolerance 0.18 --ema_decay 0.9995 --collapse_guard_best_margin 5.0

launch_cen_train CEN59_R04 CEN59_R04_r03_satfloor_rx_rescue_cons_r010 3 \
  --primary_udu_weight 0.900 --concat_sat_ce_weight 1.18 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.36 --group_ce_mode dual_worst --lambda_group_ce 0.090 --group_ce_min_domains 4 --group_ce_top_frac 0.09 --groupdro_tau 0.23 --groupdro_cap 0.32 --lambda_proto 0.034 --proto_momentum 0.984 --lambda_supcon_id 0.046 --lambda_fishr 0.000 --fishr_min_domains 4 --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.035 --use_sat_consistency --lambda_sat_cons 0.006 --sat_cons_start_epoch 118 --swad_start_epoch 46 --swad_tolerance 0.20 --collapse_guard_best_margin 5.0 --sat_view_schedule '1@0.99:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;118@0.91:rain_leo,storm_mp,mixed_orbit,clear_leo,low_elev_leo'

launch_cen_train CEN59_A05 CEN59_A05_r01_rxguard_strict845_aggr_r010 4 \
  --primary_udu_weight 0.918 --concat_sat_ce_weight 1.18 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.37 --mixstyle_late_start 118 --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.030 --group_ce_mode dual_worst --lambda_group_ce 0.094 --group_ce_min_domains 4 --group_ce_top_frac 0.09 --groupdro_tau 0.23 --groupdro_cap 0.32 --lambda_proto 0.036 --proto_momentum 0.984 --lambda_supcon_id 0.050 --lambda_fishr 0.000 --fishr_min_domains 4 --use_sat_consistency --lambda_sat_cons 0.006 --sat_cons_start_epoch 112 --swad_start_epoch 40 --swad_tolerance 0.16 --ema_decay 0.9995 --collapse_guard_best_margin 4.8 --sat_view_schedule '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;112@0.89:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'

launch_cen_train CEN59_A06 CEN59_A06_a06_satmean47_rxfloor_aggr_r010 5 \
  --primary_udu_weight 0.905 --concat_sat_ce_weight 1.24 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.36 --group_ce_mode dual_worst --lambda_group_ce 0.092 --group_ce_min_domains 4 --group_ce_top_frac 0.09 --groupdro_tau 0.23 --groupdro_cap 0.32 --lambda_proto 0.036 --proto_momentum 0.984 --lambda_supcon_id 0.050 --lambda_fishr 0.000 --fishr_min_domains 4 --mixstyle_late_start 116 --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.030 --use_sat_consistency --lambda_sat_cons 0.009 --sat_cons_start_epoch 104 --swad_start_epoch 44 --swad_tolerance 0.18 --collapse_guard_best_margin 4.8 --sat_view_schedule '1@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;104@0.93:storm_mp,mixed_orbit,rain_leo,low_elev_leo,clear_leo'

launch_cen_train CEN59_A07 CEN59_A07_r01_seed7331_confirm_aggr_r010 6 \
  --seed 7331 --primary_udu_weight 0.908 --concat_sat_ce_weight 1.18 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.36 --mixstyle_late_start 126 --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.032 --group_ce_mode dual_worst --lambda_group_ce 0.086 --group_ce_min_domains 4 --group_ce_top_frac 0.10 --groupdro_tau 0.24 --groupdro_cap 0.33 --lambda_proto 0.032 --proto_momentum 0.982 --lambda_supcon_id 0.042 --lambda_fishr 0.000 --fishr_min_domains 4 --use_sat_consistency --lambda_sat_cons 0.005 --sat_cons_start_epoch 118 --swad_start_epoch 46 --swad_tolerance 0.18 --collapse_guard_best_margin 5.0 --sat_view_schedule '1@0.98:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;118@0.90:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'

launch_cen_train CEN59_A08 CEN59_A08_r03_satdiag_rxguard_aggr_r010 7 \
  --primary_udu_weight 0.895 --concat_sat_ce_weight 1.26 --pa_orders 1,3,5 --generalization_feature feat_joint --domain_enhancer_strength 0.37 --group_ce_mode dual_worst --lambda_group_ce 0.096 --group_ce_min_domains 4 --group_ce_top_frac 0.08 --groupdro_tau 0.22 --groupdro_cap 0.31 --lambda_proto 0.038 --proto_momentum 0.985 --lambda_supcon_id 0.052 --lambda_fishr 0.000 --fishr_min_domains 4 --mixstyle_late_start 112 --mixstyle_late_min_p 0.002 --mixstyle_late_min_strength 0.028 --use_sat_consistency --lambda_sat_cons 0.010 --sat_cons_start_epoch 96 --swad_start_epoch 42 --swad_tolerance 0.17 --collapse_guard_best_margin 4.8 --sat_view_schedule '1@1.00:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit;96@0.94:storm_mp,mixed_orbit,rain_leo,low_elev_leo,clear_leo'
