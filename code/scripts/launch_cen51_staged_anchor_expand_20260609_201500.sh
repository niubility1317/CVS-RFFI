#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen51_staged_anchor_expand_20260609_201500}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-4}"
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
  local candidate_id="$1"
  local run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}

choose_init_ckpt() {
  local run_dir="$1"
  for name in best_primary_ood_model.pth best_val_model.pth best_strict_udu_model.pth latest_model.pth; do
    if [[ -f "${run_dir}/${name}" ]]; then echo "${run_dir}/${name}"; return 0; fi
  done
  echo "[ERROR] no Stage-1 checkpoint in ${run_dir}" >&2
  return 4
}

declare -A INITIAL_BY_GPU=()
declare -A LAUNCHED_BY_GPU=()

snapshot_capacity() {
  local gpu
  for gpu in 0 1 2 3 4 5 6 7; do
    INITIAL_BY_GPU[${gpu}]="$(gpu_process_count "${gpu}")"
    LAUNCHED_BY_GPU[${gpu}]=0
  done
}

reserve_gpu() {
  local candidate_id="$1" run_name="$2" gpu="$3"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] reserve candidate=${candidate_id} gpu=${gpu}"
    return 0
  fi
  local initial_count="${INITIAL_BY_GPU[${gpu}]:-0}"
  local local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( initial_count + local_count >= MAX_TRAIN_PER_GPU )); then
    printf "%s\t%s\t%s\tgpu=%s initial_count=%s local_count=%s max=%s\n" "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${initial_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" | tee -a "${LOG_ROOT}/blocked.tsv"
    return 1
  fi
  LAUNCHED_BY_GPU[${gpu}]=$(( local_count + 1 ))
  return 0
}

COMMON_ARGS=(
  '--train_mode'
  'centralized'
  '--dataset'
  'wisig'
  '--wisig_protocol'
  'cvs_day_rx'
  '--wisig_domain'
  'rx_day'
  '--wisig_equalized'
  '1'
  '--wisig_train_ratio'
  '0.1'
  '--wisig_val_ratio'
  '-1.0'
  '--wisig_split_strategy'
  'random'
  '--wisig_train_days'
  '0,1'
  '--wisig_test_days'
  '2,3'
  '--wisig_train_rxs'
  '0,1,2,3,4,5,6'
  '--wisig_test_rxs'
  '7,8,9,10,11'
  '--eval_batch_size'
  '192'
  '--batch_size'
  '96'
  '--num_workers'
  '1'
  '--cpu_threads'
  '2'
  '--cpu_interop_threads'
  '1'
  '--prefetch_factor'
  '2'
  '--test_eval_policy'
  'interval_final'
  '--test_eval_interval'
  '25'
  '--arch_family'
  'cvsincnet'
  '--slim_group'
  'none'
  '--model_variant'
  'lite_d'
  '--branch_ablation'
  'no_dac'
  '--domain_branch_ablation'
  'no_stats'
  '--domain_enhancer'
  'rcn_stats'
  '--domain_enhancer_strength'
  '0.35'
  '--id_time_stability_mode'
  'off'
  '--id_freq_stability_mode'
  'off'
  '--domain_time_stability_mode'
  'off'
  '--domain_freq_stability_mode'
  'off'
  '--exp_group'
  's3_rxrobust_no_dac'
  '--pa_orders'
  '1,3,5'
  '--collapse_guard'
  '--collapse_guard_min_epoch'
  '25'
  '--collapse_guard_best_margin'
  '10.0'
  '--collapse_guard_max_skipped_delta'
  '2'
  '--use_ema_ckpt'
  '--ema_decay'
  '0.999'
  '--use_swad_ckpt'
  '--swad_interval'
  '1'
  '--swad_tolerance'
  '0.9'
  '--primary_udu_weight'
  '0.80'
  '--label_smoothing'
  '0.0'
  '--no_enable_pa_aux'
  '--no_enable_dac_aux'
  '--no_aug_enable_pa_normal'
  '--aug_p_pa'
  '0.0'
  '--aug_p_dac'
  '0.0'
  '--lambda_cls_pa'
  '0.0'
  '--lambda_pa_joint_inv'
  '0.0'
  '--lambda_pa_kl'
  '0.0'
  '--lambda_pa_reg'
  '0.0'
)

cd "${ROOT}"
mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
snapshot_capacity
echo "[CEN51-STAGED] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU} initial_gpu_counts=${INITIAL_BY_GPU[*]}"
if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi


launch_S5_RND_CLEAN_2028() {
  local candidate_id='S5_RND_CLEAN_2028'
  local run_name='CEN51_STAGED_S5_RND_CLEAN_2028_r010'
  local gpu=0
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_CLEAN_2028_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_CLEAN_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_LATESAT_2028() {
  local candidate_id='S5_RND_LATESAT_2028'
  local run_name='CEN51_STAGED_S5_RND_LATESAT_2028_r010'
  local gpu=1
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_LATESAT_2028_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_LATESAT_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.45'
        '--lambda_adv'
        '0.22'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.022'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.004'
        '--lambda_supcon_id'
        '0.004'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.150'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.050'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.160'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_RXGUARD_2028() {
  local candidate_id='S5_RND_RXGUARD_2028'
  local run_name='CEN51_STAGED_S5_RND_RXGUARD_2028_r010'
  local gpu=2
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_RXGUARD_2028_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_RXGUARD_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.58'
        '--lambda_adv'
        '0.24'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.032'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.005'
        '--lambda_supcon_id'
        '0.005'
        '--lambda_fishr'
        '0.0003'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.120'
        '--sat_train_scenarios'
        'clear_leo,mixed_orbit'
        '--concat_sat_start_epoch'
        '35'
        '--sat_view_schedule'
        '1@0.120:clear_leo,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_STRONGSAT_NEG_2028() {
  local candidate_id='S5_RND_STRONGSAT_NEG_2028'
  local run_name='CEN51_STAGED_S5_RND_STRONGSAT_NEG_2028_r010'
  local gpu=3
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_STRONGSAT_NEG_2028_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_STRONGSAT_NEG_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.62'
        '--lambda_adv'
        '0.36'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.06'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.012'
        '--lambda_supcon_id'
        '0.012'
        '--lambda_fishr'
        '0.0015'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '3e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.080'
        '--aug_scale_max'
        '0.300'
        '--late_aug_min_scale'
        '0.080'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.180'
        '--mixstyle_strength'
        '0.300'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.550'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '1'
        '--sat_view_schedule'
        '1@0.550:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_FRONT_CLEAN_NEG_2028() {
  local candidate_id='S5_FRONT_CLEAN_NEG_2028'
  local run_name='CEN51_STAGED_S5_FRONT_CLEAN_NEG_2028_r010'
  local gpu=4
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_FRONT_CLEAN_NEG_2028_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'front'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_FRONT_CLEAN_NEG_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_CLEAN_1337() {
  local candidate_id='S5_RND_CLEAN_1337'
  local run_name='CEN51_STAGED_S5_RND_CLEAN_1337_r010'
  local gpu=5
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_CLEAN_1337_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '1337'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_CLEAN_1337_r010'
        '--epochs'
        '150'
        '--seed'
        '1337'
        '--sat_view_seed'
        '9256'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_LATESAT_1337() {
  local candidate_id='S5_RND_LATESAT_1337'
  local run_name='CEN51_STAGED_S5_RND_LATESAT_1337_r010'
  local gpu=6
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_LATESAT_1337_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '1337'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_LATESAT_1337_r010'
        '--epochs'
        '150'
        '--seed'
        '1337'
        '--sat_view_seed'
        '9256'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.45'
        '--lambda_adv'
        '0.22'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.022'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.004'
        '--lambda_supcon_id'
        '0.004'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.150'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.050'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.160'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_RXGUARD_1337() {
  local candidate_id='S5_RND_RXGUARD_1337'
  local run_name='CEN51_STAGED_S5_RND_RXGUARD_1337_r010'
  local gpu=7
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_RXGUARD_1337_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '1337'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_RXGUARD_1337_r010'
        '--epochs'
        '150'
        '--seed'
        '1337'
        '--sat_view_seed'
        '9256'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.58'
        '--lambda_adv'
        '0.24'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.032'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.005'
        '--lambda_supcon_id'
        '0.005'
        '--lambda_fishr'
        '0.0003'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.120'
        '--sat_train_scenarios'
        'clear_leo,mixed_orbit'
        '--concat_sat_start_epoch'
        '35'
        '--sat_view_schedule'
        '1@0.120:clear_leo,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_CLEAN_2030() {
  local candidate_id='S5_RND_CLEAN_2030'
  local run_name='CEN51_STAGED_S5_RND_CLEAN_2030_r010'
  local gpu=0
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_CLEAN_2030_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '2030'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_CLEAN_2030_r010'
        '--epochs'
        '150'
        '--seed'
        '2030'
        '--sat_view_seed'
        '9949'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_LATESAT_2030() {
  local candidate_id='S5_RND_LATESAT_2030'
  local run_name='CEN51_STAGED_S5_RND_LATESAT_2030_r010'
  local gpu=1
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_LATESAT_2030_r010_S1_anchor'
        '--epochs'
        '70'
        '--seed'
        '2030'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '46'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.32'
        '--lambda_adv'
        '0.12'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00022'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S5_RND_LATESAT_2030_r010'
        '--epochs'
        '150'
        '--seed'
        '2030'
        '--sat_view_seed'
        '9949'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.45'
        '--lambda_adv'
        '0.22'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.022'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.004'
        '--lambda_supcon_id'
        '0.004'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.150'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.050'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.160'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "5" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_CLEAN_2028() {
  local candidate_id='S10_RND_CLEAN_2028'
  local run_name='CEN51_STAGED_S10_RND_CLEAN_2028_r010'
  local gpu=2
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_CLEAN_2028_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_CLEAN_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_LATESAT_2028() {
  local candidate_id='S10_RND_LATESAT_2028'
  local run_name='CEN51_STAGED_S10_RND_LATESAT_2028_r010'
  local gpu=3
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_LATESAT_2028_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_LATESAT_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.45'
        '--lambda_adv'
        '0.22'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.022'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.004'
        '--lambda_supcon_id'
        '0.004'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.150'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.050'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.160'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_RXGUARD_2028() {
  local candidate_id='S10_RND_RXGUARD_2028'
  local run_name='CEN51_STAGED_S10_RND_RXGUARD_2028_r010'
  local gpu=4
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_RXGUARD_2028_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_RXGUARD_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.58'
        '--lambda_adv'
        '0.24'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.032'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.005'
        '--lambda_supcon_id'
        '0.005'
        '--lambda_fishr'
        '0.0003'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.120'
        '--sat_train_scenarios'
        'clear_leo,mixed_orbit'
        '--concat_sat_start_epoch'
        '35'
        '--sat_view_schedule'
        '1@0.120:clear_leo,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_PROTO_2028() {
  local candidate_id='S10_RND_PROTO_2028'
  local run_name='CEN51_STAGED_S10_RND_PROTO_2028_r010'
  local gpu=5
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_PROTO_2028_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_PROTO_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.48'
        '--lambda_adv'
        '0.24'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.036'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.008'
        '--lambda_supcon_id'
        '0.01'
        '--lambda_fishr'
        '0.0004'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '7e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.030'
        '--aug_scale_max'
        '0.180'
        '--late_aug_min_scale'
        '0.030'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.060'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.180'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '40'
        '--sat_view_schedule'
        '1@0.180:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_STRONGSAT_NEG_2028() {
  local candidate_id='S10_RND_STRONGSAT_NEG_2028'
  local run_name='CEN51_STAGED_S10_RND_STRONGSAT_NEG_2028_r010'
  local gpu=6
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_STRONGSAT_NEG_2028_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_STRONGSAT_NEG_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.62'
        '--lambda_adv'
        '0.36'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.06'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.012'
        '--lambda_supcon_id'
        '0.012'
        '--lambda_fishr'
        '0.0015'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '3e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.080'
        '--aug_scale_max'
        '0.300'
        '--late_aug_min_scale'
        '0.080'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.180'
        '--mixstyle_strength'
        '0.300'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.550'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '1'
        '--sat_view_schedule'
        '1@0.550:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_FRONT_CLEAN_NEG_2028() {
  local candidate_id='S10_FRONT_CLEAN_NEG_2028'
  local run_name='CEN51_STAGED_S10_FRONT_CLEAN_NEG_2028_r010'
  local gpu=7
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_FRONT_CLEAN_NEG_2028_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'front'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_FRONT_CLEAN_NEG_2028_r010'
        '--epochs'
        '150'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_CLEAN_1337() {
  local candidate_id='S10_RND_CLEAN_1337'
  local run_name='CEN51_STAGED_S10_RND_CLEAN_1337_r010'
  local gpu=0
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_CLEAN_1337_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '1337'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_CLEAN_1337_r010'
        '--epochs'
        '150'
        '--seed'
        '1337'
        '--sat_view_seed'
        '9256'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_LATESAT_1337() {
  local candidate_id='S10_RND_LATESAT_1337'
  local run_name='CEN51_STAGED_S10_RND_LATESAT_1337_r010'
  local gpu=1
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_LATESAT_1337_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '1337'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_LATESAT_1337_r010'
        '--epochs'
        '150'
        '--seed'
        '1337'
        '--sat_view_seed'
        '9256'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.45'
        '--lambda_adv'
        '0.22'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.022'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.004'
        '--lambda_supcon_id'
        '0.004'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.150'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.050'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.160'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S10_RND_LATESAT_2030() {
  local candidate_id='S10_RND_LATESAT_2030'
  local run_name='CEN51_STAGED_S10_RND_LATESAT_2030_r010'
  local gpu=2
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_LATESAT_2030_r010_S1_anchor'
        '--epochs'
        '80'
        '--seed'
        '2030'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '56'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.38'
        '--lambda_adv'
        '0.16'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.00016'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '40'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S10_RND_LATESAT_2030_r010'
        '--epochs'
        '150'
        '--seed'
        '2030'
        '--sat_view_seed'
        '9949'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '76'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.45'
        '--lambda_adv'
        '0.22'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.022'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.004'
        '--lambda_supcon_id'
        '0.004'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.150'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '50'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.050'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '75'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.160'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "10" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S20_RND_CLEAN_2028() {
  local candidate_id='S20_RND_CLEAN_2028'
  local run_name='CEN51_STAGED_S20_RND_CLEAN_2028_r010'
  local gpu=3
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S20_RND_CLEAN_2028_r010_S1_anchor'
        '--epochs'
        '90'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '20'
        '--test_eval_start_epoch'
        '66'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.0001'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '45'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S20_RND_CLEAN_2028_r010'
        '--epochs'
        '170'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.5'
        '--lambda_adv'
        '0.28'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.036'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.007'
        '--lambda_supcon_id'
        '0.007'
        '--lambda_fishr'
        '0.0005'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '5e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.040'
        '--aug_scale_max'
        '0.220'
        '--late_aug_min_scale'
        '0.040'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.080'
        '--mixstyle_strength'
        '0.220'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "20" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S20_RND_LATESAT_2028() {
  local candidate_id='S20_RND_LATESAT_2028'
  local run_name='CEN51_STAGED_S20_RND_LATESAT_2028_r010'
  local gpu=4
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S20_RND_LATESAT_2028_r010_S1_anchor'
        '--epochs'
        '90'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '20'
        '--test_eval_start_epoch'
        '66'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.0001'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '45'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S20_RND_LATESAT_2028_r010'
        '--epochs'
        '170'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.52'
        '--lambda_adv'
        '0.3'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.04'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.008'
        '--lambda_supcon_id'
        '0.008'
        '--lambda_fishr'
        '0.0008'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '4.5e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.050'
        '--aug_scale_max'
        '0.240'
        '--late_aug_min_scale'
        '0.050'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.100'
        '--mixstyle_strength'
        '0.240'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.280'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '35'
        '--sat_view_schedule'
        '1@0.280:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "20" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S20_RND_RXGUARD_2028() {
  local candidate_id='S20_RND_RXGUARD_2028'
  local run_name='CEN51_STAGED_S20_RND_RXGUARD_2028_r010'
  local gpu=5
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S20_RND_RXGUARD_2028_r010_S1_anchor'
        '--epochs'
        '90'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '20'
        '--test_eval_start_epoch'
        '66'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.0001'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '45'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S20_RND_RXGUARD_2028_r010'
        '--epochs'
        '170'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.62'
        '--lambda_adv'
        '0.34'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.052'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.01'
        '--lambda_supcon_id'
        '0.01'
        '--lambda_fishr'
        '0.001'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '4.5e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.050'
        '--aug_scale_max'
        '0.240'
        '--late_aug_min_scale'
        '0.050'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.090'
        '--mixstyle_strength'
        '0.230'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.220'
        '--sat_train_scenarios'
        'clear_leo,mixed_orbit'
        '--concat_sat_start_epoch'
        '25'
        '--sat_view_schedule'
        '1@0.220:clear_leo,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "20" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S30_RND_BAL_2028() {
  local candidate_id='S30_RND_BAL_2028'
  local run_name='CEN51_STAGED_S30_RND_BAL_2028_r010'
  local gpu=6
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S30_RND_BAL_2028_r010_S1_anchor'
        '--epochs'
        '90'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '30'
        '--test_eval_start_epoch'
        '66'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.0001'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '45'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S30_RND_BAL_2028_r010'
        '--epochs'
        '180'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '106'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.56'
        '--lambda_adv'
        '0.32'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.048'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.01'
        '--lambda_supcon_id'
        '0.01'
        '--lambda_fishr'
        '0.001'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '5e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.060'
        '--aug_scale_max'
        '0.260'
        '--late_aug_min_scale'
        '0.060'
        '--swad_start_epoch'
        '60'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.120'
        '--mixstyle_strength'
        '0.250'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '90'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.340'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '30'
        '--sat_view_schedule'
        '1@0.340:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "30" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S30_RND_SATFLOOR_2028() {
  local candidate_id='S30_RND_SATFLOOR_2028'
  local run_name='CEN51_STAGED_S30_RND_SATFLOOR_2028_r010'
  local gpu=7
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S30_RND_SATFLOOR_2028_r010_S1_anchor'
        '--epochs'
        '90'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '30'
        '--test_eval_start_epoch'
        '66'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.0001'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '45'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S30_RND_SATFLOOR_2028_r010'
        '--epochs'
        '180'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '106'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.56'
        '--lambda_adv'
        '0.3'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.044'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.008'
        '--lambda_supcon_id'
        '0.008'
        '--lambda_fishr'
        '0.001'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '4e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.060'
        '--aug_scale_max'
        '0.260'
        '--late_aug_min_scale'
        '0.060'
        '--swad_start_epoch'
        '60'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.120'
        '--mixstyle_strength'
        '0.250'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '90'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.380'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '25'
        '--sat_view_schedule'
        '1@0.380:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "30" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S50_RND_BAL_2028() {
  local candidate_id='S50_RND_BAL_2028'
  local run_name='CEN51_STAGED_S50_RND_BAL_2028_r010'
  local gpu=0
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S50_RND_BAL_2028_r010_S1_anchor'
        '--epochs'
        '90'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '50'
        '--test_eval_start_epoch'
        '66'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.0001'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '45'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S50_RND_BAL_2028_r010'
        '--epochs'
        '180'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '106'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.56'
        '--lambda_adv'
        '0.32'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.048'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.01'
        '--lambda_supcon_id'
        '0.01'
        '--lambda_fishr'
        '0.001'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '5e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.060'
        '--aug_scale_max'
        '0.260'
        '--late_aug_min_scale'
        '0.060'
        '--swad_start_epoch'
        '60'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.120'
        '--mixstyle_strength'
        '0.250'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '90'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.340'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '30'
        '--sat_view_schedule'
        '1@0.340:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "50" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S50_RND_CLEAN_2028() {
  local candidate_id='S50_RND_CLEAN_2028'
  local run_name='CEN51_STAGED_S50_RND_CLEAN_2028_r010'
  local gpu=1
  local direct=0
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S50_RND_CLEAN_2028_r010_S1_anchor'
        '--epochs'
        '90'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '50'
        '--test_eval_start_epoch'
        '66'
        '--no_eval_sat_channel'
        '--no_use_aug'
        '--no_use_mixstyle'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.01'
        '--lambda_cons'
        '0.0'
        '--lambda_group_ce'
        '0.0'
        '--lambda_proto'
        '0.0'
        '--lambda_supcon_id'
        '0.0'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '0.0001'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '45'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_S50_RND_CLEAN_2028_r010'
        '--epochs'
        '180'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '106'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.5'
        '--lambda_adv'
        '0.28'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.036'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.007'
        '--lambda_supcon_id'
        '0.007'
        '--lambda_fishr'
        '0.0005'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '5e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.040'
        '--aug_scale_max'
        '0.220'
        '--late_aug_min_scale'
        '0.040'
        '--swad_start_epoch'
        '60'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.080'
        '--mixstyle_strength'
        '0.220'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '90'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "50" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_D0P1_CLEAN_2028() {
  local candidate_id='D0P1_CLEAN_2028'
  local run_name='CEN51_STAGED_D0P1_CLEAN_2028_r010'
  local gpu=2
  local direct=1
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"

        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_D0P1_CLEAN_2028_r010'
        '--epochs'
        '170'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.42'
        '--lambda_adv'
        '0.2'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.018'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.003'
        '--lambda_supcon_id'
        '0.003'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--no_use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "0" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_D0P1_LATESAT_2028() {
  local candidate_id='D0P1_LATESAT_2028'
  local run_name='CEN51_STAGED_D0P1_LATESAT_2028_r010'
  local gpu=3
  local direct=1
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"

        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_D0P1_LATESAT_2028_r010'
        '--epochs'
        '170'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.45'
        '--lambda_adv'
        '0.22'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.022'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.004'
        '--lambda_supcon_id'
        '0.004'
        '--lambda_fishr'
        '0'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.150'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.050'
        '--mixstyle_strength'
        '0.180'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.160'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.160:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "0" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_D0P1_RXGUARD_2028() {
  local candidate_id='D0P1_RXGUARD_2028'
  local run_name='CEN51_STAGED_D0P1_RXGUARD_2028_r010'
  local gpu=4
  local direct=1
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"

        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_D0P1_RXGUARD_2028_r010'
        '--epochs'
        '170'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.58'
        '--lambda_adv'
        '0.24'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.032'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.005'
        '--lambda_supcon_id'
        '0.005'
        '--lambda_fishr'
        '0.0003'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '9e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.040'
        '--mixstyle_strength'
        '0.160'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.120'
        '--sat_train_scenarios'
        'clear_leo,mixed_orbit'
        '--concat_sat_start_epoch'
        '35'
        '--sat_view_schedule'
        '1@0.120:clear_leo,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "0" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_D0P1_STRONGSAT_NEG_2028() {
  local candidate_id='D0P1_STRONGSAT_NEG_2028'
  local run_name='CEN51_STAGED_D0P1_STRONGSAT_NEG_2028_r010'
  local gpu=5
  local direct=1
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"

        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_D0P1_STRONGSAT_NEG_2028_r010'
        '--epochs'
        '170'
        '--seed'
        '2028'
        '--sat_view_seed'
        '9947'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.62'
        '--lambda_adv'
        '0.36'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.06'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.012'
        '--lambda_supcon_id'
        '0.012'
        '--lambda_fishr'
        '0.0015'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '3e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.080'
        '--aug_scale_max'
        '0.300'
        '--late_aug_min_scale'
        '0.080'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.180'
        '--mixstyle_strength'
        '0.300'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.550'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '1'
        '--sat_view_schedule'
        '1@0.550:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "0" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_D0P1_BALANCED_1337() {
  local candidate_id='D0P1_BALANCED_1337'
  local run_name='CEN51_STAGED_D0P1_BALANCED_1337_r010'
  local gpu=6
  local direct=1
  if should_skip "${candidate_id}" "${run_name}"; then return 0; fi
  if ! reserve_gpu "${candidate_id}" "${run_name}" "${gpu}"; then return 0; fi
  if [[ "${DRY_RUN}" == "1" ]]; then return 0; fi
  local driver_log="${LOG_ROOT}/${run_name}.driver.out"
  (
    set -euo pipefail
    cd "${ROOT}"
    local init_ckpt=""
    if [[ "${direct}" == "0" ]]; then
      local s1_dir="${RUNS_ROOT}/${run_name}_S1_anchor"
      local s1_log="${LOG_ROOT}/${run_name}_S1_anchor.out"
      if [[ -e "${s1_dir}" || -e "${s1_log}" ]]; then echo "[BLOCKED] stage1 path collision ${candidate_id}"; exit 3; fi
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"

        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      echo "[STAGE1-CMD]"; print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local s2_dir="${RUNS_ROOT}/${run_name}"
    local s2_log="${LOG_ROOT}/${run_name}.out"
    if [[ -e "${s2_dir}" || -e "${s2_log}" ]]; then echo "[BLOCKED] stage2 path collision ${candidate_id}"; exit 3; fi
    mkdir -p "${s2_dir}"
    local s2_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--run_name'
        'CEN51_STAGED_D0P1_BALANCED_1337_r010'
        '--epochs'
        '170'
        '--seed'
        '1337'
        '--sat_view_seed'
        '9256'
        '--wisig_cap_strategy'
        'random'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '96'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--lambda_dom'
        '0.52'
        '--lambda_adv'
        '0.28'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.036'
        '--group_ce_mode'
        'smooth_dro_capped'
        '--group_ce_min_domains'
        '2'
        '--group_ce_top_frac'
        '0.20'
        '--groupdro_tau'
        '0.38'
        '--groupdro_cap'
        '0.50'
        '--lambda_proto'
        '0.007'
        '--lambda_supcon_id'
        '0.007'
        '--lambda_fishr'
        '0.0005'
        '--fishr_min_domains'
        '2'
        '--lambda_feature_norm_guard'
        '5e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--aug_scale_min'
        '0.040'
        '--aug_scale_max'
        '0.220'
        '--late_aug_min_scale'
        '0.040'
        '--swad_start_epoch'
        '56'
        '--use_aug'
        '--use_mixstyle'
        '--mixstyle_p'
        '0.080'
        '--mixstyle_strength'
        '0.220'
        '--mixstyle_mix'
        'same_tx_crossdomain'
        '--mixstyle_fallback'
        'skip'
        '--mixstyle_late_start'
        '85'
        '--mixstyle_late_ramp_epochs'
        '35'
        '--mixstyle_late_min_p'
        '0.020'
        '--mixstyle_late_min_strength'
        '0.120'
        '--use_concat_sat_channel_aug'
        '--no_use_sat_consistency'
        '--lambda_sat_cls'
        '0.0'
        '--lambda_sat_cons'
        '0.0'
        '--concat_sat_ce_weight'
        '0.0'
        '--sat_cons_start_epoch'
        '999'
        '--sat_view_prob'
        '0.300'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '30'
        '--sat_view_schedule'
        '1@0.300:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--use_proto_memory'
        --latest_save_path "${s2_dir}/latest_model.pth"
        --best_save_path "${s2_dir}/best_val_model.pth"
        --best_primary_save_path "${s2_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s2_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s2_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s2_dir}/ema_model.pth"
        --swa_save_path "${s2_dir}/swa_model.pth"
        --swad_save_path "${s2_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then s2_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    echo "[STAGE2-CMD]"; print_cmd "${s2_cmd[@]}"
    "${s2_cmd[@]}" > "${s2_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "0" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_S5_RND_CLEAN_2028
launch_S5_RND_LATESAT_2028
launch_S5_RND_RXGUARD_2028
launch_S5_RND_STRONGSAT_NEG_2028
launch_S5_FRONT_CLEAN_NEG_2028
launch_S5_RND_CLEAN_1337
launch_S5_RND_LATESAT_1337
launch_S5_RND_RXGUARD_1337
launch_S5_RND_CLEAN_2030
launch_S5_RND_LATESAT_2030
launch_S10_RND_CLEAN_2028
launch_S10_RND_LATESAT_2028
launch_S10_RND_RXGUARD_2028
launch_S10_RND_PROTO_2028
launch_S10_RND_STRONGSAT_NEG_2028
launch_S10_FRONT_CLEAN_NEG_2028
launch_S10_RND_CLEAN_1337
launch_S10_RND_LATESAT_1337
launch_S10_RND_LATESAT_2030
launch_S20_RND_CLEAN_2028
launch_S20_RND_LATESAT_2028
launch_S20_RND_RXGUARD_2028
launch_S30_RND_BAL_2028
launch_S30_RND_SATFLOOR_2028
launch_S50_RND_BAL_2028
launch_S50_RND_CLEAN_2028
launch_D0P1_CLEAN_2028
launch_D0P1_LATESAT_2028
launch_D0P1_RXGUARD_2028
launch_D0P1_STRONGSAT_NEG_2028
launch_D0P1_BALANCED_1337
echo "[CEN51-STAGED] launch submissions complete"
