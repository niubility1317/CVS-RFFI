#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen51_sinc_cvcnn_lowshot_compare_20260609_204500}"
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
  local candidate_id="$1" run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}
choose_init_ckpt() {
  local run_dir="$1"
  for name in best_primary_ood_model.pth best_val_model.pth best_strict_udu_model.pth latest_model.pth; do
    [[ -f "${run_dir}/${name}" ]] && { echo "${run_dir}/${name}"; return 0; }
  done
  echo "[ERROR] no checkpoint in ${run_dir}" >&2
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
  local initial_count="${INITIAL_BY_GPU[${gpu}]:-0}"
  local local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] reserve candidate=${candidate_id} run=${run_name} gpu=${gpu} initial=${initial_count} local=${local_count} max=${MAX_TRAIN_PER_GPU}"
    return 0
  fi
  if (( initial_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
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
  'sinc_cvcnn_lowshot_compare'
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
  '--no_use_mixstyle'
)

cd "${ROOT}"
mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
snapshot_capacity
echo "[SINC-CVCNN-COMPARE] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU} initial_gpu_counts=${INITIAL_BY_GPU[*]}"
[[ "${DRY_RUN}" == "1" || -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }


launch_SCV_K5_CLEAN_2028() {
  local candidate_id='SCV_K5_CLEAN_2028'
  local run_name='CEN51_SCV_K5_CLEAN_2028_r010'
  local gpu=0
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_K5_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_K5_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.3'
        '--lambda_adv'
        '0.1'
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
        '0.0002'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_K5_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_K5_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_K5_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.3'
        '--lambda_adv'
        '0.1'
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
        '0.0002'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        --run_name 'CEN51_SCV_K5_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_BCV_K5_CLEAN_2028() {
  local candidate_id='BCV_K5_CLEAN_2028'
  local run_name='CEN51_BCV_K5_CLEAN_2028_r010'
  local gpu=1
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
      local s1_dir="${RUNS_ROOT}/CEN51_BCV_K5_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_BCV_K5_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.3'
        '--lambda_adv'
        '0.1'
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
        '0.0002'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_BCV_K5_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_BCV_K5_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_BCV_K5_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.3'
        '--lambda_adv'
        '0.1'
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
        '0.0002'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        --run_name 'CEN51_BCV_K5_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_K10_CLEAN_2028() {
  local candidate_id='SCV_K10_CLEAN_2028'
  local run_name='CEN51_SCV_K10_CLEAN_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_K10_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_K10_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.36'
        '--lambda_adv'
        '0.14'
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
        '0.00014'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_K10_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_K10_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_K10_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.36'
        '--lambda_adv'
        '0.14'
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
        '0.00014'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        --run_name 'CEN51_SCV_K10_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_BCV_K10_CLEAN_2028() {
  local candidate_id='BCV_K10_CLEAN_2028'
  local run_name='CEN51_BCV_K10_CLEAN_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_BCV_K10_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_BCV_K10_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.36'
        '--lambda_adv'
        '0.14'
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
        '0.00014'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_BCV_K10_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_BCV_K10_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_BCV_K10_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.36'
        '--lambda_adv'
        '0.14'
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
        '0.00014'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        --run_name 'CEN51_BCV_K10_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_K20_LATESAT_2028() {
  local candidate_id='SCV_K20_LATESAT_2028'
  local run_name='CEN51_SCV_K20_LATESAT_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_K20_LATESAT_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_K20_LATESAT_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '20'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_K20_LATESAT_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_K20_LATESAT_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_K20_LATESAT_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.050'
        '--aug_scale_max'
        '0.240'
        '--late_aug_min_scale'
        '0.050'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        --run_name 'CEN51_SCV_K20_LATESAT_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_BCV_K20_LATESAT_2028() {
  local candidate_id='BCV_K20_LATESAT_2028'
  local run_name='CEN51_BCV_K20_LATESAT_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_BCV_K20_LATESAT_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_BCV_K20_LATESAT_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '20'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_BCV_K20_LATESAT_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_BCV_K20_LATESAT_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_BCV_K20_LATESAT_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.050'
        '--aug_scale_max'
        '0.240'
        '--late_aug_min_scale'
        '0.050'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        --run_name 'CEN51_BCV_K20_LATESAT_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_S5_CLEAN_2028() {
  local candidate_id='SCV_S5_CLEAN_2028'
  local run_name='CEN51_SCV_S5_CLEAN_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_S5_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_S5_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.3'
        '--lambda_adv'
        '0.1'
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
        '0.0002'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_S5_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_S5_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_S5_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '160'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '86'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--use_aug'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--lambda_dom'
        '0.4'
        '--lambda_adv'
        '0.18'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.016'
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
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '55'
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
        --run_name 'CEN51_SCV_S5_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_S5_LATESAT_2028() {
  local candidate_id='SCV_S5_LATESAT_2028'
  local run_name='CEN51_SCV_S5_LATESAT_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_S5_LATESAT_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_S5_LATESAT_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '5'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.3'
        '--lambda_adv'
        '0.1'
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
        '0.0002'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_S5_LATESAT_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_S5_LATESAT_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_S5_LATESAT_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '160'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '86'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--use_aug'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.160'
        '--late_aug_min_scale'
        '0.020'
        '--lambda_dom'
        '0.44'
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
        '7e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '55'
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
        '0.140'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.140:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        --run_name 'CEN51_SCV_S5_LATESAT_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_S10_CLEAN_2028() {
  local candidate_id='SCV_S10_CLEAN_2028'
  local run_name='CEN51_SCV_S10_CLEAN_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_S10_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_S10_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.36'
        '--lambda_adv'
        '0.14'
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
        '0.00014'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_S10_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_S10_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_S10_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '160'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '86'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--use_aug'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.140'
        '--late_aug_min_scale'
        '0.020'
        '--lambda_dom'
        '0.4'
        '--lambda_adv'
        '0.18'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.016'
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
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '55'
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
        --run_name 'CEN51_SCV_S10_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_S10_LATESAT_2028() {
  local candidate_id='SCV_S10_LATESAT_2028'
  local run_name='CEN51_SCV_S10_LATESAT_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_S10_LATESAT_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_S10_LATESAT_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '90'
        '--wisig_max_train_per_combo'
        '10'
        '--test_eval_start_epoch'
        '50'
        '--no_use_aug'
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
        '0.36'
        '--lambda_adv'
        '0.14'
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
        '0.00014'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--no_use_proto_memory'
        '--swad_start_epoch'
        '35'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_S10_LATESAT_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_S10_LATESAT_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_S10_LATESAT_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '160'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '86'
        '--eval_sat_channel'
        '--eval_sat_on'
        'test_unseen_day_unseen_rx'
        '--eval_sat_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--sat_eval_max_batches'
        '-1'
        '--use_aug'
        '--aug_scale_min'
        '0.020'
        '--aug_scale_max'
        '0.160'
        '--late_aug_min_scale'
        '0.020'
        '--lambda_dom'
        '0.44'
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
        '7e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '55'
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
        '0.140'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '45'
        '--sat_view_schedule'
        '1@0.140:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        --run_name 'CEN51_SCV_S10_LATESAT_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_S20_LATESAT_2028() {
  local candidate_id='SCV_S20_LATESAT_2028'
  local run_name='CEN51_SCV_S20_LATESAT_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_S20_LATESAT_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_S20_LATESAT_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '20'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_S20_LATESAT_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_S20_LATESAT_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_S20_LATESAT_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.050'
        '--aug_scale_max'
        '0.240'
        '--late_aug_min_scale'
        '0.050'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        --run_name 'CEN51_SCV_S20_LATESAT_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_S50_BAL_2028() {
  local candidate_id='SCV_S50_BAL_2028'
  local run_name='CEN51_SCV_S50_BAL_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_S50_BAL_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_S50_BAL_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '50'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_S50_BAL_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_S50_BAL_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_S50_BAL_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.060'
        '--aug_scale_max'
        '0.260'
        '--late_aug_min_scale'
        '0.060'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        '0.320'
        '--sat_train_scenarios'
        'clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        '--concat_sat_start_epoch'
        '30'
        '--sat_view_schedule'
        '1@0.320:clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit'
        --run_name 'CEN51_SCV_S50_BAL_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_D0P1_CLEAN_2028() {
  local candidate_id='SCV_D0P1_CLEAN_2028'
  local run_name='CEN51_SCV_D0P1_CLEAN_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_D0P1_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_D0P1_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_D0P1_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_D0P1_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_D0P1_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.040'
        '--aug_scale_max'
        '0.220'
        '--late_aug_min_scale'
        '0.040'
        '--lambda_dom'
        '0.5'
        '--lambda_adv'
        '0.26'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.034'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        --run_name 'CEN51_SCV_D0P1_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_D0P1_LATESAT_2028() {
  local candidate_id='SCV_D0P1_LATESAT_2028'
  local run_name='CEN51_SCV_D0P1_LATESAT_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_SCV_D0P1_LATESAT_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_SCV_D0P1_LATESAT_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_SCV_D0P1_LATESAT_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_SCV_D0P1_LATESAT_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_SCV_D0P1_LATESAT_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'sinc_cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.050'
        '--aug_scale_max'
        '0.240'
        '--late_aug_min_scale'
        '0.050'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        --run_name 'CEN51_SCV_D0P1_LATESAT_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "sinc_cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_BCV_D0P1_CLEAN_2028() {
  local candidate_id='BCV_D0P1_CLEAN_2028'
  local run_name='CEN51_BCV_D0P1_CLEAN_2028_r010'
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
      local s1_dir="${RUNS_ROOT}/CEN51_BCV_D0P1_CLEAN_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_BCV_D0P1_CLEAN_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_BCV_D0P1_CLEAN_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_BCV_D0P1_CLEAN_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_BCV_D0P1_CLEAN_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.040'
        '--aug_scale_max'
        '0.220'
        '--late_aug_min_scale'
        '0.040'
        '--lambda_dom'
        '0.5'
        '--lambda_adv'
        '0.26'
        '--grl_lambda'
        '1.0'
        '--lambda_orth'
        '0.02'
        '--lambda_cons'
        '0.006'
        '--lambda_group_ce'
        '0.034'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        --run_name 'CEN51_BCV_D0P1_CLEAN_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_BCV_D0P1_LATESAT_2028() {
  local candidate_id='BCV_D0P1_LATESAT_2028'
  local run_name='CEN51_BCV_D0P1_LATESAT_2028_r010'
  local gpu=7
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
      local s1_dir="${RUNS_ROOT}/CEN51_BCV_D0P1_LATESAT_2028_r010_S1_anchor"
      local s1_log="${LOG_ROOT}/CEN51_BCV_D0P1_LATESAT_2028_r010_S1_anchor.out"
      mkdir -p "${s1_dir}"
      local s1_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '120'
        '--wisig_max_train_per_combo'
        '100'
        '--test_eval_start_epoch'
        '70'
        '--no_use_aug'
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
        '0.012'
        '--lambda_proto'
        '0.002'
        '--lambda_supcon_id'
        '0.002'
        '--lambda_fishr'
        '0.0'
        '--lambda_feature_norm_guard'
        '8e-05'
        '--feature_norm_guard_mode'
        'l2'
        '--feature_norm_guard_target'
        '0.0'
        '--use_proto_memory'
        '--swad_start_epoch'
        '45'
        '--no_eval_sat_channel'
        --run_name 'CEN51_BCV_D0P1_LATESAT_2028_r010_S1_anchor'
        --latest_save_path "${s1_dir}/latest_model.pth"
        --best_save_path "${s1_dir}/best_val_model.pth"
        --best_primary_save_path "${s1_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${s1_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${s1_dir}/best_worst_rx_model.pth"
        --ema_save_path "${s1_dir}/ema_model.pth"
        --swa_save_path "${s1_dir}/swa_model.pth"
        --swad_save_path "${s1_dir}/swad_model.pth")
      print_cmd "${s1_cmd[@]}"
      "${s1_cmd[@]}" > "${s1_log}" 2>&1
      init_ckpt="$(choose_init_ckpt "${s1_dir}")"
      echo "[STAGE1-DONE] candidate=${candidate_id} init_ckpt=${init_ckpt}"
    fi
    local main_dir="${RUNS_ROOT}/CEN51_BCV_D0P1_LATESAT_2028_r010"
    local main_log="${LOG_ROOT}/CEN51_BCV_D0P1_LATESAT_2028_r010.out"
    mkdir -p "${main_dir}"
    local main_cmd=(env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}" "${COMMON_ARGS[@]}"
        '--arch_family'
        'cvcnn'
        '--seed'
        '2028'
        '--wisig_cap_strategy'
        'random'
        '--epochs'
        '180'
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
        '--use_aug'
        '--aug_scale_min'
        '0.050'
        '--aug_scale_max'
        '0.240'
        '--late_aug_min_scale'
        '0.050'
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
        '--use_proto_memory'
        '--swad_start_epoch'
        '65'
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
        --run_name 'CEN51_BCV_D0P1_LATESAT_2028_r010'
        --latest_save_path "${main_dir}/latest_model.pth"
        --best_save_path "${main_dir}/best_val_model.pth"
        --best_primary_save_path "${main_dir}/best_primary_ood_model.pth"
        --best_unseen_day_unseen_rx_save_path "${main_dir}/best_strict_udu_model.pth"
        --best_worst_rx_save_path "${main_dir}/best_worst_rx_model.pth"
        --ema_save_path "${main_dir}/ema_model.pth"
        --swa_save_path "${main_dir}/swa_model.pth"
        --swad_save_path "${main_dir}/swad_model.pth")
    if [[ -n "${init_ckpt}" ]]; then main_cmd+=(--init_checkpoint "${init_ckpt}"); fi
    print_cmd "${main_cmd[@]}"
    "${main_cmd[@]}" > "${main_log}" 2>&1
    echo "[CANDIDATE-DONE] ${candidate_id}"
  ) > "${driver_log}" 2>&1 &
  local driver_pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "cvcnn" "${gpu}" "${driver_pid}" "${driver_log}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}


launch_SCV_K5_CLEAN_2028
launch_BCV_K5_CLEAN_2028
launch_SCV_K10_CLEAN_2028
launch_BCV_K10_CLEAN_2028
launch_SCV_K20_LATESAT_2028
launch_BCV_K20_LATESAT_2028
launch_SCV_S5_CLEAN_2028
launch_SCV_S5_LATESAT_2028
launch_SCV_S10_CLEAN_2028
launch_SCV_S10_LATESAT_2028
launch_SCV_S20_LATESAT_2028
launch_SCV_S50_BAL_2028
launch_SCV_D0P1_CLEAN_2028
launch_SCV_D0P1_LATESAT_2028
launch_BCV_D0P1_CLEAN_2028
launch_BCV_D0P1_LATESAT_2028
echo "[SINC-CVCNN-COMPARE] launch submissions complete"
