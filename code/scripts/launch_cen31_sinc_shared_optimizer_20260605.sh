#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${CVS_TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-cen31_sinc_shared_optimizer_20260605_165045}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
EPOCHS="${EPOCHS:-170}"
TEST_EVAL_INTERVAL="${TEST_EVAL_INTERVAL:-10}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      echo "[ERROR] unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' \
    | wc -l \
    | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

expected_heavy_tests() {
  local epochs="$1"
  local interval="$2"
  local count=$(( epochs / interval ))
  if (( epochs % interval != 0 )); then
    count=$(( count + 1 ))
  fi
  printf '%s' "${count}"
}

COMMON_ARGS=(
  --train_mode centralized
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --epochs "${EPOCHS}"
  --test_eval_policy interval_final
  --test_eval_start_epoch 1
  --test_eval_interval "${TEST_EVAL_INTERVAL}"
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
  --sat_eval_max_batches -1
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --exp_group s3_rxrobust_no_dac
  --model_variant lite_d
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --seed 1337
  --arch_family cvsincnet
)

MIXSTYLE_ARGS=(
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
)

CEN_A31_STACK_ARGS=(
  --primary_udu_weight 0.70
  --concat_sat_ce_weight 1.28
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
)

CANDIDATES=(
  "01|0|sinc_shared_baseline_api|"
  "02|1|freq_from_sinc_energy|--freq_feature_source sinc_energy"
  "03|2|freq_from_sinc_phase_asym|--freq_feature_source sinc_phase_asym"
  "04|3|pa_from_sinc_lowrank_135|--pa_feature_source sinc_lowrank --pa_orders 1,3,5"
  "05|4|pa_from_sinc_lowrank_15|--pa_feature_source sinc_lowrank --pa_orders 1,5"
  "06|5|time_pa_no_overlap_3rd|--pa_orders 1,5"
  "07|6|fftless_freq_pa_joint|--freq_feature_source sinc_phase_asym --pa_feature_source sinc_lowrank --pa_orders 1,5"
  "08|7|sinc_shared_channel_trim|--channel_trim_scale 0.75"
  "09|0|no_rho_circularity|--no_use_circularity"
  "10|1|no_freq_stats_proj|--no_use_freq_stats"
  "11|2|no_pa_stats_proj|--no_use_pa_stats"
  "12|3|no_spectral_aux_stats_all|--no_use_circularity --no_use_freq_stats --no_use_pa_stats --no_use_aux_spectral_stats"
  "13|4|no_dsq_freq_stability|--domain_freq_stability_mode off"
  "14|5|no_freq_band_gate|--no_use_freq_band_gate"
  "15|6|domain_enhancer_off|--domain_enhancer off"
  "16|7|rcn_minimal_6stats|--domain_enhancer rcn_minimal_6stats"
)

if [[ "${#CANDIDATES[@]}" -ne 16 ]]; then
  echo "[ERROR] candidate matrix must contain exactly 16 entries; got ${#CANDIDATES[@]}" >&2
  exit 2
fi

cd "${ROOT}"
mkdir -p "${LOG_ROOT}"

echo "[CEN31-SINC] run_id=${RUN_ID} dry_run=${DRY_RUN} candidates=${#CANDIDATES[@]} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"
echo "[CEN31-SINC-EVAL-AUDIT] policy=interval_final test_eval_interval=${TEST_EVAL_INTERVAL} final_eval=true forbidden_every_epoch=true forbidden_val_improved_extra=true expected_heavy_tests=$(expected_heavy_tests "${EPOCHS}" "${TEST_EVAL_INTERVAL}")"

declare -A launched_by_gpu=()
for spec in "${CANDIDATES[@]}"; do
  IFS='|' read -r idx gpu candidate flags <<< "${spec}"
  run_name="CEN31SSO${idx}_${candidate}_r010"
  run_dir="${RUNS_ROOT}/${run_name}"
  log_path="${LOG_ROOT}/${run_name}.out"
  candidate_args=()
  if [[ -n "${flags}" ]]; then
    read -r -a candidate_args <<< "${flags}"
  fi
  cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_ARGS[@]}"
    --run_name "${run_name}"
    "${MIXSTYLE_ARGS[@]}"
    "${CEN_A31_STACK_ARGS[@]}"
    "${candidate_args[@]}"
    --latest_save_path "${run_dir}/latest_model.pth"
    --best_save_path "${run_dir}/best_val_model.pth"
    --best_primary_save_path "${run_dir}/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "${run_dir}/best_strict_udu_model.pth"
    --ema_save_path "${run_dir}/ema_model.pth"
    --swa_save_path "${run_dir}/swa_model.pth"
    --swad_save_path "${run_dir}/swad_model.pth"
  )

  echo "[CEN31-SINC-CANDIDATE] name=${candidate} idx=${idx} gpu=${gpu} run=${run_name} kind=centralized flags=${flags:-<control>}"
  printf '[CEN31-SINC-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi
  if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "[ERROR] train script not found: ${TRAIN_SCRIPT}" >&2
    exit 2
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    printf "%s\t%s\t%s\t%s\n" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    continue
  fi

  current_count="$(gpu_process_count "${gpu}")"
  local_count="${launched_by_gpu[${gpu}]:-0}"
  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then
    printf "%s\t%s\tgpu=%s active_count=%s local_count=%s max=%s\n" \
      "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    continue
  fi

  mkdir -p "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  pid="$!"
  launched_by_gpu["${gpu}"]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\n" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
done
