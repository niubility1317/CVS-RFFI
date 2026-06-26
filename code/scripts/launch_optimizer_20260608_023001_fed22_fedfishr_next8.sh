#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${ROOT}/code/train.py}"
RUN_ID="${RUN_ID:-optimizer_20260608_023001_fed22_fedfishr_next8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-0}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"
CPU_THREADS="${CPU_THREADS:-4}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-1}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
    | sed '/^$/d' | wc -l | tr -d ' '
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

should_skip() {
  local candidate_id="$1" run_name="$2"
  [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]
}

declare -A LAUNCHED_BY_GPU=()

run_cmd() {
  local candidate_id="$1" run_name="$2" gpu="$3" run_dir="$4" log_path="$5"
  shift 5
  local cmd=("$@")

  if should_skip "${candidate_id}" "${run_name}"; then
    return 0
  fi

  echo "[FED22-FISHR-CANDIDATE] lane=federated_vmb candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[FED22-FISHR-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "NON_LAUNCH_DIAGNOSTIC_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local current_count local_count
  current_count="$(gpu_process_count "${gpu}")"
  local_count="${LAUNCHED_BY_GPU[${gpu}]:-0}"
  if (( current_count + local_count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s local_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "DEFERRED_RETRY_CAPACITY" "${gpu}" "${current_count}" "${local_count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  LAUNCHED_BY_GPU["${gpu}"]=$(( local_count + 1 ))
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

THREAD_ENV=(
  "CVSRFFI_CPU_THREADS=${CPU_THREADS}"
  "CVSRFFI_CPU_INTEROP_THREADS=${CPU_INTEROP_THREADS}"
  "OMP_NUM_THREADS=${CPU_THREADS}"
  "MKL_NUM_THREADS=${CPU_THREADS}"
  "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
  "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
)

COMMON_FED_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_val_ratio 0.9
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --epochs 200
  --fl_rounds 200
  --fl_client_key receiver
  --fl_clients_per_round 1.0
  --fl_test_eval_interval 10
  --fl_test_eval_last_n 5
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches 0
  --num_workers 0
  --fl_num_workers 0
  --batch_size 128
  --eval_batch_size 256
  --seed 1337
  --model_variant lite_d
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --domain_enhancer_strength 0.35
  --primary_udu_weight 0.70
  --use_aug
  --use_mixstyle
  --mixstyle_layers time_down,t1
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_fallback skip
  --mixstyle_strength 0.60
  --mixstyle_p 0.14
  --mixstyle_late_start 120
  --mixstyle_late_ramp_epochs 40
  --mixstyle_late_min_p 0.04
  --mixstyle_late_min_strength 0.24
)

FED21_R02_BASE=(
  --train_mode fedprox
  --fl_local_objective receiver_agnostic_bex02
  --fedprox_mu 0.006
  --fl_local_epochs 3
  --lambda_fishr 0.000
  --fishr_min_domains 2
  --lambda_rx_adv 0.09
  --grl_lambda 0.16
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.00
  --fl_baseline_view_ce_only
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --sat_cons_start_epoch 58
  --fl_baseline_view_ce_weight 0.43
  --use_fed_proto_stats
  --lambda_fed_proto 0.008
  --fed_proto_momentum 0.96
  --use_fed_coral
  --lambda_fl_coral_zid_global 0.00008
  --fl_coral_stage all
  --fl_coral_start_round 180
  --fl_coral_feature z_id
  --fl_coral_cov_mode diag
  --fl_coral_min_count 2
  --fl_coral_collect_views clean
)

FISHR_COMMON=(
  --use_fed_fishr
  --fed_fishr_min_clients 2
  --fed_fishr_min_count 2
  --fed_fishr_reweight_floor 0.02
  --fed_fishr_reweight_cap 0.60
)

launch_fedfishr() {
  local candidate_id="$1" run_name="$2" gpu="$3"
  shift 3
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${TRAIN_SCRIPT}"
    "${COMMON_FED_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${run_dir}"
    --log_dir "${LOG_ROOT}/${run_name}"
    "${FED21_R02_BASE[@]}"
    "${FISHR_COMMON[@]}"
    "$@"
  )
  run_cmd "${candidate_id}" "${run_name}" "${gpu}" "${run_dir}" "${log_path}" "${cmd[@]}"
}

if [[ "${DRY_RUN}" != "1" ]]; then
  [[ -f "${TRAIN_SCRIPT}" ]] || { echo "[ERROR] missing train script: ${TRAIN_SCRIPT}" >&2; exit 2; }
fi

cd "${ROOT}"
echo "[FED22-FISHR] run_id=${RUN_ID} dry_run=${DRY_RUN} max_train_per_gpu=${MAX_TRAIN_PER_GPU}"

launch_fedfishr fedfishr_reweight_head_a015 FED22_R01_fedfishr_reweight_head_a015_r010 0 \
  --fed_fishr_mode reweight --lambda_fed_fishr 0.15 --fed_fishr_gradient_scope classifier_head --fed_fishr_sketch_dim 128 --fed_fishr_max_samples_per_class 4

launch_fedfishr fedfishr_reweight_head_a030 FED22_R02_fedfishr_reweight_head_a030_r010 1 \
  --fed_fishr_mode reweight --lambda_fed_fishr 0.30 --fed_fishr_gradient_scope classifier_head --fed_fishr_sketch_dim 128 --fed_fishr_max_samples_per_class 4

launch_fedfishr fedfishr_reweight_logit_a030 FED22_R03_fedfishr_reweight_logit_a030_r010 2 \
  --fed_fishr_mode reweight --lambda_fed_fishr 0.30 --fed_fishr_gradient_scope logit --fed_fishr_sketch_dim 0 --fed_fishr_max_samples_per_class 4

launch_fedfishr fedfishr_targetloss_head_l002 FED22_R04_fedfishr_targetloss_head_l002_r010 3 \
  --fed_fishr_mode target_loss --lambda_fed_fishr 0.02 --fed_fishr_gradient_scope classifier_head --fed_fishr_sketch_dim 128 --fed_fishr_max_samples_per_class 4 --fed_fishr_start_round 2

launch_fedfishr fedfishr_reweight_head_a060 FED22_A05_fedfishr_reweight_head_a060_r010 4 \
  --fed_fishr_mode reweight --lambda_fed_fishr 0.60 --fed_fishr_gradient_scope classifier_head --fed_fishr_sketch_dim 128 --fed_fishr_max_samples_per_class 4

launch_fedfishr fedfishr_reweight_head_sk64_m8 FED22_A06_fedfishr_reweight_head_sk64_m8_r010 5 \
  --fed_fishr_mode reweight --lambda_fed_fishr 0.30 --fed_fishr_gradient_scope classifier_head --fed_fishr_sketch_dim 64 --fed_fishr_max_samples_per_class 8

launch_fedfishr fedfishr_reweight_head_ema05_start5 FED22_A07_fedfishr_reweight_head_ema05_start5_r010 6 \
  --fed_fishr_mode reweight --lambda_fed_fishr 0.30 --fed_fishr_gradient_scope classifier_head --fed_fishr_sketch_dim 128 --fed_fishr_max_samples_per_class 4 --fed_fishr_momentum 0.5 --fed_fishr_start_round 5

launch_fedfishr fedfishr_both_head_l005 FED22_A08_fedfishr_both_head_l005_r010 7 \
  --fed_fishr_mode both --lambda_fed_fishr 0.05 --fed_fishr_gradient_scope classifier_head --fed_fishr_sketch_dim 128 --fed_fishr_max_samples_per_class 4 --fed_fishr_start_round 2 --fed_fishr_momentum 0.3
