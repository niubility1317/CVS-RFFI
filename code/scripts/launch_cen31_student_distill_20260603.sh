#!/usr/bin/env bash
set -euo pipefail

# CEN31 -> low-latency student distillation matrix.
# The CEN31 teacher is intentionally kept fixed; candidates vary only the student
# architecture and structural latency paths.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
DISTILL_SCRIPT="${DISTILL_SCRIPT:-${ROOT}/code/train_cen31_distill.py}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/optimizer_20260530_043050_centralized_next8/CEN_A31_a22_satboost_ce1p28_stack_r010/best_primary_ood_model.pth}"
RUN_ID="${RUN_ID:-cen31_student_distill_20260603}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-300}"
ONLY_CANDIDATE="${ONLY_CANDIDATE:-}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    --only=*)
      ONLY_CANDIDATE="${arg#--only=}"
      ;;
    *)
      echo "[ERROR] unknown argument: ${arg}" >&2
      exit 2
      ;;
  esac
done

COMMON_DATA_ARGS=(
  --dataset wisig
  --wisig_domain rx_day
  --wisig_out_len 256
  --wisig_train_ratio 0.1
  --wisig_guard_gap 8
  --wisig_train_days 0,1
  --wisig_test_days 2,3
  --wisig_train_rxs 0,1,2,3,4,5,6
  --wisig_test_rxs 7,8,9,10,11
  --batch_size 256
  --eval_batch_size 256
  --num_workers 4
  --prefetch_factor 2
  --epochs "${DISTILL_EPOCHS}"
  --eval_interval 10
  --eval_max_batches 0
  --seed 1337
)

COMMON_KD_ARGS=(
  --teacher_ckpt "${TEACHER_CKPT}"
  --lambda_kd 0.70
  --lambda_feature_kd 0.20
  --lambda_relation_kd 0.05
  --kd_temperature 3.0
  --kd_conf_min 0.60
  --kd_margin_min 0.05
  --kd_require_correct
  --lr 4e-4
  --lr_min 1e-6
  --wd 1e-4
  --label_smoothing 0.01
  --no_eval_sat_channel
)

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

launch() {
  local candidate_id="$1"
  local run_name="$2"
  local gpu="$3"
  shift 3

  if [[ -n "${ONLY_CANDIDATE}" && "${candidate_id}" != "${ONLY_CANDIDATE}" && "${run_name}" != "${ONLY_CANDIDATE}" ]]; then
    return 0
  fi

  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local cmd=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "PYTHONPATH=${ROOT}:${ROOT}/code:${PYTHONPATH:-}" "${PYTHON}" -u "${DISTILL_SCRIPT}"
    "${COMMON_DATA_ARGS[@]}"
    "${COMMON_KD_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${run_dir}"
    --latest_save_path "${run_dir}/latest_student.pth"
    --best_save_path "${run_dir}/best_student_primary.pth"
    --latency_profile_json "${run_dir}/latency_profile.json"
    "$@"
  )

  echo "[CEN31-STUDENT-KD] candidate=${candidate_id} run=${run_name} gpu=${gpu} dry_run=${DRY_RUN}"
  printf '[CEN31-STUDENT-KD-CMD]'
  print_cmd "${cmd[@]}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ ! -f "${DISTILL_SCRIPT}" ]]; then
    echo "[ERROR] distill script not found: ${DISTILL_SCRIPT}" >&2
    exit 2
  fi
  if [[ ! -f "${TEACHER_CKPT}" ]]; then
    echo "[ERROR] teacher checkpoint not found: ${TEACHER_CKPT}" >&2
    exit 2
  fi
  if [[ -e "${run_dir}" || -e "${log_path}" ]]; then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "BLOCKED_PATH_COLLISION" "${log_path}" "${run_dir}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  local count
  count="$(gpu_process_count "${gpu}")"
  if (( count >= MAX_TRAIN_PER_GPU )); then
    mkdir -p "${LOG_ROOT}"
    printf "%s\t%s\t%s\tgpu=%s active_count=%s max=%s\n" \
      "${candidate_id}" "${run_name}" "BLOCKED_CAPACITY" "${gpu}" "${count}" "${MAX_TRAIN_PER_GPU}" \
      | tee -a "${LOG_ROOT}/blocked.tsv"
    return 0
  fi

  mkdir -p "${LOG_ROOT}" "${run_dir}"
  nohup "${cmd[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "${candidate_id}" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" \
    | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

cd "${ROOT}"

launch KD_F_BALANCED \
  CEN31KD_lite_f_physlite_r010 \
  0 \
  --arch_family cvsincnet \
  --model_variant lite_f \
  --branch_ablation no_dac,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.20
sleep 2

launch KD_G_FAST \
  CEN31KD_lite_g_no_pa_fast_r010 \
  1 \
  --arch_family cvsincnet \
  --model_variant lite_g \
  --branch_ablation no_dac,no_pa,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer rcn_stats \
  --domain_enhancer_strength 0.15
sleep 2

launch KD_H_NANO \
  CEN31KD_lite_h_timeonly_nano_r010 \
  2 \
  --arch_family cvsincnet \
  --model_variant lite_h \
  --branch_ablation time_only \
  --domain_branch_ablation time_only \
  --domain_enhancer off \
  --domain_enhancer_strength 0.0 \
  --lambda_feature_kd 0.12 \
  --lambda_relation_kd 0.03
sleep 2

launch KD_CVCNN_SPEED_FLOOR \
  CEN31KD_cvcnn_speed_floor_r010 \
  3 \
  --arch_family cvcnn \
  --model_variant lite_h \
  --branch_ablation no_dac,no_pa,no_freq,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.0 \
  --lambda_feature_kd 0.10 \
  --lambda_relation_kd 0.03
sleep 2

launch KD_SINCCVCNN_HYBRID \
  CEN31KD_sinc_cvcnn_hybrid_r010 \
  5 \
  --arch_family sinc_cvcnn \
  --model_variant lite_h \
  --branch_ablation no_dac,no_pa,no_freq,no_stats \
  --domain_branch_ablation no_stats \
  --domain_enhancer off \
  --domain_enhancer_strength 0.0 \
  --lambda_feature_kd 0.12 \
  --lambda_relation_kd 0.03
