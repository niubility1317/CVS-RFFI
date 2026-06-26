#!/usr/bin/env bash
set -euo pipefail

# VMB stage1/stage2 CEN_A31-inspired matrix, created 2026-06-01.
# Hard constraints: WiSig train ratio 0.1, receiver clients, 200 FL rounds.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260601_014727_vmb_cen_a31_stage12}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
MAX_PROCS_PER_GPU="${MAX_PROCS_PER_GPU:-2}"
ONLY_INDEX="${ONLY_INDEX:-}"
ONLY_RUN="${ONLY_RUN:-}"
CPU_THREADS="${CPU_THREADS:-${CVSRFFI_CPU_THREADS:-4}}"
CPU_INTEROP_THREADS="${CPU_INTEROP_THREADS:-${CVSRFFI_CPU_INTEROP_THREADS:-1}}"

THREAD_ENV=(
  "CVSRFFI_CPU_THREADS=${CPU_THREADS}"
  "CVSRFFI_CPU_INTEROP_THREADS=${CPU_INTEROP_THREADS}"
  "OMP_NUM_THREADS=${CPU_THREADS}"
  "MKL_NUM_THREADS=${CPU_THREADS}"
  "OPENBLAS_NUM_THREADS=${CPU_THREADS}"
  "NUMEXPR_NUM_THREADS=${CPU_THREADS}"
)

COMMON_ARGS=(
  --dataset wisig
  --wisig_pkl "${WISIG_PKL}"
  --wisig_equalized 1
  --wisig_domain rx_day
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
  --fl_test_eval_last_n 10
  --eval_sat_channel
  --eval_sat_on main
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_eval_max_batches 0
  --num_workers 0
  --fl_num_workers 0
  --batch_size 128
  --eval_batch_size 256
  --seed 1337
)

VMB_CEN_ARGS=(
  --train_mode fedcvs_vmb
  --fl_vmb_cen_a31_profile
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_stage1_local_steps 2
  --fl_vmb_stage1_lr_mult 1.3
  --fl_vmb_batches_per_client 1
  --fl_conflict_agg cosine_clip
  --fl_vmb_server_lr 0.008
  --fl_vmb_server_momentum 0.9
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --fl_vmb_adv_warmup_rounds 15
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.0
  --sat_cons_start_epoch 1
)

RUN_NAMES=(
  VMB8_A01_ce_stage1_cen_ce100_r010
  VMB8_A02_ce_stage1_cen_ce114_r010
  VMB8_A03_ce_stage1_cen_ce128_r010
  VMB8_B01_receiver_style_ce114_r010
  VMB8_B02_receiver_style_ce128_r010
  VMB8_B03_receiver_style_unfreeze_ce114_r010
  VMB8_C01_cenlite_stylebank_ce114_r010
  VMB8_D01_full_proposal_bpc2_ce114_r010
)

cd "${ROOT}"

if [[ ! -f "${ROOT}/train.py" ]]; then
  echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
PID_FILE="${LOG_ROOT}/launch_pids.tsv"
if [[ ! -f "${PID_FILE}" ]]; then
  printf "run_name\tgpu\tpid\tlog_path\toutput_dir\n" > "${PID_FILE}"
fi

for idx in "${!RUN_NAMES[@]}"; do
  GPU="${idx}"
  RUN_NAME="${RUN_NAMES[$idx]}"
  if [[ -n "${ONLY_INDEX}" && "${ONLY_INDEX}" != "${idx}" ]]; then
    continue
  fi
  if [[ -n "${ONLY_RUN}" && "${ONLY_RUN}" != "${RUN_NAME}" ]]; then
    continue
  fi
  EXTRA_ARGS=()
  case "${RUN_NAME}" in
    VMB8_A01_ce_stage1_cen_ce100_r010)
      EXTRA_ARGS=(--fl_vmb_freeze_rx_stage2 --fl_vmb_pretrain_rounds 50 --fl_vmb_stage1_objective ce --no_fl_vmb_stage1_use_aux_losses --fl_baseline_view_ce_weight 1.00)
      ;;
    VMB8_A02_ce_stage1_cen_ce114_r010)
      EXTRA_ARGS=(--fl_vmb_freeze_rx_stage2 --fl_vmb_pretrain_rounds 50 --fl_vmb_stage1_objective ce --no_fl_vmb_stage1_use_aux_losses --fl_baseline_view_ce_weight 1.14)
      ;;
    VMB8_A03_ce_stage1_cen_ce128_r010)
      EXTRA_ARGS=(--fl_vmb_freeze_rx_stage2 --fl_vmb_pretrain_rounds 50 --fl_vmb_stage1_objective ce --no_fl_vmb_stage1_use_aux_losses --fl_baseline_view_ce_weight 1.28)
      ;;
    VMB8_B01_receiver_style_ce114_r010)
      EXTRA_ARGS=(--fl_vmb_freeze_rx_stage2 --fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_objective receiver_style_pretrain --fl_baseline_view_ce_weight 1.14)
      ;;
    VMB8_B02_receiver_style_ce128_r010)
      EXTRA_ARGS=(--fl_vmb_freeze_rx_stage2 --fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_objective receiver_style_pretrain --fl_baseline_view_ce_weight 1.28)
      ;;
    VMB8_B03_receiver_style_unfreeze_ce114_r010)
      EXTRA_ARGS=(--fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_objective receiver_style_pretrain --fl_baseline_view_ce_weight 1.14 --no_fl_vmb_freeze_rx_stage2)
      ;;
    VMB8_C01_cenlite_stylebank_ce114_r010)
      EXTRA_ARGS=(--fl_vmb_freeze_rx_stage2 --fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_objective cen_a31_lite --fl_baseline_view_ce_weight 1.14 --fl_style_replay_start_round 5 --fl_style_phys_start_round 5 --fl_style_dg_start_round 10 --fl_style_max_views 3 --fl_style_transform_mix_alpha 0.65)
      ;;
    VMB8_D01_full_proposal_bpc2_ce114_r010)
      EXTRA_ARGS=(--fl_vmb_freeze_rx_stage2 --fl_vmb_pretrain_rounds 60 --fl_vmb_stage1_objective receiver_style_pretrain --fl_baseline_view_ce_weight 1.14 --fl_vmb_batches_per_client 2 --fl_vmb_server_lr 0.006 --lambda_vmb_tx_proto 0.14 --lambda_vmb_rx_proto 0.14 --lambda_fed_proto 0.02)
      ;;
  esac

  active_pids="$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  active_count="$(printf '%s\n' "${active_pids}" | sed '/^$/d' | wc -l | tr -d ' ')"
  active_count="${active_count:-0}"
  if [[ "${DRY_RUN}" != "1" && "${active_count}" -ge "${MAX_PROCS_PER_GPU}" ]]; then
    echo "[SKIP] gpu=${GPU} active_compute=${active_count} max=${MAX_PROCS_PER_GPU} run=${RUN_NAME}" >&2
    continue
  fi

  CMD=(
    env "CUDA_VISIBLE_DEVICES=${GPU}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${RUN_NAME}"
    --output_dir "${RUNS_ROOT}/${RUN_NAME}"
    --log_dir "${LOG_ROOT}/${RUN_NAME}"
    "${VMB_CEN_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
  )

  echo "[VMB8] idx=${idx} gpu=${GPU} active_compute=${active_count} run=${RUN_NAME} dry_run=${DRY_RUN}"
  printf '[VMB8-CMD]'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    continue
  fi

  mkdir -p "${LOG_ROOT}/${RUN_NAME}" "${RUNS_ROOT}/${RUN_NAME}"
  LOG_PATH="${LOG_ROOT}/${RUN_NAME}.out"
  nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
  PID="$!"
  printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU}" "${PID}" "${LOG_PATH}" "${RUNS_ROOT}/${RUN_NAME}" | tee -a "${PID_FILE}"
done
