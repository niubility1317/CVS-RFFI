#!/usr/bin/env bash
set -euo pipefail

# One-run federated/VMB lane candidate selected by optimizer_20260529_112412.
# Parent evidence anchor: VPT_B60_proto015_satce_r010 plus negative B60_mixstyle rollback evidence.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260529_112412_vmb_lane}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
GPU="${GPU:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME="VMB_C10_proto015_late_mixstyle_decay_r010"
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
  --batch_size 128
  --eval_batch_size 256
  --seed 1337
)

VMB_ARGS=(
  --train_mode fedcvs_vmb
  --fl_local_objective receiver_agnostic_bex02
  --fl_vmb_stage auto
  --fl_vmb_pretrain_rounds 60
  --fl_vmb_stage1_objective ce
  --fl_vmb_stage1_local_steps 2
  --fl_vmb_stage1_lr_mult 1.5
  --fl_vmb_batches_per_client 1
  --fl_vmb_server_lr 0.01
  --fl_vmb_server_momentum 0.9
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --lambda_vmb_tx_proto 0.15
  --lambda_vmb_rx_proto 0.15
  --lambda_tx_adv_r 0.1
  --lambda_rx_adv 0.1
  --lambda_orth 0.1
  --use_tx_adv_on_zdom
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --fl_baseline_view_ce_only
  --fl_baseline_view_ce_weight 0.7
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.0
  --sat_cons_start_epoch 1
  --use_fed_proto_stats
  --lambda_fed_proto 0.05
  --fed_proto_momentum 0.2
  --use_mixstyle
  --mixstyle_mix same_tx_crossdomain
  --mixstyle_p 0.20
  --mixstyle_alpha 0.2
  --mixstyle_strength 0.60
  --mixstyle_late_start 150
  --mixstyle_late_ramp_epochs 30
  --mixstyle_late_min_p 0.03
  --mixstyle_late_min_strength 0.15
)

cd "${ROOT}"

CMD=(
  env "CUDA_VISIBLE_DEVICES=${GPU}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py
  "${COMMON_ARGS[@]}"
  --run_name "${RUN_NAME}"
  --output_dir "${RUNS_ROOT}/${RUN_NAME}"
  --log_dir "${LOG_ROOT}/${RUN_NAME}"
  "${VMB_ARGS[@]}"
)

echo "[VMB-LANE] root=${ROOT} run_id=${RUN_ID} run=${RUN_NAME} gpu=${GPU} dry_run=${DRY_RUN}"
printf '[VMB-LANE-CMD]'
printf ' %q' "${CMD[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

if [[ ! -f "${ROOT}/train.py" ]]; then
  echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
  exit 2
fi

active_pids="$(nvidia-smi --id="${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^$/d' || true)"
if [[ -n "${active_pids}" ]]; then
  echo "[ERROR] gpu=${GPU} already has compute process(es): ${active_pids}" >&2
  exit 4
fi

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}/${RUN_NAME}"
LOG_PATH="${LOG_ROOT}/${RUN_NAME}.out"
nohup "${CMD[@]}" > "${LOG_PATH}" 2>&1 &
PID="$!"
printf "%s\t%s\t%s\t%s\t%s\n" "${RUN_NAME}" "${GPU}" "${PID}" "${LOG_PATH}" "${RUNS_ROOT}/${RUN_NAME}" | tee "${LOG_ROOT}/launch_pids.tsv"
