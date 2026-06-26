#!/usr/bin/env bash
set -euo pipefail

# Federated/VMB next8 from optimizer_20260529_233451.
# Parent evidence: completed optimizer_20260529_112412_vmb_lane + optimizer_20260529_143047_vmb_missing7.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260529_233451_vmb_next8}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
DRY_RUN="${DRY_RUN:-0}"
SELECT="${SELECT:-all}"
MAX_TRAIN_PER_GPU="${MAX_TRAIN_PER_GPU:-2}"
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

VMB_BASE_ARGS=(
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
)

CANDIDATES=(
  "0|VMB2_C01_c03_cosclip_protoema985_r010|conservative|C03 strict/risk anchor with smoother prototype memory.|--lambda_vmb_tx_proto 0.15 --lambda_vmb_rx_proto 0.15 --lambda_fed_proto 0.005 --fed_proto_momentum 0.2 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.985 --fl_vmb_prototype_clip_norm 0.75"
  "1|VMB2_C02_c01_satfloor_ce075_r010|conservative|C01 satellite-floor anchor with slightly stronger baseline-view CE.|--lambda_vmb_tx_proto 0.10 --lambda_vmb_rx_proto 0.10 --lambda_fed_proto 0.03 --fed_proto_momentum 0.2 --fl_baseline_view_ce_weight 0.75 --fl_vmb_prototype_ema 0.97"
  "2|VMB2_C03_a08_rfdr_decay_stop170_r010|conservative|A08 RFDR/DSQ clean anchor with late MixStyle decay and stop to reduce rollback.|--use_aug --aug_p_rx_chain 0.45 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.97 --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.12 --mixstyle_alpha 0.2 --mixstyle_strength 0.45 --mixstyle_late_start 130 --mixstyle_late_ramp_epochs 35 --mixstyle_late_min_p 0.02 --mixstyle_late_min_strength 0.10 --mixstyle_stop_epoch 170"
  "3|VMB2_C04_a05_pcgrad_lowproto_r010|conservative|A05 PCGrad satellite-mean anchor with lower prototype pull for strict-UDU rescue.|--fl_vmb_pretrain_rounds 80 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.985 --lambda_vmb_tx_proto 0.12 --lambda_vmb_rx_proto 0.12 --lambda_fed_proto 0.02 --fl_baseline_view_ce_weight 0.75"
  "4|VMB2_A05_a08_logitkd_rfdr_r010|aggressive|A08 RFDR/DSQ plus confidence-gated logit anchors to reduce late clean rollback.|--use_aug --aug_p_rx_chain 0.45 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.97 --use_logit_anchors --lambda_logit_kd 0.02 --kd_temperature 2.0 --kd_reliability_gate 0.75 --kd_margin_min 0.10 --kd_anchor_ema 0.9 --kd_min_count 2"
  "5|VMB2_A06_stylebank_c03_floor_r010|aggressive|C03 anchor with low-probability delayed StyleBank to test satellite floor without A06 overreach.|--lambda_fed_proto 0.005 --fl_conflict_agg cosine_clip --use_fed_style_bank --fl_style_code_dim 8 --fl_style_max_views 1 --fl_style_sampling_policy diverse --fl_style_replay_prob 0.10 --fl_style_replay_start_round 40 --fl_style_phys_start_round 40 --fl_style_dg_start_round 60 --fl_style_transform_mix_alpha 0.4 --fl_style_bank_momentum 0.6 --use_fed_style_sat_view"
  "6|VMB2_A07_a05_satmean_mixdecay_r010|aggressive|A05 satellite-mean anchor with mild MixStyle late decay for anti-forgetting.|--fl_vmb_pretrain_rounds 80 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.985 --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.12 --mixstyle_alpha 0.2 --mixstyle_strength 0.45 --mixstyle_late_start 130 --mixstyle_late_ramp_epochs 40 --mixstyle_late_min_p 0.02 --mixstyle_late_min_strength 0.10 --fl_baseline_view_ce_weight 0.75"
  "7|VMB2_A08_proto_high_pretrain90_clip_r010|aggressive|Longer pretrain plus clipped higher prototypes to probe clean ceiling without unbounded drift.|--fl_vmb_pretrain_rounds 90 --fl_vmb_stage1_lr_mult 1.25 --lambda_vmb_tx_proto 0.20 --lambda_vmb_rx_proto 0.20 --lambda_fed_proto 0.02 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.985 --fl_vmb_prototype_clip_norm 0.50 --fl_baseline_view_ce_weight 0.80"
)

cd "${ROOT}"

if [[ ! -f "${ROOT}/train.py" ]]; then
  echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
  exit 2
fi

launch_one() {
  local gpu="$1"
  local run_name="$2"
  local group="$3"
  local description="$4"
  local extra="$5"
  local run_dir="${RUNS_ROOT}/${run_name}"
  local log_dir="${LOG_ROOT}/${run_name}"
  local log_path="${LOG_ROOT}/${run_name}.out"
  local active_count
  local extra_args=()

  read -r -a extra_args <<< "${extra}"

  CMD=(
    env "CUDA_VISIBLE_DEVICES=${gpu}" "${THREAD_ENV[@]}" PYTHONPATH=. "${PYTHON}" -u train.py
    "${COMMON_ARGS[@]}"
    --run_name "${run_name}"
    --output_dir "${run_dir}"
    --log_dir "${log_dir}"
    "${VMB_BASE_ARGS[@]}"
    "${extra_args[@]}"
  )

  echo "[VMB-NEXT8] run=${run_name} gpu=${gpu} group=${group} dry_run=${DRY_RUN}"
  echo "[VMB-NEXT8-DESC] ${description}"
  printf '[VMB-NEXT8-CMD]'
  printf ' %q' "${CMD[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi

  if [[ -e "${run_dir}" || -e "${log_dir}" || -e "${log_path}" ]]; then
    echo "[ERROR] target path exists for ${run_name}: ${run_dir} / ${log_dir} / ${log_path}" >&2
    exit 3
  fi

  active_count="$(
    nvidia-smi --id="${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
      | sed '/^$/d' \
      | wc -l \
      | tr -d ' '
  )"
  if [[ "${active_count}" -ge "${MAX_TRAIN_PER_GPU}" ]]; then
    echo "[ERROR] gpu=${gpu} has ${active_count} compute process(es), max=${MAX_TRAIN_PER_GPU}" >&2
    exit 4
  fi

  mkdir -p "${run_dir}" "${log_dir}"
  nohup "${CMD[@]}" > "${log_path}" 2>&1 &
  local pid="$!"
  printf "%s\t%s\t%s\t%s\t%s\n" "${run_name}" "${gpu}" "${pid}" "${log_path}" "${run_dir}" | tee -a "${LOG_ROOT}/launch_pids.tsv"
}

for candidate in "${CANDIDATES[@]}"; do
  IFS='|' read -r gpu run_name group description extra <<< "${candidate}"
  if [[ "${SELECT}" != "all" && ",${SELECT}," != *",${run_name},"* ]]; then
    continue
  fi
  launch_one "${gpu}" "${run_name}" "${group}" "${description}" "${extra}"
done
