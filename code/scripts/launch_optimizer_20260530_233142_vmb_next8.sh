#!/usr/bin/env bash
set -euo pipefail

# Federated/VMB next8 from optimizer_20260530_233142.
# Parent evidence: completed optimizer_20260530_173224_vmb_next8 (VMB5_*).

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260530_233142_vmb_next8}"
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
  --fl_test_eval_last_n 30
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
  --fl_vmb_stage1_objective ce
  --fl_vmb_stage1_local_steps 2
  --fl_vmb_batches_per_client 1
  --fl_vmb_domain_balanced_sampling
  --fl_vmb_domain_balanced_aggregation
  --fl_vmb_transmitter_balanced_batch
  --lambda_tx_adv_r 0.1
  --lambda_rx_adv 0.1
  --lambda_orth 0.1
  --use_tx_adv_on_zdom
  --use_sat_consistency
  --fl_sat_aug_mode baseline_view
  --fl_baseline_view_ce_only
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit
  --sat_view_prob 1.0
  --sat_cons_start_epoch 1
  --use_fed_proto_stats
)

CANDIDATES=(
  "0|VMB6_C01_a07_serverdamp_last30_r010|conservative|A07 parent with server-update damping and denser last-30 heavy eval targets final-vs-best rollback rather than another shallow KD sweep.|--fl_vmb_pretrain_rounds 90 --fl_vmb_stage1_lr_mult 1.10 --fl_vmb_server_lr 0.006 --fl_vmb_server_momentum 0.70 --lambda_vmb_tx_proto 0.12 --lambda_vmb_rx_proto 0.12 --lambda_fed_proto 0.012 --fed_proto_momentum 0.20 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.990 --fl_vmb_prototype_clip_norm 0.40 --fl_baseline_view_ce_weight 0.78 --use_aug --aug_p_rx_chain 0.25 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.05 --mixstyle_alpha 0.2 --mixstyle_strength 0.22 --mixstyle_late_start 100 --mixstyle_late_ramp_epochs 30 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.04 --mixstyle_stop_epoch 115 --use_logit_anchors --lambda_logit_kd 0.005 --kd_temperature 2.0 --kd_reliability_gate 0.92 --kd_margin_min 0.12 --kd_anchor_ema 0.94 --kd_min_count 2"
  "1|VMB6_C02_minrx_groupce_uniform_nokd_r010|conservative|Explicit min-RX guard using uniform aggregation pressure and hard group CE; removes KD to separate weak-RX rescue from anchor noise.|--fl_vmb_pretrain_rounds 90 --fl_vmb_stage1_lr_mult 1.15 --fl_vmb_server_lr 0.008 --fl_vmb_server_momentum 0.80 --fl_agg_weight uniform --lambda_vmb_tx_proto 0.12 --lambda_vmb_rx_proto 0.16 --lambda_fed_proto 0.012 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.990 --fl_vmb_prototype_clip_norm 0.35 --fl_baseline_view_ce_weight 0.75 --lambda_group_ce 0.04 --group_ce_mode dual_worst --group_ce_top_frac 0.50 --group_ce_min_domains 2 --groupdro_tau 0.45 --groupdro_cap 0.62 --use_aug --aug_p_rx_chain 0.20 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.04 --mixstyle_alpha 0.2 --mixstyle_strength 0.18 --mixstyle_late_start 105 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 125 --no_use_logit_anchors"
  "2|VMB6_C03_a08_lowtail_highbest_r010|conservative|A08 high-best parent with smaller stage2 steps and tighter anchor gate tests whether the 74.9 peak can survive to final.|--fl_vmb_pretrain_rounds 100 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_server_lr 0.005 --fl_vmb_server_momentum 0.65 --lambda_vmb_tx_proto 0.14 --lambda_vmb_rx_proto 0.14 --lambda_fed_proto 0.018 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.990 --fl_vmb_prototype_clip_norm 0.30 --fl_baseline_view_ce_weight 0.84 --use_aug --aug_p_rx_chain 0.25 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.04 --mixstyle_alpha 0.2 --mixstyle_strength 0.18 --mixstyle_late_start 90 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 100 --use_logit_anchors --lambda_logit_kd 0.006 --kd_temperature 2.0 --kd_reliability_gate 0.95 --kd_margin_min 0.14 --kd_anchor_ema 0.95 --kd_min_count 2"
  "3|VMB6_C04_bpc2_lownoise_guard_r010|conservative|Two VMB batches per client reduces stochastic gradient noise, a real mechanism not covered by VMB5 one-batch sweeps.|--fl_vmb_pretrain_rounds 105 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_batches_per_client 2 --fl_vmb_server_lr 0.006 --fl_vmb_server_momentum 0.75 --lambda_vmb_tx_proto 0.14 --lambda_vmb_rx_proto 0.14 --lambda_fed_proto 0.016 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.990 --fl_vmb_prototype_clip_norm 0.35 --fl_baseline_view_ce_weight 0.82 --use_aug --aug_p_rx_chain 0.18 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.035 --mixstyle_alpha 0.2 --mixstyle_strength 0.16 --mixstyle_late_start 105 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 130 --use_logit_anchors --lambda_logit_kd 0.006 --kd_temperature 2.0 --kd_reliability_gate 0.93 --kd_margin_min 0.13 --kd_anchor_ema 0.95 --kd_min_count 2"
  "4|VMB6_A05_stylebank_receiver_balanced_r010|aggressive|Receiver-balanced StyleBank replay directly targets the RX8 floor by injecting remote receiver-style domains rather than changing only KD/proto weights.|--fl_vmb_pretrain_rounds 90 --fl_vmb_stage1_lr_mult 1.10 --fl_vmb_server_lr 0.007 --fl_vmb_server_momentum 0.75 --lambda_vmb_tx_proto 0.12 --lambda_vmb_rx_proto 0.14 --lambda_fed_proto 0.014 --fed_proto_momentum 0.20 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.990 --fl_vmb_prototype_clip_norm 0.40 --fl_baseline_view_ce_weight 0.78 --use_aug --aug_p_rx_chain 0.22 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.04 --mixstyle_alpha 0.2 --mixstyle_strength 0.18 --mixstyle_late_start 100 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 125 --use_logit_anchors --lambda_logit_kd 0.005 --kd_temperature 2.0 --kd_reliability_gate 0.92 --kd_margin_min 0.12 --kd_anchor_ema 0.94 --kd_min_count 2 --use_fed_style_bank --use_fl_style_bank_stats --fl_style_sampling_policy receiver_balanced --fl_style_replay_start_round 40 --fl_style_phys_start_round 40 --fl_style_dg_start_round 70 --fl_style_dg_min_domains 2 --fl_style_domain_label_mode target_receiver --fl_style_max_views 2 --fl_style_replay_prob 0.20 --fl_style_transform_mix_alpha 0.50 --fl_style_bank_max_centroids 96 --fl_style_bank_merge_radius 0.08 --fl_style_phys_p_lowpass 0.40 --fl_style_phys_p_multipath 0.40"
  "5|VMB6_A06_style_satview_targetbalanced_r010|aggressive|StyleBank plus satellite-view style batches stress-tests channel robustness with an explicit mechanism absent from VMB5.|--fl_vmb_pretrain_rounds 95 --fl_vmb_stage1_lr_mult 1.10 --fl_vmb_server_lr 0.007 --fl_vmb_server_momentum 0.75 --lambda_vmb_tx_proto 0.14 --lambda_vmb_rx_proto 0.16 --lambda_fed_proto 0.018 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.990 --fl_vmb_prototype_clip_norm 0.35 --fl_baseline_view_ce_weight 0.86 --use_aug --aug_p_rx_chain 0.20 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.04 --mixstyle_alpha 0.2 --mixstyle_strength 0.18 --mixstyle_late_start 95 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 120 --use_logit_anchors --lambda_logit_kd 0.006 --kd_temperature 2.0 --kd_reliability_gate 0.94 --kd_margin_min 0.13 --kd_anchor_ema 0.95 --kd_min_count 2 --use_fed_style_bank --use_fl_style_bank_stats --use_fed_style_sat_view --fl_style_sampling_policy target_balanced --fl_style_replay_start_round 35 --fl_style_phys_start_round 35 --fl_style_dg_start_round 65 --fl_style_dg_min_domains 2 --fl_style_domain_label_mode target_receiver --fl_style_max_views 2 --fl_style_replay_prob 0.25 --fl_style_transform_mix_alpha 0.45 --fl_style_bank_max_centroids 128 --fl_style_bank_merge_radius 0.06 --fl_style_phys_p_lowpass 0.45 --fl_style_phys_p_multipath 0.45"
  "6|VMB6_A07_unfreeze_pcgrad_lowlr_r010|aggressive|Unfreezing the receiver/domain branch under low-LR PCGrad is a mechanism-level weak-RX intervention, not a VMB5 PCGrad neighbor.|--fl_vmb_pretrain_rounds 90 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_server_lr 0.004 --fl_vmb_server_momentum 0.60 --fl_vmb_weight_decay 0.00001 --no_fl_vmb_freeze_rx_stage2 --lambda_tx_adv_r 0.06 --lambda_rx_adv 0.06 --lambda_vmb_tx_proto 0.10 --lambda_vmb_rx_proto 0.18 --lambda_fed_proto 0.016 --fed_proto_momentum 0.20 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.990 --fl_vmb_prototype_clip_norm 0.35 --fl_baseline_view_ce_weight 0.78 --lambda_group_ce 0.03 --group_ce_mode dual_worst --group_ce_top_frac 0.50 --group_ce_min_domains 2 --use_aug --aug_p_rx_chain 0.20 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.035 --mixstyle_alpha 0.2 --mixstyle_strength 0.16 --mixstyle_late_start 105 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 120 --use_logit_anchors --lambda_logit_kd 0.004 --kd_temperature 2.0 --kd_reliability_gate 0.94 --kd_margin_min 0.13 --kd_anchor_ema 0.95 --kd_min_count 2"
  "7|VMB6_A08_asymproto_all_sat_r010|aggressive|Asymmetric TX/RX prototype pressure plus all-split satellite eval adds a new risk surface and improves reportable satellite/joint ranking coverage.|--eval_sat_on all --fl_vmb_pretrain_rounds 100 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_server_lr 0.006 --fl_vmb_server_momentum 0.70 --lambda_vmb_tx_proto 0.08 --lambda_vmb_rx_proto 0.22 --lambda_fed_proto 0.030 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.995 --fl_vmb_prototype_clip_norm 0.25 --fl_baseline_view_ce_weight 0.84 --use_aug --aug_p_rx_chain 0.22 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.04 --mixstyle_alpha 0.2 --mixstyle_strength 0.18 --mixstyle_late_start 95 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 120 --use_logit_anchors --lambda_logit_kd 0.006 --kd_temperature 2.0 --kd_reliability_gate 0.96 --kd_margin_min 0.14 --kd_anchor_ema 0.96 --kd_min_count 2"
)

cd "${ROOT}"

if [[ ! -f "${ROOT}/train.py" ]]; then
  echo "[ERROR] ROOT does not contain train.py: ${ROOT}" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

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

  echo "[VMB6-NEXT8] run=${run_name} gpu=${gpu} group=${group} dry_run=${DRY_RUN}"
  echo "[VMB6-NEXT8-DESC] ${description}"
  printf '[VMB6-NEXT8-CMD]'
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
