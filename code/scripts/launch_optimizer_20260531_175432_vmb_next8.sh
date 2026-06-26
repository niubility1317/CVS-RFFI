#!/usr/bin/env bash
set -euo pipefail

# Federated/VMB next8 after VMB6 completed.
# Evidence anchors: A05 has the best final strict UDU, C04 has the best
# satellite/joint/risk-adjusted result, and A08 is a near-stable proto variant.

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-optimizer_20260531_175432_vmb_next8}"
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
  "0|VMB7_C01_a05_stylebank_finalguard_r010|conservative|A05 parent with lower server LR and stricter KD gate to preserve best final strict UDU while reducing final-vs-best rollback.|--fl_vmb_pretrain_rounds 92 --fl_vmb_stage1_lr_mult 1.08 --fl_vmb_server_lr 0.0055 --fl_vmb_server_momentum 0.72 --lambda_vmb_tx_proto 0.12 --lambda_vmb_rx_proto 0.14 --lambda_fed_proto 0.014 --fed_proto_momentum 0.20 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.992 --fl_vmb_prototype_clip_norm 0.38 --fl_baseline_view_ce_weight 0.80 --use_aug --aug_p_rx_chain 0.22 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.035 --mixstyle_alpha 0.2 --mixstyle_strength 0.16 --mixstyle_late_start 105 --mixstyle_late_ramp_epochs 25 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 125 --use_logit_anchors --lambda_logit_kd 0.004 --kd_temperature 2.0 --kd_reliability_gate 0.94 --kd_margin_min 0.13 --kd_anchor_ema 0.95 --kd_min_count 2 --use_fed_style_bank --use_fl_style_bank_stats --fl_style_sampling_policy receiver_balanced --fl_style_replay_start_round 45 --fl_style_phys_start_round 45 --fl_style_dg_start_round 75 --fl_style_dg_min_domains 2 --fl_style_domain_label_mode target_receiver --fl_style_max_views 2 --fl_style_replay_prob 0.16 --fl_style_transform_mix_alpha 0.45 --fl_style_bank_max_centroids 96 --fl_style_bank_merge_radius 0.08 --fl_style_phys_p_lowpass 0.38 --fl_style_phys_p_multipath 0.38"
  "1|VMB7_C02_c04_bpc2_satfloor_r010|conservative|C04 parent with BPC2 retained and slightly higher satellite CE weight to improve the only VMB6 run above SAT 40 average.|--fl_vmb_pretrain_rounds 105 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_batches_per_client 2 --fl_vmb_server_lr 0.0055 --fl_vmb_server_momentum 0.74 --lambda_vmb_tx_proto 0.14 --lambda_vmb_rx_proto 0.15 --lambda_fed_proto 0.016 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.992 --fl_vmb_prototype_clip_norm 0.34 --fl_baseline_view_ce_weight 0.86 --use_aug --aug_p_rx_chain 0.16 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.03 --mixstyle_alpha 0.2 --mixstyle_strength 0.14 --mixstyle_late_start 110 --mixstyle_late_ramp_epochs 20 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 135 --use_logit_anchors --lambda_logit_kd 0.005 --kd_temperature 2.0 --kd_reliability_gate 0.94 --kd_margin_min 0.13 --kd_anchor_ema 0.95 --kd_min_count 2"
  "2|VMB7_C03_a08_asymproto_satkeep_r010|conservative|A08 asymmetric-prototype parent with reduced clip and stronger baseline-view CE to keep SAT near 40 without losing strict UDU.|--fl_vmb_pretrain_rounds 100 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_server_lr 0.0055 --fl_vmb_server_momentum 0.70 --lambda_vmb_tx_proto 0.08 --lambda_vmb_rx_proto 0.22 --lambda_fed_proto 0.026 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.995 --fl_vmb_prototype_clip_norm 0.24 --fl_baseline_view_ce_weight 0.88 --use_aug --aug_p_rx_chain 0.20 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.035 --mixstyle_alpha 0.2 --mixstyle_strength 0.16 --mixstyle_late_start 100 --mixstyle_late_ramp_epochs 22 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 122 --use_logit_anchors --lambda_logit_kd 0.005 --kd_temperature 2.0 --kd_reliability_gate 0.96 --kd_margin_min 0.14 --kd_anchor_ema 0.96 --kd_min_count 2"
  "3|VMB7_C04_c02_minrx_nokd_stable_r010|conservative|C02 min-RX guard without KD, but lower server LR and lighter group pressure to reduce rollback seen in VMB6.|--fl_vmb_pretrain_rounds 92 --fl_vmb_stage1_lr_mult 1.12 --fl_vmb_server_lr 0.0065 --fl_vmb_server_momentum 0.78 --fl_agg_weight uniform --lambda_vmb_tx_proto 0.12 --lambda_vmb_rx_proto 0.16 --lambda_fed_proto 0.012 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.992 --fl_vmb_prototype_clip_norm 0.34 --fl_baseline_view_ce_weight 0.78 --lambda_group_ce 0.035 --group_ce_mode dual_worst --group_ce_top_frac 0.45 --group_ce_min_domains 2 --groupdro_tau 0.45 --groupdro_cap 0.60 --use_aug --aug_p_rx_chain 0.18 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.035 --mixstyle_alpha 0.2 --mixstyle_strength 0.16 --mixstyle_late_start 108 --mixstyle_late_ramp_epochs 24 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 128 --no_use_logit_anchors"
  "4|VMB7_A05_c04_stylebank_bpc2_r010|aggressive|C04 BPC2 plus receiver-balanced StyleBank tests whether the SAT-best parent benefits from explicit receiver-style replay.|--fl_vmb_pretrain_rounds 105 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_batches_per_client 2 --fl_vmb_server_lr 0.006 --fl_vmb_server_momentum 0.75 --lambda_vmb_tx_proto 0.14 --lambda_vmb_rx_proto 0.15 --lambda_fed_proto 0.016 --fed_proto_momentum 0.20 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.992 --fl_vmb_prototype_clip_norm 0.34 --fl_baseline_view_ce_weight 0.86 --use_aug --aug_p_rx_chain 0.18 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.032 --mixstyle_alpha 0.2 --mixstyle_strength 0.15 --mixstyle_late_start 108 --mixstyle_late_ramp_epochs 22 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 132 --use_logit_anchors --lambda_logit_kd 0.005 --kd_temperature 2.0 --kd_reliability_gate 0.94 --kd_margin_min 0.13 --kd_anchor_ema 0.95 --kd_min_count 2 --use_fed_style_bank --use_fl_style_bank_stats --fl_style_sampling_policy receiver_balanced --fl_style_replay_start_round 45 --fl_style_phys_start_round 45 --fl_style_dg_start_round 75 --fl_style_dg_min_domains 2 --fl_style_domain_label_mode target_receiver --fl_style_max_views 2 --fl_style_replay_prob 0.18 --fl_style_transform_mix_alpha 0.45 --fl_style_bank_max_centroids 96 --fl_style_bank_merge_radius 0.08 --fl_style_phys_p_lowpass 0.40 --fl_style_phys_p_multipath 0.40"
  "5|VMB7_A06_a05_unfreeze_style_lowlr_r010|aggressive|A05 StyleBank plus low-LR receiver/domain unfreezing tests a direct weak-receiver adaptation path under bounded adversarial weights.|--fl_vmb_pretrain_rounds 92 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_server_lr 0.0045 --fl_vmb_server_momentum 0.62 --fl_vmb_weight_decay 0.00001 --no_fl_vmb_freeze_rx_stage2 --lambda_tx_adv_r 0.06 --lambda_rx_adv 0.06 --lambda_vmb_tx_proto 0.10 --lambda_vmb_rx_proto 0.18 --lambda_fed_proto 0.016 --fed_proto_momentum 0.20 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.992 --fl_vmb_prototype_clip_norm 0.34 --fl_baseline_view_ce_weight 0.80 --lambda_group_ce 0.025 --group_ce_mode dual_worst --group_ce_top_frac 0.45 --group_ce_min_domains 2 --use_aug --aug_p_rx_chain 0.20 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.032 --mixstyle_alpha 0.2 --mixstyle_strength 0.15 --mixstyle_late_start 108 --mixstyle_late_ramp_epochs 24 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 128 --use_logit_anchors --lambda_logit_kd 0.004 --kd_temperature 2.0 --kd_reliability_gate 0.95 --kd_margin_min 0.13 --kd_anchor_ema 0.95 --kd_min_count 2 --use_fed_style_bank --use_fl_style_bank_stats --fl_style_sampling_policy receiver_balanced --fl_style_replay_start_round 45 --fl_style_phys_start_round 45 --fl_style_dg_start_round 78 --fl_style_dg_min_domains 2 --fl_style_domain_label_mode target_receiver --fl_style_max_views 2 --fl_style_replay_prob 0.16 --fl_style_transform_mix_alpha 0.45 --fl_style_bank_max_centroids 96 --fl_style_bank_merge_radius 0.08 --fl_style_phys_p_lowpass 0.38 --fl_style_phys_p_multipath 0.38"
  "6|VMB7_A07_c04_pcgrad_groupce_bpc2_r010|aggressive|C04 BPC2 with PCGrad and explicit group CE is the strongest conflict-aware SAT/strict joint probe for this lane.|--fl_vmb_pretrain_rounds 105 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_batches_per_client 2 --fl_vmb_server_lr 0.006 --fl_vmb_server_momentum 0.74 --lambda_vmb_tx_proto 0.12 --lambda_vmb_rx_proto 0.18 --lambda_fed_proto 0.018 --fed_proto_momentum 0.20 --fl_conflict_agg pcgrad --fl_vmb_prototype_ema 0.992 --fl_vmb_prototype_clip_norm 0.32 --fl_baseline_view_ce_weight 0.86 --lambda_group_ce 0.035 --group_ce_mode dual_worst --group_ce_top_frac 0.48 --group_ce_min_domains 2 --groupdro_tau 0.42 --groupdro_cap 0.60 --use_aug --aug_p_rx_chain 0.18 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.032 --mixstyle_alpha 0.2 --mixstyle_strength 0.15 --mixstyle_late_start 108 --mixstyle_late_ramp_epochs 22 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 132 --use_logit_anchors --lambda_logit_kd 0.005 --kd_temperature 2.0 --kd_reliability_gate 0.95 --kd_margin_min 0.14 --kd_anchor_ema 0.96 --kd_min_count 2"
  "7|VMB7_A08_a08_allsat_styleproto_r010|aggressive|A08 all-satellite asymmetric proto with target-balanced StyleBank expands satellite coverage while preserving the 0.1/200/receiver contract.|--eval_sat_on all --fl_vmb_pretrain_rounds 100 --fl_vmb_stage1_lr_mult 1.05 --fl_vmb_server_lr 0.0055 --fl_vmb_server_momentum 0.70 --lambda_vmb_tx_proto 0.08 --lambda_vmb_rx_proto 0.24 --lambda_fed_proto 0.030 --fed_proto_momentum 0.20 --fl_conflict_agg cosine_clip --fl_vmb_prototype_ema 0.995 --fl_vmb_prototype_clip_norm 0.24 --fl_baseline_view_ce_weight 0.88 --use_aug --aug_p_rx_chain 0.22 --aug_rx_chain_envs 6 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7 --id_freq_stability_mode dsq --domain_freq_stability_mode dsq --use_mixstyle --mixstyle_mix same_tx_crossdomain --mixstyle_p 0.035 --mixstyle_alpha 0.2 --mixstyle_strength 0.16 --mixstyle_late_start 100 --mixstyle_late_ramp_epochs 22 --mixstyle_late_min_p 0.01 --mixstyle_late_min_strength 0.03 --mixstyle_stop_epoch 124 --use_logit_anchors --lambda_logit_kd 0.005 --kd_temperature 2.0 --kd_reliability_gate 0.96 --kd_margin_min 0.14 --kd_anchor_ema 0.96 --kd_min_count 2 --use_fed_style_bank --use_fl_style_bank_stats --use_fed_style_sat_view --fl_style_sampling_policy target_balanced --fl_style_replay_start_round 40 --fl_style_phys_start_round 40 --fl_style_dg_start_round 70 --fl_style_dg_min_domains 2 --fl_style_domain_label_mode target_receiver --fl_style_max_views 2 --fl_style_replay_prob 0.20 --fl_style_transform_mix_alpha 0.42 --fl_style_bank_max_centroids 128 --fl_style_bank_merge_radius 0.06 --fl_style_phys_p_lowpass 0.42 --fl_style_phys_p_multipath 0.42"
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

  echo "[VMB7-NEXT8] run=${run_name} gpu=${gpu} group=${group} dry_run=${DRY_RUN}"
  echo "[VMB7-NEXT8-DESC] ${description}"
  printf '[VMB7-NEXT8-CMD]'
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
