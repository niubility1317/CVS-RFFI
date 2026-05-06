#!/usr/bin/env bash
set -uo pipefail

# Multi-batch 8-GPU launcher for satellite RFFI performance/generalization optimization.
#
# Default behavior:
#   - Batch 1: satellite-channel scenario ablation.
#   - Batch 2: satellite consistency loss weight / start-epoch ablation.
#   - Batch 3: model preset and regularization interaction ablation.
#   - Batch 4: checkpoint averaging, prototype memory, and SupCon ablation.
#   - Batch 5: SmoothGroupDRO, Fishr, RX-chain randomization, and combined candidates.
#   - A later batch starts only after the current batch has finished.
#
# Usage:
#   bash run_sat_channel_ablation_8gpu.sh
#   GPU_IDS=0,1,2,3,4,5,6,7 bash run_sat_channel_ablation_8gpu.sh
#   START_BATCH=2 END_BATCH=5 bash run_sat_channel_ablation_8gpu.sh
#   STOP_ON_FAIL=1 bash run_sat_channel_ablation_8gpu.sh
#   SAT_EVAL_ON=main SAT_EVAL_MAX_BATCHES=20 bash run_sat_channel_ablation_8gpu.sh

mkdir -p logs sat_ablation_runs
SCHED_LOG="logs/sat_channel_ablation_8gpu_$(date +%Y%m%d_%H%M%S).log"

log_msg() {
  echo "$@" | tee -a "$SCHED_LOG"
}

GPU_IDS_CSV="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPU_IDS_CSV}"

if [ "${#GPU_LIST[@]}" -lt 1 ]; then
  echo "ERROR: GPU_IDS is empty. Example: GPU_IDS=\"0,1,2,3,4,5,6,7\" bash $0"
  exit 1
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_ARGS="${BASE_ARGS:---batch_size 256 --dataset wisig --wisig_domain rx_day}"
SAT_EVAL_ON="${SAT_EVAL_ON:-test_unseen_day_unseen_rx}"
SAT_EVAL_SCENARIOS="${SAT_EVAL_SCENARIOS:-clear_leo,low_elev_leo,rain_leo,storm_mp,mixed_orbit}"
SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:--1}"
SAT_BASE_ARGS="--eval_sat_channel --eval_sat_on ${SAT_EVAL_ON} --eval_sat_scenarios ${SAT_EVAL_SCENARIOS} --sat_eval_max_batches ${SAT_EVAL_MAX_BATCHES}"
START_BATCH="${START_BATCH:-1}"
END_BATCH="${END_BATCH:-5}"
STOP_ON_FAIL="${STOP_ON_FAIL:-0}"

launch_exp() {
  local gpu_id="$1"
  local tag="$2"
  local desc="$3"
  local extra_args="$4"
  local stamp
  local log
  local run_dir
  local cmd

  stamp="$(date +%Y%m%d_%H%M%S)"
  run_dir="sat_ablation_runs/${tag}"
  mkdir -p "$run_dir"
  log="logs/${tag}_${stamp}.log"

  cmd="CUDA_VISIBLE_DEVICES=${gpu_id} PYTHONUNBUFFERED=1 ${PYTHON_BIN} -u train.py ${BASE_ARGS} ${SAT_BASE_ARGS} --latest_save_path ${run_dir}/latest_model.pth --best_save_path ${run_dir}/best_model.pth ${extra_args}"

  {
    echo "================================"
    echo "[SAT-ABLATION] TAG => ${tag}"
    echo "[SAT-ABLATION] DESC => ${desc}"
    echo "[SAT-ABLATION] GPU => ${gpu_id}"
    echo "[SAT-ABLATION] BASE_ARGS => ${BASE_ARGS}"
    echo "[SAT-ABLATION] SAT_BASE_ARGS => ${SAT_BASE_ARGS}"
    echo "[SAT-ABLATION] EXTRA_ARGS => ${extra_args}"
    echo "[SAT-ABLATION] RUN_DIR => ${run_dir}"
    echo "[SAT-ABLATION] CMD => ${cmd}"
    echo "================================"
  } > "$log"

  CUDA_VISIBLE_DEVICES="${gpu_id}" PYTHONUNBUFFERED=1 \
  nohup ${PYTHON_BIN} -u train.py \
    ${BASE_ARGS} \
    ${SAT_BASE_ARGS} \
    --latest_save_path "${run_dir}/latest_model.pth" \
    --best_save_path "${run_dir}/best_model.pth" \
    ${extra_args} \
    >> "$log" 2>&1 &

  local pid=$!
  echo "$pid" > "logs/${tag}.pid"
  log_msg "[LAUNCHED] gpu=${gpu_id} tag=${tag} pid=${pid} log=${log}"
}

wait_wave() {
  local batch_id="$1"
  local wave_id="$2"
  local failed=0

  if [ "${#batch_pids[@]}" -eq 0 ]; then
    return 0
  fi

  log_msg "================================"
  log_msg "Waiting for batch ${batch_id}, wave ${wave_id}..."
  log_msg "PIDS => ${batch_pids[*]}"
  log_msg "TAGS => ${batch_tags[*]}"
  log_msg "================================"

  for i in "${!batch_pids[@]}"; do
    if wait "${batch_pids[$i]}"; then
      log_msg "[DONE] batch=${batch_id} wave=${wave_id} tag=${batch_tags[$i]} pid=${batch_pids[$i]}"
    else
      log_msg "[FAIL] batch=${batch_id} wave=${wave_id} tag=${batch_tags[$i]} pid=${batch_pids[$i]}"
      failed=1
    fi
  done

  batch_pids=()
  batch_tags=()
  return "$failed"
}

run_batch() {
  local batch_id="$1"
  local experiments="$2"
  local slot=0
  local wave_id=1
  local batch_failed=0
  local gpu_id
  local pid

  batch_pids=()
  batch_tags=()

  log_msg "================================"
  log_msg "Starting batch ${batch_id}"
  log_msg "================================"

  while IFS='|' read -r tag desc extra_args; do
    if [ -z "${tag}" ]; then
      continue
    fi
    gpu_id="${GPU_LIST[$slot]}"
    launch_exp "$gpu_id" "$tag" "$desc" "$extra_args"
    pid="$(cat "logs/${tag}.pid")"
    batch_pids+=("$pid")
    batch_tags+=("$tag")

    slot=$((slot + 1))
    if [ "$slot" -ge "${#GPU_LIST[@]}" ]; then
      wait_wave "$batch_id" "$wave_id" || batch_failed=1
      if [ "$batch_failed" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
        return 1
      fi
      slot=0
      wave_id=$((wave_id + 1))
    fi
  done <<< "${experiments}"

  wait_wave "$batch_id" "$wave_id" || batch_failed=1
  if [ "$batch_failed" -ne 0 ]; then
    log_msg "[WARN] Batch ${batch_id} finished with one or more failed jobs."
    return 1
  fi
  log_msg "Batch ${batch_id} finished successfully."
  return 0
}

should_run_batch() {
  local batch_id="$1"
  [ "$batch_id" -ge "$START_BATCH" ] && [ "$batch_id" -le "$END_BATCH" ]
}

read -r -d '' SAT_BATCH_1 <<'EOF' || true
SAT00_r19_clean_eval|Batch1 clean control: R19 best route, satellite evaluation only.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SAT01_r19_sat_cons_clear|Batch1 scenario: R19 + clear LEO consistency.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario clear_leo --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.05
SAT02_r19_sat_cons_low_elev|Batch1 scenario: R19 + low-elevation LEO consistency.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario low_elev_leo --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.05
SAT03_r19_sat_cons_rain|Batch1 scenario: R19 + rain LEO consistency.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario rain_leo --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.05
SAT04_r19_sat_cons_storm|Batch1 scenario: R19 + storm multipath consistency.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario storm_mp --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
SAT05_r19_sat_cons_mixed|Batch1 scenario: R19 + mixed-orbit consistency, broad generalization candidate.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
SAT06_r25_compact_clean_eval|Batch1 compact control: R25 clean route, satellite evaluation only.|--slim_group rxrobust_lite_d_no_dac_refined --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SAT07_r25_compact_sat_mixed|Batch1 compact candidate: R25 + mixed-orbit consistency.|--slim_group rxrobust_lite_d_no_dac_refined --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
EOF

read -r -d '' SAT_BATCH_2 <<'EOF' || true
SAT08_r19_mixed_cons_only_low|Batch2 loss weight: mixed-orbit consistency-only, low weight.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.00 --lambda_sat_cons 0.03
SAT09_r19_mixed_cons_only_high|Batch2 loss weight: mixed-orbit consistency-only, high weight.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.00 --lambda_sat_cons 0.08
SAT10_r19_mixed_cls_only|Batch2 loss weight: satellite classification only; isolates feature consistency contribution.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.10 --lambda_sat_cons 0.00
SAT11_r19_mixed_low_weight|Batch2 loss weight: lower balanced satellite loss.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.05 --lambda_sat_cons 0.02
SAT12_r19_mixed_mid_high_weight|Batch2 loss weight: moderately stronger balanced satellite loss.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.12 --lambda_sat_cons 0.06
SAT13_r19_mixed_high_weight|Batch2 loss weight: high satellite loss, checks over-regularization risk.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.16 --lambda_sat_cons 0.08
SAT14_r19_mixed_start_e001|Batch2 schedule: satellite consistency from epoch 1.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 1 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
SAT15_r19_mixed_start_e060|Batch2 schedule: delayed satellite consistency from epoch 60.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 60 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
EOF

read -r -d '' SAT_BATCH_3 <<'EOF' || true
SAT16_r21_worstrx_clean_eval|Batch3 preset control: R21 worst-RX-oriented clean route, satellite evaluation only.|--slim_group rxrobust_lite_b_no_dac_gce006 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SAT17_r21_worstrx_sat_mixed|Batch3 preset interaction: R21 + mixed-orbit consistency.|--slim_group rxrobust_lite_b_no_dac_gce006 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
SAT18_r27_full_clean_eval|Batch3 capacity control: full Lite-C upper-bound route, satellite evaluation only.|--slim_group rxrobust_balanced --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SAT19_r27_full_sat_mixed|Batch3 capacity interaction: full Lite-C + mixed-orbit consistency.|--slim_group rxrobust_balanced --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
SAT20_r19_no_mix_clean_eval|Batch3 regularizer control: R19 route with MixStyle disabled, satellite evaluation only.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --no_use_mixstyle
SAT21_r19_no_mix_sat_mixed|Batch3 regularizer interaction: R19 without MixStyle + mixed-orbit consistency.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --no_use_mixstyle --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
SAT22_r25_compact_sat_low_elev|Batch3 compact hard-channel: R25 + low-elevation LEO consistency.|--slim_group rxrobust_lite_d_no_dac_refined --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario low_elev_leo --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
SAT23_r25_compact_sat_rain|Batch3 compact hard-channel: R25 + rain LEO consistency.|--slim_group rxrobust_lite_d_no_dac_refined --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario rain_leo --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04
EOF

read -r -d '' SAT_BATCH_4 <<'EOF' || true
SAT24_r19_ema|Batch4 checkpoint averaging: R19 + EMA final checkpoint.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_ema_ckpt --ema_decay 0.999 --ema_start_epoch 40
SAT25_r19_swa|Batch4 checkpoint averaging: R19 + SWA from late training trajectory.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_swa_ckpt --swa_start_epoch 120 --swa_interval 5
SAT26_r19_swad|Batch4 checkpoint averaging: R19 + SWAD dense averaging near best primary OOD.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_swad_ckpt --swad_start_epoch 80 --swad_interval 1 --swad_tolerance 2.0
SAT27_r19_all_avg|Batch4 checkpoint averaging: R19 + EMA/SWA/SWAD together for post-training comparison.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_ema_ckpt --ema_decay 0.999 --ema_start_epoch 40 --use_swa_ckpt --swa_start_epoch 120 --swa_interval 5 --use_swad_ckpt --swad_start_epoch 80 --swad_interval 1 --swad_tolerance 2.0
SAT28_r25_ema|Batch4 checkpoint averaging: R25 compact + EMA final checkpoint.|--slim_group rxrobust_lite_d_no_dac_refined --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_ema_ckpt --ema_decay 0.999 --ema_start_epoch 40
SAT29_r25_swad|Batch4 checkpoint averaging: R25 compact + SWAD dense averaging.|--slim_group rxrobust_lite_d_no_dac_refined --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_swad_ckpt --swad_start_epoch 80 --swad_interval 1 --swad_tolerance 2.0
SAT30_r19_supcon|Batch4 representation: domain-aware SupCon on z_id.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --lambda_supcon_id 0.04 --supcon_temp 0.12 --generalization_feature z_id
SAT31_r19_proto_memory|Batch4 representation: class-conditional prototype memory bank on z_id.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_proto_memory --lambda_proto 0.05 --proto_momentum 0.95 --proto_domain_align_weight 0.5 --proto_push_weight 0.1 --generalization_feature z_id
EOF

read -r -d '' SAT_BATCH_5 <<'EOF' || true
SAT32_r19_proto_memory_joint|Batch5 prototype feature: prototype memory on id_feat_joint instead of z_id.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_proto_memory --lambda_proto 0.05 --proto_momentum 0.95 --generalization_feature id_feat_joint
SAT33_r19_proto_supcon|Batch5 representation combo: prototype memory + domain-aware SupCon.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_proto_memory --lambda_proto 0.04 --lambda_supcon_id 0.03 --supcon_temp 0.12 --generalization_feature z_id
SAT34_r19_groupdro_smooth|Batch5 weak-domain weighting: Smooth GroupDRO replacing hard top-domain CE.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --group_ce_mode smooth_dro --groupdro_tau 0.45 --groupdro_momentum 0.95
SAT35_r19_groupdro_capped|Batch5 weak-domain weighting: capped Smooth GroupDRO for lower training oscillation.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --group_ce_mode smooth_dro_capped --groupdro_tau 0.45 --groupdro_cap 0.55 --groupdro_momentum 0.95
SAT36_r19_dual_worst|Batch5 weak-domain weighting: dual rx/day worst-domain weighting for rx_day labels.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --group_ce_mode dual_worst --groupdro_num_days 4 --groupdro_tau 0.45 --groupdro_cap 0.55
SAT37_r19_fishr|Batch5 gradient statistics: Fishr-style logit-gradient variance matching.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --lambda_fishr 0.02 --fishr_min_domains 4
SAT38_r19_rx_chain|Batch5 augmentation: receiver-chain domain randomization in normal view.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --aug_p_rx_chain 0.35 --aug_rx_chain_envs 4 --aug_rx_chain_p_lowpass 0.7 --aug_rx_chain_p_multipath 0.7
SAT39_r19_combo_best_guess|Batch5 combined candidate: mixed satellite consistency + prototype + SupCon + capped GroupDRO + RX-chain + SWAD.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --use_sat_consistency --sat_train_scenario mixed_orbit --sat_cons_start_epoch 20 --lambda_sat_cls 0.08 --lambda_sat_cons 0.04 --use_proto_memory --lambda_proto 0.04 --lambda_supcon_id 0.03 --lambda_fishr 0.01 --group_ce_mode smooth_dro_capped --groupdro_tau 0.45 --groupdro_cap 0.55 --aug_p_rx_chain 0.30 --use_swad_ckpt --swad_start_epoch 80 --swad_interval 1 --swad_tolerance 2.0
SAT40_r19_groupdro_smooth|Batch5 alias/control: Smooth GroupDRO run kept for parser/tests and direct comparison naming.|--slim_group rxrobust_lite_b_no_dac_mix015 --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65 --group_ce_mode smooth_dro --groupdro_tau 0.45 --groupdro_momentum 0.95
EOF

read -r -d '' SAT_BATCH_6 <<'EOF' || true
SGC00_full_source|Batch6 SGC source: full SGC-Adapter on Lite-B no-DAC.|--preset sgc_lite_b_no_dac --stage source --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SGC01_no_amp_source|Batch6 SGC ablation: no amplitude normalization.|--preset sgc_lite_b_no_dac_no_amp --stage source --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SGC02_no_freq_source|Batch6 SGC ablation: no frequency compensation.|--preset sgc_lite_b_no_dac_no_freq --stage source --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SGC03_no_spec_source|Batch6 SGC ablation: no spectral suppression.|--preset sgc_lite_b_no_dac_no_spec --stage source --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SGC04_no_res_source|Batch6 SGC ablation: no residual compensation.|--preset sgc_lite_b_no_dac_no_res --stage source --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SGC05_no_adapter_source|Batch6 SGC baseline: no adapter.|--preset sgc_baseline_no_adapter --stage source --epochs 200 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SGC06_full_augment|Batch6 SGC augment: full SGC with mixed-orbit channel consistency.|--preset sgc_lite_b_no_dac --stage sgc_augment --train_sat_channel --train_sat_scenario mixed_orbit --lambda_feat 1.0 --lambda_res 0.01 --epochs 100 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
SGC07_full_adapt|Batch6 SGC adapt: adapter-only update from augmented checkpoint path.|--preset sgc_lite_b_no_dac --stage sgc_adapt --source_ckpt sgc_runs/sgc_lite_b_no_dac/augment/best_model.pth --adapt_lr 1e-4 --adapt_epochs 50 --lambda_res 0.01 --wisig_train_ratio 0.2 --primary_udu_weight 0.65
EOF

log_msg "================================"
log_msg "8-GPU satellite channel multi-batch ablation launcher"
log_msg "SCHED_LOG => ${SCHED_LOG}"
log_msg "GPU_IDS => ${GPU_IDS_CSV}"
log_msg "BASE_ARGS => ${BASE_ARGS}"
log_msg "SAT_EVAL_ON => ${SAT_EVAL_ON}"
log_msg "SAT_EVAL_SCENARIOS => ${SAT_EVAL_SCENARIOS}"
log_msg "SAT_EVAL_MAX_BATCHES => ${SAT_EVAL_MAX_BATCHES}"
log_msg "START_BATCH => ${START_BATCH}"
log_msg "END_BATCH => ${END_BATCH}"
log_msg "STOP_ON_FAIL => ${STOP_ON_FAIL}"
log_msg "================================"

overall_status=0

if should_run_batch 1; then
  run_batch "1" "${SAT_BATCH_1}" || overall_status=1
  if [ "$overall_status" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
fi

if should_run_batch 2; then
  run_batch "2" "${SAT_BATCH_2}" || overall_status=1
  if [ "$overall_status" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
fi

if should_run_batch 3; then
  run_batch "3" "${SAT_BATCH_3}" || overall_status=1
  if [ "$overall_status" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
fi

if should_run_batch 4; then
  run_batch "4" "${SAT_BATCH_4}" || overall_status=1
  if [ "$overall_status" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
fi

if should_run_batch 5; then
  run_batch "5" "${SAT_BATCH_5}" || overall_status=1
  if [ "$overall_status" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
fi

if should_run_batch 6; then
  run_batch "6" "${SAT_BATCH_6}" || overall_status=1
  if [ "$overall_status" -ne 0 ] && [ "$STOP_ON_FAIL" = "1" ]; then
    exit 1
  fi
fi

if [ "$overall_status" -eq 0 ]; then
  log_msg "All selected satellite ablation batches finished successfully."
else
  log_msg "Selected satellite ablation batches finished with one or more failed jobs."
fi
exit "$overall_status"
