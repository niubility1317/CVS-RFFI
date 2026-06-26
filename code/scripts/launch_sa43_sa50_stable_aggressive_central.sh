#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PY="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
RUN_ID="cvs_sa43_stable_aggressive_central_20260528_120112"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/runs/$RUN_ID"
LOG_ROOT="$ROOT/logs/$RUN_ID"

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

COMMON=(
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --primary_udu_weight 0.65
  --epochs 170
  --test_eval_policy every_epoch
  --test_eval_start_epoch 151
  --eval_sat_channel
  --eval_sat_on test_unseen_day_unseen_rx
  --eval_sat_scenarios clear_leo,low_elev_leo,rain_leo
  --sat_eval_max_batches -1
  --slim_group none
  --branch_ablation no_dac
  --domain_branch_ablation no_stats
  --domain_enhancer rcn_stats
  --exp_group s3_rxrobust_no_dac
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
  --lambda_fishr 0.02
  --fishr_min_domains 4
  --sat_train_scenarios clear_leo,low_elev_leo,rain_leo
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --concat_sat_start_epoch 1
  --sat_view_prob 1.00
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
)

launch() {
  local name="$1"
  local gpu="$2"
  local seed="$3"
  local model="$4"
  local ce="$5"
  local enh="$6"
  shift 6
  local run_dir="$RUN_ROOT/$name"
  local log="$LOG_ROOT/${name}_${STAMP}.log"
  mkdir -p "$run_dir"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u train.py \
    "${COMMON[@]}" \
    --seed "$seed" \
    --model_variant "$model" \
    --run_name "$name" \
    --domain_enhancer_strength "$enh" \
    --concat_sat_ce_weight "$ce" \
    --latest_save_path "$run_dir/latest_model.pth" \
    --best_save_path "$run_dir/best_val_model.pth" \
    --best_primary_save_path "$run_dir/best_primary_ood_model.pth" \
    --best_unseen_day_unseen_rx_save_path "$run_dir/best_strict_udu_model.pth" \
    "$@" \
    > "$log" 2>&1 &
  local pid="$!"
  echo -e "${name}\tGPU=${gpu}\tPID=${pid}\tLOG=${log}"
}

# Stable parameter-only exploration around the SA34 CE1.2 LEO route.
launch "SA43_sa34_evalgate_anchor_ce1p2_r010" 0 1337 lite_d 1.2 0.35 --domain_freq_stability_mode dsq --freq_stability_channels 2
sleep 2
launch "SA44_sa34_seed2027_ce1p2_r010" 1 2027 lite_d 1.2 0.35 --domain_freq_stability_mode dsq --freq_stability_channels 2
sleep 2
launch "SA45_sa34_seed3407_ce1p2_r010" 2 3407 lite_d 1.2 0.35 --domain_freq_stability_mode dsq --freq_stability_channels 2
sleep 2
launch "SA46_sa34_ce1p1_r010" 3 1337 lite_d 1.1 0.35 --domain_freq_stability_mode dsq --freq_stability_channels 2
sleep 2
launch "SA47_sa34_ce1p3_r010" 4 1337 lite_d 1.3 0.35 --domain_freq_stability_mode dsq --freq_stability_channels 2
sleep 2

# Aggressive optional backbone/model probes; none replaces the mature Lite-D no-DAC route.
launch "SA48_liteb_no_dac_sa34_ce1p2_r010" 5 1337 lite_b 1.2 0.35 --domain_freq_stability_mode dsq --freq_stability_channels 2
sleep 2
launch "SA49_id_phase_dsq_leo3_ce0p7_r010" 6 1337 lite_d 0.7 0.35 --id_time_stability_mode phase_delta --id_freq_stability_mode dsq
sleep 2
launch "SA50_all_phase_dsq_leo3_ce0p7_r010" 7 1337 lite_d 0.7 0.35 --id_time_stability_mode phase_delta --id_freq_stability_mode dsq --domain_time_stability_mode same --domain_freq_stability_mode same

echo "STAMP=${STAMP}"
