#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PY="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
RUN_ID="cvs_sa59_sa66_satfloor_central_20260528_225029"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/runs/$RUN_ID"
LOG_ROOT="$ROOT/logs/$RUN_ID"
DRY_RUN=0

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

mkdir -p "$RUN_ROOT" "$LOG_ROOT"
cd "$ROOT"

COMMON=(
  --batch_size 256
  --eval_batch_size 256
  --dataset wisig
  --wisig_domain rx_day
  --wisig_train_ratio 0.1
  --primary_udu_weight 0.70
  --epochs 170
  --test_eval_policy every_epoch
  --test_eval_start_epoch 81
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
  --use_concat_sat_channel_aug
  --concat_sat_ce_only
  --sat_view_prob 1.00
  --lambda_sat_cls 0.00
  --lambda_sat_cons 0.00
  --domain_freq_stability_mode dsq
  --freq_stability_channels 2
)

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

launch() {
  local name="$1"
  local gpu="$2"
  local seed="$3"
  local ce="$4"
  local scenarios="$5"
  local sat_start="$6"
  shift 6

  local run_dir="$RUN_ROOT/$name"
  local log="$LOG_ROOT/${name}_${STAMP}.log"
  local cmd=(
    "$PY" -u train.py
    "${COMMON[@]}"
    --seed "$seed"
    --run_name "$name"
    --concat_sat_ce_weight "$ce"
    --sat_train_scenarios "$scenarios"
    --concat_sat_start_epoch "$sat_start"
    --latest_save_path "$run_dir/latest_model.pth"
    --best_save_path "$run_dir/best_val_model.pth"
    --best_primary_save_path "$run_dir/best_primary_ood_model.pth"
    --best_unseen_day_unseen_rx_save_path "$run_dir/best_strict_udu_model.pth"
    "$@"
  )

  mkdir -p "$run_dir"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo -e "${name}\tGPU=${gpu}\tDRY_RUN=1\tLOG=${log}"
    printf 'CUDA_VISIBLE_DEVICES=%q nohup ' "$gpu"
    print_cmd "${cmd[@]}"
    return 0
  fi

  CUDA_VISIBLE_DEVICES="$gpu" nohup "${cmd[@]}" > "$log" 2>&1 &
  local pid="$!"
  echo -e "${name}\tGPU=${gpu}\tPID=${pid}\tLOG=${log}"
}

# Optimization exploration around the SA55 CE1.35 route.
launch "SA59_sa55_anchor_p70_ce1p35_eval81_r010" 0 1337 1.35 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA60_sa55_ce1p32_eval81_r010" 1 1337 1.32 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA61_sa55_ce1p38_eval81_r010" 2 1337 1.38 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA62_sa55_enh045_ce1p35_eval81_r010" 3 1337 1.35 clear_leo,low_elev_leo,rain_leo 1 --domain_enhancer_strength 0.45
sleep 2

# Aggressive floor probes: concentrate satellite CE on the rain/low-elevation bottleneck.
launch "SA63_floor_leo3_ce1p45_eval81_r010" 4 1337 1.45 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA64_floor_lowrain_ce1p45_eval81_r010" 5 1337 1.45 low_elev_leo,rain_leo 1
sleep 2
launch "SA65_latesat81_lowrain_ce1p55_r010" 6 1337 1.55 low_elev_leo,rain_leo 81 --mixstyle_p 0.12 --mixstyle_strength 0.55
sleep 2
launch "SA66_latesat121_rain_ce1p70_r010" 7 1337 1.70 rain_leo 121 --mixstyle_p 0.12 --mixstyle_strength 0.55

echo "STAMP=${STAMP}"
