#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PY="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
RUN_ID="cvs_sa51_sa58_sa47_eval91_central_20260528_174616"
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
  --primary_udu_weight 0.65
  --epochs 170
  --test_eval_policy every_epoch
  --test_eval_start_epoch 91
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

# Conservative SA47-centered stability and CE sweep.
launch "SA51_sa47_eval91_anchor_ce1p3_r010" 0 1337 1.30 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA52_sa47_seed2027_ce1p3_eval91_r010" 1 2027 1.30 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA53_sa47_seed3407_ce1p3_eval91_r010" 2 3407 1.30 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA54_sa47_ce1p25_eval91_r010" 3 1337 1.25 clear_leo,low_elev_leo,rain_leo 1
sleep 2
launch "SA55_sa47_ce1p35_eval91_r010" 4 1337 1.35 clear_leo,low_elev_leo,rain_leo 1
sleep 2

# Rain-floor probe under the same balanced backbone.
launch "SA56_sa47_rainonly_ce1p3_eval91_r010" 5 1337 1.30 rain_leo 1
sleep 2

# Aggressive in-run two-stage approximations: clean/DG anchor first, satellite CE later.
launch "SA57_sa47_latesat91_leo3_ce1p3_eval91_r010" 6 1337 1.30 clear_leo,low_elev_leo,rain_leo 91
sleep 2
launch "SA58_sa47_latesat91_rain_ce1p3_eval91_r010" 7 1337 1.30 rain_leo 91

echo "STAMP=${STAMP}"
