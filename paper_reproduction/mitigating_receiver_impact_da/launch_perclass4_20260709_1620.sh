#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/szu2070436088/2510044040/CV-SincNet"
PY="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
DATA="$ROOT/Dataset_WigSig/ManySig.pkl"
CONFIG="$ROOT/paper_reproduction/configs/mitigating_receiver_impact_da_manysig_paper_faithful.json"
STAMP="20260709_1620"
cd "$ROOT"
mkdir -p "$ROOT/paper_reproduction/logs"

launch_one() {
  local gpu="$1"
  local run_id="$2"
  local batch_size="$3"
  shift 3
  local out_dir="$ROOT/paper_reproduction/runs/$run_id"
  local log_path="$ROOT/paper_reproduction/logs/$run_id.out"
  mkdir -p "$out_dir"
  echo "LAUNCH run_id=$run_id gpu=$gpu batch=$batch_size log=$log_path out=$out_dir"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u -m paper_reproduction.mitigating_receiver_impact_da.train \
    --config "$CONFIG" \
    --run-table2 \
    --manysig-pkl "$DATA" \
    --checkpoint-dir "$out_dir" \
    --output "$out_dir/results.json" \
    --tasks "14-7->3-19" \
    --methods proposed \
    --epochs 20 \
    --batch-size "$batch_size" \
    --learning-rate 0.0006 \
    --source-pretrain-epochs 0 \
    --adapt-start-epoch 0 \
    --base-tau 0.7 \
    --estimate-steps 7 \
    --class-prior-mode source \
    --kl-estimator-mode mine_ma \
    --mine-update-scale 0.5 \
    --pseudo-threshold-mode paper \
    --pseudo-score-mode probability \
    --class-weight-timing current \
    --target-model-selection final \
    --seed 20260710 \
    --pseudo-quota-mode balanced_topk \
    --pseudo-quota-per-class 16 \
    "$@" \
    > "$log_path" 2>&1 &
  local pid=$!
  echo "PID run_id=$run_id pid=$pid"
}

launch_one 6 "mitigating_da_perclass4_q16_b64_final_e20_14-7_to_3-19_s20260710_${STAMP}" 64 \
  --pseudo-threshold-floor 0.5

launch_one 7 "mitigating_da_perclass4_q16_b256_final_e20_14-7_to_3-19_s20260710_${STAMP}" 256 \
  --pseudo-threshold-floor 0.5

launch_one 6 "mitigating_da_perclass4_q16_floor07_final_e20_14-7_to_3-19_b128_s20260710_${STAMP}" 128 \
  --pseudo-threshold-floor 0.7

echo "DONE launch_perclass4"
