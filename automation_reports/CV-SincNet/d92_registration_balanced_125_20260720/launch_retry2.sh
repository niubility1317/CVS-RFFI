#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
snapshot="$project/runs/d92_source_snapshot_retry2_20260720"
output="$project/runs/d92_registration_balanced_125_retry2_20260720"
logs="$project/logs/d92_registration_balanced_125_retry2_20260720"
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
cache="$project/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/cache_matrix"
authority="$project/runs/d81_comprehensive_125_v2auth_20260720/authority_controller/authority_final_retry1"
checkpoint="$project/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
runtime="$project/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/sealed_feature_runtime.pt"
method_lock="$project/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303/input/method_lock.json"
ground="$project/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
ground_sha=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c

mkdir -p "$logs"
for shard in 0 1 2 3 4 5 6 7; do
  test ! -e "$logs/shard_${shard}.out"
  test ! -e "$logs/shard_${shard}.err"
  test ! -e "$output/events/shard_00${shard}_of_008.jsonl"
  test ! -e "$output/summaries/shard_00${shard}_of_008.json"
  nohup env \
    CUDA_VISIBLE_DEVICES="$shard" \
    PYTHONPATH="$snapshot:$project" \
    "$python" -u "$snapshot/scripts/run_d92_125_stability.py" \
      --cache-root "$cache" \
      --authority-root "$authority" \
      --phase1-checkpoint "$checkpoint" \
      --sealed-runtime "$runtime" \
      --method-lock "$method_lock" \
      --output-root "$output" \
      --ground-component-dir "$ground" \
      --ground-manifest-sha256 "$ground_sha" \
      --cpu-threads 2 \
      --shard-index "$shard" \
      --shard-count 8 \
      --device cuda:0 \
      >"$logs/shard_${shard}.out" \
      2>"$logs/shard_${shard}.err" \
      </dev/null &
  printf 'shard=%s gpu=%s pid=%s\n' "$shard" "$shard" "$!"
done
