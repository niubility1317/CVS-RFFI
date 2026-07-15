#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/szu2070436088/2510044040/CV-SincNet
BASE_RUN="$ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14"
EXPERIMENT_ID=qknn_ground_effective8_fft_source_ablation_20260715_v15
OUT_ROOT="$BASE_RUN/$EXPERIMENT_ID"
LOG_ROOT="$ROOT/logs/$EXPERIMENT_ID"
CONDA_SH=/home/szu2200432017/miniconda3/etc/profile.d/conda.sh

if [[ ! -f "$CONDA_SH" ]]; then
  echo "missing conda activation script: $CONDA_SH" >&2
  exit 2
fi
for required in \
  "$BASE_RUN/effective8_adapter_fp16.pt" \
  "$BASE_RUN/training_manifest.json" \
  "$BASE_RUN/phase1_caches/source_validation/cache_set.json"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 3
  fi
done
if [[ -e "$OUT_ROOT" || -e "$LOG_ROOT" ]]; then
  echo "refusing to overwrite existing ablation output" >&2
  exit 4
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
source "$CONDA_SH"

weights=(0.5 0.7 1.0 2.0)
labels=(w0p5 w0p7 w1p0 w2p0)
gpus=(0 1 2 3)
manifest="$OUT_ROOT/launch_manifest.tsv"
printf 'weight\tgpu\tpid\tlog\tout_dir\n' > "$manifest"

for index in "${!weights[@]}"; do
  weight="${weights[$index]}"
  label="${labels[$index]}"
  gpu="${gpus[$index]}"
  out_dir="$OUT_ROOT/$label"
  log="$LOG_ROOT/$label.log"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" conda run --no-capture-output -n CVS-RFFI \
    python "$ROOT/paper_reproduction/scripts/validate_cvs_ground_lora_multiview.py" \
      --ckpt "$ROOT/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth" \
      --adapter_state "$BASE_RUN/effective8_adapter_fp16.pt" \
      --training_manifest "$BASE_RUN/training_manifest.json" \
      --source_cache_set "$BASE_RUN/phase1_caches/source_validation/cache_set.json" \
      --out_dir "$out_dir" \
      --source_train_rxs '1-1,1-19,14-7,18-2,19-2,2-1' \
      --source_val_rxs '2-19' \
      --num_old_classes 6 \
      --batch_size 256 \
      --max_mean_backbone_forwards 3.0 \
      --min_extra_view_rate 0.05 \
      --fft_weight "$weight" \
      --device cuda:0 > "$log" 2>&1 &
  pid=$!
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$weight" "$gpu" "$pid" "$log" "$out_dir" >> "$manifest"
done

cat "$manifest"
