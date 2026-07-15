#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/szu2070436088/2510044040/CV-SincNet
BASE_RUN="$ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14"
EXPERIMENT_ID=qknn_ground_adapt_layer_loss_ablation_20260715_v16
OUT_ROOT="$ROOT/runs/$EXPERIMENT_ID"
LOG_ROOT="$ROOT/logs/$EXPERIMENT_ID"
CONDA_SH=/home/szu2200432017/miniconda3/etc/profile.d/conda.sh
CHECKPOINT="$ROOT/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
SOURCE_TRAIN="$BASE_RUN/phase1_caches/source_train/cache_set.json"
SOURCE_VALIDATION="$BASE_RUN/phase1_caches/source_validation/cache_set.json"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "missing conda activation script: $CONDA_SH" >&2
  exit 2
fi
for required in "$CHECKPOINT" "$SOURCE_TRAIN" "$SOURCE_VALIDATION"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 3
  fi
done
if [[ -e "$OUT_ROOT" || -e "$LOG_ROOT" ]]; then
  echo "refusing to overwrite existing adapt ablation output" >&2
  exit 4
fi

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
source "$CONDA_SH"

labels=(
  p4_r16_e8_std
  h4_r16_e8_std
  e8_r8_e8_std
  e8_r16_e8_std
  p4_r16_e8_k1
  h4_r16_e8_k1
  e8_r8_e8_k1
  e8_r16_e8_k1
)
scopes=(
  projection_feature feat_joint effective_feature effective_feature
  projection_feature feat_joint effective_feature effective_feature
)
ranks=(16 16 8 16 16 16 8 16)
profiles=(std std std std k1 k1 k1 k1)
gpus=(0 1 2 3 4 5 6 7)
manifest="$OUT_ROOT/launch_manifest.tsv"
printf 'label\tscope\trank\tepochs\tprofile\tgpu\tpid\tlog\tout_dir\n' > "$manifest"

for index in "${!labels[@]}"; do
  label="${labels[$index]}"
  scope="${scopes[$index]}"
  rank="${ranks[$index]}"
  profile="${profiles[$index]}"
  gpu="${gpus[$index]}"
  run_dir="$OUT_ROOT/$label"
  log="$LOG_ROOT/$label.log"
  state="$run_dir/adapter_fp16.pt"
  training_manifest="$run_dir/training_manifest.json"
  validation_dir="$run_dir/source_validation"
  if [[ "$profile" == k1 ]]; then
    relation=1.0
    prototype_gram=0.5
    worst_k=1.0
    view_consistency=0.5
    reference_margin=10.0
    teacher_kl=0.20
  else
    relation=0.5
    prototype_gram=0.25
    worst_k=0.5
    view_consistency=0.25
    reference_margin=7.5
    teacher_kl=0.16
  fi
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    conda activate CVS-RFFI
    python "$ROOT/code/scripts/train_apply_phase1_iq_preadapter_20260703.py" \
      --ckpt "$CHECKPOINT" \
      --runs_root "$run_dir" \
      --source_tx_ids '14-10,14-7,20-15,20-19,6-15,8-20' \
      --source_rxs '1-1,1-19,14-7,18-2,19-2,2-1' \
      --source_leo_weak_cache_set_manifest "$SOURCE_TRAIN" \
      --wisig_out_len 256 \
      --num_old_classes 6 \
      --feature_name z_id \
      --sat_scenarios 'leo_clear_weak,leo_low_elev_weak,leo_rain_weak' \
      --star_ground_channel_impl simplified_leo_residual \
      --batch_size 256 \
      --epochs 8 \
      --no-input_adapter_enabled \
      --model_adapter_mode lora_effective_feature \
      --lora_effective_scope "$scope" \
      --lora_rank "$rank" \
      --lora_alpha "$rank" \
      --adapter_state_out "$state" \
      --adapter_manifest_out "$training_manifest" \
      --source_only_ground_lora \
      --input_repair raw \
      --lr 0.0005 \
      --weight_decay 0.0001 \
      --mse_weight 1 \
      --cos_weight 2 \
      --proto_ce_weight 0.2 \
      --logit_ce_weight 0 \
      --leo_reference_identity_weight 22 \
      --leo_reference_cos_weight 1 \
      --feature_margin_weight 4.5 \
      --leo_reference_margin_weight "$reference_margin" \
      --feature_margin_tolerance 0.01 \
      --teacher_logit_distill_weight "$teacher_kl" \
      --multiview_consistency_weight "$view_consistency" \
      --relation_preservation_weight "$relation" \
      --prototype_gram_weight "$prototype_gram" \
      --prototype_gram_max_cosine 0.65 \
      --worst_k_risk_weight "$worst_k" \
      --worst_k_values '1,2,5,10,20' \
      --worst_k_tau 0.2 \
      --worst_k_proto_temperature 0.07 \
      --distill_temperature 2 \
      --residual_weight 0 \
      --proto_temperature 0.07 \
      --grad_clip 1 \
      --log_every 1 \
      --device cuda:0 \
      --seed 4070391
    python "$ROOT/paper_reproduction/scripts/validate_cvs_ground_lora_multiview.py" \
      --ckpt "$CHECKPOINT" \
      --adapter_state "$state" \
      --training_manifest "$training_manifest" \
      --source_cache_set "$SOURCE_VALIDATION" \
      --out_dir "$validation_dir" \
      --source_train_rxs '1-1,1-19,14-7,18-2,19-2,2-1' \
      --source_val_rxs '2-19' \
      --num_old_classes 6 \
      --batch_size 256 \
      --max_mean_backbone_forwards 3.0 \
      --min_extra_view_rate 0.05 \
      --fft_weight 2.0 \
      --device cuda:0
  ) > "$log" 2>&1 &
  pid=$!
  printf '%s\t%s\t%s\t8\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$scope" "$rank" "$profile" "$gpu" "$pid" "$log" "$run_dir" >> "$manifest"
done

cat "$manifest"
