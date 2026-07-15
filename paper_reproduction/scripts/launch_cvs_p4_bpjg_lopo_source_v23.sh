#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/szu2070436088/2510044040/CV-SincNet
EXPERIMENT_ID=qknnv42_p4_bpjg_lopo_source_k1_layer_20260715_v23
OUT_ROOT="$ROOT/runs/$EXPERIMENT_ID"
LOG_ROOT="$ROOT/logs/$EXPERIMENT_ID"
CONDA_SH=/home/szu2200432017/miniconda3/etc/profile.d/conda.sh
CHECKPOINT="$ROOT/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
CHECKPOINT_SHA256=2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98
BASE_RUN="$ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14"
SOURCE_VALIDATION="$BASE_RUN/phase1_caches/source_validation/cache_set.json"
GROUND_STATE="$ROOT/runs/qknn_ground_adapt_layer_loss_ablation_20260715_v16/p4_r16_e8_k1/adapter_fp16.pt"
GROUND_SHA256=95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446
TRAINER="$ROOT/paper_reproduction/scripts/train_export_cvs_support_lora_adapter.py"
TRAINER_SHA256=f985f5e5f718f1c60ab75e6b41684bf4962edce454c1612a7d2f7c0e14406f7e
SCREEN="$ROOT/paper_reproduction/scripts/screen_cvs_p4_bpjg_lopo_source.py"
SCREEN_SHA256=f19be0b4c3745c4c950161199faa82008b001f4904640fc8a6129e00a8fd1834

for required in "$CONDA_SH" "$CHECKPOINT" "$SOURCE_VALIDATION" "$GROUND_STATE" "$TRAINER" "$SCREEN"; do
  if [[ ! -f "$required" ]]; then
    echo "missing K1 layer-screen artifact: $required" >&2
    exit 2
  fi
done
observed_trainer_sha256="$(sha256sum "$TRAINER" | awk '{print $1}')"
observed_screen_sha256="$(sha256sum "$SCREEN" | awk '{print $1}')"
if [[ "$observed_trainer_sha256" != "$TRAINER_SHA256" ]]; then
  echo "support trainer SHA256 drift: $observed_trainer_sha256" >&2
  exit 3
fi
if [[ "$observed_screen_sha256" != "$SCREEN_SHA256" ]]; then
  echo "source screen SHA256 drift: $observed_screen_sha256" >&2
  exit 3
fi

labels=(JP8_LR005 JP8_LR010 JP8_LR020 JG8_LR005)
scopes=(joint_projection joint_projection joint_projection joint_gate)
lrs=(0.005 0.01 0.02 0.005)
gpus=(0 1 2 3)

mkdir -p "$OUT_ROOT" "$LOG_ROOT"
manifest="$OUT_ROOT/launch_manifest.tsv"
if [[ -e "$manifest" ]]; then
  echo "refusing to overwrite existing v23 manifest" >&2
  exit 4
fi
printf 'label\tscope\tlr\trank\tepochs\tk\tgpu\tpid\tlog\tstatus_file\tout_dir\tlaunch_only\n' > "$manifest"
for index in 0 1 2 3; do
  label="${labels[$index]}"
  out_dir="$OUT_ROOT/$label"
  log="$LOG_ROOT/$label.log"
  status_file="$LOG_ROOT/$label.status.tsv"
  if [[ -e "$out_dir" || -e "$log" || -e "$status_file" ]]; then
    echo "refusing to overwrite existing v23 arm: $label" >&2
    exit 4
  fi
done

source "$CONDA_SH"
for index in 0 1 2 3; do
  label="${labels[$index]}"
  scope="${scopes[$index]}"
  lr="${lrs[$index]}"
  gpu="${gpus[$index]}"
  out_dir="$OUT_ROOT/$label"
  log="$LOG_ROOT/$label.log"
  status_file="$LOG_ROOT/$label.status.tsv"
  (
    set +e
    finalize() {
      rc=$?
      if [[ "$rc" -eq 0 ]]; then
        status=PASS
      elif [[ "$rc" -eq 5 ]]; then
        status=SOURCE_SCREEN_NEGATIVE
      else
        status=INFRASTRUCTURE_FAILURE
      fi
      printf 'exit_code\tstatus\n%s\t%s\n' "$rc" "$status" > "$status_file"
      if [[ -d "$out_dir" ]]; then
        cp "$status_file" "$out_dir/process_status.tsv"
      fi
      trap - EXIT
      exit 0
    }
    trap finalize EXIT
    export CUDA_VISIBLE_DEVICES="$gpu"
    conda activate CVS-RFFI
    rc=$?
    if [[ "$rc" -ne 0 ]]; then
      exit "$rc"
    fi
    python -u -m paper_reproduction.scripts.screen_cvs_p4_bpjg_lopo_source \
      --ckpt "$CHECKPOINT" \
      --ckpt_sha256 "$CHECKPOINT_SHA256" \
      --ground_adapter_state "$GROUND_STATE" \
      --ground_adapter_sha256 "$GROUND_SHA256" \
      --source_cache_set "$SOURCE_VALIDATION" \
      --out_dir "$out_dir" \
      --source_train_rxs '1-1,1-19,14-7,18-2,19-2,2-1' \
      --source_val_rxs '2-19' \
      --class_count 6 \
      --k_shot 1 \
      --support_pool_max_k 20 \
      --seed 4070391 \
      --scope "$scope" \
      --rank 8 \
      --learning_rate "$lr" \
      --epochs 5 \
      --max_optimizer_steps 50 \
      --batch_size 256 \
      --device cuda:0
    exit $?
  ) > "$log" 2>&1 &
  pid=$!
  printf '%s\t%s\t%s\t8\t5\t1\t%s\t%s\t%s\t%s\t%s\ttrue\n' \
    "$label" "$scope" "$lr" "$gpu" "$pid" "$log" "$status_file" "$out_dir" >> "$manifest"
done

cat "$manifest"
