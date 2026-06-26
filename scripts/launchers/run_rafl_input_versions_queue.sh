#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WISIG_PKL="${WISIG_PKL:-$ROOT/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/fedbase_paper}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/fedbase_paper}"
GPU_IDS="${GPU_IDS:-0,1}"
RAFL_PROFILE="${RAFL_PROFILE:-cvs_adapter}"
if [[ "$RAFL_PROFILE" == "strict_paper" ]]; then
  VARIANTS="${VARIANTS:-paper_52x126}"
else
  VARIANTS="${VARIANTS:-paper_52x126,wisig_complex}"
fi
PYTHON="${PYTHON:-python}"
if [[ -z "${TRAIN_SCRIPT:-}" ]]; then
  if [[ -f "$ROOT/code/train.py" ]]; then
    TRAIN_SCRIPT="$ROOT/code/train.py"
  else
    TRAIN_SCRIPT="$ROOT/train.py"
  fi
fi
CONDA_ENV="${CONDA_ENV:-}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-2}"
DRY_RUN="${DRY_RUN:-1}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --launch)
      DRY_RUN=0
      ;;
    --profile)
      RAFL_PROFILE="$2"
      shift
      ;;
    --variants)
      VARIANTS="$2"
      shift
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

IFS=',' read -r -a variant_list <<< "$VARIANTS"
IFS=',' read -r -a gpu_list <<< "$GPU_IDS"

if [[ "$DRY_RUN" != "1" ]]; then
  capacity=$((${#gpu_list[@]} * MAX_JOBS_PER_GPU))
  if [[ ${#variant_list[@]} -gt $capacity ]]; then
    echo "Refusing launch: variants=${#variant_list[@]} exceeds capacity=$capacity from GPU_IDS=$GPU_IDS and MAX_JOBS_PER_GPU=$MAX_JOBS_PER_GPU" >&2
    exit 3
  fi
fi

echo "[RAFL-INPUT-VERSIONS] root=$ROOT train_script=$TRAIN_SCRIPT wisig_pkl=$WISIG_PKL run_root=$RUN_ROOT log_root=$LOG_ROOT variants=$VARIANTS profile=$RAFL_PROFILE gpus=$GPU_IDS dry_run=$DRY_RUN conda_env=${CONDA_ENV:-none} max_jobs_per_gpu=$MAX_JOBS_PER_GPU"

variant_extra_args() {
  case "$1" in
    paper_52x126)
      echo "--rafl_input_version paper_52x126 --rafl_spec_n_fft 52 --rafl_spec_hop_length 2 --rafl_spec_win_length 52 --rafl_spec_freq_bins 52 --rafl_spec_time_bins 126"
      ;;
    wisig_native)
      echo "--rafl_input_version wisig_native"
      ;;
    wisig_complex)
      echo "--rafl_input_version wisig_complex --rafl_spec_n_fft 52 --rafl_spec_hop_length 2 --rafl_spec_win_length 52"
      ;;
    *)
      echo "Unsupported RAFL input variant: $1" >&2
      return 2
      ;;
  esac
}

idx=0
for variant in "${variant_list[@]}"; do
  variant="$(echo "$variant" | tr '[:upper:]' '[:lower:]' | xargs)"
  gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
  run_name="${TIMESTAMP}_fedbase_rafl_${variant}_cvsrffi_r010"
  out_dir="$RUN_ROOT/$run_name"
  log_dir="$LOG_ROOT/$run_name"
  variant_args="$(variant_extra_args "$variant")"
  base_rounds=200
  selected_args="--rafl_selected_clients 0 --rafl_candidate_clients 0 --rafl_candidate_fraction 1.0"
  profile_args="--fedbase_paper_profile cvs_adapter"
  if [[ "$RAFL_PROFILE" == "strict_paper" ]]; then
    base_rounds=300
    selected_args="--rafl_selected_clients 5 --rafl_candidate_clients 10 --rafl_candidate_fraction 1.0"
    profile_args="--fedbase_paper_profile strict_paper"
  fi
  env_prefix=""
  if [[ -n "$CONDA_ENV" ]]; then
    env_prefix="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && "
  fi
  cmd="cd \"$ROOT\" && ${env_prefix}CUDA_VISIBLE_DEVICES=$gpu \"$PYTHON\" -u \"$TRAIN_SCRIPT\" --dataset wisig --wisig_pkl \"$WISIG_PKL\" --train_mode rafl --wisig_train_ratio 0.1 --epochs 200 --fl_rounds $base_rounds --fl_client_key receiver --eval_sat_channel --eval_sat_on main --run_name \"$run_name\" --output_dir \"$out_dir\" --log_dir \"$log_dir\" --batch_size 64 --fl_local_epochs 5 --fl_clients_per_round 0.5 --lr 0.001 --wd 0.0 --fedbase_paper_method RAFL $profile_args --rafl_lambda_rx 0.1 --rafl_momentum 0.0 --rafl_client_selection label_loss_driven --rafl_selection_dataset internal_train_split $selected_args --rafl_selection_eval_ratio 0.1 $variant_args"
  echo "CMD $cmd"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$out_dir" "$log_dir"
    nohup bash -lc "$cmd" > "$log_dir/stdout.log" 2>&1 &
    echo "PID $! variant=$variant gpu=$gpu log=$log_dir/stdout.log"
  fi
  idx=$((idx + 1))
done
