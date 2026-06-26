#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WISIG_PKL="${WISIG_PKL:-$ROOT/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/fedbase_paper}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/fedbase_paper}"
GPU_IDS="${GPU_IDS:-0,1}"
METHODS="${METHODS:-fedriei,fedfa,fucl,rafl}"
FEDBASE_PROFILE="${FEDBASE_PROFILE:-cvs_adapter}"
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
      FEDBASE_PROFILE="$2"
      shift
      ;;
    --methods)
      METHODS="$2"
      shift
      ;;
    *)
      echo "Unsupported argument: $1" >&2
      exit 2
      ;;
  esac
  shift
done

IFS=',' read -r -a method_list <<< "$METHODS"
IFS=',' read -r -a gpu_list <<< "$GPU_IDS"

if [[ "$DRY_RUN" != "1" ]]; then
  capacity=$((${#gpu_list[@]} * MAX_JOBS_PER_GPU))
  if [[ ${#method_list[@]} -gt $capacity ]]; then
    echo "Refusing launch: methods=${#method_list[@]} exceeds capacity=$capacity from GPU_IDS=$GPU_IDS and MAX_JOBS_PER_GPU=$MAX_JOBS_PER_GPU" >&2
    exit 3
  fi
fi

echo "[FEDBASE-PAPER-MATRIX] root=$ROOT train_script=$TRAIN_SCRIPT wisig_pkl=$WISIG_PKL run_root=$RUN_ROOT log_root=$LOG_ROOT methods=$METHODS profile=$FEDBASE_PROFILE gpus=$GPU_IDS dry_run=$DRY_RUN conda_env=${CONDA_ENV:-none} max_jobs_per_gpu=$MAX_JOBS_PER_GPU"

method_extra_args() {
  case "$1" in
    fedriei)
      echo "--batch_size 16 --fl_local_epochs 1 --lr 0.0001 --wd 0.0 --fedbase_paper_method FedRIEI --fedriei_lambda_mi 1.2 --fedriei_lambda_ie 1.2 --fedriei_gradient_compression none --fedriei_compression_noise_std 0.01 --fl_test_eval_last_n 5"
      ;;
    fedfa)
      if [[ "$FEDBASE_PROFILE" == "strict_paper" ]]; then
        echo "--batch_size 64 --fl_local_epochs 4 --lr 0.01 --wd 0.0 --fl_agg_weight uniform --fedbase_paper_method FedFA --fedbase_paper_profile strict_paper --fedfa_align_lambda 0.03"
      else
        echo "--batch_size 64 --fl_local_epochs 4 --lr 0.01 --wd 0.0 --fl_agg_weight uniform --fedbase_paper_method FedFA --fedbase_paper_profile cvs_adapter --fedfa_align_lambda 0.03"
      fi
      ;;
    fucl)
      strict_args=""
      if [[ "$FEDBASE_PROFILE" == "strict_paper" ]]; then
        strict_args="--fedbase_paper_profile strict_paper"
      else
        strict_args="--fedbase_paper_profile cvs_adapter"
      fi
      echo "--batch_size 128 --fl_local_epochs 1 --lr 0.001 --wd 0.0 --fedbase_paper_method FUCL $strict_args --fedbase_feature_dim 128 --fucl_temperature 0.05 --fucl_pretrain_lr 0.0003 --fucl_finetune_lr 0.001 --fucl_finetune_epochs 20 --fucl_local_validation_ratio 0.1 --fucl_local_lr_patience 10 --fucl_local_lr_decay 0.1 --fucl_local_early_stop_patience 20 --fucl_local_max_epochs 200 --fucl_sample_rate_hz 500000 --fucl_tdl_rms_delay_min_ns 5 --fucl_tdl_rms_delay_max_ns 300 --fucl_tdl_doppler_min_hz 0 --fucl_tdl_doppler_max_hz 5 --fucl_tdl_snr_min_db 0 --fucl_tdl_snr_max_db 80 --fucl_cis_n_fft 64 --fucl_cis_hop_length 32 --fucl_cis_win_length 64 --fucl_cis_crop_fraction 0.30 --fucl_cis_freq_bins 26 --fucl_cis_time_bins 126 --fucl_cis_normalize none"
      ;;
    rafl)
      if [[ "$FEDBASE_PROFILE" == "strict_paper" ]]; then
        echo "--batch_size 64 --fl_local_epochs 5 --fl_clients_per_round 0.5 --lr 0.001 --wd 0.0 --fedbase_paper_method RAFL --fedbase_paper_profile strict_paper --rafl_lambda_rx 0.1 --rafl_momentum 0.0 --rafl_client_selection label_loss_driven --rafl_selection_dataset internal_train_split --rafl_input_version paper_52x126 --rafl_spec_n_fft 52 --rafl_spec_hop_length 2 --rafl_spec_win_length 52 --rafl_spec_freq_bins 52 --rafl_spec_time_bins 126 --rafl_selected_clients 5 --rafl_candidate_clients 10 --rafl_candidate_fraction 1.0 --rafl_selection_eval_ratio 0.1"
      else
        echo "--batch_size 64 --fl_local_epochs 5 --fl_clients_per_round 0.5 --lr 0.001 --wd 0.0 --fedbase_paper_method RAFL --fedbase_paper_profile cvs_adapter --rafl_lambda_rx 0.1 --rafl_momentum 0.0 --rafl_client_selection label_loss_driven --rafl_selection_dataset internal_train_split --rafl_input_version wisig_complex --rafl_spec_n_fft 52 --rafl_spec_hop_length 2 --rafl_spec_win_length 52 --rafl_selected_clients 0 --rafl_candidate_clients 0 --rafl_candidate_fraction 1.0 --rafl_selection_eval_ratio 0.1"
      fi
      ;;
    *)
      echo "Unsupported method: $1" >&2
      return 2
      ;;
  esac
}

method_rounds() {
  if [[ "$FEDBASE_PROFILE" != "strict_paper" ]]; then
    echo "200"
    return
  fi
  case "$1" in
    fedfa)
      echo "40"
      ;;
    fucl)
      echo "5"
      ;;
    rafl)
      echo "300"
      ;;
    *)
      echo "200"
      ;;
  esac
}

idx=0
for method in "${method_list[@]}"; do
  method="$(echo "$method" | tr '[:upper:]' '[:lower:]' | xargs)"
  gpu="${gpu_list[$((idx % ${#gpu_list[@]}))]}"
  run_name="${TIMESTAMP}_fedbase_${method}_cvsrffi_r010"
  out_dir="$RUN_ROOT/$run_name"
  log_dir="$LOG_ROOT/$run_name"
  extra_args="$(method_extra_args "$method")"
  rounds="$(method_rounds "$method")"
  env_prefix=""
  if [[ -n "$CONDA_ENV" ]]; then
    env_prefix="source \"$CONDA_SH\" && conda activate \"$CONDA_ENV\" && "
  fi
  cmd="cd \"$ROOT\" && ${env_prefix}CUDA_VISIBLE_DEVICES=$gpu \"$PYTHON\" -u \"$TRAIN_SCRIPT\" --dataset wisig --wisig_pkl \"$WISIG_PKL\" --train_mode $method --wisig_train_ratio 0.1 --epochs 200 --fl_rounds $rounds --fl_client_key receiver --eval_sat_channel --eval_sat_on main --run_name \"$run_name\" --output_dir \"$out_dir\" --log_dir \"$log_dir\" $extra_args"
  echo "CMD $cmd"
  if [[ "$DRY_RUN" != "1" ]]; then
    mkdir -p "$out_dir" "$log_dir"
    nohup bash -lc "$cmd" > "$log_dir/stdout.log" 2>&1 &
    echo "PID $! method=$method gpu=$gpu log=$log_dir/stdout.log"
  fi
  idx=$((idx + 1))
done
