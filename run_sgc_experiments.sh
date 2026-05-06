#!/usr/bin/env bash
set -euo pipefail

# Three-stage SGC-Adapter launcher for the CVS-RFFI root project.
#
# Examples:
#   bash run_sgc_experiments.sh source
#   bash run_sgc_experiments.sh augment
#   bash run_sgc_experiments.sh adapt
#   SGC_PRESET=sgc_lite_b_no_dac_no_freq GPU_ID=1 bash run_sgc_experiments.sh source
#   RUN_ALL_ABLATIONS=1 bash run_sgc_experiments.sh source

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-0}"
STAGE="${1:-source}"
SGC_PRESET="${SGC_PRESET:-sgc_lite_b_no_dac}"
BASE_ARGS="${BASE_ARGS:---dataset wisig --wisig_domain rx_day --batch_size 256 --wisig_train_ratio 0.2}"
EPOCHS_SOURCE="${EPOCHS_SOURCE:-200}"
EPOCHS_AUGMENT="${EPOCHS_AUGMENT:-100}"
ADAPT_EPOCHS="${ADAPT_EPOCHS:-50}"
SOURCE_CKPT="${SOURCE_CKPT:-sgc_runs/${SGC_PRESET}/source/best_model.pth}"
AUGMENT_CKPT="${AUGMENT_CKPT:-sgc_runs/${SGC_PRESET}/augment/best_model.pth}"
SAT_SCENARIO="${SAT_SCENARIO:-mixed_orbit}"
PSEUDO_THRESHOLD="${PSEUDO_THRESHOLD:-0.85}"
RUN_ALL_ABLATIONS="${RUN_ALL_ABLATIONS:-0}"

PRESETS=(
  "sgc_lite_b_no_dac"
  "sgc_lite_b_no_dac_no_amp"
  "sgc_lite_b_no_dac_no_freq"
  "sgc_lite_b_no_dac_no_spec"
  "sgc_lite_b_no_dac_no_res"
  "sgc_baseline_no_adapter"
)

mkdir -p logs sgc_runs

run_one() {
  local preset="$1"
  local stage="$2"
  local run_dir="sgc_runs/${preset}/${stage}"
  local log_path="logs/sgc_${preset}_${stage}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "${run_dir}"

  case "${stage}" in
    source)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u train.py \
        ${BASE_ARGS} \
        --preset "${preset}" \
        --stage source \
        --epochs "${EPOCHS_SOURCE}" \
        --latest_save_path "${run_dir}/latest_model.pth" \
        --best_save_path "${run_dir}/best_model.pth" \
        2>&1 | tee "${log_path}"
      ;;
    augment|sgc_augment)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u train.py \
        ${BASE_ARGS} \
        --preset "${preset}" \
        --stage sgc_augment \
        --source_ckpt "${SOURCE_CKPT}" \
        --train_sat_channel \
        --train_sat_scenario "${SAT_SCENARIO}" \
        --sat_view_source main \
        --lambda_feat 1.0 \
        --lambda_res 0.01 \
        --epochs "${EPOCHS_AUGMENT}" \
        --latest_save_path "${run_dir}/latest_model.pth" \
        --best_save_path "${run_dir}/best_model.pth" \
        2>&1 | tee "${log_path}"
      ;;
    adapt|sgc_adapt)
      CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u train.py \
        ${BASE_ARGS} \
        --preset "${preset}" \
        --stage sgc_adapt \
        --source_ckpt "${AUGMENT_CKPT}" \
        --pseudo_label_threshold "${PSEUDO_THRESHOLD}" \
        --lambda_proto 1.0 \
        --lambda_cons 0.5 \
        --lambda_ent 0.01 \
        --lambda_res 0.01 \
        --adapt_lr 1e-4 \
        --adapt_epochs "${ADAPT_EPOCHS}" \
        --latest_save_path "${run_dir}/latest_model.pth" \
        --best_save_path "${run_dir}/best_model.pth" \
        2>&1 | tee "${log_path}"
      ;;
    *)
      echo "Unknown SGC stage: ${stage}. Use source, augment, or adapt." >&2
      return 2
      ;;
  esac
}

if [ "${RUN_ALL_ABLATIONS}" = "1" ]; then
  for preset in "${PRESETS[@]}"; do
    run_one "${preset}" "${STAGE}"
  done
else
  run_one "${SGC_PRESET}" "${STAGE}"
fi
