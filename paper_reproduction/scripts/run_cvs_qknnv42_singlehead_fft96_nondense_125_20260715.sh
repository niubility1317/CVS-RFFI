#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
GPU="${GPU:-6}"
SWEEP_ID="${SWEEP_ID:-qknnv42_nondense_adapter_epoch_sweep_20260715_104409}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/${SWEEP_ID}/singlehead_fft96}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${SWEEP_ID}/singlehead_fft96}"
CVS_CONFIG="${CVS_CONFIG:-${ROOT}/paper_reproduction/configs/cvs_qknnv42_singlehead_fft96_nondense_stage2c_20260715_n607.json}"

export PYTHONPATH="${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
mkdir -p "$OUT_ROOT" "$LOG_ROOT"
cd "$ROOT"

CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m paper_reproduction.scripts.run_cvs_publication_matrix \
  --phase stage2c \
  --config "${ROOT}/paper_reproduction/configs/cvs_stage2c_publication_base_n607.json" \
  --cvs-config "$CVS_CONFIG" \
  --output-root "$OUT_ROOT" --log-root "$LOG_ROOT" \
  --methods cvs_qknnv42 --execute

echo "[QKNN-SINGLEHEAD-FFT96-NONDENSE-DONE] run_root=${OUT_ROOT}"
