#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-cvs_qknnv42_fft96_singleview_125_20260714}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/paper_reproduction/logs/${RUN_ID}}"

bash "${ROOT}/paper_reproduction/scripts/export_cvs_publication_adv3b02_fft96_singleview_20260714.sh"

env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "$PYTHON" -u -m \
  paper_reproduction.scripts.run_cvs_publication_matrix \
  --phase stage2c \
  --config "${ROOT}/paper_reproduction/configs/cvs_stage2c_publication_base_n607.json" \
  --cvs-config "${ROOT}/paper_reproduction/configs/cvs_qknnv42_fft96_singleview_stage2c_20260714_n607.json" \
  --output-root "$OUT_ROOT" \
  --log-root "$LOG_ROOT" \
  --methods cvs_qknnv42 \
  --execute

echo "[FFT96-SINGLEVIEW-125-DONE] run_root=${OUT_ROOT}"
