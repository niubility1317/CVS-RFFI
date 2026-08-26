#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
RUN_ID="${RUN_ID:-PHASE1_JMRS02_RX2_NUMERICAL_REPAIR_S20260824_20260826D}"
PYTHON_BIN="${PYTHON_BIN:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CHECKPOINT="${CHECKPOINT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/phase1_jmrs02_rx2_20260826/${RUN_ID}}"
SMOKE_ROOT="${SMOKE_ROOT:-${ROOT}/runs/phase1_jmrs02_rx2_20260826/${RUN_ID}_SMOKE}"
DEVICE="${DEVICE:-cuda:1}"
ROWS="B0,RX0,RX2"

if [[ -e "${OUTPUT_ROOT}" || -e "${SMOKE_ROOT}" ]]; then
  echo "refusing to overwrite existing RX2 output: ${OUTPUT_ROOT} or ${SMOKE_ROOT}" >&2
  exit 23
fi

"${PYTHON_BIN}" code/audit_phase1_jmrs02_j1.py \
  --output_dir "${SMOKE_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --wisig_pkl "${WISIG_PKL}" \
  --rows "${ROWS}" \
  --device "${DEVICE}" \
  --seed 20260826 \
  --sat_seed 20260824 \
  --focused_rx2 \
  --smoke_only

"${PYTHON_BIN}" code/audit_phase1_jmrs02_j1.py \
  --output_dir "${OUTPUT_ROOT}" \
  --checkpoint "${CHECKPOINT}" \
  --wisig_pkl "${WISIG_PKL}" \
  --rows "${ROWS}" \
  --device "${DEVICE}" \
  --seed 20260826 \
  --sat_seed 20260824 \
  --batch_size 128 \
  --eval_batch_size 256 \
  --inner_epochs 0 \
  --outer_epochs 20 \
  --learning_rate 3e-4 \
  --focused_rx2

"${PYTHON_BIN}" code/score_phase1_jmrs02_rx2.py \
  --predictions "${OUTPUT_ROOT}/predictions.jsonl" \
  --truth "${OUTPUT_ROOT}/truth.jsonl" \
  --output_dir "${OUTPUT_ROOT}/score"
