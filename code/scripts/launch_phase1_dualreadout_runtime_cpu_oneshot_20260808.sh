#!/usr/bin/env bash
set -euo pipefail

: "${PYTHON:?PYTHON is required}"
: "${B_CHECKPOINT:?B_CHECKPOINT is required}"
: "${B_CHECKPOINT_SHA256:?B_CHECKPOINT_SHA256 is required}"
: "${C_CHECKPOINT:?C_CHECKPOINT is required}"
: "${C_CHECKPOINT_SHA256:?C_CHECKPOINT_SHA256 is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

"${PYTHON}" scripts/phase1_dualreadout_bundle_v2.py export-runtime \
  --checkpoint "${B_CHECKPOINT}" \
  --expected-checkpoint-sha256 "${B_CHECKPOINT_SHA256}" \
  --input-len 256 \
  --runtime-out "${OUTPUT_ROOT}/inputs/angular.ts" \
  --receipt-out "${OUTPUT_ROOT}/inputs/angular_parity.json" \
  --device cpu

"${PYTHON}" scripts/phase1_dualreadout_bundle_v2.py export-runtime \
  --checkpoint "${C_CHECKPOINT}" \
  --expected-checkpoint-sha256 "${C_CHECKPOINT_SHA256}" \
  --input-len 256 \
  --runtime-out "${OUTPUT_ROOT}/inputs/robust.ts" \
  --receipt-out "${OUTPUT_ROOT}/inputs/robust_parity.json" \
  --device cpu
