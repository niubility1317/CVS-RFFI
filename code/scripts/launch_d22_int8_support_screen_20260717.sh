#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/szu2070436088/2510044040/CV-SincNet
BASE="$PROJECT/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303"
CAP="$BASE/phase2_capsules/rx_20_1/seed_713101/k_10/new_5_retry3"
AUTH="$BASE/input/runtime_authorization_k10_new5"
COMP="$PROJECT/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
RUN="$PROJECT/runs/d20_int8_maxold_fftrf_20260717"
OUTPUT="$RUN/output/support_screen_v1"
LOG_ROOT="$PROJECT/logs/d20_int8_maxold_fftrf_20260717"
LOG="$LOG_ROOT/support_screen_v1.log"
PID_FILE="$RUN/support_screen_v1.pid"
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python

if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "server Python missing: $PYTHON" >&2
  exit 3
fi

mkdir -p "$(dirname "$OUTPUT")" "$LOG_ROOT"
cd "$PROJECT"

CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup "$PYTHON" \
  code/scripts/run_d19_support_only_ciaf.py \
  --before-root "$CAP/predictor/before/enrollment_only" \
  --before-seal "$CAP/seals/before_enrollment.seal.json" \
  --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 \
  --before-formal-policy "$AUTH/formal_execution_policy.json" \
  --before-formal-policy-authorization "$AUTH/before_formal_policy_authorization.v2.json" \
  --before-signed-policy-authorization-envelope "$AUTH/before_signed_policy_authorization_envelope.v2.json" \
  --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e \
  --after-root "$CAP/predictor/after/enrollment_only" \
  --after-seal "$CAP/seals/after_enrollment.seal.json" \
  --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff \
  --after-formal-policy "$AUTH/formal_execution_policy.json" \
  --after-formal-policy-authorization "$AUTH/after_formal_policy_authorization.v2.json" \
  --after-signed-policy-authorization-envelope "$AUTH/after_signed_policy_authorization_envelope.v2.json" \
  --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 \
  --component-dir "$COMP" \
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c \
  --class-binding "$RUN/input/class_binding.json" \
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f \
  --output "$OUTPUT" \
  --device cuda:0 \
  --mode development_select_unverified_component \
  >"$LOG" 2>&1 &

pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "D22_PID=$pid"
echo "D22_LOG=$LOG"
echo "D22_OUTPUT=$OUTPUT"
