#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/szu2070436088/2510044040/CV-SincNet
BASE="$PROJECT/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303"
CAP="$BASE/phase2_capsules/rx_20_1/seed_713101/k_10/new_5_retry3"
AUTH="$BASE/input/runtime_authorization_k10_new5"
COMP="$PROJECT/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
D22_RUN="$PROJECT/runs/d20_int8_maxold_fftrf_20260717"
RUN="$PROJECT/runs/d36_compiled_joint_int8_20260718"
OUTPUT="$RUN/output/support_screen_v1"
LOG_ROOT="$PROJECT/logs/d36_compiled_joint_int8_20260718"
LOG="$LOG_ROOT/support_screen_v1.log"
PID_FILE="$RUN/support_screen_v1.pid"
PYCACHE_ROOT="$RUN/pycache_support_screen_v1"
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
GPU="${D36_GPU:-0}"

RUNNER=code/scripts/run_d25_support_only_concat.py
D36_CORE=code/cvsrffi/stage2_d36_compiled_joint_int8.py
D36_FISHER=code/cvsrffi/stage2_b3_fisher_closed_form.py
EXPECTED_RUNNER_SHA256=aa879e5529075f43f30727f999dc0a23881e1adb64fecce39e0e4ade4d42550c
EXPECTED_D36_CORE_SHA256=f38630824abd2ef35c71fd425b8a055dc5b52d7d77d221bb171fa2d6b13234a0
EXPECTED_D36_FISHER_SHA256=2cc05c0f2ef10c231698fdfd183ba84bcc1554a493e0d9bbbe318ac021f5d8ef

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    echo "source closure mismatch: $path expected=$expected actual=$actual" >&2
    exit 4
  fi
}

if [[ -e "$OUTPUT" ]]; then
  echo "output already exists: $OUTPUT" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "server Python missing: $PYTHON" >&2
  exit 3
fi

cd "$PROJECT"
verify_sha256 "$RUNNER" "$EXPECTED_RUNNER_SHA256"
verify_sha256 "$D36_CORE" "$EXPECTED_D36_CORE_SHA256"
verify_sha256 "$D36_FISHER" "$EXPECTED_D36_FISHER_SHA256"

mkdir -p "$(dirname "$OUTPUT")" "$LOG_ROOT"

CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTHONPYCACHEPREFIX="$PYCACHE_ROOT" nohup "$PYTHON" \
  "$RUNNER" \
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
  --class-binding "$D22_RUN/input/class_binding.json" \
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f \
  --output "$OUTPUT" \
  --device cuda:0 \
  --mode development_select_unverified_component \
  --candidate-set d36_v1 \
  >"$LOG" 2>&1 &

pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "D36_PID=$pid"
echo "D36_GPU=$GPU"
echo "D36_LOG=$LOG"
echo "D36_OUTPUT=$OUTPUT"
