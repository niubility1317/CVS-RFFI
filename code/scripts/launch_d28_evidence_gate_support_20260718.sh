#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/szu2070436088/2510044040/CV-SincNet
BASE="$PROJECT/runs/d18_formal_k10_new5_rx20_1_seed713101_20260717_085303"
CAP="$BASE/phase2_capsules/rx_20_1/seed_713101/k_10/new_5_retry3"
AUTH="$BASE/input/runtime_authorization_k10_new5"
COMP="$PROJECT/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
D22_RUN="$PROJECT/runs/d20_int8_maxold_fftrf_20260717"
RUN="$PROJECT/runs/d28_evidence_gate_20260718"
OUTPUT="$RUN/output/support_screen_v2"
LOG_ROOT="$PROJECT/logs/d28_evidence_gate_20260718"
LOG="$LOG_ROOT/support_screen_v2.log"
PID_FILE="$RUN/support_screen_v2.pid"
PYCACHE_ROOT="$RUN/pycache_support_screen_v2"
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
GPU="${D28_GPU:-0}"

RUNNER=code/scripts/run_d25_support_only_concat.py
D28_CORE=code/cvsrffi/stage2_support_evidence_gate.py
D27_CORE=code/cvsrffi/stage2_multimodal_compact_diag.py
D25_CORE=code/cvsrffi/stage2_multimodal_concat_fusion.py
D24=code/cvsrffi/stage2_uncertainty_proto_fusion.py
CIAF=code/cvsrffi/stage2_ciaf.py
CONTROL=code/scripts/run_d19_support_only_ciaf.py
DIAG_OPERATOR=code/cvsrffi/stage2_diag_cosine_exploration.py
EXPECTED_RUNNER_SHA256=d380ff5923c917a82dfde4d7090548feff1625c7bbd3470b2d23d5bdd6a412af
EXPECTED_D28_CORE_SHA256=d18110786598c05d120fec4c278bf31de555a7b19fb00d3780d5a6c5f3e52e5f
EXPECTED_D27_CORE_SHA256=553d6361a728490c26963944df8353f1bc64bf1540b2ab6709f2f25bedd6f1ff
EXPECTED_D25_CORE_SHA256=2c43008c1f14f6a6173c3680b3af8a8b4015dfde662b0d4fcfb11e74829dac1e
EXPECTED_D24_SHA256=2ed2067c4636447f9e013bab2b99d6bc94e149ed5152907fc363b7e802bd2b86
EXPECTED_CIAF_SHA256=f46c5007cb1c0279bf2b27169ad79989eba908f32658c5a4d7f819916381aeb1
EXPECTED_CONTROL_SHA256=7e46db1e99ac40f4e9d7679dcb7f668553d928a0672a7bcf07022383949c8553
# Preserve the already verified remote operator; do not overwrite the local
# unrelated in-progress diagnostic edits.
EXPECTED_DIAG_OPERATOR_SHA256=14ec919395f9bf9f13214c677b1a3d640764214668d1d00e9109f5b149ec41ca

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
verify_sha256 "$D28_CORE" "$EXPECTED_D28_CORE_SHA256"
verify_sha256 "$D27_CORE" "$EXPECTED_D27_CORE_SHA256"
verify_sha256 "$D25_CORE" "$EXPECTED_D25_CORE_SHA256"
verify_sha256 "$D24" "$EXPECTED_D24_SHA256"
verify_sha256 "$CIAF" "$EXPECTED_CIAF_SHA256"
verify_sha256 "$CONTROL" "$EXPECTED_CONTROL_SHA256"
verify_sha256 "$DIAG_OPERATOR" "$EXPECTED_DIAG_OPERATOR_SHA256"

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
  --candidate-set d28_v1_evidence_gate \
  >"$LOG" 2>&1 &

pid=$!
printf '%s\n' "$pid" >"$PID_FILE"
echo "D28_PID=$pid"
echo "D28_GPU=$GPU"
echo "D28_LOG=$LOG"
echo "D28_OUTPUT=$OUTPUT"
