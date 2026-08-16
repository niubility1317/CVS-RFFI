#!/usr/bin/env bash
set -euo pipefail

# Versioned six-fold entry for the unchanged ADV3B02 formal profile.  The
# command source is mechanically inherited from v1; this wrapper changes only
# the immutable run identity/paths and requires a completed v2 technical
# receipt before any formal root can be created.

V1_LAUNCHER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_phase1_adv3b02_clic6_v1_20260813.sh"
V1_RUN_ID="phase1_adv3b02_clic6_20260813_v1"
RUN_ID="${RUN_ID:-phase1_adv3b02_clic6_20260816_v2}"
SMOKE_ROOT_NAME=".smoke_phase1_adv3b02_clic6_20260816_v2_F1"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
WISIG_PKL="${WISIG_PKL:-${PROJECT_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/${RUN_ID}}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${CODE_ROOT}/SSDG/train_ssdg.py}"

[[ -f "${V1_LAUNCHER}" ]] || {
  echo "missing mechanically reused v1 formal launcher: ${V1_LAUNCHER}" >&2
  exit 2
}
[[ "${RUN_ID}" == "phase1_adv3b02_clic6_20260816_v2" ]] || {
  echo "RUN_ID is immutable for this launcher: ${RUN_ID}" >&2
  exit 2
}

# Dry-run/contract inspection is intentionally non-mutating and must remain
# available before the smoke exists so the trainer can independently recover
# the formal F1 parser profile.
requires_smoke_receipt=1
for argument in "$@"; do
  case "${argument}" in
    --dry-run|--print-contract|--validate-contract-file|--validate-contract-file=*)
      requires_smoke_receipt=0
      ;;
  esac
done

validate_v2_smoke_receipt() {
  local receipt_path="$1"
  command -v python3 >/dev/null 2>&1 || return 1
  python3 - "${receipt_path}" <<'PY'
import json
import math
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as stream:
    payload = json.load(stream)

required = {
    "schema": "cvs.phase1.adv3b02_technical_smoke.v2",
    "completed": True,
    "claim": "NO_PERFORMANCE_RESULT",
    "base_candidate": "ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL",
    "run_id": "phase1_adv3b02_clic6_20260816_v2",
    "candidate_id": "F1_ADV3B02_CLIC",
    "fold": "F1",
    "raw_batch_cap": 4,
    "target_effective_steps": 3,
    "effective_forward_steps": 3,
    "effective_backward_steps": 3,
    "optimizer_attempts": 3,
    "optimizer_effective_steps": 3,
    "skipped_nonfinite_loss_batches": 0,
}
for field, expected in required.items():
    if payload.get(field) != expected:
        raise SystemExit(f"receipt field mismatch: {field}")

for field in (
    "source_val_rows_opened",
    "query_rows_opened",
    "target_rows_opened",
    "test_rows_opened",
    "selection_feedback_count",
):
    value = payload.get(field)
    if type(value) is not int or value != 0:
        raise SystemExit(f"receipt access counter must be exact integer zero: {field}")

raw_batches = payload.get("raw_batches_observed")
if not isinstance(raw_batches, int) or raw_batches not in {3, 4}:
    raise SystemExit("receipt raw batch count is outside the fixed v2 bound")
grad_skips = payload.get("skipped_nonfinite_grad_batches")
if grad_skips not in {0, 1} or payload.get("handled_grad_skip_count") != grad_skips:
    raise SystemExit("receipt gradient skip count violates the v2 bound")
records = payload.get("raw_batch_records")
if not isinstance(records, list) or len(records) != raw_batches:
    raise SystemExit("receipt raw batch records do not close")
if [record.get("raw_batch_index") for record in records] != list(range(1, raw_batches + 1)):
    raise SystemExit("receipt raw batch record indices do not close")
if sum(not bool(record.get("loss_finite")) for record in records) != 0:
    raise SystemExit("receipt contains a nonfinite loss")
if sum(not bool(record.get("grad_finite")) for record in records) != grad_skips:
    raise SystemExit("receipt gradient finiteness does not match skip count")
if sum(bool(record.get("optimizer_attempted")) for record in records) != 3:
    raise SystemExit("receipt optimizer attempt count does not close")
if sum(bool(record.get("optimizer_effective")) for record in records) != 3:
    raise SystemExit("receipt optimizer effective count does not close")
for record in records:
    if not bool(record.get("grad_finite")) and (
        bool(record.get("optimizer_attempted")) or bool(record.get("optimizer_effective"))
    ):
        raise SystemExit("receipt attempts an optimizer step after a nonfinite gradient")
    for field in ("amp_scale_before", "amp_scale_after"):
        try:
            if not math.isfinite(float(record[field])):
                raise ValueError(field)
        except (KeyError, TypeError, ValueError):
            raise SystemExit(f"receipt AMP scale is not finite: {field}")
PY
}

if [[ "${requires_smoke_receipt}" == "1" ]]; then
  receipt_path="${PROJECT_ROOT}/runs/${SMOKE_ROOT_NAME}/F1_ADV3B02_CLIC/phase1_adv3b02_technical_smoke_v2_receipt.json"
  [[ -f "${receipt_path}" ]] && validate_v2_smoke_receipt "${receipt_path}" || {
    echo "formal v2 requires a complete v2 technical smoke receipt: ${receipt_path}" >&2
    exit 2
  }
fi

# The inherited script uses environment variables for all runtime roots.  They
# must be exported because it is intentionally executed from a pipe rather
# than sourced, which also prevents it from mutating this wrapper's state.
export RUN_ID PROJECT_ROOT CODE_ROOT PYTHON WISIG_PKL RUN_ROOT LOG_ROOT TRAIN_SCRIPT
sed "s/${V1_RUN_ID}/${RUN_ID}/g" "${V1_LAUNCHER}" | bash -s -- "$@"
