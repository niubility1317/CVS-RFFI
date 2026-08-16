#!/usr/bin/env bash
set -euo pipefail

# Identity-only v2 wrapper for the already frozen six-fold blind publisher.
# The stopped v1 output is never resumed.  A completed file-backed F1 smoke
# receipt is mandatory before the inherited formal entry may create output.
V1_LAUNCHER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_phase1_adv3b02_target_prediction6_v1_20260816.sh"
V1_RUN_ID="phase1_adv3b02_target_prediction_20260816_v1"
RUN_ID="phase1_adv3b02_target_prediction_20260816_v2"
SMOKE_ID=".smoke_phase1_adv3b02_target_prediction_20260816_v2_F1"
ADV_TRAINING_RUN_ID="phase1_adv3b02_clic6_20260816_v2"
PROJECT_ROOT="${PROJECT_ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SMOKE_ENTRY="${CODE_ROOT}/smoke_phase1_adv3b02_target_prediction_f1.py"
ADV_F1_ROOT="${PROJECT_ROOT}/runs/${ADV_TRAINING_RUN_ID}/F1_ADV3B02_CLIC"
CHECKPOINT="${ADV_F1_ROOT}/final_ssdg.pth"
COMPLETION="${ADV_F1_ROOT}/phase1_training_completion_receipt.json"
SMOKE_F1_ROOT="${PROJECT_ROOT}/runs/${SMOKE_ID}/F1_ADV3B02_CLIC"
TRAIN_CONFIG="${SMOKE_F1_ROOT}/train_data_config.json"
RECEIPT="${SMOKE_F1_ROOT}/technical_smoke_receipt.json"
DRY_RUN=0

for argument in "$@"; do
  case "${argument}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "invalid argument: ${argument}" >&2; exit 2 ;;
  esac
done

[[ -f "${V1_LAUNCHER}" ]] || {
  echo "missing frozen v1 formal target-prediction launcher" >&2
  exit 2
}

if [[ "${DRY_RUN}" == "0" ]]; then
  [[ -x "${PYTHON}" && -f "${SMOKE_ENTRY}" ]] || {
    echo "formal v2 requires the versioned F1 target-prediction smoke entry" >&2
    exit 2
  }
  [[ -f "${TRAIN_CONFIG}" && -f "${RECEIPT}" ]] || {
    echo "formal v2 requires a complete F1 target-prediction smoke receipt: ${RECEIPT}" >&2
    exit 2
  }
  env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="${CODE_ROOT}" \
    PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "${PYTHON}" -u "${SMOKE_ENTRY}" \
    --validate-receipt \
    --checkpoint "${CHECKPOINT}" \
    --completion-receipt-json "${COMPLETION}" \
    --train-config-output "${TRAIN_CONFIG}" \
    --receipt-output "${RECEIPT}" >/dev/null
fi

# Reuse the reviewed v1 formal implementation by rendering only its exact run
# identity substitution into a short-lived file, then execute that file path.
# No streamed shell source, process-stdin script body, or SSH-transmitted body
# is involved.
TRANSFORMED_LAUNCHER="$(mktemp "${TMPDIR:-/tmp}/adv-target-prediction-v2.XXXXXX.sh")"
cleanup_transformed_launcher() {
  rm -f -- "${TRANSFORMED_LAUNCHER}"
}
trap cleanup_transformed_launcher EXIT
sed "s/${V1_RUN_ID}/${RUN_ID}/g" "${V1_LAUNCHER}" >"${TRANSFORMED_LAUNCHER}"
grep -Fq "${RUN_ID}" "${TRANSFORMED_LAUNCHER}" || {
  echo "failed to render ADV target prediction v2 run identity" >&2
  exit 2
}
if grep -Fq "${V1_RUN_ID}" "${TRANSFORMED_LAUNCHER}"; then
  echo "stopped v1 ADV target prediction identity remains in v2 launcher" >&2
  exit 2
fi

export PROJECT_ROOT CODE_ROOT PYTHON
bash "${TRANSFORMED_LAUNCHER}" "$@"
