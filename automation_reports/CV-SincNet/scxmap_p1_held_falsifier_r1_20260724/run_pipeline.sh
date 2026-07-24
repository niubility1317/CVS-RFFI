#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_ID="scxmap_p1_held_falsifier_r1_20260724"
ROOT="/home/szu2070436088/2510044040/CV-SincNet"
EXPECTED_RUN_ROOT="${ROOT}/runs/${RUN_ID}"
RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
SOURCE_ROOT="${RUN_ROOT}/source"
R8_ROOT="${ROOT}/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8"
ARCHIVE="${R8_ROOT}/output/archive/phase1_singleobs_dual_feature_archive.npz"
MANIFEST="${R8_ROOT}/output/archive/phase1_singleobs_dual_feature_archive.manifest.json"
COVERAGE="${R8_ROOT}/output/coverage_receipt.json"
CHECKPOINT="${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
MODULE_FILE="${SOURCE_ROOT}/code/cvsrffi/scxmap_phase1_held_falsifier.py"
TRANSFORM_FILE="${SOURCE_ROOT}/code/cvsrffi/stage2_scxmap_transform.py"
OUTPUT_ROOT="${RUN_ROOT}/output"
EXIT_FILE="${RUN_ROOT}/pipeline.exit"

die() {
  printf 'SCXMAP_HELD_ERROR=%s\n' "$1" >&2
  exit "$2"
}

write_exit() {
  local status="$?"
  trap - EXIT
  if ! (set -o noclobber; printf '%s\n' "${status}" > "${EXIT_FILE}"); then
    printf 'SCXMAP_HELD_ERROR=exit_receipt_write_failed\n' >&2
    status=74
  fi
  exit "${status}"
}

require_sha256() {
  local expected="$1"
  local path="$2"
  [[ -f "${path}" && ! -L "${path}" ]] || die "missing_or_symlink:${path}" 71
  local actual
  actual="$(sha256sum -- "${path}" | awk '{print $1}')"
  [[ "${actual}" == "${expected}" ]] || die "sha256_drift:${path}:${actual}" 72
}

[[ "${RUN_ROOT}" == "${EXPECTED_RUN_ROOT}" ]] || die "run_root_drift:${RUN_ROOT}" 64
[[ ! -e "${EXIT_FILE}" ]] || die "immutable_exit_receipt_exists" 74
trap write_exit EXIT
[[ -x "${PYTHON}" ]] || die "python_missing:${PYTHON}" 70
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES must bind one runner-selected GPU}"
[[ ! -e "${OUTPUT_ROOT}" ]] || die "immutable_output_exists:${OUTPUT_ROOT}" 73

require_sha256 "ce4edb7badaa1fe39efb324e8ec3f3d7f191f54051918f6028381f529a5df976" "${MODULE_FILE}"
require_sha256 "8298ed9f879715e77805e48d2272a7fa640a758554dec18f6f3e189187626944" "${TRANSFORM_FILE}"
require_sha256 "dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0" "${ARCHIVE}"
require_sha256 "34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4" "${MANIFEST}"
require_sha256 "c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17" "${COVERAGE}"
require_sha256 "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98" "${CHECKPOINT}"

mkdir -p -- "${OUTPUT_ROOT}"
export PYTHONPATH="${SOURCE_ROOT}/code"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

SMOKE="${OUTPUT_ROOT}/real_support_only_smoke.json"
PACKET="${OUTPUT_ROOT}/packet.json"
TRUTH="${OUTPUT_ROOT}/truth.json"
QUERY="${OUTPUT_ROOT}/query.npz"
BUILD_RECEIPT="${OUTPUT_ROOT}/build_receipt.json"
PREDICTION="${OUTPUT_ROOT}/prediction.json"
SCORE="${OUTPUT_ROOT}/score.json"

"${PYTHON}" -m cvsrffi.scxmap_phase1_held_falsifier support-smoke \
  --archive "${ARCHIVE}" \
  --manifest "${MANIFEST}" \
  --coverage "${COVERAGE}" \
  --coverage-sha256 "c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17" \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-sha256 "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98" \
  --output "${SMOKE}"

"${PYTHON}" -m cvsrffi.scxmap_phase1_held_falsifier build \
  --archive "${ARCHIVE}" \
  --manifest "${MANIFEST}" \
  --coverage "${COVERAGE}" \
  --coverage-sha256 "c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17" \
  --packet "${PACKET}" \
  --truth "${TRUTH}" \
  --query "${QUERY}" \
  --build-receipt "${BUILD_RECEIPT}"

BUILD_RECEIPT_SHA256="$(sha256sum -- "${BUILD_RECEIPT}" | awk '{print $1}')"
TRUTH_SHA256="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["truth_sha256"])' "${TRUTH}")"

"${PYTHON}" -m cvsrffi.scxmap_phase1_held_falsifier predict \
  --packet "${PACKET}" \
  --query "${QUERY}" \
  --build-receipt "${BUILD_RECEIPT}" \
  --build-receipt-sha256 "${BUILD_RECEIPT_SHA256}" \
  --output "${PREDICTION}"

PREDICTION_COMMIT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["COMMIT"])' "${PREDICTION}")"

"${PYTHON}" -m cvsrffi.scxmap_phase1_held_falsifier score \
  --packet "${PACKET}" \
  --prediction "${PREDICTION}" \
  --truth "${TRUTH}" \
  --query "${QUERY}" \
  --build-receipt "${BUILD_RECEIPT}" \
  --build-receipt-sha256 "${BUILD_RECEIPT_SHA256}" \
  --truth-sha256 "${TRUTH_SHA256}" \
  --commit "${PREDICTION_COMMIT}" \
  --output "${SCORE}"

"${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); assert len(d["metrics"])==54; assert [x["K"] for x in d["summary_by_K"]]==[1,5,10]; assert d["formal_phase2_eligible"] is False; assert d["bundle_created"] is False; assert d["target25_release_authorized"] is False; assert isinstance(d["proxy_gate_pass"], bool); print("score_rows=54"); print("target25_release_authorized=false"); print("proxy_gate_pass="+str(d["proxy_gate_pass"]).lower())' "${SCORE}"

sha256sum -- "${SMOKE}" "${PACKET}" "${TRUTH}" "${QUERY}" "${BUILD_RECEIPT}" "${PREDICTION}" "${SCORE}" > "${OUTPUT_ROOT}/sha256sums.txt"
printf '%s\n' "SCXMAP_HELD_ARTIFACTS_COMPLETE" > "${OUTPUT_ROOT}/complete.marker"
printf 'SCXMAP_HELD_COMPLETE run_id=%s build_receipt_sha256=%s truth_sha256=%s prediction_commit=%s\n' "${RUN_ID}" "${BUILD_RECEIPT_SHA256}" "${TRUTH_SHA256}" "${PREDICTION_COMMIT}"
