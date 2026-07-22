#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_ID="svrn_qknn_bcrr_k5_scorefix1_b0baa0dc_20260723_070006"
EXPECTED_RUN_ROOT="/home/szu2070436088/2510044040/CV-SincNet/runs/${RUN_ID}"
RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
SOURCE_ROOT="${RUN_ROOT}/source_b0baa0dc"
PARENT_ROOT="/home/szu2070436088/2510044040/CV-SincNet/runs/svrn_qknn_bcrr_k5_held_r2_165ca031_20260723"
CORE_FILE="${SOURCE_ROOT}/code/cvsrffi/stage2_svrn_bcr.py"
HELD_FILE="${SOURCE_ROOT}/code/cvsrffi/svrn_bcr_fixed_held_spike.py"
TEST_FILE="${SOURCE_ROOT}/tests/test_svrn_bcr_fixed_held_spike.py"
OUTPUT_ROOT="${RUN_ROOT}/output"
EXIT_FILE="${RUN_ROOT}/pipeline.exit"
PACKET="${PARENT_ROOT}/output/packet.json"
TRUTH="${PARENT_ROOT}/output/truth.json"
QUERY="${PARENT_ROOT}/output/query.npz"
PREDICTION="${PARENT_ROOT}/output/prediction.json"
SCORE="${OUTPUT_ROOT}/score.json"

die() {
  printf 'SVRN_QKNN_BCRR_K5_SCOREFIX1_ERROR=%s\n' "$1" >&2
  exit "$2"
}

write_exit() {
  local status="$?"
  trap - EXIT
  if ! (set -o noclobber; printf '%s\n' "${status}" > "${EXIT_FILE}"); then
    printf 'SVRN_QKNN_BCRR_K5_SCOREFIX1_ERROR=exit_receipt_write_failed\n' >&2
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

require_sha256 "aa5401306cab361cdb06a41b7c11af3dc8b1aea0a00fe9e75b475c5d283deaf4" "${CORE_FILE}"
require_sha256 "1e71fd2934360a3d3f1082e4a3841bc307334807bc3496455b7c9b29d2366366" "${HELD_FILE}"
require_sha256 "ef0fee40e393b3917e83d0dc955053f599989067b500c55253a7a67cdad2445a" "${TEST_FILE}"
require_sha256 "ef15a8488d40ac70d129db9ac15c796418b4afe5fa64624883eab0f66fd4e95b" "${PACKET}"
require_sha256 "9745068bc5961ebe90f6305c672cacc8ce338d745579e1c97d4ea503cbd06d8" "${TRUTH}"
require_sha256 "be089f42be790a73cd7a95d68cb13956a64735019b10f6cd4ba32199c33c56c9" "${QUERY}"
require_sha256 "0f9313e632884e9987caaa262e2e7d261338bfe9b7f84beae85753571b72e06e" "${PREDICTION}"

mkdir -- "${OUTPUT_ROOT}"
export PYTHONPATH="${SOURCE_ROOT}/code"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

TRUTH_SHA256="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["truth_sha256"])' "${TRUTH}")"
PREDICTION_COMMIT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["COMMIT"])' "${PREDICTION}")"
[[ "${TRUTH_SHA256}" == "637e845fec201627118181a5eb256861b86e76880c101d1b6a5452563cce64b4" ]] || die "truth_internal_sha_drift:${TRUTH_SHA256}" 72
[[ "${PREDICTION_COMMIT}" == "2524a1aa291cb05ed055625c496f8abc12fc692b57736070334b65ce1c68211a" ]] || die "prediction_commit_drift:${PREDICTION_COMMIT}" 72

"${PYTHON}" -m cvsrffi.svrn_bcr_fixed_held_spike score \
  --packet "${PACKET}" \
  --prediction "${PREDICTION}" \
  --truth "${TRUTH}" \
  --truth-sha256 "${TRUTH_SHA256}" \
  --commit "${PREDICTION_COMMIT}" \
  --output "${SCORE}"

"${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); m=d["metrics"]; assert d["candidate_revision"]=="SVRN-qKNN-BCRR/r3-held"; assert d["COMMIT"]=="2524a1aa291cb05ed055625c496f8abc12fc692b57736070334b65ce1c68211a"; assert d["truth_sha256"]=="637e845fec201627118181a5eb256861b86e76880c101d1b6a5452563cce64b4"; assert len(m)==72; assert len({r["row_id"] for r in m})==18; assert {r["arm"] for r in m}=={"M0","M_DA","M_OTHER","M_JOINT"}; print("parent_prediction_slices=18"); print("score_metric_rows=72"); print("verdict="+d["decision"]["verdict"])' "${SCORE}"

sha256sum -- "${PACKET}" "${TRUTH}" "${QUERY}" "${PREDICTION}" > "${OUTPUT_ROOT}/parent_sha256sums.txt"
sha256sum -- "${SCORE}" > "${OUTPUT_ROOT}/sha256sums.txt"
printf '%s\n' "SVRN_QKNN_BCRR_K5_SCOREFIX1_ARTIFACTS_COMPLETE" > "${OUTPUT_ROOT}/complete.marker"
printf 'SVRN_QKNN_BCRR_K5_SCOREFIX1_COMPLETE run_id=%s truth_sha256=%s prediction_commit=%s\n' "${RUN_ID}" "${TRUTH_SHA256}" "${PREDICTION_COMMIT}"
