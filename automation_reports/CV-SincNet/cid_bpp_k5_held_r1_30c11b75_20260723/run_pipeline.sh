#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_ID="cid_bpp_k5_held_r1_30c11b75_20260723"
EXPECTED_RUN_ROOT="/home/szu2070436088/2510044040/CV-SincNet/runs/${RUN_ID}"
RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
SOURCE_ROOT="${RUN_ROOT}/source_30c11b75"
R8_ROOT="/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8"
ARCHIVE="${R8_ROOT}/output/archive/phase1_singleobs_dual_feature_archive.npz"
MANIFEST="${R8_ROOT}/output/archive/phase1_singleobs_dual_feature_archive.manifest.json"
COVERAGE="${R8_ROOT}/output/coverage_receipt.json"
PARITY="${R8_ROOT}/output/runtime/base_parity_receipt.json"
SELECTOR_FILE="${SOURCE_ROOT}/code/cvsrffi/cid_bpp_phase1_nested_lodo.py"
MODULE_FILE="${SOURCE_ROOT}/code/cvsrffi/cid_bpp_fixed_held_spike.py"
TEST_FILE="${SOURCE_ROOT}/tests/test_cid_bpp_fixed_held_spike.py"
OUTPUT_ROOT="${RUN_ROOT}/output"
EXIT_FILE="${RUN_ROOT}/pipeline.exit"

die() {
  printf 'CID_BPP_HELD_ERROR=%s\n' "$1" >&2
  exit "$2"
}

write_exit() {
  local status="$?"
  trap - EXIT
  if ! (set -o noclobber; printf '%s\n' "${status}" > "${EXIT_FILE}"); then
    printf 'CID_BPP_HELD_ERROR=exit_receipt_write_failed\n' >&2
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

require_sha256 "05958df26be904884b19b2fbbcdcff5c61a78612a5e064f997dfd611901c9a59" "${SELECTOR_FILE}"
require_sha256 "515927c376608a3c94247270c475c3c6b4ee137e694f88bea591b5cb7ed2aed7" "${MODULE_FILE}"
require_sha256 "a10d541fe2d40af84abf881db927ea0c2ace31ee5d626345478a17db12ffb67b" "${TEST_FILE}"
require_sha256 "dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0" "${ARCHIVE}"
require_sha256 "34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4" "${MANIFEST}"
require_sha256 "c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17" "${COVERAGE}"
require_sha256 "b93219c40b79be8ecdf8c0a51d77710d8119f8899331ae7e2518b77adfeac60b" "${PARITY}"

mkdir -p -- "${OUTPUT_ROOT}"
export PYTHONPATH="${SOURCE_ROOT}/code"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PACKET="${OUTPUT_ROOT}/packet.json"
TRUTH="${OUTPUT_ROOT}/truth.json"
QUERY="${OUTPUT_ROOT}/query.npz"
PREDICTION="${OUTPUT_ROOT}/prediction.json"
SCORE="${OUTPUT_ROOT}/score.json"

"${PYTHON}" -m cvsrffi.cid_bpp_fixed_held_spike build \
  --archive "${ARCHIVE}" \
  --manifest "${MANIFEST}" \
  --coverage "${COVERAGE}" \
  --coverage-sha256 "c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17" \
  --packet "${PACKET}" \
  --truth "${TRUTH}" \
  --query "${QUERY}"

TRUTH_SHA256="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["truth_sha256"])' "${TRUTH}")"

"${PYTHON}" -m cvsrffi.cid_bpp_fixed_held_spike predict \
  --packet "${PACKET}" \
  --query "${QUERY}" \
  --output "${PREDICTION}"

PREDICTION_COMMIT="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["COMMIT"])' "${PREDICTION}")"

"${PYTHON}" -m cvsrffi.cid_bpp_fixed_held_spike score \
  --packet "${PACKET}" \
  --prediction "${PREDICTION}" \
  --truth "${TRUTH}" \
  --truth-sha256 "${TRUTH_SHA256}" \
  --commit "${PREDICTION_COMMIT}" \
  --output "${SCORE}"

"${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1], encoding="utf-8")); m=d["metrics"]; assert len(m)==72; assert len({r["row_id"] for r in m})==18; assert {r["arm"] for r in m}=={"M0","M_DA","M_HEAD","M_JOINT"}; print("score_metric_rows=72"); print("prediction_slices=18")' "${SCORE}"

sha256sum -- "${PACKET}" "${TRUTH}" "${QUERY}" "${PREDICTION}" "${SCORE}" > "${OUTPUT_ROOT}/sha256sums.txt"
printf '%s\n' "CID_BPP_HELD_ARTIFACTS_COMPLETE" > "${OUTPUT_ROOT}/complete.marker"
printf 'CID_BPP_HELD_COMPLETE run_id=%s truth_sha256=%s prediction_commit=%s\n' "${RUN_ID}" "${TRUTH_SHA256}" "${PREDICTION_COMMIT}"
