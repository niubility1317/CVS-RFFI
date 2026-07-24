#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_ID="scxmap_p1_held_falsifier_r1_20260724"
ROOT="/home/szu2070436088/2510044040/CV-SincNet"
EXPECTED_RUN_ROOT="${ROOT}/runs/${RUN_ID}"
RUN_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
SOURCE_ROOT="${RUN_ROOT}/source"
RELEASE_RECEIPT="${RUN_ROOT}/release_receipt.json"
R8_ROOT="${ROOT}/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8"
ARCHIVE="${R8_ROOT}/output/archive/phase1_singleobs_dual_feature_archive.npz"
MANIFEST="${R8_ROOT}/output/archive/phase1_singleobs_dual_feature_archive.manifest.json"
COVERAGE="${R8_ROOT}/output/coverage_receipt.json"
CHECKPOINT="${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
MODULE_FILE="${SOURCE_ROOT}/code/cvsrffi/scxmap_phase1_held_falsifier.py"
TRANSFORM_FILE="${SOURCE_ROOT}/code/cvsrffi/stage2_scxmap_transform.py"
R2_FILE="${SOURCE_ROOT}/code/cvsrffi/r2a_fixed_held_four_arm.py"
QKNN_FILE="${SOURCE_ROOT}/code/cvsrffi/stage2_zid_student_t_qknn.py"
EXPORTER_FILE="${SOURCE_ROOT}/code/scripts/export_phase1_singleobs_dual_feature_archive.py"
BASELINE_FILE="${SOURCE_ROOT}/code/baseline_origin_sat_view.py"
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
: "${RELEASE_RECEIPT_SHA256:?RELEASE_RECEIPT_SHA256 must freeze the local handoff receipt}"
[[ "${CUDA_VISIBLE_DEVICES}" =~ ^[0-7]$ ]] || die "single_gpu_binding_drift:${CUDA_VISIBLE_DEVICES}" 65
[[ ! -e "${OUTPUT_ROOT}" ]] || die "immutable_output_exists:${OUTPUT_ROOT}" 73

require_sha256 "c54383a5e8df4ce353ccaeacf3232eace6b85c34e33c405725f13fddac3f22d0" "${MODULE_FILE}"
require_sha256 "cdf3f67c24631c1e3023b490826af1c1571412d05a0894e608b261cd89ccc247" "${TRANSFORM_FILE}"
require_sha256 "51e5d187805ed5f58d7088431e9f99d878fd5687fbecc08cd9140e51963e2bc8" "${R2_FILE}"
require_sha256 "19d25bf311c3a4f32ff38bd74ae03205e71bf5b44feaead8a134fa8502fac297" "${QKNN_FILE}"
require_sha256 "dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0" "${ARCHIVE}"
require_sha256 "34213331d20594dceface61680ab0fea8ffc40ee72d7e13c844763c70fef26d4" "${MANIFEST}"
require_sha256 "c6e25ebeaed32b577e3321e78cd569acff934a7c804d0cb621b26e68f26d0c17" "${COVERAGE}"
require_sha256 "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98" "${CHECKPOINT}"
require_sha256 "${RELEASE_RECEIPT_SHA256}" "${RELEASE_RECEIPT}"

"${PYTHON}" -c 'import hashlib,json,os,pathlib,re,sys
receipt_path=pathlib.Path(sys.argv[1]).resolve()
run_root=pathlib.Path(sys.argv[2]).resolve()
receipt=json.loads(receipt_path.read_text(encoding="utf-8"))
expected={"schema","run_id","release_commit","source_archive","files"}
assert set(receipt)==expected
assert receipt["schema"]=="cvs.scxmap-held.release-receipt.v1"
assert receipt["run_id"]=="scxmap_p1_held_falsifier_r1_20260724"
assert re.fullmatch(r"[0-9a-f]{40}",receipt["release_commit"])
archive=receipt["source_archive"]
assert set(archive)=={"relative_path","sha256","size_bytes"}
assert re.fullmatch(r"[0-9a-f]{64}",archive["sha256"])
assert isinstance(archive["size_bytes"],int) and archive["size_bytes"]>0
files=receipt["files"]
assert isinstance(files,dict) and len(files)>=7
def checked(relative,expected_sha,expected_size=None):
    assert isinstance(relative,str) and relative and not relative.startswith(("/", "\\"))
    path=(run_root/relative).resolve()
    assert os.path.commonpath((str(run_root),str(path)))==str(run_root)
    assert path.is_file() and not path.is_symlink()
    data=path.read_bytes()
    assert hashlib.sha256(data).hexdigest()==expected_sha
    if expected_size is not None: assert len(data)==expected_size
checked(archive["relative_path"],archive["sha256"],archive["size_bytes"])
for relative,sha in files.items():
    assert re.fullmatch(r"[0-9a-f]{64}",sha)
    checked(relative,sha)
print("release_commit="+receipt["release_commit"])
print("release_file_count="+str(len(files)))' "${RELEASE_RECEIPT}" "${RUN_ROOT}" || die "release_receipt_validation_failed" 75

[[ -d "${SOURCE_ROOT}" && ! -L "${SOURCE_ROOT}" ]] || die "source_root_drift" 76
if find "${SOURCE_ROOT}" -type f -perm /222 -print -quit | grep -q .; then
  die "source_tree_is_writable" 76
fi
require_sha256 "31a6a464f470ae9bdb6cbc8814581ff6c73403d5c99b497a224b3f783831fe64" "${EXPORTER_FILE}"
require_sha256 "5df3db9184f627ed8d3f5076cfdca401f0f32d20e1f0ddb95787549dc8180eec" "${BASELINE_FILE}"

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
