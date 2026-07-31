#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

readonly RUN_ROOT="/home/szu2070436088/2510044040/CV-SincNet/runs/d105_phase1_sourceheld_a0bdbba6_20260731_r6"
readonly SOURCE_ROOT="${RUN_ROOT}/source"
readonly PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
readonly BUILDER="${SOURCE_ROOT}/code/scripts/build_d105_phase1_bundle.py"
readonly RUNTIME_MANIFEST="${SOURCE_ROOT}/configs/d105_candidate_runtime_manifest_20260731.json"
readonly METHOD_LOCK="${SOURCE_ROOT}/configs/d105_candidate_method_lock_20260731.json"
readonly REVOCATION_MANIFEST="${RUN_ROOT}/input/d105_d102_revocation_manifest.json"
readonly REVOCATION_SIGNATURE="${RUN_ROOT}/input/d105_d102_revocation_manifest.ed25519"
readonly CACHE_SET="/home/szu2070436088/2510044040/CV-SincNet/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json"
readonly SELECTION_SALT="/home/szu2070436088/2510044040/CV-SincNet/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json"
readonly CHECKPOINT="/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
readonly REFERENCE_DUAL="/home/szu2070436088/2510044040/CV-SincNet/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8/output/archive/phase1_singleobs_dual_feature_archive.npz"
readonly TAP_ROOT="${RUN_ROOT}/output/strict_tap"
readonly HELD_ROOT="${RUN_ROOT}/output/source_held"
readonly PREDICTIONS="${HELD_ROOT}/source_held_predictions.json"
readonly TRUTH_OPEN="${HELD_ROOT}/source_held_truth_open_receipt.json"
readonly SCORES="${HELD_ROOT}/source_held_scores.json"
readonly GATE="${HELD_ROOT}/derived_source_held_gate.json"
readonly COMPONENT="${RUN_ROOT}/output/component"
readonly EXIT_FILE="${RUN_ROOT}/logs/pipeline_stage1.exit"

export PYTHONPATH="${SOURCE_ROOT}/code"

record_exit() {
  local rc=$?
  trap - EXIT
  local temporary="${EXIT_FILE}.tmp.$$"
  printf '%s\n' "${rc}" > "${temporary}"
  chmod 0444 "${temporary}"
  mv -n "${temporary}" "${EXIT_FILE}"
  exit "${rc}"
}
trap record_exit EXIT

for required in \
  "${BUILDER}" \
  "${RUNTIME_MANIFEST}" \
  "${METHOD_LOCK}" \
  "${REVOCATION_MANIFEST}" \
  "${REVOCATION_SIGNATURE}" \
  "${CACHE_SET}" \
  "${SELECTION_SALT}" \
  "${CHECKPOINT}" \
  "${REFERENCE_DUAL}"; do
  [[ -f "${required}" && ! -L "${required}" ]]
done

for output in \
  "${TAP_ROOT}" \
  "${HELD_ROOT}" \
  "${COMPONENT}" \
  "${EXIT_FILE}"; do
  [[ ! -e "${output}" && ! -L "${output}" ]]
done

mkdir -p "${RUN_ROOT}/output" "${HELD_ROOT}"

"${PYTHON}" "${BUILDER}" tap-cache \
  --cache-set "${CACHE_SET}" \
  --cache-set-sha256 125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74 \
  --selection-salt-receipt "${SELECTION_SALT}" \
  --selection-salt-receipt-sha256 38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0 \
  --checkpoint "${CHECKPOINT}" \
  --checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --candidate-runtime-manifest "${RUNTIME_MANIFEST}" \
  --candidate-method-lock "${METHOD_LOCK}" \
  --d102-revocation-manifest "${REVOCATION_MANIFEST}" \
  --d102-revocation-signature "${REVOCATION_SIGNATURE}" \
  --reference-dual-archive "${REFERENCE_DUAL}" \
  --reference-dual-archive-sha256 dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0 \
  --output-dir "${TAP_ROOT}" \
  --device cuda:0 \
  --batch-size 128

"${PYTHON}" "${BUILDER}" predict-source-held \
  --strict-tap-archive "${TAP_ROOT}/d105_phase1_strict_tap.npz" \
  --strict-tap-receipt "${TAP_ROOT}/d105_phase1_strict_tap_receipt.json" \
  --candidate-runtime-manifest "${RUNTIME_MANIFEST}" \
  --candidate-method-lock "${METHOD_LOCK}" \
  --output-prediction-manifest "${PREDICTIONS}"

"${PYTHON}" "${BUILDER}" open-truth \
  --strict-tap-archive "${TAP_ROOT}/d105_phase1_strict_tap.npz" \
  --strict-tap-receipt "${TAP_ROOT}/d105_phase1_strict_tap_receipt.json" \
  --candidate-runtime-manifest "${RUNTIME_MANIFEST}" \
  --candidate-method-lock "${METHOD_LOCK}" \
  --source-held-prediction-manifest "${PREDICTIONS}" \
  --output-receipt "${TRUTH_OPEN}"

"${PYTHON}" "${BUILDER}" score-source-held \
  --strict-tap-archive "${TAP_ROOT}/d105_phase1_strict_tap.npz" \
  --strict-tap-receipt "${TAP_ROOT}/d105_phase1_strict_tap_receipt.json" \
  --candidate-runtime-manifest "${RUNTIME_MANIFEST}" \
  --candidate-method-lock "${METHOD_LOCK}" \
  --source-held-prediction-manifest "${PREDICTIONS}" \
  --source-held-truth-open-receipt "${TRUTH_OPEN}" \
  --output-score-artifact "${SCORES}"

"${PYTHON}" "${BUILDER}" derive-gate \
  --strict-tap-archive "${TAP_ROOT}/d105_phase1_strict_tap.npz" \
  --strict-tap-receipt "${TAP_ROOT}/d105_phase1_strict_tap_receipt.json" \
  --candidate-runtime-manifest "${RUNTIME_MANIFEST}" \
  --candidate-method-lock "${METHOD_LOCK}" \
  --source-held-prediction-manifest "${PREDICTIONS}" \
  --source-held-truth-open-receipt "${TRUTH_OPEN}" \
  --source-held-score-artifact "${SCORES}" \
  --output-receipt "${GATE}"

"${PYTHON}" "${BUILDER}" build \
  --strict-tap-archive "${TAP_ROOT}/d105_phase1_strict_tap.npz" \
  --strict-tap-receipt "${TAP_ROOT}/d105_phase1_strict_tap_receipt.json" \
  --candidate-runtime-manifest "${RUNTIME_MANIFEST}" \
  --candidate-method-lock "${METHOD_LOCK}" \
  --source-held-prediction-manifest "${PREDICTIONS}" \
  --source-held-truth-open-receipt "${TRUTH_OPEN}" \
  --source-held-score-artifact "${SCORES}" \
  --d102-revocation-manifest "${REVOCATION_MANIFEST}" \
  --d102-revocation-signature "${REVOCATION_SIGNATURE}" \
  --output-dir "${COMPONENT}"
