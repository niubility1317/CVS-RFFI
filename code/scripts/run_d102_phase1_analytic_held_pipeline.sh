#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_ID="d102_rb_metabias4_phase1_analytic_held_20260724_r2"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_ID"
SOURCE_ROOT="$RUN_ROOT/source"
OUTPUT_ROOT="$RUN_ROOT/output"
LOG_ROOT="$RUN_ROOT/logs"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
CHECKPOINT="$PROJECT_ROOT/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
CACHE_SET="$PROJECT_ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json"
SELECTION_SALT="$PROJECT_ROOT/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json"
DUAL_ARCHIVE="$PROJECT_ROOT/runs/rchm_bpp_p1_dual_archive_rawsha_r8_de9a6476_20260723_r8/output/archive/phase1_singleobs_dual_feature_archive.npz"
METHOD_LOCK="$SOURCE_ROOT/docs/D102_RB_METABIAS4_PHASE1_ANALYTIC_HELD_LOCK.json"
STATUS_PATH="$LOG_ROOT/pipeline.exit"
PID_PATH="$LOG_ROOT/pipeline.pid"

CHECKPOINT_SHA="2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
CACHE_SET_SHA="125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74"
SELECTION_SALT_SHA="38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0"
DUAL_ARCHIVE_SHA="dd2a2b0c8ab1a1d8edbeed81e78ffb79c253240998a9ac2404b75699f4ca68d0"
RUNTIME_SHA="e1b21bee74941dfb550b67698a75f485937bc39431ed7859baaa20d44a4899f3"
METHOD_LOCK_SHA="9640267c2913e452a89be39e1b41e8b19d3371499afbed1efe8c9e3b7ad0e52f"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  printf 'CUDA_VISIBLE_DEVICES must be assigned by the runner\n' >&2
  exit 70
}
[[ -d "$SOURCE_ROOT" && -d "$LOG_ROOT" ]] || {
  printf 'precreated immutable source/log root missing\n' >&2
  exit 75
}
[[ ! -e "$OUTPUT_ROOT" ]] || {
  printf 'immutable output already exists: %s\n' "$OUTPUT_ROOT" >&2
  exit 73
}
[[ ! -e "$STATUS_PATH" && ! -e "$PID_PATH" ]] || {
  printf 'immutable status/PID marker already exists\n' >&2
  exit 74
}

export PYTHONPATH="$SOURCE_ROOT/code:$SOURCE_ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ! -e "$STATUS_PATH" ]]; then
    local temporary="$LOG_ROOT/pipeline.exit.tmp.$$"
    printf '%s\n' "$rc" > "$temporary"
    mv "$temporary" "$STATUS_PATH"
  fi
  exit "$rc"
}
trap on_exit EXIT

check_sha() {
  local path=$1
  local expected=$2
  local observed
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'missing regular input: %s\n' "$path" >&2
    return 71
  }
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || {
    printf 'sha256 drift: %s observed=%s expected=%s\n' \
      "$path" "$observed" "$expected" >&2
    return 72
  }
}

check_sha "$CHECKPOINT" "$CHECKPOINT_SHA"
check_sha "$CACHE_SET" "$CACHE_SET_SHA"
check_sha "$SELECTION_SALT" "$SELECTION_SALT_SHA"
check_sha "$DUAL_ARCHIVE" "$DUAL_ARCHIVE_SHA"
check_sha "$METHOD_LOCK" "$METHOD_LOCK_SHA"
check_sha "$SOURCE_ROOT/code/cvsrffi/phase1_rb_metabias4_bundle.py" \
  "7e2eb67a592a94de8be1091c29b6df796d8122ffe76b3a7424e985d694ae8c5f"
check_sha "$SOURCE_ROOT/code/cvsrffi/rb_metabias4_phase1_held_falsifier.py" \
  "a40b6979cea27c3b5f089ce3c57afae92798727da45fa70408081ad80fe92df5"
check_sha "$SOURCE_ROOT/code/scripts/build_phase1_rb_metabias4_bundle.py" \
  "9b0857a7e4df19321c6ba36ca230fea040ae1a96749a06dd419361fc458a1f1b"
check_sha "$SOURCE_ROOT/code/scripts/run_rb_metabias4_phase1_held.py" \
  "8e401086c135fbb5deed0f9e53b73df60e98ce3bc6c05b302ee6fbabfda3c3b8"
check_sha "$SOURCE_ROOT/code/scripts/export_phase1_jp4_tap_archive.py" \
  "196deaf7c8ebb70a021fb21da8788a85ba8c02fbd334fa7b460127341b06daaa"

temporary="$LOG_ROOT/pipeline.pid.tmp.$$"
printf '%s\n' "$BASHPID" > "$temporary"
mv "$temporary" "$PID_PATH"
mkdir "$OUTPUT_ROOT"
mkdir "$OUTPUT_ROOT/tap" "$OUTPUT_ROOT/held" "$OUTPUT_ROOT/bundle"

"$PYTHON" -u "$SOURCE_ROOT/code/scripts/export_phase1_jp4_tap_archive.py" \
  --cache-set "$CACHE_SET" \
  --cache-set-sha256 "$CACHE_SET_SHA" \
  --selection-salt-receipt "$SELECTION_SALT" \
  --selection-salt-receipt-sha256 "$SELECTION_SALT_SHA" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --reference-archive "$DUAL_ARCHIVE" \
  --reference-archive-sha256 "$DUAL_ARCHIVE_SHA" \
  --output-dir "$OUTPUT_ROOT/tap" \
  --device cuda:0 \
  --batch-size 256

TAP_ARCHIVE="$OUTPUT_ROOT/tap/phase1_jp4_tap_archive.npz"
TAP_SHA=$(sha256sum "$TAP_ARCHIVE" | awk '{print $1}')

"$PYTHON" -u "$SOURCE_ROOT/code/scripts/run_rb_metabias4_phase1_held.py" \
  --tap-archive "$TAP_ARCHIVE" \
  --tap-archive-sha256 "$TAP_SHA" \
  --dual-archive "$DUAL_ARCHIVE" \
  --dual-archive-sha256 "$DUAL_ARCHIVE_SHA" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --runtime-sha256 "$RUNTIME_SHA" \
  --method-lock-sha256 "$METHOD_LOCK_SHA" \
  --output "$OUTPUT_ROOT/held/phase1_analytic_held.json"

"$PYTHON" -u "$SOURCE_ROOT/code/scripts/build_phase1_rb_metabias4_bundle.py" \
  --tap-archive "$TAP_ARCHIVE" \
  --tap-archive-sha256 "$TAP_SHA" \
  --dual-archive "$DUAL_ARCHIVE" \
  --dual-archive-sha256 "$DUAL_ARCHIVE_SHA" \
  --checkpoint-sha256 "$CHECKPOINT_SHA" \
  --runtime-sha256 "$RUNTIME_SHA" \
  --method-lock-sha256 "$METHOD_LOCK_SHA" \
  --output-dir "$OUTPUT_ROOT/bundle"

sha256sum \
  "$OUTPUT_ROOT/tap/phase1_jp4_tap_archive.npz" \
  "$OUTPUT_ROOT/tap/phase1_jp4_tap_archive.manifest.json" \
  "$OUTPUT_ROOT/held/phase1_analytic_held.json" \
  "$OUTPUT_ROOT/held/phase1_analytic_held.json.sha256" \
  "$OUTPUT_ROOT/bundle/phase1_rb_metabias4_bundle.npz" \
  "$OUTPUT_ROOT/bundle/phase1_rb_metabias4_bundle.manifest.json" \
  "$OUTPUT_ROOT/bundle/phase1_rb_metabias4_bundle.seal.sha256" \
  > "$OUTPUT_ROOT/sha256sums.txt"

printf 'D102_PHASE1_ANALYTIC_HELD_ARTIFACTS_COMPLETE run_id=%s output=%s\n' \
  "$RUN_ID" "$OUTPUT_ROOT"
