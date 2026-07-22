#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_ID="rchm_bpp_p1_dual_archive_9ca1a59a_20260722_r1"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_ID"
SOURCE_ROOT="$RUN_ROOT/source_9ca1a59a"
OUTPUT_ROOT="$RUN_ROOT/output"
LOG_ROOT="$RUN_ROOT/logs"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
CHECKPOINT="$PROJECT_ROOT/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
ADAPTER="$PROJECT_ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/effective8_adapter_fp16.pt"
CACHE_SET="$PROJECT_ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json"
SELECTION_SALT="$PROJECT_ROOT/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json"
STATUS_PATH="$LOG_ROOT/pipeline.exit"
PID_PATH="$LOG_ROOT/pipeline.pid"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || {
  printf 'CUDA_VISIBLE_DEVICES must be assigned by the runner\n' >&2
  exit 70
}
[[ -d "$LOG_ROOT" ]] || {
  printf 'precreated log directory is missing: %s\n' "$LOG_ROOT" >&2
  exit 75
}
export PYTHONPATH="$SOURCE_ROOT/code:$SOURCE_ROOT"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export CVSRFFI_CPU_THREADS=2
export CVSRFFI_CPU_INTEROP_THREADS=1

on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ! -e "$STATUS_PATH" ]]; then
    local tmp="$LOG_ROOT/pipeline.exit.tmp.$$"
    printf '%s\n' "$rc" > "$tmp"
    mv "$tmp" "$STATUS_PATH"
  fi
  exit "$rc"
}
trap on_exit EXIT

check_sha() {
  local path=$1
  local expected=$2
  [[ -f "$path" && ! -L "$path" ]] || {
    printf 'missing regular input: %s\n' "$path" >&2
    return 71
  }
  local observed
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || {
    printf 'sha256 drift: %s observed=%s expected=%s\n' "$path" "$observed" "$expected" >&2
    return 72
  }
}

[[ ! -e "$OUTPUT_ROOT" ]] || {
  printf 'immutable output already exists: %s\n' "$OUTPUT_ROOT" >&2
  exit 73
}
[[ ! -e "$STATUS_PATH" ]] || {
  printf 'immutable exit marker already exists: %s\n' "$STATUS_PATH" >&2
  exit 74
}
[[ ! -e "$PID_PATH" ]] || {
  printf 'immutable pid marker already exists: %s\n' "$PID_PATH" >&2
  exit 76
}
PID_TMP="$LOG_ROOT/pipeline.pid.tmp.$$"
printf '%s\n' "$BASHPID" > "$PID_TMP"
mv "$PID_TMP" "$PID_PATH"

check_sha "$CHECKPOINT" "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
check_sha "$ADAPTER" "9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b"
check_sha "$CACHE_SET" "125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74"
check_sha "$SELECTION_SALT" "38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0"
check_sha "$SOURCE_ROOT/code/scripts/export_phase1_singleobs_dual_feature_archive.py" "44ceff9d1afb0c6a1832ef0d09bfb19f24ce1190481387ddf984b3ef7bdc8b4b"
check_sha "$SOURCE_ROOT/code/scripts/export_phase1_singleobs_feature_archive.py" "81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687"
check_sha "$SOURCE_ROOT/code/scripts/export_adv3b02_dual_feature_torchscript.py" "6c637520ca0e5877740a6b9a45dafb7d52ad0d881da4282538e32524c865ba7a"
check_sha "$SOURCE_ROOT/code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py" "606d0e27a826f917e4e28171775e2cb0b8f8edfd68b50e7ab5ba554be175d069"
check_sha "$SOURCE_ROOT/code/cvsrffi/dual_feature_forward.py" "1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56"
check_sha "$SOURCE_ROOT/code/cvsrffi/leo_weak_cache.py" "656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86"

mkdir -p "$OUTPUT_ROOT/runtime"

"$PYTHON" -u "$SOURCE_ROOT/code/scripts/export_adv3b02_dual_feature_torchscript.py" \
  --checkpoint "$CHECKPOINT" \
  --adapter-state "$ADAPTER" \
  --input-len 256 \
  --base-runtime-out "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" \
  --candidate-runtime-out "$OUTPUT_ROOT/runtime/candidate_dual_runtime.pt" \
  --export-receipt-out "$OUTPUT_ROOT/runtime/dual_export_receipt.json" \
  --device cuda:0 \
  --parity-seed 20260721 \
  --parity-rows 8 \
  --runtime-batch-size 256 \
  --max-abs-tolerance 1e-4

EXPORT_SHA=$(sha256sum "$OUTPUT_ROOT/runtime/dual_export_receipt.json" | awk '{print $1}')
BASE_RUNTIME_SHA=$(sha256sum "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" | awk '{print $1}')

"$PYTHON" -u "$SOURCE_ROOT/code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py" \
  --checkpoint "$CHECKPOINT" \
  --adapter-state "$ADAPTER" \
  --runtime "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" \
  --export-receipt "$OUTPUT_ROOT/runtime/dual_export_receipt.json" \
  --expected-export-receipt-sha256 "$EXPORT_SHA" \
  --runtime-role base \
  --receipt-out "$OUTPUT_ROOT/runtime/base_parity_receipt.json" \
  --vector-audit-out "$OUTPUT_ROOT/runtime/base_parity_vector.json" \
  --input-len 256 \
  --parity-seed 20260721 \
  --parity-rows 8 \
  --device cuda:0 \
  --max-abs-tolerance 1e-5

PARITY_SHA=$(sha256sum "$OUTPUT_ROOT/runtime/base_parity_receipt.json" | awk '{print $1}')

"$PYTHON" -u "$SOURCE_ROOT/code/scripts/export_phase1_singleobs_dual_feature_archive.py" \
  --cache-set "$CACHE_SET" \
  --cache-set-sha256 "125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74" \
  --selection-salt-receipt "$SELECTION_SALT" \
  --selection-salt-receipt-sha256 "38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0" \
  --runtime "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" \
  --runtime-sha256 "$BASE_RUNTIME_SHA" \
  --runtime-role base \
  --export-receipt "$OUTPUT_ROOT/runtime/dual_export_receipt.json" \
  --export-receipt-sha256 "$EXPORT_SHA" \
  --parity-receipt "$OUTPUT_ROOT/runtime/base_parity_receipt.json" \
  --parity-receipt-sha256 "$PARITY_SHA" \
  --class-ids "14-10,14-7,20-15,20-19,6-15,8-20" \
  --output-dir "$OUTPUT_ROOT/archive" \
  --device cuda:0 \
  --batch-size 256

"$PYTHON" - "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.npz" \
  "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.manifest.json" \
  "$OUTPUT_ROOT/coverage_receipt.json" <<'PY'
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


archive_path = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
if output_path.exists():
    raise FileExistsError(f"refusing to overwrite coverage receipt: {output_path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


with np.load(archive_path, allow_pickle=False) as archive:
    labels = np.asarray(archive["labels"]).astype(str)
    receivers = np.asarray(archive["receiver_ids"]).astype(str)
    days = np.asarray(archive["day_ids"]).astype(str)
    physical = np.asarray(archive["physical_ids"]).astype(str)
    scenarios = np.asarray(archive["scenario_names"]).astype(str)
    classes = np.asarray(archive["class_ids"]).astype(str)
    observations = np.asarray(archive["observation_ids"]).astype(str)

row_count = len(labels)
if not all(len(value) == row_count for value in (receivers, days, physical, scenarios, observations)):
    raise ValueError("metadata row count drift")
if len(set(physical.tolist())) != row_count or len(set(observations.tolist())) != row_count:
    raise ValueError("physical/observation uniqueness drift")

receiver_values = sorted(set(receivers.tolist()))
day_values = sorted(set(days.tolist()))
class_values = classes.tolist()
scene_values = sorted(set(scenarios.tolist()))
cell_counts = {}
for receiver in receiver_values:
    for day in day_values:
        for class_id in class_values:
            key = f"{receiver}|{day}|{class_id}"
            cell_counts[key] = int(
                np.sum((receivers == receiver) & (days == day) & (labels == class_id))
            )
cell_values = list(cell_counts.values())
expected_class_values = ["14-10", "14-7", "20-15", "20-19", "6-15", "8-20"]
expected_scene_values = [
    "leo_clear_weak",
    "leo_low_elev_weak",
    "leo_rain_weak",
]
if row_count != 8400:
    raise ValueError(f"pre-registered row_count mismatch: {row_count} != 8400")
if len(set(physical.tolist())) != 8400 or len(set(observations.tolist())) != 8400:
    raise ValueError("pre-registered physical/observation unique count mismatch")
if class_values != expected_class_values:
    raise ValueError(
        f"pre-registered class registry mismatch: {class_values!r}"
    )
if sorted(set(labels.tolist())) != expected_class_values:
    raise ValueError("archive labels do not match the pre-registered class registry")
if len(receiver_values) != 7 or len(day_values) != 4:
    raise ValueError(
        "pre-registered receiver/day count mismatch: "
        f"{len(receiver_values)}/{len(day_values)} != 7/4"
    )
if scene_values != expected_scene_values:
    raise ValueError(
        f"pre-registered scenario registry mismatch: {scene_values!r}"
    )
if len(cell_values) != 168:
    raise ValueError(
        f"pre-registered receiver-day-class cell count mismatch: "
        f"{len(cell_values)} != 168"
    )
if any(value == 0 for value in cell_values):
    raise ValueError("pre-registered receiver-day-class coverage contains a zero cell")
if min(cell_values) <= 10:
    raise ValueError(
        "receiver-day-class coverage cannot leave at least one query after K10 support"
    )
payload = {
    "schema": "cvs.phase1.singleobs_dual_feature_coverage_receipt.v1",
    "status": "DESCRIPTIVE_ONLY_NO_HELD_FOLD_DECISION",
    "artifact_stage": "phase1_offline_before_target_access",
    "archive_sha256": sha256_file(archive_path),
    "manifest_sha256": sha256_file(manifest_path),
    "metadata_arrays_read": [
        "labels",
        "receiver_ids",
        "day_ids",
        "physical_ids",
        "scenario_names",
        "class_ids",
        "observation_ids",
    ],
    "feature_arrays_read": [],
    "row_count": row_count,
    "physical_id_unique_count": len(set(physical.tolist())),
    "observation_id_unique_count": len(set(observations.tolist())),
    "class_ids": class_values,
    "receiver_ids": receiver_values,
    "day_ids": day_values,
    "scenario_names": scene_values,
    "counts_by_class": dict(sorted(Counter(labels.tolist()).items())),
    "counts_by_receiver": dict(sorted(Counter(receivers.tolist()).items())),
    "counts_by_day": dict(sorted(Counter(days.tolist()).items())),
    "counts_by_scenario": dict(sorted(Counter(scenarios.tolist()).items())),
    "counts_by_receiver_day_class": cell_counts,
    "receiver_day_class_cell_count": len(cell_values),
    "receiver_day_class_zero_cell_count": sum(value == 0 for value in cell_values),
    "receiver_day_class_min_count": min(cell_values),
    "receiver_day_class_max_count": max(cell_values),
    "pre_registered_coverage_gate_passed": True,
    "k_values_described_only": [1, 5, 10],
    "min_rows_remaining_after_support_by_k": {
        str(k): min(cell_values) - k for k in (1, 5, 10)
    },
    "target_access": False,
    "query_access": False,
    "held_fold_selected": False,
}
serialized = (
    json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
    + "\n"
).encode("utf-8")
with output_path.open("xb") as handle:
    handle.write(serialized)
    handle.flush()
    os.fsync(handle.fileno())
PY

sha256sum \
  "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" \
  "$OUTPUT_ROOT/runtime/candidate_dual_runtime.pt" \
  "$OUTPUT_ROOT/runtime/dual_export_receipt.json" \
  "$OUTPUT_ROOT/runtime/base_parity_receipt.json" \
  "$OUTPUT_ROOT/runtime/base_parity_vector.json" \
  "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.npz" \
  "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.manifest.json" \
  "$OUTPUT_ROOT/coverage_receipt.json" \
  > "$OUTPUT_ROOT/sha256sums.txt"

printf 'PIPELINE_ARTIFACTS_COMPLETE run_id=%s output=%s\n' "$RUN_ID" "$OUTPUT_ROOT"
