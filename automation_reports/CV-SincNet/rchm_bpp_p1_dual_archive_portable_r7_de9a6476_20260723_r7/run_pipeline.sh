#!/usr/bin/env bash
set -euo pipefail
umask 077

RUN_ID="rchm_bpp_p1_dual_archive_portable_r7_de9a6476_20260723_r7"
PROJECT_ROOT="/home/szu2070436088/2510044040/CV-SincNet"
RUN_ROOT="$PROJECT_ROOT/runs/$RUN_ID"
SOURCE_ROOT="$RUN_ROOT/source_de9a6476"
OUTPUT_ROOT="$RUN_ROOT/output"
LOG_ROOT="$RUN_ROOT/logs"
PYTHON="/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python"
CHECKPOINT="$PROJECT_ROOT/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
ADAPTER="$PROJECT_ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/effective8_adapter_fp16.pt"
CACHE_SET="$PROJECT_ROOT/runs/qknn_ground_effective8_r16_e12_leoonly_20260715_v14/phase1_caches/source_validation/cache_set.json"
SELECTION_SALT="$PROJECT_ROOT/runs/d99_d100_phase1_export_7932dbf9_cuda0_20260721_r4/input/d99_d100_phase1_selection_salt.json"
STATUS_PATH="$LOG_ROOT/pipeline.exit"
PID_PATH="$LOG_ROOT/pipeline.pid"

[[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || { printf 'CUDA_VISIBLE_DEVICES must be assigned by the runner\n' >&2; exit 70; }
[[ -d "$LOG_ROOT" ]] || { printf 'precreated log directory is missing: %s\n' "$LOG_ROOT" >&2; exit 75; }
export PYTHONPATH="$SOURCE_ROOT/code:$SOURCE_ROOT"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export CVSRFFI_CPU_THREADS=2 CVSRFFI_CPU_INTEROP_THREADS=1
on_exit() {
  local rc=$?
  trap - EXIT
  if [[ ! -e "$STATUS_PATH" ]]; then
    local tmp="$LOG_ROOT/pipeline.exit.tmp.$$"
    printf '%s\n' "$rc" > "$tmp"; mv "$tmp" "$STATUS_PATH"
  fi
  exit "$rc"
}
trap on_exit EXIT
check_sha() {
  local path=$1 expected=$2 observed
  [[ -f "$path" && ! -L "$path" ]] || { printf 'missing regular input: %s\n' "$path" >&2; return 71; }
  observed=$(sha256sum "$path" | awk '{print $1}')
  [[ "$observed" == "$expected" ]] || { printf 'sha256 drift: %s observed=%s expected=%s\n' "$path" "$observed" "$expected" >&2; return 72; }
}
[[ ! -e "$OUTPUT_ROOT" ]] || { printf 'immutable output already exists: %s\n' "$OUTPUT_ROOT" >&2; exit 73; }
[[ ! -e "$STATUS_PATH" ]] || { printf 'immutable exit marker already exists: %s\n' "$STATUS_PATH" >&2; exit 74; }
[[ ! -e "$PID_PATH" ]] || { printf 'immutable pid marker already exists: %s\n' "$PID_PATH" >&2; exit 76; }
PID_TMP="$LOG_ROOT/pipeline.pid.tmp.$$"; printf '%s\n' "$BASHPID" > "$PID_TMP"; mv "$PID_TMP" "$PID_PATH"

check_sha "$CHECKPOINT" "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98"
check_sha "$ADAPTER" "9e43d6decc3b05dcc526c0897127d959a768573a674236de657a12f619c6931b"
check_sha "$CACHE_SET" "125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74"
check_sha "$SELECTION_SALT" "38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0"
check_sha "$SOURCE_ROOT/code/scripts/export_phase1_singleobs_dual_feature_archive.py" "31a6a464f470ae9bdb6cbc8814581ff6c73403d5c99b497a224b3f783831fe64"
check_sha "$SOURCE_ROOT/code/scripts/export_phase1_singleobs_feature_archive.py" "81209f0761f47fe264437740897adad95cab3b7160e4af05fb42a5ce92196687"
check_sha "$SOURCE_ROOT/code/scripts/export_adv3b02_dual_feature_torchscript.py" "e2cbc0ce19402e8c665489fb2b13bb63f988c14e138cbc39dd20f7c9e2b12090"
check_sha "$SOURCE_ROOT/code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py" "0a3e80b226997b353c577de94aa8c8e92fb25f5bf13d8b81c9bc27f448ef284b"
check_sha "$SOURCE_ROOT/code/cvsrffi/dual_feature_forward.py" "1694c29b9a94142b8ba1bb6e5ff540b56ab60ef3fd155747bb0584de5142cc56"
check_sha "$SOURCE_ROOT/code/cvsrffi/leo_weak_cache.py" "656b5851de412310cb15751883341a6c1e7934a94759455cf9dad54f094a5a86"
mkdir -p "$OUTPUT_ROOT/runtime"

"$PYTHON" -u "$SOURCE_ROOT/code/scripts/export_adv3b02_dual_feature_torchscript.py" \
  --checkpoint "$CHECKPOINT" --adapter-state "$ADAPTER" --input-len 256 \
  --base-runtime-out "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" \
  --candidate-runtime-out "$OUTPUT_ROOT/runtime/candidate_dual_runtime.pt" \
  --export-receipt-out "$OUTPUT_ROOT/runtime/dual_export_receipt.json" \
  --device cuda:0 --parity-seed 20260721 --parity-rows 8 --runtime-batch-size 256 --max-abs-tolerance 1e-5
EXPORT_SHA=$(sha256sum "$OUTPUT_ROOT/runtime/dual_export_receipt.json" | awk '{print $1}')
BASE_RUNTIME_SHA=$(sha256sum "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" | awk '{print $1}')
"$PYTHON" -u "$SOURCE_ROOT/code/scripts/verify_adv3b02_dual_runtime_checkpoint_parity.py" \
  --checkpoint "$CHECKPOINT" --adapter-state "$ADAPTER" --runtime "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" \
  --export-receipt "$OUTPUT_ROOT/runtime/dual_export_receipt.json" --expected-export-receipt-sha256 "$EXPORT_SHA" --runtime-role base \
  --receipt-out "$OUTPUT_ROOT/runtime/base_parity_receipt.json" --vector-audit-out "$OUTPUT_ROOT/runtime/base_parity_vector.json" \
  --input-len 256 --parity-seed 20260721 --parity-rows 8 --device cuda:0 --max-abs-tolerance 1e-5
PARITY_SHA=$(sha256sum "$OUTPUT_ROOT/runtime/base_parity_receipt.json" | awk '{print $1}')
"$PYTHON" -u "$SOURCE_ROOT/code/scripts/export_phase1_singleobs_dual_feature_archive.py" \
  --cache-set "$CACHE_SET" --cache-set-sha256 "125bb312972fd82edab9b1566a1ebddcd077b9a00c5255a55da22afb453b8d74" \
  --selection-salt-receipt "$SELECTION_SALT" --selection-salt-receipt-sha256 "38ffbdda293cd2eead31c481237a459581c862572041ea472b38391a1b4bddb0" \
  --runtime "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" --runtime-sha256 "$BASE_RUNTIME_SHA" --runtime-role base \
  --export-receipt "$OUTPUT_ROOT/runtime/dual_export_receipt.json" --export-receipt-sha256 "$EXPORT_SHA" \
  --parity-receipt "$OUTPUT_ROOT/runtime/base_parity_receipt.json" --parity-receipt-sha256 "$PARITY_SHA" \
  --class-ids "14-10,14-7,20-15,20-19,6-15,8-20" --output-dir "$OUTPUT_ROOT/archive" --device cuda:0 --batch-size 256

"$PYTHON" - "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.npz" "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.manifest.json" "$OUTPUT_ROOT/coverage_receipt.json" <<'PY'
from collections import Counter
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
a,m,o=map(Path,sys.argv[1:])
if o.exists(): raise FileExistsError(f"refusing to overwrite coverage receipt: {o}")
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for b in iter(lambda:f.read(1048576),b""): h.update(b)
 return h.hexdigest()
with np.load(a,allow_pickle=False) as z:
 labels=np.asarray(z["labels"]).astype(str); receivers=np.asarray(z["receiver_ids"]).astype(str); days=np.asarray(z["day_ids"]).astype(str)
 physical=np.asarray(z["physical_ids"]).astype(str); scenes=np.asarray(z["scenario_names"]).astype(str); classes=np.asarray(z["class_ids"]).astype(str); observations=np.asarray(z["observation_ids"]).astype(str)
n=len(labels)
if not all(len(x)==n for x in (receivers,days,physical,scenes,observations)): raise ValueError("metadata row count drift")
if len(set(physical.tolist()))!=n or len(set(observations.tolist()))!=n: raise ValueError("physical/observation uniqueness drift")
rv,dv,cv,sv=sorted(set(receivers.tolist())),sorted(set(days.tolist())),classes.tolist(),sorted(set(scenes.tolist()))
cc={f"{r}|{d}|{c}":int(np.sum((receivers==r)&(days==d)&(labels==c))) for r in rv for d in dv for c in cv}; vals=list(cc.values())
ec=["14-10","14-7","20-15","20-19","6-15","8-20"]; es=["leo_clear_weak","leo_low_elev_weak","leo_rain_weak"]
if n!=8400: raise ValueError(f"pre-registered row_count mismatch: {n} != 8400")
if len(set(physical.tolist()))!=8400 or len(set(observations.tolist()))!=8400: raise ValueError("pre-registered physical/observation unique count mismatch")
if cv!=ec or sorted(set(labels.tolist()))!=ec: raise ValueError("pre-registered class registry mismatch")
if len(rv)!=7 or len(dv)!=4 or sv!=es: raise ValueError("pre-registered receiver/day/scenario mismatch")
if len(vals)!=168 or any(x==0 for x in vals): raise ValueError("pre-registered receiver-day-class coverage mismatch")
if min(vals)<=10: raise ValueError("receiver-day-class coverage cannot leave at least one query after K10 support")
p={"schema":"cvs.phase1.singleobs_dual_feature_coverage_receipt.v1","status":"DESCRIPTIVE_ONLY_NO_HELD_FOLD_DECISION","artifact_stage":"phase1_offline_before_target_access","archive_sha256":sha(a),"manifest_sha256":sha(m),"metadata_arrays_read":["labels","receiver_ids","day_ids","physical_ids","scenario_names","class_ids","observation_ids"],"feature_arrays_read":[],"row_count":n,"physical_id_unique_count":len(set(physical.tolist())),"observation_id_unique_count":len(set(observations.tolist())),"class_ids":cv,"receiver_ids":rv,"day_ids":dv,"scenario_names":sv,"counts_by_class":dict(sorted(Counter(labels.tolist()).items())),"counts_by_receiver":dict(sorted(Counter(receivers.tolist()).items())),"counts_by_day":dict(sorted(Counter(days.tolist()).items())),"counts_by_scenario":dict(sorted(Counter(scenes.tolist()).items())),"counts_by_receiver_day_class":cc,"receiver_day_class_cell_count":len(vals),"receiver_day_class_zero_cell_count":sum(x==0 for x in vals),"receiver_day_class_min_count":min(vals),"receiver_day_class_max_count":max(vals),"pre_registered_coverage_gate_passed":True,"k_values_described_only":[1,5,10],"min_rows_remaining_after_support_by_k":{str(k):min(vals)-k for k in (1,5,10)},"target_access":False,"query_access":False,"held_fold_selected":False}
with o.open("xb") as f:
 b=(json.dumps(p,ensure_ascii=True,allow_nan=False,sort_keys=True,indent=2)+"\n").encode("utf-8"); f.write(b);f.flush();os.fsync(f.fileno())
PY
sha256sum "$OUTPUT_ROOT/runtime/base_dual_runtime.pt" "$OUTPUT_ROOT/runtime/candidate_dual_runtime.pt" "$OUTPUT_ROOT/runtime/dual_export_receipt.json" "$OUTPUT_ROOT/runtime/base_parity_receipt.json" "$OUTPUT_ROOT/runtime/base_parity_vector.json" "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.npz" "$OUTPUT_ROOT/archive/phase1_singleobs_dual_feature_archive.manifest.json" "$OUTPUT_ROOT/coverage_receipt.json" > "$OUTPUT_ROOT/sha256sums.txt"
printf 'PIPELINE_ARTIFACTS_COMPLETE run_id=%s output=%s\n' "$RUN_ID" "$OUTPUT_ROOT"
