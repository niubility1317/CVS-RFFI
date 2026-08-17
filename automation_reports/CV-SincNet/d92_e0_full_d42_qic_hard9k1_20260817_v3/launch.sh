#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
run_id=d92_e0_full_d42_qic_hard9k1_20260817_v3
fresh_run_retry=false
source_root="$project/runs/d92_qic_hard9_k1_source_fa75cf8e_20260817_v3"
runtime_archive="$project/runs/d92_qic_hard9_k1_source_fa75cf8e_20260817_v3.tar.gz"
driver="$project/runs/d92_qic_hard9_k1_driver_${run_id}.sh"
output_root="$project/runs/d92_qic_hard9_k1_20260817_v3"
logs_root="$project/logs/${run_id}"
code_root="$source_root/code"
config="$source_root/configs/stage2_d92_qic_hard9_k1_v3.json"
runner="$code_root/scripts/run_d92_qic_hard9_k1.py"
source_manifest="$code_root/D92_QIC_HARD9_K1_SOURCE_MANIFEST.sha256"
archive_sha256=61d424bda548ec04b49a1763b83080a0dff04c11b94d6a707e72d98cd00bddbe
archive_size_bytes=320797
archive_member_count=50
coordinator_id="task3-${run_id}"

require_file() {
  test -f "$1" || {
    printf 'required file missing: %s\n' "$1" >&2
    exit 64
  }
}

script_path=$(readlink -f "${BASH_SOURCE[0]}")
test "$script_path" = "$driver"
require_file "$driver"
require_file "$runtime_archive"
test ! -L "$runtime_archive"
test "$(sha256sum "$runtime_archive" | awk '{print $1}')" = "$archive_sha256"
test "$(stat -c '%s' "$runtime_archive")" = "$archive_size_bytes"
test "$(tar -tzf "$runtime_archive" | wc -l | tr -d ' ')" = "$archive_member_count"
if tar -tzf "$runtime_archive" | grep -Eq '(^/|(^|/)\.\.(\/|$))'; then
  printf '%s\n' 'unsafe archive member' >&2
  exit 65
fi

test ! -e "$source_root"
test ! -e "$output_root"
test ! -e "$logs_root"
mkdir "$source_root"
tar -xzf "$runtime_archive" -C "$source_root"
require_file "$source_manifest"
(cd "$source_root" && sha256sum -c "$source_manifest")

mkdir "$logs_root"
require_file "$config"
require_file "$runner"

env PYTHONPATH="$code_root:$project" "$python" -m py_compile \
  "$code_root/cvsrffi/stage2_d92_qic_hard9_k1.py" \
  "$code_root/cvsrffi/stage2_d92_qic_hard9_k1_analysis.py" \
  "$code_root/cvsrffi/stage2_d92_d42_quantization_intercept_closure.py" \
  "$code_root/scripts/run_d92_qic_hard9_k1.py" \
  "$code_root/scripts/analyze_d92_qic_hard9_k1.py" \
  >"$logs_root/import_closure.out" 2>"$logs_root/import_closure.err"

env PYTHONPATH="$code_root:$project" "$python" -c '
import importlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
names = (
    "cvsrffi.stage2_d92_qic_hard9_k1",
    "cvsrffi.stage2_d92_qic_hard9_k1_analysis",
    "cvsrffi.stage2_d92_d42_quantization_intercept_closure",
    "cvsrffi.stage2_d92_e0d_query_evaluation",
    "cvsrffi.stage2_d92_e0d_slim",
    "cvsrffi.stage2_d92_registration_balanced_covariance",
    "scripts.run_d92_qic_hard9_k1",
    "scripts.analyze_d92_qic_hard9_k1",
    "scripts.run_d92_e0d_prediction",
)
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime import escaped frozen source root: {name} -> {path}")
' "$code_root" >"$logs_root/import_path_check.out" 2>"$logs_root/import_path_check.err"

env PYTHONPATH="$code_root:$project" "$python" -u "$runner" prepare \
  --config "$config" \
  >"$logs_root/prepare.out" 2>"$logs_root/prepare.err"
env PYTHONPATH="$code_root:$project" "$python" -c '
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert value.get("status") == "QIC_HARD9_K1_MATRIX_PREPARED"
assert value.get("runtime_source_verification_mode") == "sha256_only"
assert value.get("e0_resource_source_mode") == "embedded_preregistered_projection"
declared = value.get("e0_resource_fit_audit_declared_sha256")
assert isinstance(declared, dict) and len(declared) == 10
assert all(isinstance(item, str) and len(item) == 64 for item in declared.values())
assert value.get("job_count") == 10
assert value.get("scene_arm_count") == 30
' "$logs_root/prepare.out"

matrix_manifest="$output_root/matrix_manifest.json"
matrix_manifest_sha256=$(sha256sum "$matrix_manifest" | awk '{print $1}')

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$code_root:$project" "$python" -u "$runner" smoke \
  --matrix-manifest "$matrix_manifest" \
  --matrix-manifest-sha256 "$matrix_manifest_sha256" \
  --device cuda:0 \
  --cpu-threads 2 \
  >"$logs_root/smoke.out" 2>"$logs_root/smoke.err"
env PYTHONPATH="$code_root:$project" "$python" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
value = json.loads((root / "smoke" / "smoke_receipt.json").read_text(encoding="utf-8"))
assert value.get("status") == "D92_QIC_HARD9_K1_REAL_CHECKPOINT_TRUTH_FREE_SMOKE_PASS"
assert value.get("truth_open") is False
assert value.get("query_truth_joined_only_after_immutable_predictions") is True
assert value.get("prediction_and_scorer_processes_isolated") is True
assert value.get("e0_resource_source_mode") == "embedded_preregistered_projection"
declared = value.get("e0_resource_fit_audit_declared_sha256")
assert isinstance(declared, str) and len(declared) == 64
for key in ("query_truth_access", "query_fit_access", "query_update_access", "query_selection_access", "query_role_oracle_access", "query_class_quota_access", "query_global_reassignment"):
    assert value.get(key) is False, key
' "$output_root"

for shard in 0 1 2 3 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$code_root:$project" "$python" -u "$runner" run-shard \
    --matrix-manifest "$matrix_manifest" \
    --matrix-manifest-sha256 "$matrix_manifest_sha256" \
    --shard-index "$shard" \
    --shard-count 8 \
    --device cuda:0 \
    --cpu-threads 2 \
    >"$logs_root/shard_${shard}.out" 2>"$logs_root/shard_${shard}.err" \
    </dev/null &
  printf 'shard=%s gpu=%s pid=%s\n' "$shard" "$shard" "$!"
done

shard_status=0
for pid in $(jobs -pr); do
  wait "$pid" || shard_status=1
done
if test -f "$output_root/SYSTEMIC_TECHNICAL_FAILURE_STOP.json"; then
  env PYTHONPATH="$code_root:$project" "$python" -u "$runner" coordinator-stop \
    --output-root "$output_root" \
    --coordinator-id "$coordinator_id" \
    --grace-seconds 1.0 \
    >"$logs_root/coordinator-stop.out" 2>"$logs_root/coordinator-stop.err"
fi
test "$shard_status" = 0

env PYTHONPATH="$code_root:$project" "$python" -c '
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summaries = sorted((root / "summaries").glob("shard_*.json"))
assert len(summaries) == 8
for path in summaries:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value.get("shard_index") in range(8)
    assert value.get("selected_job_count") >= 0
    assert value.get("completed_job_count") >= 0
    assert value.get("failed_job_count") >= 0
' "$output_root"
