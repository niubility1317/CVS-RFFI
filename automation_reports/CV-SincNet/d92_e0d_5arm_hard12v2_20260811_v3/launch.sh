#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
source_root="$project/runs/d92_e0d_source_snapshot_20260811_v3"
code_root="$source_root/code"
runtime_archive="$source_root/d92_e0d_runtime_closure_053b4347.tar.gz"
context="$project/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json"
method_lock="$source_root/stage2_d92_e0d_5arm_hard12v2_v1.json"
smoke="$project/runs/d92_e0d_truthfree_smoke_20260811_v3"
output="$project/runs/d92_e0d_5arm_hard12v2_20260811_v3"
logs="$project/logs/d92_e0d_5arm_hard12v2_20260811_v3"

require_file() {
  test -f "$1" || {
    printf 'required file missing: %s\n' "$1" >&2
    exit 64
  }
}

require_file "$runtime_archive"
test "$(sha256sum "$runtime_archive" | awk '{print $1}')" = \
  "933df4742c328c6d97669f7d040f05ef4f1d0d5d7559504ecaaf154c9b2aac93"
require_file "$method_lock"
test "$(sha256sum "$method_lock" | awk '{print $1}')" = \
  "b80f967e1fc070a730a7b193f691036339930af022682fe2fca81c2e4d229f86"
test ! -e "$code_root"
test ! -e "$smoke"
test ! -e "$output"
test ! -e "$logs"
tar -xzf "$runtime_archive" -C "$source_root"
require_file "$code_root/cvsrffi/__init__.py"
require_file "$code_root/scripts/probe_d81_ground_nuisance_cauchy_center.py"
require_file "$code_root/scripts/probe_d92_registration_balanced_covariance.py"
mkdir -p "$logs"
cd "$code_root"

env PYTHONPATH="$code_root:$project" "$python" -c '
import importlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
names = (
    "cvsrffi.stage2_registration_resource_probe",
    "cvsrffi.stage2_d92_e0d_slim",
    "cvsrffi.stage2_d92_e0d_query_evaluation",
    "cvsrffi.stage2_d92_e0d_hard12",
    "scripts.probe_d81_ground_nuisance_cauchy_center",
    "scripts.probe_d92_registration_balanced_covariance",
    "scripts.run_d92_e0d_prediction",
    "scripts.score_d92_be_prediction",
    "scripts.run_d92_e0d_hard12v2",
)
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime import escaped frozen source root: {name} -> {path}")
' "$code_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err"

env PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_e0d_hard12v2.py prepare \
  --context-manifest "$context" \
  --method-lock "$method_lock" \
  --output-root "$output" \
  >"$logs/prepare.out" 2>"$logs/prepare.err"

manifest="$output/matrix_manifest.json"
manifest_sha=$("$python" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$manifest")

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_e0d_hard12v2.py smoke \
  --matrix-manifest "$manifest" \
  --matrix-manifest-sha256 "$manifest_sha" \
  --output-root "$smoke" \
  --device cuda:0 \
  --cpu-threads 2 \
  >"$logs/smoke.out" 2>"$logs/smoke.err"

for shard in 0 1 2 3 4 5 6 7; do
  nohup env \
    CUDA_VISIBLE_DEVICES="$shard" \
    PYTHONPATH="$code_root:$project" \
    "$python" -u scripts/run_d92_e0d_hard12v2.py run-shard \
      --matrix-manifest "$manifest" \
      --matrix-manifest-sha256 "$manifest_sha" \
      --shard-index "$shard" \
      --shard-count 8 \
      --device cuda:0 \
      --cpu-threads 2 \
      >"$logs/shard_${shard}.out" \
      2>"$logs/shard_${shard}.err" \
      </dev/null &
  printf 'shard=%s gpu=%s pid=%s\n' "$shard" "$shard" "$!"
done

