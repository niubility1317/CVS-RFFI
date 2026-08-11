#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
source_root="$project/runs/d92_be_source_snapshot_20260811_v1"
context="$project/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json"
method_lock="$source_root/configs/stage2_d92_be_2x2_hard12_v1.json"
smoke="$project/runs/d92_be_truthfree_smoke_20260811_v1"
output="$project/runs/d92_be_2x2_hard12_20260811_v1"
logs="$project/logs/d92_be_2x2_hard12_20260811_v1"

test -f "$source_root/cvsrffi/__init__.py"
test -f "$source_root/scripts/probe_d81_ground_nuisance_cauchy_center.py"
test -f "$source_root/scripts/probe_d92_registration_balanced_covariance.py"
test ! -e "$smoke"
test ! -e "$output"
test ! -e "$logs"
mkdir -p "$logs"
cd "$source_root"

env PYTHONPATH="$source_root:$project" "$python" -c '
import importlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
names = (
    "cvsrffi.stage2_registration_resource_probe",
    "cvsrffi.stage2_d92_be_slim",
    "cvsrffi.stage2_d92_be_query_evaluation",
    "cvsrffi.stage2_d92_be_hard12",
    "scripts.probe_d81_ground_nuisance_cauchy_center",
    "scripts.probe_d92_registration_balanced_covariance",
    "scripts.run_d92_be_prediction",
    "scripts.score_d92_be_prediction",
    "scripts.run_d92_be_hard12",
)
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime import escaped frozen source root: {name} -> {path}")
' "$source_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err"

env PYTHONPATH="$source_root:$project" "$python" -u scripts/run_d92_be_hard12.py prepare \
  --context-manifest "$context" \
  --method-lock "$method_lock" \
  --output-root "$output" \
  >"$logs/prepare.out" 2>"$logs/prepare.err"

manifest="$output/matrix_manifest.json"
manifest_sha=$("$python" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$manifest")

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$source_root:$project" "$python" -u scripts/run_d92_be_hard12.py smoke \
  --matrix-manifest "$manifest" \
  --matrix-manifest-sha256 "$manifest_sha" \
  --output-root "$smoke" \
  --device cuda:0 \
  --cpu-threads 2 \
  >"$logs/smoke.out" 2>"$logs/smoke.err"

for shard in 0 1 2 3 4 5 6 7; do
  nohup env \
    CUDA_VISIBLE_DEVICES="$shard" \
    PYTHONPATH="$source_root:$project" \
    "$python" -u scripts/run_d92_be_hard12.py run-shard \
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
