#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
source_root="$project/runs/d92_newguard_source_snapshot_20260812_v1"
code_root="$source_root/code"
runtime_archive="$source_root/d92_newguard_runtime_closure_2d2ea755.tar.gz"
context="$project/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json"
method_lock="$source_root/configs/stage2_d92_full_bidirectional_newguard_hard11_v1.json"
output="$project/runs/d92_e0_full_bidirectional_newguard_hard11_20260812_v1"
logs="$project/logs/d92_e0_full_bidirectional_newguard_hard11_20260812_v1"

require_file() {
  test -f "$1" || {
    printf 'required file missing: %s\n' "$1" >&2
    exit 64
  }
}

require_file "$runtime_archive"
test "$(sha256sum "$runtime_archive" | awk '{print $1}')" = \
  "48074423354da375dde3b6488396bde31daadb490f2afc14cb93208469cee607"
require_file "$method_lock"
test "$(sha256sum "$method_lock" | awk '{print $1}')" = \
  "d41b116b2bb7fb8be1fb56512e9e47e7915e94b5fae57776ced9c875ceb5f523"
require_file "$context"
test ! -e "$code_root"
test ! -e "$output"
test ! -e "$logs"

tar -xzf "$runtime_archive" -C "$source_root"
require_file "$code_root/cvsrffi/__init__.py"
require_file "$code_root/cvsrffi/stage2_d92_bidirectional_newguard.py"
require_file "$code_root/cvsrffi/stage2_d92_e0d_slim.py"
require_file "$code_root/cvsrffi/stage2_d92_e0d_query_evaluation.py"
require_file "$code_root/cvsrffi/stage2_d92_newguard_hard11.py"
require_file "$code_root/scripts/probe_d81_ground_nuisance_cauchy_center.py"
require_file "$code_root/scripts/probe_d92_registration_balanced_covariance.py"
require_file "$code_root/scripts/run_d92_e0d_prediction.py"
require_file "$code_root/scripts/score_d92_be_prediction.py"
require_file "$code_root/scripts/run_d92_newguard_hard11.py"
mkdir -p "$logs"
cd "$code_root"

env PYTHONPATH="$code_root:$project" "$python" -c '
import importlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1]).resolve()
names = (
    "cvsrffi.stage2_registration_resource_probe",
    "cvsrffi.stage2_d92_bidirectional_newguard",
    "cvsrffi.stage2_d92_e0d_slim",
    "cvsrffi.stage2_d92_e0d_query_evaluation",
    "cvsrffi.stage2_d92_newguard_hard11",
    "scripts.probe_d81_ground_nuisance_cauchy_center",
    "scripts.probe_d92_registration_balanced_covariance",
    "scripts.run_d92_e0d_prediction",
    "scripts.score_d92_be_prediction",
    "scripts.run_d92_newguard_hard11",
)
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime import escaped frozen source root: {name} -> {path}")
' "$code_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err"

env PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_newguard_hard11.py prepare \
  --context-manifest "$context" \
  --method-lock "$method_lock" \
  --output-root "$output" \
  >"$logs/prepare.out" 2>"$logs/prepare.err"

manifest="$output/matrix_manifest.json"
manifest_sha=$("$python" -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$manifest")

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_newguard_hard11.py smoke \
  --matrix-manifest "$manifest" \
  --matrix-manifest-sha256 "$manifest_sha" \
  --output-root "$output/smoke" \
  --device cuda:0 \
  --cpu-threads 2 \
  >"$logs/smoke.out" 2>"$logs/smoke.err"

for shard in 0 1 2 3 4 5 6 7; do
  nohup env \
    CUDA_VISIBLE_DEVICES="$shard" \
    PYTHONPATH="$code_root:$project" \
    "$python" -u scripts/run_d92_newguard_hard11.py run-shard \
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
