#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
source_root="$project/runs/d92_tcra_safe_v2_hard9_source_86a26b24_20260812_v1"
code_root="$source_root/code"
runtime_archive="$source_root/d92_tcra_safe_v2_hard9_runtime_86a26b24.tar.gz"
context="$project/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json"
method_lock="$source_root/configs/stage2_d92_tcra_safe_v2_hard10_v1.json"
output="$project/runs/d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v1"
logs="$project/logs/d92_e0_full_d42_tcra_safe_v2_hard9k1_20260812_v1"

require_file() { test -f "$1" || { printf 'required file missing: %s\n' "$1" >&2; exit 64; }; }
require_file "$runtime_archive"
test "$(sha256sum "$runtime_archive" | awk '{print $1}')" = "3338d85ef75d680d9c3e0feb46af5434ef0596b8fa3d0f6ae19ab8dc28f0ecff"
require_file "$method_lock"
test "$(sha256sum "$method_lock" | awk '{print $1}')" = "9740ebd8f7368ea73bf8bdfb1ff57735e7407f89dab7b51a834988c4d6f9f13e"
require_file "$context"
test ! -e "$code_root"
test ! -e "$output"
test ! -e "$logs"

tar -xzf "$runtime_archive" -C "$source_root"
for required in \
  "$code_root/cvsrffi/__init__.py" \
  "$code_root/cvsrffi/stage2_d92_d42_tail_class_row_ascent.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_slim.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_query_evaluation.py" \
  "$code_root/cvsrffi/stage2_d92_tcra_hard10.py" \
  "$code_root/scripts/probe_d81_ground_nuisance_cauchy_center.py" \
  "$code_root/scripts/probe_d92_registration_balanced_covariance.py" \
  "$code_root/scripts/run_d92_e0d_prediction.py" \
  "$code_root/scripts/score_d92_be_prediction.py" \
  "$code_root/scripts/run_d92_tcra_hard10.py"; do
  require_file "$required"
done

mkdir -p "$logs"
cd "$code_root"
env PYTHONPATH="$code_root:$project" "$python" -c '
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
for name in (
    "cvsrffi.stage2_d92_d42_tail_class_row_ascent",
    "cvsrffi.stage2_d92_e0d_slim",
    "cvsrffi.stage2_d92_e0d_query_evaluation",
    "cvsrffi.stage2_d92_tcra_hard10",
    "scripts.run_d92_tcra_hard10",
):
    path = pathlib.Path(importlib.import_module(name).__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime import escaped frozen source root: {name} -> {path}")
' "$code_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err"

env PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_tcra_hard10.py prepare \
  --context-manifest "$context" --method-lock "$method_lock" --output-root "$output" \
  >"$logs/prepare.out" 2>"$logs/prepare.err"
manifest="$output/matrix_manifest.json"
manifest_sha=$($python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$manifest")

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_tcra_hard10.py smoke \
  --matrix-manifest "$manifest" --matrix-manifest-sha256 "$manifest_sha" --output-root "$output/smoke" \
  --device cuda:0 --cpu-threads 2 >"$logs/smoke.out" 2>"$logs/smoke.err"

for shard in 0 1 2 3 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_tcra_hard10.py run-shard \
    --matrix-manifest "$manifest" --matrix-manifest-sha256 "$manifest_sha" --shard-index "$shard" \
    --shard-count 8 --device cuda:0 --cpu-threads 2 >"$logs/shard_${shard}.out" 2>"$logs/shard_${shard}.err" </dev/null &
  printf 'shard=%s gpu=%s pid=%s\n' "$shard" "$shard" "$!"
done
