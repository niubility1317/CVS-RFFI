#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
source_root="$project/runs/d92_tpce_source_snapshot_ecae572d_20260812_v1"
code_root="$source_root/code"
runtime_archive="$source_root/d92_tpce_runtime_closure_ecae572d_r2.tar.gz"
context="$project/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json"
method_lock="$source_root/configs/stage2_d92_full_d42_tpce_hard11_v1.json"
output="$project/runs/d92_e0_full_d42_tpce_hard11_20260812_v1"
logs="$project/logs/d92_e0_full_d42_tpce_hard11_20260812_v1"

require_file() {
  test -f "$1" || { printf 'required file missing: %s\n' "$1" >&2; exit 64; }
}

require_file "$runtime_archive"
test "$(sha256sum "$runtime_archive" | awk '{print $1}')" = "a527062a64be9b68307164b77f793e25dea9c6c786cf056730c0ec84ef9abb14"
require_file "$method_lock"
test "$(sha256sum "$method_lock" | awk '{print $1}')" = "58dabf7ed4510c74aa2beff4031a2bbe745be940d2dc1b8361300ecf07f7f23c"
require_file "$context"
test ! -e "$code_root"
test ! -e "$output"
test ! -e "$logs"

tar -xzf "$runtime_archive" -C "$source_root"
for required in \
  "$code_root/cvsrffi/__init__.py" \
  "$code_root/cvsrffi/stage2_d92_d42_tail_pair_code_exchange.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_slim.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_query_evaluation.py" \
  "$code_root/cvsrffi/stage2_d92_tpce_hard11.py" \
  "$code_root/cvsrffi/stage2_d92_tpce_hard11_analysis.py" \
  "$code_root/scripts/probe_d81_ground_nuisance_cauchy_center.py" \
  "$code_root/scripts/probe_d92_registration_balanced_covariance.py" \
  "$code_root/scripts/run_d92_e0d_prediction.py" \
  "$code_root/scripts/score_d92_be_prediction.py" \
  "$code_root/scripts/run_d92_tpce_hard11.py" \
  "$code_root/scripts/analyze_d92_tpce_hard11.py"; do
  require_file "$required"
done
mkdir -p "$logs"
cd "$code_root"

env PYTHONPATH="$code_root:$project" "$python" -c '
import importlib, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
names = (
    "cvsrffi.stage2_registration_resource_probe",
    "cvsrffi.stage2_d92_d42_tail_pair_code_exchange",
    "cvsrffi.stage2_d92_e0d_slim",
    "cvsrffi.stage2_d92_e0d_query_evaluation",
    "cvsrffi.stage2_d92_tpce_hard11",
    "cvsrffi.stage2_d92_tpce_hard11_analysis",
    "scripts.probe_d81_ground_nuisance_cauchy_center",
    "scripts.probe_d92_registration_balanced_covariance",
    "scripts.run_d92_e0d_prediction",
    "scripts.score_d92_be_prediction",
    "scripts.run_d92_tpce_hard11",
    "scripts.analyze_d92_tpce_hard11",
)
for name in names:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime import escaped frozen source root: {name} -> {path}")
' "$code_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err"

env PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_tpce_hard11.py prepare \
  --context-manifest "$context" --method-lock "$method_lock" --output-root "$output" \
  >"$logs/prepare.out" 2>"$logs/prepare.err"
manifest="$output/matrix_manifest.json"
manifest_sha=$($python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$manifest")

env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_tpce_hard11.py smoke \
  --matrix-manifest "$manifest" --matrix-manifest-sha256 "$manifest_sha" --output-root "$output/smoke" \
  --device cuda:0 --cpu-threads 2 >"$logs/smoke.out" 2>"$logs/smoke.err"

for shard in 0 1 2 3 4 5 6 7; do
  nohup env CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_tpce_hard11.py run-shard \
    --matrix-manifest "$manifest" --matrix-manifest-sha256 "$manifest_sha" --shard-index "$shard" \
    --shard-count 8 --device cuda:0 --cpu-threads 2 >"$logs/shard_${shard}.out" 2>"$logs/shard_${shard}.err" </dev/null &
  printf 'shard=%s gpu=%s pid=%s\n' "$shard" "$shard" "$!"
done
