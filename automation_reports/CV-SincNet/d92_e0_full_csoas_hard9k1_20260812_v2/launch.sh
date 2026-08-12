#!/usr/bin/env bash
set -euo pipefail
project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
source_root="$project/runs/d92_csoas_hard9_source_1fab89eb_20260812_v2"
code_root="$source_root/code"
runtime_archive="$source_root/d92_csoas_hard9_runtime_1fab89eb.tar.gz"
context="$project/runs/d131_d92_lite160_qtie_target125_20260804_r3/prepared/target125_context.json"
method_lock="$source_root/configs/stage2_d92_csoas_hard10_v1.json"
output="$project/runs/d92_e0_full_csoas_hard9k1_20260812_v2"
logs="$project/logs/d92_e0_full_csoas_hard9k1_20260812_v2"
require_file(){ test -f "$1" || { printf 'required file missing: %s\n' "$1" >&2; exit 64; }; }
require_file "$runtime_archive"
test "$(sha256sum "$runtime_archive" | awk '{print $1}')" = "de74fe49d8d24432898e44fddfc3c8a9f2f2444b2d70421e7d69d786c9a25d78"
require_file "$method_lock"
test "$(sha256sum "$method_lock" | awk '{print $1}')" = "6fcd29dfab77c99745df336f32425dfdc0a0a0a99469c92766a4751fa92e427e"
require_file "$context"
test "$(sha256sum "$context" | awk '{print $1}')" = "067a6365e9c859161407ab62ba6349d7beb93083f59f78fd7c780c6d8924731f"
test ! -e "$code_root"; test ! -e "$output"; test ! -e "$logs"
tar -xzf "$runtime_archive" -C "$source_root"
for required in \
 "$code_root/cvsrffi/__init__.py" \
 "$code_root/cvsrffi/stage2_d92_cauchy_scatter_oas.py" \
 "$code_root/cvsrffi/stage2_d92_e0d_slim.py" \
 "$code_root/cvsrffi/stage2_d92_e0d_query_evaluation.py" \
 "$code_root/cvsrffi/stage2_d92_csoas_hard10.py" \
 "$code_root/cvsrffi/stage2_d92_csoas_hard10_analysis.py" \
 "$code_root/scripts/probe_d81_ground_nuisance_cauchy_center.py" \
 "$code_root/scripts/probe_d92_registration_balanced_covariance.py" \
 "$code_root/scripts/run_d92_e0d_prediction.py" \
 "$code_root/scripts/score_d92_be_prediction.py" \
 "$code_root/scripts/run_d92_csoas_hard10.py" \
 "$code_root/scripts/analyze_d92_csoas_hard10.py"; do require_file "$required"; done
mkdir -p "$logs"; cd "$code_root"
env PYTHONPATH="$code_root:$project" "$python" -c '
import importlib,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve()
for name in ("cvsrffi.stage2_d92_cauchy_scatter_oas","cvsrffi.stage2_d92_e0d_slim","cvsrffi.stage2_d92_e0d_query_evaluation","cvsrffi.stage2_d92_csoas_hard10","cvsrffi.stage2_d92_csoas_hard10_analysis","scripts.run_d92_csoas_hard10","scripts.analyze_d92_csoas_hard10"):
 p=pathlib.Path(importlib.import_module(name).__file__).resolve()
 if root not in p.parents: raise SystemExit(f"runtime import escaped root: {name}->{p}")
' "$code_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err"
env PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_csoas_hard10.py prepare --context-manifest "$context" --method-lock "$method_lock" --output-root "$output" >"$logs/prepare.out" 2>"$logs/prepare.err"
manifest="$output/matrix_manifest.json"
manifest_sha=$($python -c 'import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$manifest")
env CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_csoas_hard10.py smoke --matrix-manifest "$manifest" --matrix-manifest-sha256 "$manifest_sha" --output-root "$output/smoke" --device cuda:0 --cpu-threads 2 >"$logs/smoke.out" 2>"$logs/smoke.err"
for shard in 0 1 2 3 4 5 6 7; do
 nohup env CUDA_VISIBLE_DEVICES="$shard" PYTHONPATH="$code_root:$project" "$python" -u scripts/run_d92_csoas_hard10.py run-shard --matrix-manifest "$manifest" --matrix-manifest-sha256 "$manifest_sha" --shard-index "$shard" --shard-count 8 --device cuda:0 --cpu-threads 2 >"$logs/shard_${shard}.out" 2>"$logs/shard_${shard}.err" </dev/null &
 printf 'shard=%s gpu=%s pid=%s\n' "$shard" "$shard" "$!"
done
