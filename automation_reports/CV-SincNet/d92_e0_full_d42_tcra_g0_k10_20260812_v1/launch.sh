#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
run_id=d92_e0_full_d42_tcra_g0_k10_20260812_v1
source_root="$project/runs/d92_tcra_g0_source_b2934f62_20260812_v1"
code_root="$source_root/code"
runtime_archive="$source_root/d92_tcra_g0_runtime_b2934f62.tar.gz"
output="$project/runs/$run_id"
logs="$project/logs/$run_id"
source_job="$project/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5"
ground="$project/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
ground_sha=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c

require_file() { test -f "$1" || { printf 'required file missing: %s\n' "$1" >&2; exit 64; }; }
require_dir() { test -d "$1" || { printf 'required directory missing: %s\n' "$1" >&2; exit 64; }; }
require_sha() { test "$(sha256sum "$1" | awk '{print $1}')" = "$2" || { printf 'sha mismatch: %s\n' "$1" >&2; exit 65; }; }

require_file "$runtime_archive"
require_sha "$runtime_archive" aeed383cb79892cc4c84c9a02bf9bc543503962dcffc10bbb3299c4bd94bb973
require_dir "$source_job/offline/predictor/before/enrollment_only"
require_dir "$source_job/offline/predictor/before/apply_only_staging"
require_dir "$source_job/offline/predictor/after/enrollment_only"
require_dir "$source_job/offline/predictor/after/apply_only_staging"
require_file "$source_job/offline/seals/before_enrollment.seal.json"
require_sha "$source_job/offline/seals/before_enrollment.seal.json" e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9
require_file "$source_job/apply_seals/before_apply.seal.json"
require_sha "$source_job/apply_seals/before_apply.seal.json" 736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473
require_file "$source_job/offline/seals/after_enrollment.seal.json"
require_sha "$source_job/offline/seals/after_enrollment.seal.json" 2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286
require_file "$source_job/apply_seals/after_apply.seal.json"
require_sha "$source_job/apply_seals/after_apply.seal.json" afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a
require_file "$ground/manifest.json"
require_sha "$ground/manifest.json" "$ground_sha"
test ! -e "$code_root"
test ! -e "$output"
test ! -e "$logs"

tar -xzf "$runtime_archive" -C "$source_root"
for required in \
  "$code_root/cvsrffi/__init__.py" \
  "$code_root/cvsrffi/stage2_d92_d42_tail_class_row_ascent.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_slim.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_query_evaluation.py" \
  "$code_root/scripts/probe_d81_ground_nuisance_cauchy_center.py" \
  "$code_root/scripts/probe_d92_registration_balanced_covariance.py" \
  "$code_root/scripts/run_d92_e0d_prediction.py"; do
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
    "scripts.run_d92_e0d_prediction",
):
    path = pathlib.Path(importlib.import_module(name).__file__).resolve()
    if root not in path.parents:
        raise SystemExit(f"runtime import escaped frozen source root: {name} -> {path}")
' "$code_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err"

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$code_root:$project" \
  "$python" -u scripts/run_d92_e0d_prediction.py \
  --before-enrollment-package-root "$source_job/offline/predictor/before/enrollment_only" \
  --before-enrollment-seal-path "$source_job/offline/seals/before_enrollment.seal.json" \
  --before-enrollment-seal-sha256 e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9 \
  --before-apply-package-root "$source_job/offline/predictor/before/apply_only_staging" \
  --before-apply-seal-path "$source_job/apply_seals/before_apply.seal.json" \
  --before-apply-seal-sha256 736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473 \
  --after-enrollment-package-root "$source_job/offline/predictor/after/enrollment_only" \
  --after-enrollment-seal-path "$source_job/offline/seals/after_enrollment.seal.json" \
  --after-enrollment-seal-sha256 2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286 \
  --after-apply-package-root "$source_job/offline/predictor/after/apply_only_staging" \
  --after-apply-seal-path "$source_job/apply_seals/after_apply.seal.json" \
  --after-apply-seal-sha256 afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a \
  --ground-component-dir "$ground" \
  --ground-manifest-sha256 "$ground_sha" \
  --arm E0_FULL_D42_TAIL_CLASS_ROW_ASCENT \
  --output-root "$output" \
  --device cuda:0 >"$logs/prediction.out" 2>"$logs/prediction.err"

"$python" - "$output/diag/after/fit_audit.json" "$logs/g0_validation.json" <<'PY'
import json, pathlib, sys

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
rows = json.loads(source.read_text(encoding="utf-8"))
expected = {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
if not isinstance(rows, list) or len(rows) != 3 or {row.get("scenario") for row in rows} != expected:
    raise SystemExit("TCRA G0 scenario closure failed")
walls = []
summary = []
for row in rows:
    prefix = "d92_e0d_tcra_"
    tolerance = float(row[prefix + "guard_tolerance"])
    old_gains = [float(value) for value in row[prefix + "old_tail_gain_by_class"]]
    checks = (
        row[prefix + "active"] is True,
        row[prefix + "fallback_active"] is False,
        row[prefix + "fallback_reason"] is None,
        row[prefix + "state_postprocess_mode"] == "d42_tcra",
        row[prefix + "direct_state_publish"] is True,
        row[prefix + "requantize_call_count"] == 0,
        row[prefix + "e0_state_sha256"] != row[prefix + "final_state_sha256"],
        row[prefix + "modified_state_field_names"] == ["coef2_qint8"],
        row[prefix + "support_guard_pass"] is True,
        len(old_gains) == 6 and all(value > tolerance for value in old_gains),
        float(row[prefix + "pooled_new_cross_tail_gain"]) > tolerance,
        float(row[prefix + "pooled_new_allclass_tail_gain"]) >= -tolerance,
        float(row[prefix + "old_to_new_hinge_delta"]) <= tolerance,
        float(row[prefix + "new_to_old_hinge_delta"]) <= tolerance,
        row[prefix + "component_fit_count"] == 0,
        row["after_total_component_fit_count"] == 2,
        row["after_actual_component_inventory"]["actual_component_fit_count"] == 1,
        row[prefix + "persistent_state_bytes_delta"] == 0,
        row[prefix + "query_rows_used"] == 0,
        row[prefix + "query_macs"] == 0,
    )
    forbidden = (
        "fit_access", "update_access", "selection_access", "truth_access",
        "role_oracle_access", "class_quota_access", "global_reassignment",
    )
    if not all(checks) or any(row[prefix + "query_" + name] is not False for name in forbidden):
        raise SystemExit(f"TCRA G0 mechanism closure failed: {row.get('scenario')}")
    if any(row[name] is not False for name in (
        "query_fit_access", "query_update_access", "query_selection_access",
        "query_truth_access", "query_role_oracle_access", "query_class_quota_access",
        "query_global_reassignment",
    )):
        raise SystemExit(f"TCRA G0 query boundary failed: {row.get('scenario')}")
    wall = int(row["after_registration_resource"]["registration_wall_time_ns"])
    if wall <= 0:
        raise SystemExit("TCRA G0 wall receipt invalid")
    walls.append(wall)
    summary.append({
        "scenario": row["scenario"],
        "wall_time_ns": wall,
        "generated": int(row[prefix + "generated_atomic_ascent_count"]),
        "selected": int(row[prefix + "selected_atomic_ascent_count"]),
        "old_tail_min_gain": min(old_gains),
        "pooled_new_cross_tail_gain": float(row[prefix + "pooled_new_cross_tail_gain"]),
    })
p90_nearest_rank = max(walls)
if p90_nearest_rank > 150_000_000:
    raise SystemExit(f"TCRA G0 resource gate failed: {p90_nearest_rank}")
receipt = {
    "schema": "cvs.phase2.d92_tcra.truth_free_g0_validation.v1",
    "status": "D92_TCRA_G0_ACTIVE_RESOURCE_PASS",
    "performance_claim": False,
    "truth_or_scorer_used": False,
    "wall_p90_nearest_rank_ns": p90_nearest_rank,
    "rows": summary,
}
destination.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY

printf 'D92_TCRA_G0_ACTIVE_RESOURCE_PASS\n'
