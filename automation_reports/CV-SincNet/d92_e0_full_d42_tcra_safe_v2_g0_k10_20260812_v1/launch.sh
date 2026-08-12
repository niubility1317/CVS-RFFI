#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
run_id=d92_e0_full_d42_tcra_safe_v2_g0_k10_20260812_v1
source_root="$project/runs/d92_tcra_safe_v2_g0_source_6a74c410_20260812_v1"
code_root="$source_root/code"
archive="$source_root/d92_tcra_safe_v2_g0_runtime_6a74c410.tar.gz"
output="$project/runs/$run_id"
logs="$project/logs/$run_id"
job="$project/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5"
ground="$project/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
ground_sha=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c

rf() { test -f "$1" || { printf 'required file missing: %s\n' "$1" >&2; exit 64; }; }
rd() { test -d "$1" || { printf 'required directory missing: %s\n' "$1" >&2; exit 64; }; }
rs() { test "$(sha256sum "$1" | awk '{print $1}')" = "$2" || { printf 'sha mismatch: %s\n' "$1" >&2; exit 65; }; }

rf "$archive"; rs "$archive" 24ea05944806503085755fccbaa2c6e451653ecaedc1d95843c332eb95fc00fc
for path in \
  "$job/offline/predictor/before/enrollment_only" \
  "$job/offline/predictor/before/apply_only_staging" \
  "$job/offline/predictor/after/enrollment_only" \
  "$job/offline/predictor/after/apply_only_staging"; do rd "$path"; done
rf "$job/offline/seals/before_enrollment.seal.json"; rs "$job/offline/seals/before_enrollment.seal.json" e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9
rf "$job/apply_seals/before_apply.seal.json"; rs "$job/apply_seals/before_apply.seal.json" 736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473
rf "$job/offline/seals/after_enrollment.seal.json"; rs "$job/offline/seals/after_enrollment.seal.json" 2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286
rf "$job/apply_seals/after_apply.seal.json"; rs "$job/apply_seals/after_apply.seal.json" afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a
rf "$ground/manifest.json"; rs "$ground/manifest.json" "$ground_sha"
test ! -e "$code_root"; test ! -e "$output"; test ! -e "$logs"

tar -xzf "$archive" -C "$source_root"
for path in \
  "$code_root/cvsrffi/__init__.py" \
  "$code_root/cvsrffi/stage2_d92_d42_tail_class_row_ascent.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_slim.py" \
  "$code_root/cvsrffi/stage2_d92_e0d_query_evaluation.py" \
  "$code_root/scripts/probe_d81_ground_nuisance_cauchy_center.py" \
  "$code_root/scripts/probe_d92_registration_balanced_covariance.py" \
  "$code_root/scripts/run_d92_e0d_prediction.py"; do rf "$path"; done
mkdir -p "$logs"
cd "$code_root"

env PYTHONPATH="$code_root:$project" "$python" - "$code_root" >"$logs/import_closure.out" 2>"$logs/import_closure.err" <<'PY'
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
        raise SystemExit(f"runtime import escaped frozen root: {name} -> {path}")
PY

env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH="$code_root:$project" \
  "$python" -u scripts/run_d92_e0d_prediction.py \
  --before-enrollment-package-root "$job/offline/predictor/before/enrollment_only" \
  --before-enrollment-seal-path "$job/offline/seals/before_enrollment.seal.json" \
  --before-enrollment-seal-sha256 e3da38668a1e6ec4053e65e669e2a1845bb43198891644d830950e4550b5cea9 \
  --before-apply-package-root "$job/offline/predictor/before/apply_only_staging" \
  --before-apply-seal-path "$job/apply_seals/before_apply.seal.json" \
  --before-apply-seal-sha256 736852188c32255647b8105bc7a68d4cc92ca73615e4734d0ed5f4bdd0f04473 \
  --after-enrollment-package-root "$job/offline/predictor/after/enrollment_only" \
  --after-enrollment-seal-path "$job/offline/seals/after_enrollment.seal.json" \
  --after-enrollment-seal-sha256 2600a21ee9a2f95a8d17fa1f4d2263b0e04d243424e3257474502953ed6d9286 \
  --after-apply-package-root "$job/offline/predictor/after/apply_only_staging" \
  --after-apply-seal-path "$job/apply_seals/after_apply.seal.json" \
  --after-apply-seal-sha256 afbdc2ebae59fcc311b0cd44aafd27898d7c4af65c9ef03c1085154c8d13020a \
  --ground-component-dir "$ground" --ground-manifest-sha256 "$ground_sha" \
  --arm E0_FULL_D42_TAIL_CLASS_ROW_ASCENT --output-root "$output" --device cuda:0 \
  >"$logs/prediction.out" 2>"$logs/prediction.err"

"$python" - "$output/after/fit_audit.json" "$logs/g0_validation.json" <<'PY'
import json, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
if not isinstance(rows, list) or len(rows) != 3 or {r.get("scenario") for r in rows} != expected:
    raise SystemExit("TCRA-v2 scenario closure failed")
summary, walls = [], []
for row in rows:
    p = "d92_e0d_tcra_"
    tol = float(row[p + "guard_tolerance"])
    old = [float(v) for v in row[p + "old_tail_gain_by_class"]]
    required = (
        row[p + "active"] is True,
        row[p + "fallback_active"] is False,
        row[p + "fallback_reason"] is None,
        row[p + "final_gate_revision"] == "safe_directional_v2",
        row[p + "safe_directional_pass"] is True,
        row[p + "support_guard_pass"] is True,
        row[p + "state_postprocess_mode"] == "d42_tcra",
        row[p + "e0_state_sha256"] != row[p + "final_state_sha256"],
        row[p + "modified_state_field_names"] == ["coef2_qint8"],
        len(old) == 6 and min(old + [float(row[p + "pooled_new_cross_tail_gain"])]) >= -tol,
        float(row[p + "pooled_new_allclass_tail_gain"]) >= -tol,
        float(row[p + "old_to_new_hinge_delta"]) <= tol,
        float(row[p + "new_to_old_hinge_delta"]) <= tol,
        max(old) > tol,
        sum(old) > tol,
        abs(float(row[p + "old_tail_gain_sum"]) - sum(old)) <= 1e-12,
        int(row[p + "old_tail_strict_positive_count"]) == sum(v > tol for v in old),
        int(row[p + "selected_atomic_ascent_count"]) > 0,
        row[p + "component_fit_count"] == 0,
        row["after_total_component_fit_count"] == 2,
        row["after_actual_component_inventory"]["actual_component_fit_count"] == 1,
        row[p + "persistent_state_bytes_delta"] == 0,
        row[p + "query_rows_used"] == 0,
        row[p + "query_macs"] == 0,
    )
    forbidden = (
        "fit_access", "update_access", "selection_access", "truth_access",
        "role_oracle_access", "class_quota_access", "global_reassignment",
    )
    if not all(required) or any(row[p + "query_" + name] is not False for name in forbidden):
        raise SystemExit(f"TCRA-v2 mechanism closure failed: {row.get('scenario')}")
    if any(row["query_" + name] is not False for name in forbidden):
        raise SystemExit(f"TCRA-v2 query boundary failed: {row.get('scenario')}")
    wall = int(row["after_registration_resource"]["registration_wall_time_ns"])
    walls.append(wall)
    summary.append({
        "scenario": row["scenario"], "wall_time_ns": wall,
        "generated": int(row[p + "generated_atomic_ascent_count"]),
        "selected": int(row[p + "selected_atomic_ascent_count"]),
        "old_tail_min_gain": min(old), "old_tail_gain_sum": sum(old),
        "pooled_new_cross_tail_gain": float(row[p + "pooled_new_cross_tail_gain"]),
    })
p90 = max(walls)
if p90 > 150_000_000:
    raise SystemExit(f"TCRA-v2 G0 resource gate failed: {p90}")
receipt = {
    "schema": "cvs.phase2.d92_tcra_safe_v2.truth_free_g0_validation.v1",
    "status": "D92_TCRA_SAFE_V2_G0_ACTIVE_RESOURCE_PASS",
    "performance_claim": False, "truth_or_scorer_used": False,
    "wall_p90_nearest_rank_ns": p90, "rows": summary,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY

printf 'D92_TCRA_SAFE_V2_G0_ACTIVE_RESOURCE_PASS\n'
