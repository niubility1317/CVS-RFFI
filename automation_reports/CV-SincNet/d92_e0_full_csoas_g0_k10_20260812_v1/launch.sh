#!/usr/bin/env bash
set -euo pipefail

project=/home/szu2070436088/2510044040/CV-SincNet
python=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
run_id=d92_e0_full_csoas_g0_k10_20260812_v1
scientific_commit=b8ebd4f4522fcc3e9e6b7dd18d722c329021f181
source_root="$project/runs/d92_csoas_g0_source_b8ebd4f4_20260812_v1"
code_root="$source_root/code"
archive="$source_root/d92_csoas_g0_runtime_b8ebd4f4.tar.gz"
output="$project/runs/$run_id"
logs="$project/logs/$run_id"
job="$project/runs/d92_registration_balanced_125_retry2_20260720/jobs/rx_7_7__seed_713106__k_10__new_5"
ground="$project/runs/d19_ciaf_int8_proto_20260717_1039/input/int8_component"
ground_sha=15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c

rf() { test -f "$1" || { printf 'required file missing: %s\n' "$1" >&2; exit 64; }; }
rd() { test -d "$1" || { printf 'required directory missing: %s\n' "$1" >&2; exit 64; }; }
rs() { test "$(sha256sum "$1" | awk '{print $1}')" = "$2" || { printf 'sha mismatch: %s\n' "$1" >&2; exit 65; }; }

rf "$archive"; rs "$archive" 4b0b434a26b47511cb0ddeb9f2455bc81964d8fcef312e75e57879547b631ca5
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
  "$code_root/cvsrffi/stage2_d92_cauchy_scatter_oas.py" \
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
    "cvsrffi.stage2_d92_cauchy_scatter_oas",
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
  --arm E0_FULL_CSOAS --output-root "$output" --device cuda:0 \
  >"$logs/prediction.out" 2>"$logs/prediction.err"

"$python" - "$output/after/fit_audit.json" "$logs/g0_validation.json" <<'PY'
import json, math, pathlib, sys
rows = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
e0_sha = {
    "leo_clear_weak": "f68f4dba37fc89d475d9a5d9444c6e314b11ebdc0063c8cff5cddfb657016bc9",
    "leo_low_elev_weak": "091154fd6b5c3786f7097865b9fd1b8ba4ebed0ab81ab12f787e6ee0d5037d11",
    "leo_rain_weak": "db62c36f8e88a7dac344c03fd7793f514423e77fb725809003bfdefe6e9abb2a",
}
e0_peak = {
    "leo_clear_weak": 1_327_104,
    "leo_low_elev_weak": 1_060_864,
    "leo_rain_weak": 57_344,
}
if not isinstance(rows, list) or len(rows) != 3 or {r.get("scenario") for r in rows} != expected:
    raise SystemExit("CSOAS scenario closure failed")
summary, walls = [], []
for row in rows:
    scenario = row["scenario"]
    inventory = row["after_actual_component_inventory"]
    resource = row["after_registration_resource"]
    method_query = (
        "d92_csoas_query_fit_access", "d92_csoas_query_update_access",
        "d92_csoas_query_selection_access", "d92_csoas_query_truth_access",
        "d92_csoas_query_role_oracle_access", "d92_csoas_query_class_quota_access",
        "d92_csoas_query_global_reassignment",
    )
    top_query = (
        "query_fit_access", "query_update_access", "query_selection_access",
        "query_truth_access", "query_role_oracle_access", "query_class_quota_access",
        "query_global_reassignment",
    )
    quant_error = float(row["resource_audit"]["final_coefficient_quantization_error_max"])
    state_sha = row["after_state_fingerprint_sha256"]
    peak = int(resource["registration_incremental_peak_working_set_bytes"])
    wall = int(resource["registration_wall_time_ns"])
    required = (
        row["arm_id"] == "E0_FULL_CSOAS",
        row["candidate_id"] == "d92_e0_full_csoas",
        row["after_registered_d_mode_effective"] == "csoas_full",
        row["d92_csoas_active"] is True,
        row["d92_csoas_fallback_active"] is False,
        row["d92_csoas_fallback_reason"] is None,
        int(row["d92_csoas_candidate_attempt_fit_count"]) == 1,
        int(row["d92_csoas_fallback_reference_fit_count"]) == 0,
        row["d92_csoas_candidate_statistic_receipt_available"] is True,
        row["d92_csoas_paired_e0_codec_state_equal"] is None,
        row["d92_e0d_csoas_g0_eligible"] is False,
        row["d92_e0d_csoas_g0_block_reason"] == "PENDING_DEPLOYED_CODEC_PAIRED_E0",
        int(row["after_total_component_fit_count"]) == 2,
        int(inventory["actual_component_fit_count"]) == 1,
        int(inventory["full_component_fit_count"]) == 1,
        int(inventory["block3_component_fit_count"]) == 0,
        state_sha != e0_sha[scenario],
        math.isfinite(quant_error) and quant_error > 0.0,
        int(row["after_state_bytes"]) == 8_583,
        int(row["query_macs"]) == 11 * 288,
        peak <= e0_peak[scenario] + 524_288,
    )
    if not all(required) or any(row[name] is not False for name in method_query + top_query):
        raise SystemExit(f"CSOAS mechanism/resource closure failed: {scenario}")
    walls.append(wall)
    summary.append({
        "scenario": scenario,
        "wall_time_ns": wall,
        "incremental_peak_working_set_bytes": peak,
        "paired_e0_incremental_peak_working_set_bytes": e0_peak[scenario],
        "candidate_state_sha256": state_sha,
        "paired_e0_state_sha256": e0_sha[scenario],
        "d42_persisted_state_quantum_change": True,
        "final_coefficient_quantization_error_max": quant_error,
    })
p90 = max(walls)
if p90 > 150_000_000:
    raise SystemExit(f"CSOAS G0 wall gate failed: {p90}")
receipt = {
    "schema": "cvs.phase2.d92_csoas.truth_free_g0_validation.v1",
    "status": "D92_CSOAS_G0_ACTIVE_NON_E0_RESOURCE_PASS",
    "performance_claim": False,
    "truth_or_scorer_used": False,
    "wall_p90_nearest_rank_ns": p90,
    "wall_target_120ms_pass": p90 <= 120_000_000,
    "wall_hard_150ms_pass": True,
    "paired_e0_source": "immutable_target125_same_outer_fit_audit",
    "rows": summary,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
PY

printf 'D92_CSOAS_G0_ACTIVE_NON_E0_RESOURCE_PASS\n'
