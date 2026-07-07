#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_adv3b02_stage2c_normsep_protocol_20260707}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2c_seennew_exempt_floor_audit_20260707}"
CASE_ID="${CASE_ID:-PHASE2_STAGE2C_RX7_14}"
SOURCE_RUNS_ROOT="${SOURCE_RUNS_ROOT:-${ROOT}/runs/${SOURCE_RUN_ID}}"
RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
LOG_ROOT="${ROOT}/logs/${RUN_ID}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-70}"
QKNN_K="${QKNN_K:-8}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

declare -a VARIANTS=("STAGE2C_NORM_SEP" "STAGE2C_HEAD_SEP")
declare -a PROFILES=(
  "AUDIT_FLOORLESS_X0P000R000 0 0.00 0.00"
  "AUDIT_SUPPORT1_X1P000R000 1 0.00 0.00"
)

mkdir -p "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}"
echo "[STAGE2C-SEENNEW-EXEMPT-FLOOR-AUDIT] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-SEENNEW-EXEMPT-FLOOR-AUDIT] variants=${VARIANTS[*]} k=10 profiles=${#PROFILES[@]}"
echo "[STAGE2C-SEENNEW-EXEMPT-FLOOR-AUDIT] diagnostic_only=true include_event_results=true"

run_one() {
  local variant="$1"
  local profile="$2"
  local exempt_support="$3"
  local exempt_pvalue="$4"
  local exempt_reliability="$5"
  local feature_npz="${SOURCE_RUNS_ROOT}/${CASE_ID}/${variant}/features_stage2c_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${variant}/${profile}"
  local out_json="${out_dir}/stage2c_qknn_${profile}_k10.json"
  local out_csv="${out_dir}/stage2c_qknn_${profile}_k10.summary.csv"
  local out_log="${LOG_ROOT}/${variant}_${profile}_k10.out"
  mkdir -p "${out_dir}"
  echo "[STAGE2C-SEENNEW-EXEMPT-FLOOR-AUDIT-RUN] variant=${variant} profile=${profile} k=10 exemption=${exempt_support}/${exempt_pvalue}/${exempt_reliability}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py" \
    --feature_npz "${feature_npz}" \
    --output_json "${out_json}" \
    --output_summary_csv "${out_csv}" \
    --k_shot 10 \
    --query_per_class "${QUERY_PER_CLASS}" \
    --qknn_k "${QKNN_K}" \
    --support_selection_policy stable_first \
    --collab_counts all \
    --support_calibration_mode leave_one_out \
    --score_threshold_combine min \
    --support_quantile 0.01 \
    --proxy_quantile 0.99 \
    --unknown_quantile 0.92 \
    --unknown_risk_threshold 0.92 \
    --accept_margin_threshold -0.05 \
    --consensus_gap_threshold -0.02 \
    --consensus_score_threshold -0.02 \
    --scorer_component_vote_threshold 0.75 \
    --class_score_threshold_enabled \
    --fusion_policy scorer_cvs \
    --seen_new_rescue_enabled \
    --seen_new_rescue_risk_scale 0.25 \
    --seen_new_rescue_min_score -0.02 \
    --seen_new_rescue_min_margin -0.05 \
    --seen_new_rescue_min_agreement 0.50 \
    --conformal_rescue_enabled \
    --conformal_rescue_min_pvalue 0.0 \
    --conformal_rescue_risk_scale 0.25 \
    --conformal_rescue_min_agreement 0.50 \
    --rescue_unknown_veto_enabled \
    --rescue_unknown_veto_event_risk 0.80 \
    --rescue_unknown_veto_label_risk 0.80 \
    --rescue_unknown_veto_shell_risk 0.85 \
    --rescue_unknown_veto_component_agreement 0.70 \
    --rescue_unknown_veto_min_sources 2 \
    --rescue_unknown_veto_action unknown_reject \
    --rescue_unknown_veto_seen_new_exemption_enabled \
    --rescue_unknown_veto_seen_new_min_support_count "${exempt_support}" \
    --rescue_unknown_veto_seen_new_min_pvalue "${exempt_pvalue}" \
    --rescue_unknown_veto_seen_new_min_receiver_class_reliability "${exempt_reliability}" \
    --old_gate_max_effective_unknown_risk 0.92 \
    --old_gate_max_component_agreement 1.0 \
    --old_gate_min_support_density 0.0 \
    --seen_new_gate_max_effective_unknown_risk 0.92 \
    --seen_new_gate_max_component_agreement 1.0 \
    --seen_new_gate_min_support_density 0.0 \
    --candidate_set_min_receivers 1 \
    --candidate_set_min_top1_receivers 1 \
    --candidate_set_max_label_unknown_risk 0.92 \
    --candidate_set_max_event_unknown_risk 0.92 \
    --candidate_set_max_label_risk_component_agreement 1.0 \
    --candidate_set_unknown_reject_risk 0.92 \
    --max_event_bytes 2048 \
    --max_event_latency_ms 25 \
    --include_event_results \
    > "${out_log}" 2>&1
}

for variant in "${VARIANTS[@]}"; do
  for row in "${PROFILES[@]}"; do
    read -r profile exempt_support exempt_pvalue exempt_reliability <<<"${row}"
    run_one "${variant}" "${profile}" "${exempt_support}" "${exempt_pvalue}" "${exempt_reliability}"
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[STAGE2C-SEENNEW-EXEMPT-FLOOR-AUDIT-DRY-RUN-DONE]"
  exit 0
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/stage2c_seennew_exempt_floor_audit_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
event_rows = []
for path in sorted(case_dir.glob("*/*/stage2c_qknn_*_k10.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    variant = path.parent.parent.name
    profile = path.parent.name
    for row in payload.get("summary_rows", []):
        item = dict(row)
        item["variant"] = variant
        item["profile"] = profile
        item["json_path"] = str(path)
        rows.append(item)
    metrics = payload.get("counts", {}).get("3", {})
    events = metrics.get("event_results", [])
    seen_new_candidates = [
        e for e in events
        if e.get("output_label_set") == "seen_new"
        or e.get("predicted_label_set") == "seen_new"
    ]
    vetoed_seen_new = [e for e in seen_new_candidates if e.get("rescue_unknown_veto_hit")]
    exempt_seen_new = [
        e for e in seen_new_candidates
        if e.get("rescue_unknown_veto_seen_new_exemption_passed")
    ]
    rescued_seen_new = [
        e for e in seen_new_candidates
        if e.get("seen_new_rescue_applied") or e.get("conformal_rescue_applied")
    ]
    event_rows.append({
        "variant": variant,
        "profile": profile,
        "event_count": len(events),
        "seen_new_candidate_events": len(seen_new_candidates),
        "rescued_seen_new_candidate_events": len(rescued_seen_new),
        "vetoed_seen_new_candidate_events": len(vetoed_seen_new),
        "exempt_seen_new_candidate_events": len(exempt_seen_new),
        "max_seen_new_support": max(
            [float(e.get("label_class_conformal_support_count", 0.0)) for e in seen_new_candidates] or [0.0]
        ),
        "max_seen_new_pvalue": max(
            [float(e.get("label_class_conformal_pvalue", 0.0)) for e in seen_new_candidates] or [0.0]
        ),
        "max_seen_new_receiver_reliability": max(
            [float(e.get("label_receiver_class_reliability", 0.0)) for e in seen_new_candidates] or [0.0]
        ),
    })
rows.sort(
    key=lambda r: (
        float(r.get("seen_new_acc", 0.0)),
        float(r.get("old_acc", 0.0)),
        -float(r.get("unknown_FAR", 1.0)),
    ),
    reverse=True,
)
out = {"rows": rows, "event_audit": event_rows}
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
csv_path = out_path.with_suffix(".csv")
if rows:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
event_csv_path = out_path.with_name(out_path.stem + "_event_audit.csv")
if event_rows:
    with event_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)
print(json.dumps({"rows": rows, "event_audit": event_rows, "out": str(out_path), "csv": str(csv_path), "event_csv": str(event_csv_path)}, ensure_ascii=False, indent=2))
PY

echo "[STAGE2C-SEENNEW-EXEMPT-FLOOR-AUDIT-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
