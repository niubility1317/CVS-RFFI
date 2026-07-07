#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_adv3b02_stage2c_normsep_protocol_20260707}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2c_policy_router_sweep_20260707}"
CASE_ID="${CASE_ID:-PHASE2_STAGE2C_RX7_14}"
SOURCE_RUNS_ROOT="${SOURCE_RUNS_ROOT:-${ROOT}/runs/${SOURCE_RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
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
declare -a K_SHOTS=("5" "10")
declare -a PROFILES=(
  "SR_CLASS_SCORE_GUARD"
  "SR_CENTER_GUARD"
  "CS_CLASS_SCORE_BAL"
  "KGR_CLASS_SCORE_BAL"
)

mkdir -p "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}"
echo "[STAGE2C-POLICY-ROUTER] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-POLICY-ROUTER] profiles=${PROFILES[*]} k=${K_SHOTS[*]}"
echo "[STAGE2C-POLICY-ROUTER] diagnostic_only=true unknown_query_eval_only=true"

profile_args() {
  local profile="$1"
  PROFILE_ARGS=(
    --support_calibration_mode leave_one_out
    --score_threshold_combine min
    --support_quantile 0.01
    --proxy_quantile 0.99
    --unknown_quantile 0.90
    --unknown_risk_threshold 0.90
    --accept_margin_threshold -0.05
    --consensus_gap_threshold -0.02
    --consensus_score_threshold -0.02
    --old_gate_max_effective_unknown_risk 0.90
    --old_gate_max_component_agreement 1.0
    --old_gate_min_support_density 0.0
    --seen_new_gate_max_effective_unknown_risk 0.90
    --seen_new_gate_max_component_agreement 1.0
    --seen_new_gate_min_support_density 0.0
    --candidate_set_min_receivers 1
    --candidate_set_min_top1_receivers 1
    --candidate_set_min_conformal_pvalue 0.0
    --candidate_set_max_label_risk_component_agreement 1.0
    --max_event_bytes 2048
    --max_event_latency_ms 25
  )
  case "${profile}" in
    SR_CLASS_SCORE_GUARD)
      PROFILE_ARGS+=(
        --fusion_policy support_router_cvs
        --class_score_threshold_enabled
        --candidate_set_max_label_unknown_risk 0.84
        --candidate_set_max_event_unknown_risk 0.88
        --candidate_set_unknown_reject_risk 0.84
      )
      ;;
    SR_CENTER_GUARD)
      PROFILE_ARGS+=(
        --fusion_policy support_router_cvs
        --feature_adapter_policy support_center
        --feature_adapter_strength 0.50
        --candidate_set_max_label_unknown_risk 0.84
        --candidate_set_max_event_unknown_risk 0.88
        --candidate_set_unknown_reject_risk 0.84
      )
      ;;
    CS_CLASS_SCORE_BAL)
      PROFILE_ARGS+=(
        --fusion_policy candidate_set_cvs
        --class_score_threshold_enabled
        --unknown_risk_threshold 0.92
        --old_gate_max_effective_unknown_risk 0.92
        --seen_new_gate_max_effective_unknown_risk 0.92
        --candidate_set_max_label_unknown_risk 0.86
        --candidate_set_max_event_unknown_risk 0.88
        --candidate_set_unknown_reject_risk 0.88
      )
      ;;
    KGR_CLASS_SCORE_BAL)
      PROFILE_ARGS+=(
        --fusion_policy known_guarded_rescue_cvs
        --class_score_threshold_enabled
        --unknown_risk_threshold 0.92
        --old_gate_max_effective_unknown_risk 0.92
        --seen_new_gate_max_effective_unknown_risk 0.92
        --candidate_set_max_label_unknown_risk 0.86
        --candidate_set_max_event_unknown_risk 0.88
        --candidate_set_unknown_reject_risk 0.84
      )
      ;;
    *)
      echo "[ERROR] unknown profile ${profile}" >&2
      exit 2
      ;;
  esac
}

run_one() {
  local variant="$1"
  local profile="$2"
  local k="$3"
  local feature_npz="${SOURCE_RUNS_ROOT}/${CASE_ID}/${variant}/features_stage2c_leo_repaired.npz"
  local out_dir="${RUNS_ROOT}/${CASE_ID}/${variant}/${profile}"
  local out_json="${out_dir}/stage2c_qknn_${profile}_k${k}.json"
  local out_csv="${out_dir}/stage2c_qknn_${profile}_k${k}.summary.csv"
  local out_log="${LOG_ROOT}/${variant}_${profile}_k${k}.out"
  mkdir -p "${out_dir}"
  profile_args "${profile}"
  echo "[STAGE2C-POLICY-ROUTER-RUN] variant=${variant} profile=${profile} k=${k} feature=${feature_npz}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_frozen_manytx_unknown_diagnostic.py" \
    --feature_npz "${feature_npz}" \
    --output_json "${out_json}" \
    --output_summary_csv "${out_csv}" \
    --k_shot "${k}" \
    --query_per_class "${QUERY_PER_CLASS}" \
    --qknn_k "${QKNN_K}" \
    --support_selection_policy stable_first \
    --collab_counts all \
    "${PROFILE_ARGS[@]}" \
    > "${out_log}" 2>&1
}

for variant in "${VARIANTS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    for k in "${K_SHOTS[@]}"; do
      run_one "${variant}" "${profile}" "${k}"
    done
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[STAGE2C-POLICY-ROUTER-DRY-RUN-DONE]"
  exit 0
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/stage2c_policy_router_sweep_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/*/stage2c_qknn_*_k*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload.get("summary_rows", []):
        item = dict(row)
        item["variant"] = path.parent.parent.name
        item["profile"] = path.parent.name
        item["json_path"] = str(path)
        item["k_shot"] = payload.get("protocol_safety", {}).get("k_shot")
        item["threshold_scope"] = payload.get("protocol_safety", {}).get("threshold_scope")
        item["unknown_query_eval_only"] = payload.get("protocol_safety", {}).get("unknown_query_eval_only")
        item["far_feasible"] = float(item.get("unknown_FAR", 1.0)) <= 0.05
        rows.append(item)
rows.sort(
    key=lambda r: (
        bool(r.get("far_feasible", False)),
        float(r.get("old_acc", 0.0)),
        float(r.get("min_old_class_acc", 0.0)),
        float(r.get("seen_new_acc", 0.0)),
        -float(r.get("unknown_FAR", 1.0)),
    ),
    reverse=True,
)
feasible = [r for r in rows if r.get("far_feasible")]
out = {"rows": rows, "best_feasible": feasible[:10], "best_overall": rows[:10]}
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
csv_path = out_path.with_suffix(".csv")
if rows:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"best_feasible": feasible[:5], "best_overall": rows[:5], "out": str(out_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
PY

echo "[STAGE2C-POLICY-ROUTER-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
