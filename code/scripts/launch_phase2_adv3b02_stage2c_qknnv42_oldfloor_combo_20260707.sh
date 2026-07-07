#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
SOURCE_RUN_ID="${SOURCE_RUN_ID:-phase2_adv3b02_stage2c_normsep_protocol_20260707}"
RUN_ID="${RUN_ID:-phase2_adv3b02_stage2c_qknnv42_oldfloor_combo_20260707}"
CASE_ID="${CASE_ID:-PHASE2_STAGE2C_RX7_14}"
SOURCE_RUNS_ROOT="${SOURCE_RUNS_ROOT:-${ROOT}/runs/${SOURCE_RUN_ID}}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
QUERY_PER_CLASS="${QUERY_PER_CLASS:-70}"
QKNN_K="${QKNN_K:-8}"
INCLUDE_EVENT_RESULTS="${INCLUDE_EVENT_RESULTS:-1}"
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
  "ORBIT_BASE"
  "OLDFLOOR_STRICT"
  "OLDFLOOR_BALANCED"
  "OLDFLOOR_RELAXED"
  "OLDFLOOR_RELAXED_SEEN_RESCUE_VETO"
)

mkdir -p "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}"
echo "[STAGE2C-QKNNV42-OLDFLOOR-COMBO] run_id=${RUN_ID} source_run_id=${SOURCE_RUN_ID} dry_run=${DRY_RUN}"
echo "[STAGE2C-QKNNV42-OLDFLOOR-COMBO] variants=${VARIANTS[*]} k=${K_SHOTS[*]} profiles=${PROFILES[*]}"
echo "[STAGE2C-QKNNV42-OLDFLOOR-COMBO] diagnostic_only=true unknown_query_eval_only=true no_training=true"

profile_args() {
  local profile="$1"
  PROFILE_ARGS=(
    --support_calibration_mode leave_one_out
    --score_threshold_combine min
    --support_quantile 0.01
    --proxy_quantile 0.99
    --unknown_quantile 0.92
    --unknown_risk_threshold 0.92
    --accept_margin_threshold -0.05
    --consensus_gap_threshold -0.02
    --consensus_score_threshold -0.02
    --scorer_component_vote_threshold 0.75
    --class_score_threshold_enabled
    --fusion_policy orbit_coproto
    --old_gate_max_effective_unknown_risk 0.92
    --old_gate_max_component_agreement 1.0
    --old_gate_min_support_density 0.0
    --seen_new_gate_max_effective_unknown_risk 0.92
    --seen_new_gate_max_component_agreement 1.0
    --seen_new_gate_min_support_density 0.0
    --candidate_set_min_receivers 1
    --candidate_set_min_top1_receivers 1
    --candidate_set_max_label_unknown_risk 0.92
    --candidate_set_max_event_unknown_risk 0.92
    --candidate_set_max_label_risk_component_agreement 1.0
    --candidate_set_unknown_reject_risk 0.92
    --orbit_min_trust 0.30
    --orbit_unknown_veto_risk 0.92
    --max_event_bytes 2048
    --max_event_latency_ms 25
  )
  case "${profile}" in
    ORBIT_BASE)
      ;;
    OLDFLOOR_STRICT)
      PROFILE_ARGS+=(
        --orbit_old_floor_rescue_enabled
        --orbit_old_floor_max_rank 2
        --orbit_old_floor_min_receivers 1
        --orbit_old_floor_min_pvalue 0.70
        --orbit_old_floor_min_receiver_class_reliability 0.30
        --orbit_old_floor_min_support_density 0.0
        --orbit_old_floor_min_margin 0.0
        --orbit_old_floor_max_label_unknown_risk 0.70
        --orbit_old_floor_max_event_unknown_risk 0.70
        --orbit_old_floor_max_shell_risk 0.10
        --orbit_old_floor_max_component_agreement 0.10
        --orbit_old_floor_min_trust 0.55
      )
      ;;
    OLDFLOOR_BALANCED)
      PROFILE_ARGS+=(
        --orbit_old_floor_rescue_enabled
        --orbit_old_floor_max_rank 3
        --orbit_old_floor_min_receivers 1
        --orbit_old_floor_min_pvalue 0.45
        --orbit_old_floor_min_receiver_class_reliability 0.28
        --orbit_old_floor_min_support_density 0.0
        --orbit_old_floor_min_margin 0.0
        --orbit_old_floor_max_label_unknown_risk 0.78
        --orbit_old_floor_max_event_unknown_risk 0.78
        --orbit_old_floor_max_shell_risk 0.18
        --orbit_old_floor_max_component_agreement 0.34
        --orbit_old_floor_min_trust 0.45
      )
      ;;
    OLDFLOOR_RELAXED)
      PROFILE_ARGS+=(
        --orbit_old_floor_rescue_enabled
        --orbit_old_floor_max_rank 3
        --orbit_old_floor_min_receivers 1
        --orbit_old_floor_min_pvalue 0.20
        --orbit_old_floor_min_receiver_class_reliability 0.25
        --orbit_old_floor_min_support_density 0.0
        --orbit_old_floor_min_margin 0.0
        --orbit_old_floor_max_label_unknown_risk 0.88
        --orbit_old_floor_max_event_unknown_risk 0.82
        --orbit_old_floor_max_shell_risk 0.35
        --orbit_old_floor_max_component_agreement 0.50
        --orbit_old_floor_min_trust 0.35
      )
      ;;
    OLDFLOOR_RELAXED_SEEN_RESCUE_VETO)
      PROFILE_ARGS+=(
        --orbit_old_floor_rescue_enabled
        --orbit_old_floor_max_rank 3
        --orbit_old_floor_min_receivers 1
        --orbit_old_floor_min_pvalue 0.20
        --orbit_old_floor_min_receiver_class_reliability 0.25
        --orbit_old_floor_min_support_density 0.0
        --orbit_old_floor_min_margin 0.0
        --orbit_old_floor_max_label_unknown_risk 0.88
        --orbit_old_floor_max_event_unknown_risk 0.82
        --orbit_old_floor_max_shell_risk 0.35
        --orbit_old_floor_max_component_agreement 0.50
        --orbit_old_floor_min_trust 0.35
        --seen_new_rescue_enabled
        --seen_new_rescue_risk_scale 0.25
        --seen_new_rescue_min_score -0.02
        --seen_new_rescue_min_margin -0.05
        --seen_new_rescue_min_agreement 0.50
        --conformal_rescue_enabled
        --conformal_rescue_min_pvalue 0.0
        --conformal_rescue_risk_scale 0.25
        --conformal_rescue_min_agreement 0.50
        --rescue_unknown_veto_enabled
        --rescue_unknown_veto_event_risk 0.80
        --rescue_unknown_veto_label_risk 0.80
        --rescue_unknown_veto_shell_risk 0.85
        --rescue_unknown_veto_component_agreement 0.70
        --rescue_unknown_veto_min_sources 2
        --rescue_unknown_veto_action unknown_reject
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
  echo "[STAGE2C-QKNNV42-OLDFLOOR-COMBO-RUN] variant=${variant} profile=${profile} k=${k} feature=${feature_npz}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  EXTRA_ARGS=()
  if [[ "${INCLUDE_EVENT_RESULTS}" == "1" ]]; then
    EXTRA_ARGS+=(--include_event_results)
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
    "${EXTRA_ARGS[@]}" \
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
  echo "[STAGE2C-QKNNV42-OLDFLOOR-COMBO-DRY-RUN-DONE]"
  exit 0
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/stage2c_qknnv42_oldfloor_combo_summary.json" <<'PY'
import csv
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/*/stage2c_qknn_*_k*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("counts", {})
    for row in payload.get("summary_rows", []):
        count_key = str(row.get("collab_count"))
        metrics = counts.get(count_key, {})
        item = dict(row)
        item["variant"] = path.parent.parent.name
        item["profile"] = path.parent.name
        item["json_path"] = str(path)
        item["k_shot"] = payload.get("protocol_safety", {}).get("k_shot")
        item["threshold_scope"] = payload.get("protocol_safety", {}).get("threshold_scope")
        item["unknown_query_eval_only"] = payload.get("protocol_safety", {}).get("unknown_query_eval_only")
        item["orbit_old_floor_rescue_count"] = int(metrics.get("orbit_old_floor_rescue_count", 0))
        item["orbit_old_floor_rescue_by_role"] = json.dumps(
            metrics.get("orbit_old_floor_rescue_by_role", {}),
            sort_keys=True,
            ensure_ascii=False,
        )
        item["rescue_unknown_veto_by_role"] = json.dumps(
            metrics.get("rescue_unknown_veto_by_role", {}),
            sort_keys=True,
            ensure_ascii=False,
        )
        item["far_feasible_005"] = float(item.get("unknown_FAR", 1.0)) <= 0.05
        item["far_feasible_010"] = float(item.get("unknown_FAR", 1.0)) <= 0.10
        item["old80_feasible"] = (
            float(item.get("old_acc", 0.0)) >= 0.80
            and float(item.get("min_old_class_acc", 0.0)) > 0.0
        )
        item["balanced_score"] = (
            2.0 * float(item.get("old_acc", 0.0))
            + float(item.get("min_old_class_acc", 0.0))
            + float(item.get("seen_new_acc", 0.0))
            + float(item.get("min_seen_new_class_acc", 0.0))
            - 2.0 * float(item.get("unknown_FAR", 1.0))
        )
        rows.append(item)

rows.sort(
    key=lambda r: (
        bool(r.get("far_feasible_010", False)),
        bool(r.get("old80_feasible", False)),
        float(r.get("balanced_score", -999.0)),
        float(r.get("old_acc", 0.0)),
        float(r.get("min_old_class_acc", 0.0)),
        float(r.get("seen_new_acc", 0.0)),
        float(r.get("min_seen_new_class_acc", 0.0)),
        -float(r.get("unknown_FAR", 1.0)),
    ),
    reverse=True,
)
old80_far010 = [r for r in rows if r.get("old80_feasible") and r.get("far_feasible_010")]
far005 = [r for r in rows if r.get("far_feasible_005")]
out = {
    "rows": rows,
    "best_balanced": rows[:12],
    "best_old80_far010": old80_far010[:12],
    "best_far005": far005[:12],
}
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
csv_path = out_path.with_suffix(".csv")
if rows:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({"best_balanced": rows[:5], "best_old80_far010": old80_far010[:5], "best_far005": far005[:5], "out": str(out_path), "csv": str(csv_path)}, ensure_ascii=False, indent=2))
PY

echo "[STAGE2C-QKNNV42-OLDFLOOR-COMBO-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
