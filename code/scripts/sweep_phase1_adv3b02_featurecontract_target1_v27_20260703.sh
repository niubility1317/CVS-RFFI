#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_featurecontract_target1_v27_20260703}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
SEED="${SEED:-4070327}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3,4,5,6,7,0,1}"

mkdir -p "${LOG_ROOT}"

CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
TRAIN_SAT_CLEAR="${SOURCE_PAIR_DIR}/source_leo_clear_weak.npz"
TRAIN_SAT_LOW="${SOURCE_PAIR_DIR}/source_leo_low_elev_weak.npz"
TRAIN_SAT_RAIN="${SOURCE_PAIR_DIR}/source_leo_rain_weak.npz"
for required in "${CLEAN_NPZ}" "${TRAIN_SAT_CLEAR}" "${TRAIN_SAT_LOW}" "${TRAIN_SAT_RAIN}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[V27-MISSING] ${required}" >&2
    exit 2
  fi
done

IFS=',' read -r -a CELL_GPUS <<<"${GPUS_CSV}"

declare -a CELLS=(
  "phase1_adv3b02_multiview_keepold_rx20_1_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx20_1_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx3_19_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx3_19_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_14_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_7_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx7_7_u1_20260702"
  "phase1_adv3b02_multiview_keepold_rx8_8_u10_20260702"
  "phase1_adv3b02_multiview_keepold_rx8_8_u1_20260702"
)

declare -a ADAPTERS=(
  "LEOADAPT9_FEAT_LINR linear_residual 96 0.08 0.00 180 0.0005 0.45 0.90 0.55 9.00 4.00 1.00 3.00 0.015 0.020 7.00 7.00 0.000 0.75 tx,rx"
  "LEOADAPT9_FEAT_MLP mlp_residual 128 0.08 0.03 180 0.00045 0.45 0.90 0.55 9.00 4.00 1.00 3.00 0.015 0.020 7.00 7.00 0.000 0.75 tx,rx"
  "LEOADAPT9_COMPACT_MLP mlp_residual 160 0.10 0.04 200 0.00045 0.55 1.10 0.45 8.00 3.50 2.50 2.50 0.020 0.010 8.00 7.00 0.000 1.00 tx,rx"
  "LEOADAPT9_MARGIN_LINR linear_residual 96 0.06 0.00 180 0.0005 0.35 0.75 0.75 10.00 6.00 0.75 4.50 0.010 0.020 8.00 9.00 0.000 1.00 tx,rx"
  "LEOADAPT9_CONSERVE_LINR linear_residual 96 0.04 0.00 160 0.0005 0.40 0.80 0.90 12.00 3.00 0.50 3.00 0.020 0.025 10.00 10.00 0.000 0.50 tx,rx"
  "LEOADAPT9_CONSERVE_MLP mlp_residual 128 0.05 0.02 160 0.00045 0.40 0.80 0.90 12.00 3.00 0.50 3.00 0.020 0.025 10.00 10.00 0.000 0.50 tx,rx"
)

run_cell() {
  local idx="$1"
  local run_id="${CELLS[$idx]}"
  local gpu="${CELL_GPUS[$idx]}"
  local run_dir="${RUNS_ROOT}/${run_id}"
  local sat_npz="${run_dir}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz"
  local clean_apply_npz="${run_dir}/ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz"
  mkdir -p "${LOG_ROOT}/${run_id}"
  if [[ ! -f "${sat_npz}" || ! -f "${clean_apply_npz}" ]]; then
    echo "[V27-SKIP] run_id=${run_id} missing sat_or_clean_npz"
    return 0
  fi
  echo "[V27-CELL] run_id=${run_id} gpu=${gpu} start=$(date -Is)"
  for adapter_row in "${ADAPTERS[@]}"; do
    read -r ADAPT_NAME KIND HIDDEN ALPHA DROPOUT EPOCHS LR PAIR_W COS_W RES_W CLEAN_ID_W FEAT_MARGIN_W FEAT_COMPACT_W CLEAN_FEAT_MARGIN_W FEAT_MARGIN_TOL FEAT_COMPACT_TOL UNK_ID_W UNK_FEAT_NOINC_W UNK_FEAT_SLACK GROUP_W GROUP_FIELDS <<<"${adapter_row}"
    local adapt_dir="${run_dir}/${ADAPT_NAME}"
    mkdir -p "${adapt_dir}"
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/fit_apply_phase1_leo_feature_adapter.py" \
      --clean_npz "${CLEAN_NPZ}" \
      --sat_npz "${sat_npz}" \
      --train_sat_npz "${TRAIN_SAT_CLEAR}" \
      --train_sat_npz "${TRAIN_SAT_LOW}" \
      --train_sat_npz "${TRAIN_SAT_RAIN}" \
      --source_unknown_npz "${sat_npz}" \
      --unknown_roles "proxy_unknown" \
      --out_npz "${adapt_dir}/features_leo_repaired.npz" \
      --clean_apply_npz "${clean_apply_npz}" \
      --clean_out_npz "${adapt_dir}/features_clean_repaired.npz" \
      --source_tx_ids "${SOURCE_TX_IDS}" \
      --adapter_kind "${KIND}" \
      --hidden_dim "${HIDDEN}" \
      --alpha "${ALPHA}" \
      --dropout "${DROPOUT}" \
      --epochs "${EPOCHS}" \
      --batch_size 512 \
      --lr "${LR}" \
      --weight_decay 0.0001 \
      --pair_weight "${PAIR_W}" \
      --cos_weight "${COS_W}" \
      --proto_ce_weight 0.00 \
      --residual_weight "${RES_W}" \
      --clean_identity_weight "${CLEAN_ID_W}" \
      --margin_retention_weight 0.00 \
      --clean_margin_weight 0.00 \
      --feature_margin_weight "${FEAT_MARGIN_W}" \
      --feature_compactness_weight "${FEAT_COMPACT_W}" \
      --clean_feature_margin_weight "${CLEAN_FEAT_MARGIN_W}" \
      --feature_margin_tolerance "${FEAT_MARGIN_TOL}" \
      --feature_compactness_tolerance "${FEAT_COMPACT_TOL}" \
      --unknown_repulsion_weight 0.00 \
      --unknown_identity_weight "${UNK_ID_W}" \
      --unknown_oldness_nonincrease_weight 0.00 \
      --unknown_feature_sim_nonincrease_weight "${UNK_FEAT_NOINC_W}" \
      --unknown_feature_sim_slack "${UNK_FEAT_SLACK}" \
      --unknown_source_quantile 0.05 \
      --unknown_batch_size 512 \
      --group_floor_weight "${GROUP_W}" \
      --group_floor_fields "${GROUP_FIELDS}" \
      --seed "${SEED}" \
      --device "cuda:0" \
      --output_json "${adapt_dir}/adapter_metrics.json" \
      --adapter_out "${adapt_dir}/adapter.pt" \
      > "${LOG_ROOT}/${run_id}/${ADAPT_NAME}_fit_apply.out" 2>&1
  done
  echo "[V27-CELL-DONE] run_id=${run_id} gpu=${gpu} end=$(date -Is)"
}

echo "[PHASE1-FEATURECONTRACT-TARGET1-V27] start=$(date -Is) gpus=${GPUS_CSV} seed=${SEED}"
pids=()
for idx in "${!CELLS[@]}"; do
  mkdir -p "${LOG_ROOT}/${CELLS[$idx]}"
  run_cell "${idx}" > "${LOG_ROOT}/${CELLS[$idx]}/cell_driver.out" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "[V27-FAILED] one or more cells failed" >&2
  exit 3
fi

VARIANTS="LEOADAPT9_FEAT_LINR,LEOADAPT9_FEAT_MLP,LEOADAPT9_COMPACT_MLP,LEOADAPT9_MARGIN_LINR,LEOADAPT9_CONSERVE_LINR,LEOADAPT9_CONSERVE_MLP"
SUMMARY_CSV="${LOG_ROOT}/target1_strong_v27_summary.csv"
METRICS_JSON="${LOG_ROOT}/target1_strong_v27_metrics.json"
env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
  "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_target1_strong_repair_audit_20260703.py" \
  --runs_root "${RUNS_ROOT}" \
  --source_clean_npz "${CLEAN_NPZ}" \
  --out_csv "${SUMMARY_CSV}" \
  --metrics_json "${METRICS_JSON}" \
  --source_tx_ids "${SOURCE_TX_IDS}" \
  --variants "${VARIANTS}" \
  > "${LOG_ROOT}/target1_strong_v27_eval.out" 2>&1

"${PYTHON}" - "${SUMMARY_CSV}" "${LOG_ROOT}/target1_strong_v27_best.json" <<'PY'
import csv
import json
import math
import sys
from pathlib import Path

csv_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
with csv_path.open("r", encoding="utf-8", newline="") as f:
    for row in csv.DictReader(f):
        for col in [
            "target_old_closed_acc",
            "target_old_delta_pp_vs_identity",
            "target_old_min_scenario_acc",
            "target_old_min_tx_acc",
            "clean_repaired_drop_pp",
            "target_unknown_far_source05",
            "target_unknown_far_delta_vs_identity",
            "target_unknown_oldness_delta_vs_identity",
            "target_old_margin_delta_vs_identity",
            "target_old_true_dist_delta_vs_identity",
        ]:
            try:
                row[col] = float(row[col])
            except Exception:
                row[col] = math.nan
        row["passes_strong_target1"] = str(row.get("passes_strong_target1", "")).lower() == "true"
        rows.append(row)

candidates = [r for r in rows if r.get("variant") != "LEOADAPT3_IDENTITY"]
passes = [r for r in candidates if r["passes_strong_target1"]]

def truth(row, key):
    return str(row.get(key, "")).lower() == "true"

def top(frame, key, n=15):
    return sorted(frame, key=key)[:n]

best_by_run = {}
for row in candidates:
    gate_count = sum(truth(row, k) for k in [
        "passes_old_recovery_gate",
        "passes_clean_fidelity_gate",
        "passes_floor_gate",
        "passes_margin_gate",
        "passes_unknown_safety_gate",
    ])
    key = row["run_id"]
    score = (gate_count, row["target_old_closed_acc"], row["target_old_min_tx_acc"], -abs(row["target_unknown_far_delta_vs_identity"]))
    if key not in best_by_run or score > best_by_run[key][0]:
        best_by_run[key] = (score, row)

summary = {
    "rows": len(rows),
    "candidate_rows": len(candidates),
    "strong_target1_pass": len(passes),
    "pass_rows": passes[:30],
    "gate_counts": {
        key: sum(truth(r, key) for r in candidates)
        for key in [
            "passes_old_recovery_gate",
            "passes_clean_fidelity_gate",
            "passes_floor_gate",
            "passes_margin_gate",
            "passes_unknown_safety_gate",
            "passes_strong_target1",
        ]
    },
    "best_by_run": [item[1] for item in sorted(best_by_run.values(), key=lambda x: x[1]["run_id"])],
    "best_by_target_old_acc": top(candidates, lambda r: (-r["target_old_closed_acc"], r["clean_repaired_drop_pp"], r["target_unknown_far_source05"])),
    "best_by_tx_floor": top(candidates, lambda r: (-r["target_old_min_tx_acc"], -r["target_old_closed_acc"], r["target_unknown_far_source05"])),
    "best_unknown_safe": top(candidates, lambda r: (r["target_unknown_far_delta_vs_identity"], -r["target_old_delta_pp_vs_identity"])),
}
out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ["rows", "candidate_rows", "strong_target1_pass", "gate_counts"]}, ensure_ascii=False, sort_keys=True))
PY

echo "[PHASE1-FEATURECONTRACT-TARGET1-V27-DONE] end=$(date -Is) summary=${SUMMARY_CSV}"
