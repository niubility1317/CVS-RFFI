#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_satphysmv11_constrained_matrix_20260702}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"

mkdir -p "${MATRIX_LOG_ROOT}"

declare -a CELLS=(
  "phase1_adv3b02_satphysmv11_rx20_1_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satphysmv11_rx20_1_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satphysmv11_rx3_19_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satphysmv11_rx3_19_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satphysmv11_rx7_14_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satphysmv11_rx7_14_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satphysmv11_rx7_7_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satphysmv11_rx7_7_u1_20260702 1-16,4-10"
  "phase1_adv3b02_satphysmv11_rx8_8_u10_20260702 10-1,10-10"
  "phase1_adv3b02_satphysmv11_rx8_8_u1_20260702 1-16,4-10"
)

declare -a POLICIES=(
  "SATPHY11C_MLP_M10_PROXY05 mlp 64 350 0.003 margin 10.0 1.0 1.0 proxy_far 1.0000 0.05 all"
  "SATPHY11C_MLP_M20_PROXY05 mlp 64 350 0.003 margin 20.0 1.0 1.0 proxy_far 1.0000 0.05 all"
  "SATPHY11C_MLP_M50_PROXY05 mlp 64 350 0.003 margin 50.0 1.0 1.0 proxy_far 1.0000 0.05 all"
  "SATPHY11C_MLP_M100_PROXY05 mlp 64 350 0.003 margin 100.0 1.0 1.0 proxy_far 1.0000 0.05 all"
  "SATPHY11C_MLP_M20_PROXY02 mlp 64 350 0.003 margin 20.0 1.0 1.0 proxy_far 1.0000 0.02 all"
  "SATPHY11C_MLP_M50_PROXY02 mlp 64 350 0.003 margin 50.0 1.0 1.0 proxy_far 1.0000 0.02 all"
  "SATPHY11C_MLP_M20_MIN02 mlp 64 350 0.003 margin 20.0 1.0 1.0 min_source_proxy 1.0000 0.02 all"
  "SATPHY11C_MLP_M50_MIN02 mlp 64 350 0.003 margin 50.0 1.0 1.0 min_source_proxy 1.0000 0.02 all"
  "SATPHY11C_MLP_M20_SRC9999 mlp 64 350 0.003 margin 20.0 1.0 1.0 source_accept 0.9999 0.05 all"
  "SATPHY11C_MLP_M50_SRC9999 mlp 64 350 0.003 margin 50.0 1.0 1.0 source_accept 0.9999 0.05 all"
  "SATPHY11C_MLP_M100_SRC9999 mlp 64 350 0.003 margin 100.0 1.0 1.0 source_accept 0.9999 0.05 all"
  "SATPHY11C_LIN_M20_PROXY05 linear 64 500 0.010 margin 20.0 1.0 1.0 proxy_far 1.0000 0.05 all"
  "SATPHY11C_LIN_M50_PROXY05 linear 64 500 0.010 margin 50.0 1.0 1.0 proxy_far 1.0000 0.05 all"
  "SATPHY11C_MLP_M20_COR_PROXY05 mlp 64 350 0.003 margin 20.0 1.0 1.0 proxy_far 1.0000 0.05 correct"
  "SATPHY11C_MLP_M50_COR_PROXY05 mlp 64 350 0.003 margin 50.0 1.0 1.0 proxy_far 1.0000 0.05 correct"
  "SATPHY11C_MLP_M50_COR_SRC9999 mlp 64 350 0.003 margin 50.0 1.0 1.0 source_accept 0.9999 0.05 correct"
)

echo "[PHASE1-SATPHYSMV11-CONSTRAINED-SWEEP] start=$(date -Is)"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  FEATURE_NPZ="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATPHYSMV11/features_satphysmv11.npz"
  test -f "${FEATURE_NPZ}"
  mkdir -p "${LOG_ROOT}"
  echo "[PHASE1-SATPHYSMV11-CONSTRAINED-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"

  for policy_row in "${POLICIES[@]}"; do
    read -r NAME HEAD HIDDEN EPOCHS LR LOSS SRC_W PROXY_W MARGIN TH_POLICY SRC_Q PROXY_Q TRAIN_SET <<<"${policy_row}"
    OUT_DIR="${RUNS_ROOT}/${NAME}"
    mkdir -p "${OUT_DIR}"
    extra_flags=()
    if [[ "${TRAIN_SET}" == "correct" ]]; then
      extra_flags+=(--train_known_correct_only)
    fi
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_multiview_reject.py" \
      --feature_npz "${FEATURE_NPZ}" \
      --source_tx_ids "${SOURCE_TX_IDS}" \
      --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
      --train_known_roles source \
      --proxy_unknown_roles proxy_unknown \
      --known_query_roles target_old \
      --unknown_query_roles target_unknown \
      --threshold_policy "${TH_POLICY}" \
      --source_accept_quantile "${SRC_Q}" \
      --proxy_far_quantile "${PROXY_Q}" \
      --head_type "${HEAD}" \
      --loss_mode "${LOSS}" \
      --source_loss_weight "${SRC_W}" \
      --proxy_loss_weight "${PROXY_W}" \
      --margin "${MARGIN}" \
      --hidden_dim "${HIDDEN}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}" \
      --l2 0.0001 \
      --seed 4070233 \
      --unknown_far_target 0.05 \
      --max_old_drop_pp 2.0 \
      --output_json "${OUT_DIR}/metrics.json" \
      --score_table_csv "${OUT_DIR}/score_table.csv" \
      "${extra_flags[@]}" \
      > "${LOG_ROOT}/${NAME}_satphysmv11_constrained.out" 2>&1
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/satphysmv11_constrained_sweep_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(runs_root.glob("phase1_adv3b02_satphysmv11_*_20260702/SATPHY11C_*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    threshold = data.get("threshold", {})
    training = data.get("training", {})
    manifest = data.get("manifest", {})
    rows.append({
        "run_id": metrics_path.parent.parent.name,
        "policy": metrics_path.parent.name,
        "head_type": training.get("head_type", ""),
        "loss_mode": training.get("loss_mode", ""),
        "source_loss_weight": training.get("source_loss_weight", ""),
        "threshold_policy": threshold.get("threshold_policy", ""),
        "unknown_FAR": data.get("unknown_FAR"),
        "passes_unknown_far_target": data.get("passes_unknown_far_target"),
        "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
        "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
        "old_drop_pp_vs_closed": data.get("old_drop_pp_vs_closed"),
        "passes_old_drop_target": data.get("passes_old_drop_target"),
        "passes_dual_target": data.get("passes_dual_target"),
        "known_coverage": data.get("known_coverage"),
        "known_accepted_accuracy": data.get("known_accepted_accuracy"),
        "source_accept_quantile": threshold.get("source_accept_quantile"),
        "proxy_far_quantile": threshold.get("proxy_far_quantile"),
        "source_accept_rate_at_threshold": threshold.get("source_accept_rate_at_threshold"),
        "proxy_false_accept_rate_at_threshold": threshold.get("proxy_false_accept_rate_at_threshold"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
        "train_known_correct_only": data.get("train_known_correct_only"),
        "satellite_tta_policy": manifest.get("satellite_tta_policy"),
        "satellite_tta_view_count": manifest.get("satellite_tta_view_count"),
    })
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_id", "policy"])
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-SATPHYSMV11-CONSTRAINED-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/satphysmv11_constrained_sweep_summary.csv"
