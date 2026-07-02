#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_leo_feature_adapter_matrix_20260703}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
SEED="${SEED:-4070301}"

mkdir -p "${MATRIX_LOG_ROOT}"

declare -a CELLS=(
  "phase1_adv3b02_multiview_keepold_rx20_1_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx20_1_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx3_19_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx3_19_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx7_14_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx7_14_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx7_7_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx7_7_u1_20260702 1-16,4-10"
  "phase1_adv3b02_multiview_keepold_rx8_8_u10_20260702 10-1,10-10"
  "phase1_adv3b02_multiview_keepold_rx8_8_u1_20260702 1-16,4-10"
)

declare -a ADAPTERS=(
  "LEOADAPT_IDENTITY identity 64 1.0 0.00 1 0.0001 1.0 0.0 0.0 0.0"
  "LEOADAPT_LINR_COS linear_residual 64 0.75 0.00 140 0.0010 1.0 0.75 0.20 0.02"
  "LEOADAPT_MLP64_BAL mlp_residual 64 0.75 0.05 180 0.0010 1.0 0.75 0.25 0.02"
  "LEOADAPT_MLP128_CE mlp_residual 128 0.60 0.05 180 0.0008 0.8 0.50 0.60 0.03"
  "LEOADAPT_AFFINE affine 64 1.00 0.00 120 0.0005 1.0 0.50 0.20 0.05"
)

declare -a HEAD_POLICIES=(
  "ADAPT_LIN_SRC9999 linear 64 450 0.020 0.9999 source_accept"
  "ADAPT_LIN_MIN05 linear 64 450 0.020 1.0000 min_source_proxy"
  "ADAPT_MLP64_SRC9999 mlp 64 320 0.005 0.9999 source_accept"
  "ADAPT_MLP64_MIN05 mlp 64 320 0.005 1.0000 min_source_proxy"
)

declare -a PROTO_POLICIES=(
  "ADAPT_PROTO_COS_SRC9999 cosine 0.0 0.0 0.0 source_accept 0.9999 0.05"
  "ADAPT_PROTO_COS_MIN05 cosine 0.0 0.0 0.0 min_source_proxy 1.0000 0.05"
  "ADAPT_PROTO_MAH_MIN05 diag_mahalanobis 0.0 0.0 0.0 min_source_proxy 1.0000 0.05"
)

echo "[PHASE1-LEO-FEATURE-ADAPTER-SWEEP] start=$(date -Is) seed=${SEED}"
for cell in "${CELLS[@]}"; do
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  CLEAN_NPZ="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_MULTIVIEW/clean.npz"
  SAT_NPZ="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz"
  if [[ ! -f "${CLEAN_NPZ}" || ! -f "${SAT_NPZ}" ]]; then
    echo "[PHASE1-LEO-FEATURE-ADAPTER-SKIP] run_id=${RUN_ID} missing_clean_or_sat=1 clean=${CLEAN_NPZ} sat=${SAT_NPZ}"
    continue
  fi
  echo "[PHASE1-LEO-FEATURE-ADAPTER-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"
  for adapter_row in "${ADAPTERS[@]}"; do
    read -r ADAPT_NAME KIND HIDDEN ALPHA DROPOUT EPOCHS LR PAIR_W COS_W CE_W RES_W <<<"${adapter_row}"
    ADAPT_DIR="${RUNS_ROOT}/${ADAPT_NAME}"
    ADAPT_NPZ="${ADAPT_DIR}/features_leo_repaired.npz"
    mkdir -p "${ADAPT_DIR}" "${LOG_ROOT}"
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/fit_apply_phase1_leo_feature_adapter.py" \
      --clean_npz "${CLEAN_NPZ}" \
      --sat_npz "${SAT_NPZ}" \
      --out_npz "${ADAPT_NPZ}" \
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
      --proto_ce_weight "${CE_W}" \
      --residual_weight "${RES_W}" \
      --seed "${SEED}" \
      --device "cuda:0" \
      --output_json "${ADAPT_DIR}/adapter_metrics.json" \
      --adapter_out "${ADAPT_DIR}/adapter.pt" \
      > "${LOG_ROOT}/${ADAPT_NAME}_fit_apply.out" 2>&1

    for policy_row in "${HEAD_POLICIES[@]}"; do
      read -r NAME HEAD_TYPE HIDDEN_DIM HEAD_EPOCHS HEAD_LR SRC_Q TH_POLICY <<<"${policy_row}"
      OUT_DIR="${ADAPT_DIR}/${NAME}"
      mkdir -p "${OUT_DIR}"
      env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
        "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_multiview_reject.py" \
        --feature_npz "${ADAPT_NPZ}" \
        --source_tx_ids "${SOURCE_TX_IDS}" \
        --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
        --train_known_roles source \
        --proxy_unknown_roles proxy_unknown \
        --known_query_roles target_old \
        --unknown_query_roles target_unknown \
        --threshold_policy "${TH_POLICY}" \
        --source_accept_quantile "${SRC_Q}" \
        --proxy_far_quantile 0.05 \
        --head_type "${HEAD_TYPE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --epochs "${HEAD_EPOCHS}" \
        --lr "${HEAD_LR}" \
        --l2 0.0001 \
        --seed "${SEED}" \
        --unknown_far_target 0.05 \
        --max_old_drop_pp 2.0 \
        --output_json "${OUT_DIR}/metrics.json" \
        --score_table_csv "${OUT_DIR}/score_table.csv" \
        > "${LOG_ROOT}/${ADAPT_NAME}_${NAME}.out" 2>&1
    done

    for policy_row in "${PROTO_POLICIES[@]}"; do
      read -r NAME METRIC CONF_W ENT_W MARGIN_W TH_POLICY SRC_Q PROXY_Q <<<"${policy_row}"
      OUT_DIR="${ADAPT_DIR}/${NAME}"
      mkdir -p "${OUT_DIR}"
      env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" \
        "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_prototype_reject.py" \
        --feature_npz "${ADAPT_NPZ}" \
        --source_tx_ids "${SOURCE_TX_IDS}" \
        --unknown_tx_ids "${UNKNOWN_TX_IDS}" \
        --train_known_roles source \
        --proxy_unknown_roles proxy_unknown \
        --known_query_roles target_old \
        --unknown_query_roles target_unknown \
        --metric "${METRIC}" \
        --confidence_weight "${CONF_W}" \
        --entropy_weight "${ENT_W}" \
        --margin_weight "${MARGIN_W}" \
        --threshold_policy "${TH_POLICY}" \
        --source_accept_quantile "${SRC_Q}" \
        --proxy_far_quantile "${PROXY_Q}" \
        --unknown_far_target 0.05 \
        --max_old_drop_pp 2.0 \
        --output_json "${OUT_DIR}/metrics.json" \
        --score_table_csv "${OUT_DIR}/score_table.csv" \
        > "${LOG_ROOT}/${ADAPT_NAME}_${NAME}.out" 2>&1
    done
  done
done

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/leo_feature_adapter_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for adapter_metrics in sorted(runs_root.glob("phase1_adv3b02_multiview_keepold_*_20260702/LEOADAPT_*/adapter_metrics.json")):
    adapt = json.loads(adapter_metrics.read_text(encoding="utf-8"))
    val = adapt.get("val_alignment", {})
    train = adapt.get("train_alignment", {})
    for metrics_path in sorted(adapter_metrics.parent.glob("ADAPT_*/metrics.json")):
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
        threshold = data.get("threshold", {})
        training = data.get("training", {})
        scoring = data.get("scoring", {})
        rows.append({
            "run_id": adapter_metrics.parent.parent.name,
            "adapter": adapter_metrics.parent.name,
            "reject_policy": metrics_path.parent.name,
            "adapter_kind": adapt.get("adapter_kind"),
            "source_pair_count": adapt.get("source_pair_count"),
            "val_pair_mse_before": val.get("pair_mse_before"),
            "val_pair_mse_after": val.get("pair_mse_after"),
            "val_pair_cos_before": val.get("pair_cos_before"),
            "val_pair_cos_after": val.get("pair_cos_after"),
            "val_proto_acc_before": val.get("proto_acc_before"),
            "val_proto_acc_after": val.get("proto_acc_after"),
            "train_pair_mse_before": train.get("pair_mse_before"),
            "train_pair_mse_after": train.get("pair_mse_after"),
            "phase": data.get("phase"),
            "head_type": training.get("head_type", ""),
            "metric": scoring.get("metric", ""),
            "threshold_policy": threshold.get("threshold_policy"),
            "unknown_FAR": data.get("unknown_FAR"),
            "passes_unknown_far_target": data.get("passes_unknown_far_target"),
            "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
            "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
            "old_drop_pp_vs_closed": data.get("old_drop_pp_vs_closed"),
            "passes_old_drop_target": data.get("passes_old_drop_target"),
            "passes_dual_target": data.get("passes_dual_target"),
            "known_coverage": data.get("known_coverage"),
            "known_accepted_accuracy": data.get("known_accepted_accuracy"),
            "known_query_count": data.get("known_query_count"),
            "unknown_query_count": data.get("unknown_query_count"),
        })
fields = list(rows[0].keys()) if rows else ["run_id", "adapter", "reject_policy"]
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print({"summary_csv": str(out), "rows": len(rows)})
PY
echo "[PHASE1-LEO-FEATURE-ADAPTER-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/leo_feature_adapter_summary.csv"
