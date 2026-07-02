#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
MATRIX_LOG_ROOT="${MATRIX_LOG_ROOT:-${ROOT}/logs/phase1_adv3b02_global_source_leo_adapter_matrix_20260703}"
SOURCE_PAIR_DIR="${SOURCE_PAIR_DIR:-${ROOT}/runs/phase1_adv3b02_global_source_leo_pairs_20260703}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
EXPORT_SOURCE_TX_IDS="${EXPORT_SOURCE_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
SEED="${SEED:-4070311}"
GPU="${GPU:-0}"
CELL_SHARD_INDEX="${CELL_SHARD_INDEX:-0}"
CELL_SHARD_COUNT="${CELL_SHARD_COUNT:-1}"
DO_EXPORT="${DO_EXPORT:-0}"
RUN_CELLS="${RUN_CELLS:-1}"

mkdir -p "${MATRIX_LOG_ROOT}" "${SOURCE_PAIR_DIR}"

if [[ "${DO_EXPORT}" == "1" ]]; then
  echo "[PHASE1-GLOBAL-SOURCE-LEO-EXPORT] start=$(date -Is) gpu=${GPU} out=${SOURCE_PAIR_DIR}"
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/export_phase1_source_leo_pair_features.py" \
    --ckpt "${TEACHER_CKPT}" \
    --wisig_pkl "${WISIG_PKL}" \
    --out_dir "${SOURCE_PAIR_DIR}" \
    --feature_name z_id \
    --source_tx_ids "${EXPORT_SOURCE_TX_IDS}" \
    --source_rxs "${CEN51_TRAIN_RXS}" \
    --wisig_equalized 1 \
    --wisig_domain rx_day \
    --wisig_out_len 256 \
    --max_samples_per_combo 0 \
    --max_samples_per_tx 1600 \
    --batch_size 512 \
    --device "cuda:0" \
    --seed "${SEED}" \
    --sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak \
    --star_ground_channel_impl simplified_leo_residual \
    > "${MATRIX_LOG_ROOT}/source_pair_export.out" 2>&1
  echo "[PHASE1-GLOBAL-SOURCE-LEO-EXPORT-DONE] end=$(date -Is)"
  if [[ "${RUN_CELLS}" == "0" ]]; then
    exit 0
  fi
fi

CLEAN_NPZ="${SOURCE_PAIR_DIR}/source_clean.npz"
TRAIN_SAT_CLEAR="${SOURCE_PAIR_DIR}/source_leo_clear_weak.npz"
TRAIN_SAT_LOW="${SOURCE_PAIR_DIR}/source_leo_low_elev_weak.npz"
TRAIN_SAT_RAIN="${SOURCE_PAIR_DIR}/source_leo_rain_weak.npz"
for required in "${CLEAN_NPZ}" "${TRAIN_SAT_CLEAR}" "${TRAIN_SAT_LOW}" "${TRAIN_SAT_RAIN}"; do
  if [[ ! -f "${required}" ]]; then
    echo "[PHASE1-GLOBAL-SOURCE-LEO-MISSING] ${required}"
    exit 2
  fi
done

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
  "LEOADAPT3_IDENTITY identity 64 1.0 0.00 1 0.0001 1.0 0.0 0.0 0.0"
  "LEOADAPT3_MEANSHIFT mean_shift 64 1.0 0.00 1 0.0001 1.0 0.0 0.0 0.0"
  "LEOADAPT3_NORMSHIFT norm_mean_shift 64 0.7 0.00 1 0.0001 1.0 0.0 0.0 0.0"
  "LEOADAPT3_LINR_COS linear_residual 96 0.45 0.00 180 0.0007 0.7 2.0 1.2 0.06"
  "LEOADAPT3_MLP_ID mlp_residual 128 0.30 0.05 160 0.0005 0.5 2.0 1.5 0.08"
)

declare -a HEAD_POLICIES=(
  "ADAPT3_LIN_SRC9999 linear 64 450 0.020 0.9999 source_accept"
  "ADAPT3_LIN_MIN05 linear 64 450 0.020 1.0000 min_source_proxy"
  "ADAPT3_MLP64_SRC9999 mlp 64 320 0.005 0.9999 source_accept"
  "ADAPT3_MLP64_MIN05 mlp 64 320 0.005 1.0000 min_source_proxy"
)

declare -a PROTO_POLICIES=(
  "ADAPT3_PROTO_COS_SRC9999 cosine 0.0 0.0 0.0 source_accept 0.9999 0.05"
  "ADAPT3_PROTO_COS_MIN05 cosine 0.0 0.0 0.0 min_source_proxy 1.0000 0.05"
  "ADAPT3_PROTO_MAH_MIN05 diag_mahalanobis 0.0 0.0 0.0 min_source_proxy 1.0000 0.05"
)

echo "[PHASE1-GLOBAL-SOURCE-LEO-ADAPTER-SWEEP] start=$(date -Is) seed=${SEED} gpu=${GPU} shard=${CELL_SHARD_INDEX}/${CELL_SHARD_COUNT}"
for cell_idx in "${!CELLS[@]}"; do
  if (( CELL_SHARD_COUNT > 1 && (cell_idx % CELL_SHARD_COUNT) != CELL_SHARD_INDEX )); then
    continue
  fi
  cell="${CELLS[$cell_idx]}"
  read -r RUN_ID UNKNOWN_TX_IDS <<<"${cell}"
  RUNS_ROOT="${ROOT}/runs/${RUN_ID}"
  LOG_ROOT="${ROOT}/logs/${RUN_ID}"
  SAT_NPZ="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_SATUNKNOWN_SINGLEVIEW/features_satunknown_singleview.npz"
  if [[ ! -f "${SAT_NPZ}" ]]; then
    echo "[PHASE1-GLOBAL-SOURCE-LEO-SKIP] run_id=${RUN_ID} missing_sat_npz=${SAT_NPZ}"
    continue
  fi
  echo "[PHASE1-GLOBAL-SOURCE-LEO-CELL] run_id=${RUN_ID} unknown=${UNKNOWN_TX_IDS}"
  for adapter_row in "${ADAPTERS[@]}"; do
    read -r ADAPT_NAME KIND HIDDEN ALPHA DROPOUT EPOCHS LR PAIR_W COS_W CE_W RES_W <<<"${adapter_row}"
    ADAPT_DIR="${RUNS_ROOT}/${ADAPT_NAME}"
    ADAPT_NPZ="${ADAPT_DIR}/features_leo_repaired.npz"
    mkdir -p "${ADAPT_DIR}" "${LOG_ROOT}"
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU}" \
      "${PYTHON}" -u "${ROOT}/code/scripts/fit_apply_phase1_leo_feature_adapter.py" \
      --clean_npz "${CLEAN_NPZ}" \
      --sat_npz "${SAT_NPZ}" \
      --train_sat_npz "${TRAIN_SAT_CLEAR}" \
      --train_sat_npz "${TRAIN_SAT_LOW}" \
      --train_sat_npz "${TRAIN_SAT_RAIN}" \
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
        --metric "${METRIC}" \
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

"${PYTHON}" - <<'PY' "${ROOT}/runs" "${MATRIX_LOG_ROOT}/global_source_leo_adapter_summary.csv"
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for adapter_metrics in sorted(runs_root.glob("phase1_adv3b02_multiview_keepold_*_20260702/LEOADAPT3_*/adapter_metrics.json")):
    adapt = json.loads(adapter_metrics.read_text(encoding="utf-8"))
    val = adapt.get("val_alignment", {})
    train = adapt.get("train_alignment", {})
    for metrics_path in sorted(adapter_metrics.parent.glob("ADAPT3_*/metrics.json")):
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
echo "[PHASE1-GLOBAL-SOURCE-LEO-ADAPTER-SWEEP-DONE] end=$(date -Is) summary=${MATRIX_LOG_ROOT}/global_source_leo_adapter_summary.csv"
