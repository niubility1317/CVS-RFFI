#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase1_adv3b02_logits_open_set_keepold_20260702}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"
SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
EVAL_SOURCE_TX_IDS="${EVAL_SOURCE_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
CEN51_TRAIN_RXS="${CEN51_TRAIN_RXS:-0,1,2,3,4,5,6}"
TARGET_RECEIVER_IDS="${TARGET_RECEIVER_IDS:-20-1}"
UNKNOWN_TX_IDS="${UNKNOWN_TX_IDS:-10-1,10-10}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

TARGET_RECEIVER_IDS="$(printf '%s' "${TARGET_RECEIVER_IDS}" | tr -d '\r')"
UNKNOWN_TX_IDS="$(printf '%s' "${UNKNOWN_TX_IDS}" | tr -d '\r')"
EVAL_SOURCE_TX_IDS="$(printf '%s' "${EVAL_SOURCE_TX_IDS}" | tr -d '\r')"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

FEATURE_DIR="${RUNS_ROOT}/ADV3B02_CORE90_SOFT_E200_PHASE1_LOGITS"
FEATURE_NPZ="${FEATURE_DIR}/features.npz"

echo "[PHASE1-LOGITS-OSR] run_id=${RUN_ID} dry_run=${DRY_RUN} gpu=${GPU}"
echo "[PHASE1-LOGITS-OSR] objective=unknown_FAR<=0.05 and old_retention_vs_closed>=0.95"
if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${RUNS_ROOT}" "${LOG_ROOT}" "${FEATURE_DIR}"
fi

EXPORT_CMD=(
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${GPU}"
  "${PYTHON}" -u "${ROOT}/code/export_spaceborne_features.py"
  --ckpt "${TEACHER_CKPT}"
  --wisig_pkl "${WISIG_PKL}"
  --new_wisig_pkl "${NEW_WISIG_PKL}"
  --out_npz "${FEATURE_NPZ}"
  --feature_name z_id
  --source_tx_ids "${SOURCE_TX_IDS}"
  --source_rxs "${CEN51_TRAIN_RXS}"
  --target_old_tx_ids "${TARGET_OLD_TX_IDS}"
  --target_old_rxs "${TARGET_RECEIVER_IDS}"
  --target_old_channel_view satellite
  --target_old_sat_scenarios leo_clear_weak,leo_low_elev_weak,leo_rain_weak
  --unknown_tx_ids "${UNKNOWN_TX_IDS}"
  --star_ground_channel_impl simplified_leo_residual
  --wisig_equalized 1
  --wisig_domain rx_day
  --wisig_out_len 256
  --max_samples_per_combo 0
  --max_samples_per_tx 200
  --batch_size 512
  --device "cuda:0"
  --seed 4070202
)

printf "[PHASE1-LOGITS-EXPORT-CMD] "; printf "%q " "${EXPORT_CMD[@]}"; printf "\n"
if [[ "${DRY_RUN}" != "1" ]]; then
  "${EXPORT_CMD[@]}" > "${LOG_ROOT}/feature_export.out" 2>&1
fi

declare -a POLICIES=(
  "KEEP999_ALL 0.001 0.001 0.999 all"
  "KEEP99_ALL 0.01 0.01 0.99 all"
  "KEEP999_CONF_ONLY 0.001 0.001 0.999 conf"
  "KEEP99_CONF_ONLY 0.01 0.01 0.99 conf"
  "KEEP95_CONF_ONLY 0.05 0.05 0.95 conf"
  "KEEP99_MARGIN_ONLY 0.01 0.01 0.99 margin"
  "KEEP99_ENERGY_ONLY 0.01 0.01 0.99 energy"
  "KEEP99_CONF_MARGIN 0.01 0.01 0.99 conf_margin"
)

for row in "${POLICIES[@]}"; do
  read -r NAME CONF_Q MARGIN_Q ENERGY_Q GATES <<<"${row}"
  OUT_DIR="${RUNS_ROOT}/${NAME}"
  if [[ "${DRY_RUN}" != "1" ]]; then
    mkdir -p "${OUT_DIR}"
  fi
  EXTRA_ARGS=()
  case "${GATES}" in
    all) ;;
    conf) EXTRA_ARGS+=(--disable_margin_gate --disable_energy_gate) ;;
    margin) EXTRA_ARGS+=(--disable_conf_gate --disable_energy_gate) ;;
    energy) EXTRA_ARGS+=(--disable_conf_gate --disable_margin_gate) ;;
    conf_margin) EXTRA_ARGS+=(--disable_energy_gate) ;;
    *) echo "[ERROR] unknown gates: ${GATES}" >&2; exit 2 ;;
  esac
  EVAL_CMD=(
    env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}"
    "${PYTHON}" -u "${ROOT}/code/scripts/eval_phase1_logits_open_set_reject.py"
    --feature_npz "${FEATURE_NPZ}"
    --source_tx_ids "${EVAL_SOURCE_TX_IDS}"
    --unknown_tx_ids "${UNKNOWN_TX_IDS}"
    --known_query_roles target_old
    --unknown_query_roles target_unknown
    --calibration_roles source
    --conf_quantile "${CONF_Q}"
    --margin_quantile "${MARGIN_Q}"
    --energy_quantile "${ENERGY_Q}"
    --unknown_far_target 0.05
    --output_json "${OUT_DIR}/metrics.json"
    --score_table_csv "${OUT_DIR}/score_table.csv"
    "${EXTRA_ARGS[@]}"
  )
  printf "[PHASE1-LOGITS-EVAL-CMD] policy=%s gates=%s " "${NAME}" "${GATES}"; printf "%q " "${EVAL_CMD[@]}"; printf "\n"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${EVAL_CMD[@]}" > "${LOG_ROOT}/${NAME}.out" 2>&1
  fi
done

SUMMARY_CSV="${RUNS_ROOT}/summary_phase1_logits_open_set_keepold.csv"
if [[ "${DRY_RUN}" != "1" ]]; then
  "${PYTHON}" - <<'PY' "${RUNS_ROOT}" "${SUMMARY_CSV}"
import csv, json, sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
rows = []
for metrics_path in sorted(root.glob("*/metrics.json")):
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    gate = data.get("gate_policy", {})
    rows.append({
        "policy": metrics_path.parent.name,
        "unknown_FAR": data.get("unknown_FAR"),
        "passes_unknown_far_target": data.get("passes_unknown_far_target"),
        "unknown_reject_rate": data.get("unknown_reject_rate"),
        "known_closed_accuracy_no_reject": data.get("known_closed_accuracy_no_reject"),
        "known_full_accuracy_after_reject": data.get("known_full_accuracy_after_reject"),
        "known_coverage": data.get("known_coverage"),
        "known_accepted_accuracy": data.get("known_accepted_accuracy"),
        "old_retention_vs_closed": data.get("old_retention_vs_closed"),
        "use_confidence_gate": gate.get("use_confidence_gate"),
        "use_margin_gate": gate.get("use_margin_gate"),
        "use_energy_gate": gate.get("use_energy_gate"),
        "AUROC_unknown": data.get("AUROC_unknown"),
        "FPR95": data.get("FPR95"),
        "known_query_count": data.get("known_query_count"),
        "unknown_query_count": data.get("unknown_query_count"),
    })
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["policy"])
    writer.writeheader()
    writer.writerows(rows)
print(json.dumps({"summary_csv": str(out), "rows": len(rows)}, ensure_ascii=False))
PY
fi

echo "[PHASE1-LOGITS-OSR-DONE] run_id=${RUN_ID} summary=${SUMMARY_CSV}"
