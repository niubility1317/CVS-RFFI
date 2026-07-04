#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
RUN_ID="${RUN_ID:-phase2_adv3b02_manynew10_supcon_repr_20260704}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/logs/${RUN_ID}}"
TEACHER_CKPT="${TEACHER_CKPT:-${ROOT}/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth}"
WISIG_PKL="${WISIG_PKL:-${ROOT}/Dataset_WigSig/ManySig.pkl}"
NEW_WISIG_PKL="${NEW_WISIG_PKL:-${ROOT}/Dataset_WigSig/ManyTx.pkl}"

SOURCE_TX_IDS="${SOURCE_TX_IDS:-0,1,2,3,4,5}"
EVAL_OLD_TX_IDS="${EVAL_OLD_TX_IDS:-14-10,14-7,20-15,20-19,6-15,8-20}"
TARGET_OLD_TX_IDS="${TARGET_OLD_TX_IDS:-0,1,2,3,4,5}"
SOURCE_RXS="${SOURCE_RXS:-0,1,2,3,4,5,6}"
TARGET_RX="${TARGET_RX:-7-14}"
TARGET_NEW_TX_IDS="${TARGET_NEW_TX_IDS:-10-10,11-10,18-5,19-3,2-13,2-5,3-8,4-10,8-18,8-3}"
PROXY_UNKNOWN_RXS="${PROXY_UNKNOWN_RXS:-1-1,1-19,14-7,18-2,19-2,2-1}"
PROXY_UNKNOWN_TX_IDS="${PROXY_UNKNOWN_TX_IDS:-1-1,1-10,1-11,1-12,1-14,1-15,1-16,1-18,1-19,1-2,1-8,10-11,10-17,10-4,10-7,11-1,11-17,11-19,11-20,11-4,11-7,12-19,12-20,12-7,13-14,13-19,13-20,13-3,13-7,14-11,14-12,14-13,14-14,14-20,14-8,14-9,15-1,15-19,15-6,16-1,16-16,16-19,16-20,16-5,17-10,17-11,18-1,18-10,18-11,18-12,18-13,18-14,18-15,18-16,18-17,18-2,18-20,18-4,18-7,18-8,18-9,19-1,19-10,19-11,19-12,19-13,19-14,19-19,19-2,19-20,19-4,19-6,19-7,19-8,19-9,2-12,2-14,2-15,2-16,2-17,2-19,2-20,2-3,2-4,2-6,2-7,2-8,20-1,20-12,20-14,20-16,20-18,20-20,20-3,20-4,20-5,20-7,20-8,3-1,3-13,3-18,3-19,3-2,3-20,4-1,4-11,5-1,5-16,5-20,5-5,6-1,6-6,7-10,7-11,7-20,7-7,7-8,7-9,8-1,8-13,8-7,8-8,9-1,9-20,9-7}"

SAT_SCENARIOS="${SAT_SCENARIOS:-leo_clear_weak,leo_low_elev_weak,leo_rain_weak}"
SEED="${SEED:-4070491}"
GPUS_CSV="${GPUS_CSV:-0,1}"
DRY_RUN="${DRY_RUN:-0}"

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "[ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

IFS=',' read -r -a GPUS <<<"${GPUS_CSV}"
declare -a VARIANTS=(
  "MANYNEW10_SUPCON_HEAD id_feature_head 40 0.00012 1.10 2.20 16.00 4.00 5.50 0.004 0.20 0.14 0.07 0.12 0.10"
  "MANYNEW10_SUPCON_NORM id_norm_late_feature 40 0.00008 0.90 1.80 18.00 3.50 6.00 0.003 0.18 0.12 0.07 0.10 0.18"
)

if (( ${#GPUS[@]} < ${#VARIANTS[@]} )); then
  echo "[MANYNEW10-SUPCON-GPU-COUNT] need ${#VARIANTS[@]} GPUs but got ${GPUS_CSV}" >&2
  exit 2
fi

CASE_ID="PHASE2_MANYNEW10_RX${TARGET_RX//-/_}"
CELLS_ARG="${CASE_ID}:${TARGET_RX}:${TARGET_NEW_TX_IDS}"

mkdir -p "${LOG_ROOT}" "${RUNS_ROOT}"
echo "[MANYNEW10-SUPCON-REPR] run_id=${RUN_ID} dry_run=${DRY_RUN} target_rx=${TARGET_RX} gpus=${GPUS_CSV}"
echo "[MANYNEW10-SUPCON-REPR] target_new=${TARGET_NEW_TX_IDS}"
echo "[MANYNEW10-SUPCON-REPR] proxy_pool_excludes_target_new=true"
echo "[MANYNEW10-SUPCON-REPR] success=old_acc>=0.80 and every seen-new class>=0.75"

run_variant() {
  local idx="$1"
  local row="${VARIANTS[$idx]}"
  local gpu="${GPUS[$idx]}"
  read -r NAME MODE EPOCHS LR MSE_W COS_W CLEAN_W FEAT_MARGIN_W CLEAN_MARGIN_W MARGIN_TOL PROTO_CE_W PROXY_SEP_W PROXY_MAX_COS SUPCON_W DISTILL_W <<<"${row}"
  local log_path="${LOG_ROOT}/${NAME}_train_export.out"
  echo "[MANYNEW10-SUPCON-VARIANT] name=${NAME} mode=${MODE} gpu=${gpu} start=$(date -Is)"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[MANYNEW10-SUPCON-DRY-RUN] ${NAME} -> ${log_path}"
    return 0
  fi
  env "PYTHONPATH=${ROOT}/code:${ROOT}:${PYTHONPATH:-}" "CUDA_VISIBLE_DEVICES=${gpu}" \
    "${PYTHON}" -u "${ROOT}/code/scripts/train_apply_phase1_iq_preadapter_20260703.py" \
    --ckpt "${TEACHER_CKPT}" \
    --wisig_pkl "${WISIG_PKL}" \
    --new_wisig_pkl "${NEW_WISIG_PKL}" \
    --runs_root "${RUNS_ROOT}" \
    --out_subdir "${NAME}" \
    --out_name "features_leo_repaired.npz" \
    --clean_out_name "features_clean_repaired.npz" \
    --identity_subdir "MANYNEW10_IDENTITY" \
    --export_clean_control \
    --export_identity \
    --cells "${CELLS_ARG}" \
    --source_tx_ids "${SOURCE_TX_IDS}" \
    --target_old_tx_ids "${TARGET_OLD_TX_IDS}" \
    --source_rxs "${SOURCE_RXS}" \
    --proxy_unknown_tx_ids "${PROXY_UNKNOWN_TX_IDS}" \
    --proxy_unknown_rxs "${PROXY_UNKNOWN_RXS}" \
    --sat_scenarios "${SAT_SCENARIOS}" \
    --star_ground_channel_impl simplified_leo_residual \
    --no-input_adapter_enabled \
    --model_adapter_mode "${MODE}" \
    --input_repair raw \
    --clean_input_repair_mode raw \
    --epochs "${EPOCHS}" \
    --hidden_dim 32 \
    --alpha 0.00 \
    --lr "${LR}" \
    --mse_weight "${MSE_W}" \
    --cos_weight "${COS_W}" \
    --proto_ce_weight "${PROTO_CE_W}" \
    --logit_ce_weight 0.0 \
    --clean_identity_weight "${CLEAN_W}" \
    --feature_margin_weight "${FEAT_MARGIN_W}" \
    --clean_feature_margin_weight "${CLEAN_MARGIN_W}" \
    --feature_margin_tolerance "${MARGIN_TOL}" \
    --proxy_unknown_separation_weight "${PROXY_SEP_W}" \
    --proxy_unknown_max_cos "${PROXY_MAX_COS}" \
    --proxy_unknown_supcon_weight "${SUPCON_W}" \
    --proxy_unknown_supcon_temperature 0.07 \
    --max_proxy_unknown_train_samples_per_tx 500 \
    --teacher_logit_distill_weight "${DISTILL_W}" \
    --distill_temperature 2.0 \
    --residual_weight 0.0 \
    --batch_size 384 \
    --max_source_samples_per_tx 1000 \
    --max_export_samples_per_tx 80 \
    --device cuda:0 \
    --seed "${SEED}" \
    > "${log_path}" 2>&1
  echo "[MANYNEW10-SUPCON-VARIANT-DONE] name=${NAME} gpu=${gpu} end=$(date -Is)"
}

pids=()
for idx in "${!VARIANTS[@]}"; do
  run_variant "${idx}" > "${LOG_ROOT}/variant_${idx}.driver.out" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" != "0" ]]; then
  echo "[MANYNEW10-SUPCON-TRAIN-FAILED] one or more variants failed" >&2
  exit 3
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[MANYNEW10-SUPCON-DRY-RUN-DONE]"
  exit 0
fi

for row in "${VARIANTS[@]}"; do
  read -r NAME _rest <<<"${row}"
  FEATURE_NPZ="${RUNS_ROOT}/${CASE_ID}/${NAME}/features_leo_repaired.npz"
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_newtx_pair_sweep.py" \
    --feature_npz "${FEATURE_NPZ}" \
    --output_json "${RUNS_ROOT}/${CASE_ID}/${NAME}/manynew10_k20_eval.json" \
    --output_csv "${RUNS_ROOT}/${CASE_ID}/${NAME}/manynew10_k20_eval.csv" \
    --candidate_new_tx_ids "${TARGET_NEW_TX_IDS}" \
    --old_tx_ids "${EVAL_OLD_TX_IDS}" \
    --old_roles target_old \
    --new_roles target_unknown \
    --combo_size 10 \
    --methods knn1,knn3,proto \
    --k_old 20 \
    --k_new 20 \
    --query_per_old 60 \
    --query_per_new 60 \
    --old_target 0.80 \
    --seen_new_target 0.75 \
    --seed 422947 \
    > "${LOG_ROOT}/${NAME}_manynew10_eval.out" 2>&1
done

IDENTITY_NPZ="${RUNS_ROOT}/${CASE_ID}/MANYNEW10_IDENTITY/features_leo_repaired.npz"
if [[ -f "${IDENTITY_NPZ}" ]]; then
  "${PYTHON}" -u "${ROOT}/code/scripts/phase2_newtx_pair_sweep.py" \
    --feature_npz "${IDENTITY_NPZ}" \
    --output_json "${RUNS_ROOT}/${CASE_ID}/MANYNEW10_IDENTITY/manynew10_k20_eval.json" \
    --output_csv "${RUNS_ROOT}/${CASE_ID}/MANYNEW10_IDENTITY/manynew10_k20_eval.csv" \
    --candidate_new_tx_ids "${TARGET_NEW_TX_IDS}" \
    --old_tx_ids "${EVAL_OLD_TX_IDS}" \
    --old_roles target_old \
    --new_roles target_unknown \
    --combo_size 10 \
    --methods knn1,knn3,proto \
    --k_old 20 \
    --k_new 20 \
    --query_per_old 60 \
    --query_per_new 60 \
    --old_target 0.80 \
    --seen_new_target 0.75 \
    --seed 422947 \
    > "${LOG_ROOT}/MANYNEW10_IDENTITY_manynew10_eval.out" 2>&1
fi

"${PYTHON}" - "${RUNS_ROOT}/${CASE_ID}" "${LOG_ROOT}/manynew10_eval_summary.json" <<'PY'
import json
import sys
from pathlib import Path

case_dir = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = []
for path in sorted(case_dir.glob("*/manynew10_k20_eval.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload.get("combo_rows", payload.get("rows", []))
    if best:
        row = dict(best[0])
        row["variant"] = path.parent.name
        row["json_path"] = str(path)
        rows.append(row)
rows.sort(key=lambda r: (r.get("passes_joint_target", False), r.get("min_seen_new_class_acc", 0.0), r.get("seen_new_acc", 0.0), r.get("old_acc", 0.0)), reverse=True)
out_path.write_text(json.dumps({"rows": rows, "best": rows[:5]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"best": rows[:3], "out": str(out_path)}, ensure_ascii=False, indent=2))
PY

echo "[MANYNEW10-SUPCON-REPR-DONE] run_id=${RUN_ID} runs=${RUNS_ROOT} logs=${LOG_ROOT}"
