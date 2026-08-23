#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823.json}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER="${WORKER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
RESOURCE_SLOT_LIMIT="${RESOURCE_SLOT_LIMIT:-1}"
RESOURCE_POLL_SECONDS="${RESOURCE_POLL_SECONDS:-60}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
[[ -f "${MATRIX}" && -f "${WORKER}" ]] || { echo "[RC4-QB-ERROR] matrix/worker missing" >&2; exit 2; }

mapfile -t ROWS < <("${CONTROL_PYTHON}" -c '
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert d["schema"]=="cvs.phase1.fasttrust_rc4_qb_e200_matrix.v1"
assert d["run_id"]=="phase1_adv3b02_fasttrust_rc4_qb_e200_s392002_20260823_r1"
assert d["seed"]==392002 and d["epochs"]==200 and d["unlabeled_batch_size"]==256
assert d["rc4_lambda_domain"]==0.16 and d["eval_batch_size"]==512
h=d["source_val_heavy_eval"]
assert h=={"start_epoch":1,"interval":5,"final_window":20,"final_interval":1}
rows=d["rows"]
assert len(rows)==3 and [r["gpu"] for r in rows]==[0,1,2]
assert len({r["candidate"] for r in rows})==3
assert all(not r["negative"] for r in rows)
for r in rows:
 print("\t".join([
  str(r["gpu"]),r["candidate"],
  *[str(bool(r[k])).lower() for k in ("hard","partial","partial_set","partial_conditional","negative","class_rx_cap")],
  str(r["total_budget"]),str(bool(r["calibrated_partial_threshold"])).lower(),
  str(d["rc4_lambda_domain"]),str(d["eval_batch_size"]),
  str(h["start_epoch"]),str(h["interval"]),str(h["final_window"]),str(h["final_interval"]),
  d["base_checkpoint"]
 ]))
' "${MATRIX}")

run_row() {
  local gpu="$1" candidate="$2" hard="$3" partial="$4" partial_set="$5" partial_conditional="$6"
  local negative="$7" cap="$8" total_budget="$9" calibrated_threshold="${10}"
  local domain_scale="${11}" eval_batch="${12}" heavy_start="${13}" heavy_interval="${14}"
  local heavy_window="${15}" heavy_final_interval="${16}" checkpoint="${17}"
  checkpoint="${checkpoint%$'\r'}"
  local extra=()
  [[ "${DRY_RUN}" == 1 ]] && extra+=(--dry-run)
  env ROOT="${ROOT}" CODE_ROOT="${CODE_ROOT}" PYTHON="${PYTHON}" CONTROL_PYTHON="${CONTROL_PYTHON}" \
    RUN_ID="${RUN_ID}" RUNS_ROOT="${RUNS_ROOT}" GPU="${gpu}" INIT_MODE=adv3b02_core90 \
    TOTAL_EPOCHS=200 LABEL_EPOCHS=130 PSEUDO_EPOCHS=70 \
    MUSE_S2A_START=17 MUSE_S2B_START=41 MUSE_S3A_START=69 MUSE_S3B_START=161 MUSE_S3C_START=181 \
    RC4_IDENTITY_START=11 RC4_CONSOLIDATION_START=181 RC4_CALIBRATION_EPOCHS=1,41,91,161 \
    RC4_TAIL_TRANSITION_START=91 RC4_TAIL_TRANSITION_EPOCHS=20 RC4_TAIL_TRANSITION_FLOOR=0.25 \
    BASE_CKPT="${ROOT}/${checkpoint}" CANDIDATE_ID_OVERRIDE="${candidate}" ABLATION=NONE \
    MUSE_UNLABELED_BATCH_SIZE=256 FASTTRUST_RC4=true SAT_ANCHOR_SSL=false \
    RC4_USE_ANCHOR=true RC4_USE_CALIBRATION=true RC4_ENABLE_HARD="${hard}" \
    RC4_ENABLE_PARTIAL="${partial}" RC4_ENABLE_PARTIAL_SET="${partial_set}" \
    RC4_ENABLE_PARTIAL_CONDITIONAL="${partial_conditional}" RC4_ENABLE_NEGATIVE="${negative}" \
    RC4_PARTIAL_EFFECTIVE_BUDGET=0 RC4_NEGATIVE_EFFECTIVE_BUDGET=0 \
    RC4_TOTAL_IDENTITY_EFFECTIVE_BUDGET="${total_budget}" \
    RC4_USE_CALIBRATED_PARTIAL_THRESHOLD="${calibrated_threshold}" \
    RC4_CLASS_RX_CAP="${cap}" RC4_SATELLITE=false RC4_LAMBDA_DOMAIN="${domain_scale}" \
    RC4_NONFINITE_GUARD_MIN_COUNT=8 RC4_NONFINITE_GUARD_FRACTION=0.05 \
    EVAL_BATCH_SIZE="${eval_batch}" SOURCE_VAL_HEAVY_EVAL_START_EPOCH="${heavy_start}" \
    SOURCE_VAL_HEAVY_EVAL_INTERVAL="${heavy_interval}" SOURCE_VAL_HEAVY_EVAL_FINAL_WINDOW="${heavy_window}" \
    SOURCE_VAL_HEAVY_EVAL_FINAL_INTERVAL="${heavy_final_interval}" \
    bash "${WORKER}" --only=M3 "${extra[@]}"
}

wait_for_gpu_slot() {
  local gpu="$1" uuid count
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F ', ' -v gpu="${gpu}" '$1==gpu {print $2}')"
  [[ -n "${uuid}" ]] || { echo "[RC4-QB-ERROR] cannot resolve GPU${gpu} UUID" >&2; return 2; }
  while true; do
    count="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -v uuid="${uuid}" '$1==uuid {n++} END{print n+0}')"
    if (( count < RESOURCE_SLOT_LIMIT )); then
      echo "[RC4-QB-SLOT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} action=launch"
      return 0
    fi
    echo "[RC4-QB-SLOT-WAIT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} poll=${RESOURCE_POLL_SECONDS}s"
    sleep "${RESOURCE_POLL_SECONDS}"
  done
}

echo "[RC4-QB-RUN] run_id=${RUN_ID} rows=${#ROWS[@]} dry_run=${DRY_RUN} epochs=200 U=256"
if [[ "${DRY_RUN}" == 1 ]]; then
  for row in "${ROWS[@]}"; do
    IFS=$'\t' read -r gpu candidate hard partial partial_set partial_conditional negative cap total_budget calibrated_threshold domain_scale eval_batch heavy_start heavy_interval heavy_window heavy_final_interval checkpoint <<<"${row}"
    echo "[RC4-QB-ROW] gpu=${gpu} candidate=${candidate} H=${hard} P=${partial} N=${negative} total_budget=${total_budget} calibrated_P=${calibrated_threshold}"
    run_row "$gpu" "$candidate" "$hard" "$partial" "$partial_set" "$partial_conditional" "$negative" "$cap" "$total_budget" "$calibrated_threshold" "$domain_scale" "$eval_batch" "$heavy_start" "$heavy_interval" "$heavy_window" "$heavy_final_interval" "$checkpoint"
  done
  exit 0
fi
[[ ! -e "${RUNS_ROOT}" ]] || { echo "[RC4-QB-ERROR] refusing overwrite: ${RUNS_ROOT}" >&2; exit 3; }
mkdir -p "${RUNS_ROOT}/dispatcher_logs"
pids=(); names=()
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r gpu candidate hard partial partial_set partial_conditional negative cap total_budget calibrated_threshold domain_scale eval_batch heavy_start heavy_interval heavy_window heavy_final_interval checkpoint <<<"${row}"
  (wait_for_gpu_slot "$gpu" && run_row "$gpu" "$candidate" "$hard" "$partial" "$partial_set" "$partial_conditional" "$negative" "$cap" "$total_budget" "$calibrated_threshold" "$domain_scale" "$eval_batch" "$heavy_start" "$heavy_interval" "$heavy_window" "$heavy_final_interval" "$checkpoint") >"${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 &
  pids+=("$!"); names+=("${candidate}")
done
failed=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "[RC4-QB-WORKER-FAILED] ${names[$i]}" >&2; failed=1; }
done
[[ "${failed}" == 0 ]] || exit 4
echo "[RC4-QB-COMPLETE] ${RUN_ID}"
