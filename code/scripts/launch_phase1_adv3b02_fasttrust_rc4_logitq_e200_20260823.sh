#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_rc4_logitq_e200_s392002_20260823.json}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fasttrust_rc4_logitq_e200_s392002_20260823}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER="${WORKER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
RESOURCE_SLOT_LIMIT="${RESOURCE_SLOT_LIMIT:-2}"
RESOURCE_POLL_SECONDS="${RESOURCE_POLL_SECONDS:-60}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
[[ -f "${MATRIX}" && -f "${WORKER}" ]] || { echo "[RC4-ERROR] matrix/worker missing" >&2; exit 2; }

mapfile -t ROWS < <("${CONTROL_PYTHON}" -c '
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert d["schema"]=="cvs.phase1.fasttrust_rc4_logitq_e200_matrix.v1"
assert d["seed"]==392002 and d["epochs"]==200 and d["unlabeled_batch_size"]==256
rows=d["rows"]
assert len(rows)==6 and {r["gpu"] for r in rows}==set(range(6))
assert len({r["candidate"] for r in rows})==6
for r in rows:
 print("\t".join([str(r["gpu"]),r["candidate"]]+[str(bool(r[k])).lower() for k in ("hard","partial","partial_set","partial_conditional","negative","class_rx_cap","satellite")]+[str(d["partial_effective_budget"]),str(d["negative_effective_budget"]),d["base_checkpoint"]]))
' "${MATRIX}")

run_row() {
  local gpu="$1" candidate="$2" hard="$3" partial="$4" partial_set="$5" partial_conditional="$6" negative="$7" cap="$8" satellite="$9" p_budget="${10}" n_budget="${11}" checkpoint="${12}"
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
    RC4_PARTIAL_EFFECTIVE_BUDGET="${p_budget}" RC4_NEGATIVE_EFFECTIVE_BUDGET="${n_budget}" \
    RC4_CLASS_RX_CAP="${cap}" RC4_SATELLITE="${satellite}" bash "${WORKER}" --only=M3 "${extra[@]}"
}

wait_for_gpu_slot() {
  local gpu="$1" uuid count
  uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits | awk -F ', ' -v gpu="${gpu}" '$1==gpu {print $2}')"
  [[ -n "${uuid}" ]] || { echo "[RC4-ERROR] cannot resolve GPU${gpu} UUID" >&2; return 2; }
  while true; do
    count="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>/dev/null | awk -v uuid="${uuid}" '$1==uuid {n++} END{print n+0}')"
    if (( count < RESOURCE_SLOT_LIMIT )); then
      echo "[RC4-SLOT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} action=launch"
      return 0
    fi
    echo "[RC4-SLOT-WAIT] gpu=${gpu} compute_apps=${count} limit=${RESOURCE_SLOT_LIMIT} poll=${RESOURCE_POLL_SECONDS}s"
    sleep "${RESOURCE_POLL_SECONDS}"
  done
}

echo "[RC4-RUN] run_id=${RUN_ID} rows=${#ROWS[@]} dry_run=${DRY_RUN} epochs=200 U=256"
if [[ "${DRY_RUN}" == 1 ]]; then
  for row in "${ROWS[@]}"; do IFS=$'\t' read -r gpu candidate hard partial partial_set partial_conditional negative cap satellite p_budget n_budget checkpoint <<<"${row}"; echo "[RC4-ROW] gpu=${gpu} candidate=${candidate} H=${hard} Pset=${partial_set} Pcond=${partial_conditional} N=${negative} cap=${cap} sat=${satellite}"; run_row "$gpu" "$candidate" "$hard" "$partial" "$partial_set" "$partial_conditional" "$negative" "$cap" "$satellite" "$p_budget" "$n_budget" "$checkpoint"; done
  exit 0
fi
[[ ! -e "${RUNS_ROOT}" ]] || { echo "[RC4-ERROR] refusing overwrite: ${RUNS_ROOT}" >&2; exit 3; }
mkdir -p "${RUNS_ROOT}/dispatcher_logs"
pids=(); names=()
for row in "${ROWS[@]}"; do IFS=$'\t' read -r gpu candidate hard partial partial_set partial_conditional negative cap satellite p_budget n_budget checkpoint <<<"${row}"; (wait_for_gpu_slot "$gpu" && run_row "$gpu" "$candidate" "$hard" "$partial" "$partial_set" "$partial_conditional" "$negative" "$cap" "$satellite" "$p_budget" "$n_budget" "$checkpoint") >"${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 & pids+=("$!"); names+=("${candidate}"); done
failed=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[RC4-WORKER-FAILED] ${names[$i]}" >&2; failed=1; }; done
[[ "${failed}" == 0 ]] || exit 4
echo "[RC4-COMPLETE] ${RUN_ID}"
