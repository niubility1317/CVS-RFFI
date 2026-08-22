#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_fasttrust_rc4_s392002_20260822.json}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fasttrust_rc4_s392002_20260822}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER="${WORKER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
[[ -f "${MATRIX}" && -f "${WORKER}" ]] || { echo "[RC4-ERROR] matrix/worker missing" >&2; exit 2; }

mapfile -t ROWS < <("${CONTROL_PYTHON}" -c '
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert d["schema"]=="cvs.phase1.fasttrust_rc4_matrix.v1" and d["seed"]==392002 and d["epochs"]==200
rows=d["rows"]
assert len(rows)==8 and {r["gpu"] for r in rows}==set(range(8)) and len({r["candidate"] for r in rows})==8
for r in rows:
 print("\t".join([str(r["gpu"]),r["candidate"]]+[str(bool(r[k])).lower() for k in ("anchor","calibration","hard","partial","negative","class_rx_cap","satellite")]+[d["base_checkpoint"]]))
' "${MATRIX}")

run_row() {
  local gpu="$1" candidate="$2" anchor="$3" calibration="$4" hard="$5" partial="$6" negative="$7" cap="$8" satellite="$9" checkpoint="${10}"
  checkpoint="${checkpoint%$'\r'}"
  local extra=()
  [[ "${DRY_RUN}" == 1 ]] && extra+=(--dry-run)
  env ROOT="${ROOT}" CODE_ROOT="${CODE_ROOT}" PYTHON="${PYTHON}" CONTROL_PYTHON="${CONTROL_PYTHON}" \
    RUN_ID="${RUN_ID}" RUNS_ROOT="${RUNS_ROOT}" GPU="${gpu}" INIT_MODE=adv3b02_core90 \
    BASE_CKPT="${ROOT}/${checkpoint}" CANDIDATE_ID_OVERRIDE="${candidate}" ABLATION=NONE \
    MUSE_UNLABELED_BATCH_SIZE=256 FASTTRUST_RC4=true SAT_ANCHOR_SSL=false \
    RC4_USE_ANCHOR="${anchor}" RC4_USE_CALIBRATION="${calibration}" RC4_ENABLE_HARD="${hard}" \
    RC4_ENABLE_PARTIAL="${partial}" RC4_ENABLE_NEGATIVE="${negative}" RC4_CLASS_RX_CAP="${cap}" \
    RC4_SATELLITE="${satellite}" bash "${WORKER}" --only=M3 "${extra[@]}"
}

echo "[RC4-RUN] run_id=${RUN_ID} rows=${#ROWS[@]} dry_run=${DRY_RUN} U=256"
if [[ "${DRY_RUN}" == 1 ]]; then
  for row in "${ROWS[@]}"; do IFS=$'\t' read -r gpu candidate anchor calibration hard partial negative cap satellite checkpoint <<<"${row}"; echo "[RC4-ROW] gpu=${gpu} candidate=${candidate} H=${hard} P=${partial} N=${negative} cap=${cap} sat=${satellite}"; run_row "$gpu" "$candidate" "$anchor" "$calibration" "$hard" "$partial" "$negative" "$cap" "$satellite" "$checkpoint"; done
  exit 0
fi
[[ ! -e "${RUNS_ROOT}" ]] || { echo "[RC4-ERROR] refusing overwrite: ${RUNS_ROOT}" >&2; exit 3; }
mkdir -p "${RUNS_ROOT}/dispatcher_logs"
pids=(); names=()
for row in "${ROWS[@]}"; do IFS=$'\t' read -r gpu candidate anchor calibration hard partial negative cap satellite checkpoint <<<"${row}"; (run_row "$gpu" "$candidate" "$anchor" "$calibration" "$hard" "$partial" "$negative" "$cap" "$satellite" "$checkpoint") >"${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 & pids+=("$!"); names+=("${candidate}"); done
failed=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "[RC4-WORKER-FAILED] ${names[$i]}" >&2; failed=1; }; done
[[ "${failed}" == 0 ]] || exit 4
echo "[RC4-COMPLETE] ${RUN_ID}"
