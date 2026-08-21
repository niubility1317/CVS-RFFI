#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${ROOT}/configs/phase1_adv3b02_fasttrust16_s392002_20260821.json}"
RUN_ID="${RUN_ID:-phase1_adv3b02_fasttrust16_s392002_20260821}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER_LAUNCHER="${WORKER_LAUNCHER:-${ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
ONLY_CANDIDATE=""
GPU_FILTER=""
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;
    --gpu=*) GPU_FILTER="${arg#--gpu=}" ;;
    *) echo "[FASTTRUST-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -n "${GPU_FILTER}" && ! "${GPU_FILTER}" =~ ^[0-7]$ ]]; then
  echo "[FASTTRUST-ERROR] --gpu must be in 0..7" >&2
  exit 2
fi
if [[ ! -f "${MATRIX}" ]]; then
  echo "[FASTTRUST-ERROR] matrix missing: ${MATRIX}" >&2
  exit 2
fi
if [[ ! -f "${WORKER_LAUNCHER}" ]]; then
  echo "[FASTTRUST-ERROR] worker launcher missing: ${WORKER_LAUNCHER}" >&2
  exit 2
fi

map_candidate() {
  local candidate="$1"
  local muse="$2"
  LEVEL=""
  ABLATION_NAME="NONE"
  case "${muse}" in
    off) LEVEL="M0" ;;
    fast_hml) LEVEL="M2" ;;
    fasttrust_full) LEVEL="M3" ;;
    *) echo "[FASTTRUST-ERROR] unsupported muse mapping: ${muse}" >&2; return 2 ;;
  esac
  case "${candidate}" in
    R3_FAST_HML_UPROTO_U256) ABLATION_NAME="U_PROTO" ;;
    R4_NO_U_SAT_ID_U256) ABLATION_NAME="NO_U_SATELLITE_ID" ;;
    R4_NO_PROTO_EVIDENCE_U256) ABLATION_NAME="NO_PROTO_EVIDENCE" ;;
    R4_NO_U_PROTO_UPDATE_U256) ABLATION_NAME="NO_U_PROTO_UPDATE" ;;
    R4_NO_TEMPORAL_U256) ABLATION_NAME="NO_TEMPORAL" ;;
    R4_NO_PRIOR_U256) ABLATION_NAME="NO_PRIOR" ;;
    R4_NUISANCE_DETACHED_U256) ABLATION_NAME="NUISANCE_DETACHED" ;;
    R4_NO_NUISANCE_U256) ABLATION_NAME="NO_NUISANCE" ;;
    R4_NO_CROSSRX_U256) ABLATION_NAME="NO_CROSSRX" ;;
    R4_NO_CLASS_CAP_U256) ABLATION_NAME="NO_CLASS_CAP" ;;
  esac
}

mapfile -t MATRIX_ROWS < <(
  "${CONTROL_PYTHON}" -c '
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if data.get("schema") != "cvs.phase1.fasttrust_matrix.v1":
    raise SystemExit("unsupported matrix schema")
if int(data.get("seed", -1)) != 392002 or int(data.get("epochs", -1)) != 200:
    raise SystemExit("matrix seed/epoch drift")
rows = data.get("rows", [])
if len(rows) != 16:
    raise SystemExit("matrix must contain exactly 16 rows")
counts = {gpu: 0 for gpu in range(8)}
seen = set()
for row in rows:
    gpu = int(row["gpu"])
    candidate = str(row["candidate"])
    if gpu not in counts or candidate in seen:
        raise SystemExit("invalid or duplicate matrix row")
    counts[gpu] += 1
    seen.add(candidate)
    print("\t".join((
        str(gpu), str(row["slot"]), candidate, str(row["init"]),
        str(row["muse"]), str(int(row["u_batch"])),
        ",".join(str(item) for item in row.get("changes", [])),
        str(data["base_checkpoint"]),
    )))
if any(value != 2 for value in counts.values()):
    raise SystemExit("matrix must assign exactly two rows per GPU")
' "${MATRIX}"
)

SELECTED_ROWS=()
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r gpu slot candidate init_mode muse u_batch changes base_checkpoint <<< "${row}"
  if [[ -n "${ONLY_CANDIDATE}" && "${candidate}" != "${ONLY_CANDIDATE}" ]]; then
    continue
  fi
  if [[ -n "${GPU_FILTER}" && "${gpu}" != "${GPU_FILTER}" ]]; then
    continue
  fi
  SELECTED_ROWS+=("${row}")
done
if [[ "${#SELECTED_ROWS[@]}" -eq 0 ]]; then
  echo "[FASTTRUST-ERROR] selection matched no matrix row" >&2
  exit 2
fi

run_row() {
  local gpu="$1"
  local candidate="$2"
  local init_mode="$3"
  local muse="$4"
  local u_batch="$5"
  local base_checkpoint="$6"
  map_candidate "${candidate}" "${muse}"
  local args=("--only=${LEVEL}")
  if [[ "${DRY_RUN}" == "1" ]]; then
    args+=(--dry-run)
  fi
  env \
    ROOT="${ROOT}" \
    PYTHON="${PYTHON}" \
    CONTROL_PYTHON="${CONTROL_PYTHON}" \
    RUN_ID="${RUN_ID}" \
    RUNS_ROOT="${RUNS_ROOT}" \
    GPU="${gpu}" \
    INIT_MODE="${init_mode}" \
    BASE_CKPT="${ROOT}/${base_checkpoint}" \
    CANDIDATE_ID_OVERRIDE="${candidate}" \
    MUSE_UNLABELED_BATCH_SIZE="${u_batch}" \
    ABLATION="${ABLATION_NAME}" \
    bash "${WORKER_LAUNCHER}" "${args[@]}"
}

echo "[FASTTRUST-RUN] run_id=${RUN_ID} matrix=${MATRIX} selected=${#SELECTED_ROWS[@]} dry_run=${DRY_RUN} mode=two_concurrent_per_gpu"
if [[ "${DRY_RUN}" == "1" ]]; then
  for row in "${SELECTED_ROWS[@]}"; do
    IFS=$'\t' read -r gpu slot candidate init_mode muse u_batch changes base_checkpoint <<< "${row}"
    echo "[FASTTRUST-ROW] gpu=${gpu} slot=${slot} candidate=${candidate} init=${init_mode} muse=${muse} u_batch=${u_batch} changes=${changes}"
    run_row "${gpu}" "${candidate}" "${init_mode}" "${muse}" "${u_batch}" "${base_checkpoint}"
  done
  exit 0
fi

if [[ -e "${RUNS_ROOT}" ]]; then
  echo "[FASTTRUST-ERROR] refusing to overwrite existing run root: ${RUNS_ROOT}" >&2
  exit 3
fi
mkdir -p "${RUNS_ROOT}/dispatcher_logs"

PIDS=()
CANDIDATES=()
for row in "${SELECTED_ROWS[@]}"; do
  IFS=$'\t' read -r gpu slot candidate init_mode muse u_batch changes base_checkpoint <<< "${row}"
  echo "[FASTTRUST-ROW] gpu=${gpu} slot=${slot} candidate=${candidate} init=${init_mode} muse=${muse} u_batch=${u_batch} changes=${changes}"
  (
    run_row "${gpu}" "${candidate}" "${init_mode}" "${muse}" "${u_batch}" "${base_checkpoint}"
  ) > "${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 &
  PIDS+=("$!")
  CANDIDATES+=("${candidate}")
done

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[FASTTRUST-WORKER-COMPLETE] candidate=${CANDIDATES[$index]}"
  else
    echo "[FASTTRUST-WORKER-FAILED] candidate=${CANDIDATES[$index]} log=${RUNS_ROOT}/dispatcher_logs/${CANDIDATES[$index]}.log" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  exit 4
fi
echo "[FASTTRUST-COMPLETE] run_id=${RUN_ID} status=ALL_SELECTED_ROWS_COMPLETE"
