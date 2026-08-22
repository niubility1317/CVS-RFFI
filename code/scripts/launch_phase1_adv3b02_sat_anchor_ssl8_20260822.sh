#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
CODE_ROOT="${CODE_ROOT:-${ROOT}}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
CONTROL_PYTHON="${CONTROL_PYTHON:-${PYTHON}}"
MATRIX="${MATRIX:-${CODE_ROOT}/configs/phase1_adv3b02_sat_anchor_ssl8_s392002_20260822.json}"
RUN_ID="${RUN_ID:-phase1_adv3b02_sat_anchor_ssl8_s392002_20260822}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT}/runs/${RUN_ID}}"
WORKER_LAUNCHER="${WORKER_LAUNCHER:-${CODE_ROOT}/code/scripts/launch_phase1_adv3b02_muse_ssdg_20260819.sh}"
ONLY_CANDIDATE=""
GPU_FILTER=""
DRY_RUN=0

for arg in "$@"; do
  case "${arg}" in
    --dry-run) DRY_RUN=1 ;;
    --only=*) ONLY_CANDIDATE="${arg#--only=}" ;;
    --gpu=*) GPU_FILTER="${arg#--gpu=}" ;;
    *) echo "[SAT-ANCHOR-ERROR] unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

if [[ -n "${GPU_FILTER}" && ! "${GPU_FILTER}" =~ ^[0-7]$ ]]; then
  echo "[SAT-ANCHOR-ERROR] --gpu must be in 0..7" >&2
  exit 2
fi
[[ -f "${MATRIX}" ]] || { echo "[SAT-ANCHOR-ERROR] matrix missing: ${MATRIX}" >&2; exit 2; }
[[ -f "${WORKER_LAUNCHER}" ]] || { echo "[SAT-ANCHOR-ERROR] worker missing: ${WORKER_LAUNCHER}" >&2; exit 2; }

mapfile -t MATRIX_ROWS < <(
  "${CONTROL_PYTHON}" -c '
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if data.get("schema") != "cvs.phase1.sat_anchor_matrix.v1":
    raise SystemExit("unsupported matrix schema")
if int(data.get("seed", -1)) != 392002 or int(data.get("epochs", -1)) != 200:
    raise SystemExit("matrix seed/epoch drift")
rows = data.get("rows", [])
if len(rows) != 8:
    raise SystemExit("matrix must contain exactly 8 rows")
seen_gpu, seen_candidate = set(), set()
for row in rows:
    gpu = int(row["gpu"])
    candidate = str(row["candidate"])
    if gpu not in range(8) or gpu in seen_gpu or candidate in seen_candidate:
        raise SystemExit("matrix requires one unique candidate per GPU")
    seen_gpu.add(gpu)
    seen_candidate.add(candidate)
    print("\t".join((
        str(gpu), candidate, str(row["muse_level"]),
        str(bool(row["sat_anchor_ssl"])).lower(),
        str(float(row["lambda_pair"])),
        str(int(row["pair_interval"])),
        str(float(row["lambda_satellite"])),
        str(float(row["lambda_clean_kl"])),
        str(float(row["fill_to_fraction"])),
        str(bool(row["receiver_cap"])).lower(),
        str(bool(row["adapter"])).lower(),
        str(row["gradient_scope"]),
        str(data["base_checkpoint"]),
    )))
if seen_gpu != set(range(8)):
    raise SystemExit("matrix must cover GPU0..GPU7 exactly once")
' "${MATRIX}"
)

SELECTED_ROWS=()
for row in "${MATRIX_ROWS[@]}"; do
  IFS=$'\t' read -r gpu candidate muse_level sat_anchor_ssl lambda_pair pair_interval lambda_satellite lambda_clean_kl fill receiver_cap adapter gradient_scope base_checkpoint <<< "${row}"
  [[ -z "${ONLY_CANDIDATE}" || "${candidate}" == "${ONLY_CANDIDATE}" ]] || continue
  [[ -z "${GPU_FILTER}" || "${gpu}" == "${GPU_FILTER}" ]] || continue
  SELECTED_ROWS+=("${row}")
done
if [[ "${#SELECTED_ROWS[@]}" -eq 0 ]]; then
  echo "[SAT-ANCHOR-ERROR] selection matched no matrix row" >&2
  exit 2
fi

run_row() {
  local gpu="$1" candidate="$2" muse_level="$3" sat_anchor_ssl="$4"
  local lambda_pair="$5" pair_interval="$6"
  local lambda_satellite="$7" lambda_clean_kl="$8" fill="$9"
  local receiver_cap="${10}" adapter="${11}" gradient_scope="${12}" base_checkpoint="${13}"
  local args=("--only=${muse_level}")
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)
  env \
    ROOT="${ROOT}" CODE_ROOT="${CODE_ROOT}" PYTHON="${PYTHON}" CONTROL_PYTHON="${CONTROL_PYTHON}" \
    RUN_ID="${RUN_ID}" RUNS_ROOT="${RUNS_ROOT}" GPU="${gpu}" \
    INIT_MODE=adv3b02_core90 BASE_CKPT="${ROOT}/${base_checkpoint}" \
    CANDIDATE_ID_OVERRIDE="${candidate}" MUSE_UNLABELED_BATCH_SIZE=256 ABLATION=NONE \
    SAT_ANCHOR_SSL="${sat_anchor_ssl}" SAT_ANCHOR_LAMBDA_PAIR="${lambda_pair}" \
    SAT_ANCHOR_PAIR_INTERVAL="${pair_interval}" \
    SAT_ANCHOR_LAMBDA_SATELLITE="${lambda_satellite}" \
    SAT_ANCHOR_LAMBDA_CLEAN_KL="${lambda_clean_kl}" \
    SAT_ANCHOR_FILL_TO_FRACTION="${fill}" SAT_ANCHOR_RECEIVER_CAP="${receiver_cap}" \
    SAT_ANCHOR_ADAPTER="${adapter}" SAT_ANCHOR_GRADIENT_SCOPE="${gradient_scope}" \
    bash "${WORKER_LAUNCHER}" "${args[@]}"
}

echo "[SAT-ANCHOR-RUN] run_id=${RUN_ID} selected=${#SELECTED_ROWS[@]} dry_run=${DRY_RUN} mode=one_training_process_per_gpu"
if [[ "${DRY_RUN}" == "1" ]]; then
  for row in "${SELECTED_ROWS[@]}"; do
    IFS=$'\t' read -r gpu candidate muse_level sat_anchor_ssl lambda_pair pair_interval lambda_satellite lambda_clean_kl fill receiver_cap adapter gradient_scope base_checkpoint <<< "${row}"
    echo "[SAT-ANCHOR-ROW] gpu=${gpu} candidate=${candidate} muse=${muse_level} enabled=${sat_anchor_ssl} pair=${lambda_pair}@${pair_interval} sat=${lambda_satellite} anchor=${lambda_clean_kl} fill=${fill} rx_cap=${receiver_cap} adapter=${adapter} scope=${gradient_scope}"
    run_row "${gpu}" "${candidate}" "${muse_level}" "${sat_anchor_ssl}" "${lambda_pair}" "${pair_interval}" "${lambda_satellite}" "${lambda_clean_kl}" "${fill}" "${receiver_cap}" "${adapter}" "${gradient_scope}" "${base_checkpoint}"
  done
  exit 0
fi

if [[ -e "${RUNS_ROOT}" ]]; then
  echo "[SAT-ANCHOR-ERROR] refusing to overwrite existing run root: ${RUNS_ROOT}" >&2
  exit 3
fi
mkdir -p "${RUNS_ROOT}/dispatcher_logs"

PIDS=()
CANDIDATES=()
for row in "${SELECTED_ROWS[@]}"; do
  IFS=$'\t' read -r gpu candidate muse_level sat_anchor_ssl lambda_pair pair_interval lambda_satellite lambda_clean_kl fill receiver_cap adapter gradient_scope base_checkpoint <<< "${row}"
  echo "[SAT-ANCHOR-ROW] gpu=${gpu} candidate=${candidate} muse=${muse_level} enabled=${sat_anchor_ssl} pair=${lambda_pair}@${pair_interval} sat=${lambda_satellite} anchor=${lambda_clean_kl} fill=${fill} rx_cap=${receiver_cap} adapter=${adapter} scope=${gradient_scope}"
  (
    run_row "${gpu}" "${candidate}" "${muse_level}" "${sat_anchor_ssl}" "${lambda_pair}" "${pair_interval}" "${lambda_satellite}" "${lambda_clean_kl}" "${fill}" "${receiver_cap}" "${adapter}" "${gradient_scope}" "${base_checkpoint}"
  ) > "${RUNS_ROOT}/dispatcher_logs/${candidate}.log" 2>&1 &
  PIDS+=("$!")
  CANDIDATES+=("${candidate}")
done

failed=0
for index in "${!PIDS[@]}"; do
  if wait "${PIDS[$index]}"; then
    echo "[SAT-ANCHOR-WORKER-COMPLETE] candidate=${CANDIDATES[$index]}"
  else
    echo "[SAT-ANCHOR-WORKER-FAILED] candidate=${CANDIDATES[$index]}" >&2
    failed=1
  fi
done
[[ "${failed}" -eq 0 ]] || exit 4
echo "[SAT-ANCHOR-COMPLETE] run_id=${RUN_ID} status=ALL_SELECTED_ROWS_COMPLETE"
