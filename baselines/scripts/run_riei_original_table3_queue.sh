#!/usr/bin/env bash
set -euo pipefail

# RIEI original paper Table III WiSig protocol launcher.
# Dry-run is the default so users can inspect the full queue before launching.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

WISIG_PKL="${WISIG_PKL:-${REPO_ROOT}/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/runs/riei_original_table3_seed1337}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/riei_original_table3_seed1337}"
GPU_IDS="${GPU_IDS:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-1}"

usage() {
  cat <<'EOF'
Options:
  --gpu-ids CSV     GPU ids forwarded to run_cvs_baseline_queue.sh
  --wisig-pkl PATH  Dataset_WigSig/ManySig.pkl path
  --python PATH     Python executable forwarded to the child launcher
  --run-root PATH   Root for run outputs
  --log-root PATH   Root for logs
  --seed N          Random seed
  --dry-run         Print commands only (default)
  --launch          Execute the 12 runs sequentially
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --gpu-ids) GPU_IDS="$2"; shift 2 ;;
    --wisig-pkl) WISIG_PKL="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --log-root) LOG_ROOT="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --launch) DRY_RUN=0; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

declare -a TABLE3_ROWS=(
  "rx1_1_rx7_7_to_rx1_19|1-1,7-7|1-19"
  "rx1_1_rx8_8_to_rx1_19|1-1,8-8|1-19"
  "rx1_1_rx14_7_to_rx1_19|1-1,14-7|1-19"
  "rx7_7_rx8_8_to_rx1_19|7-7,8-8|1-19"
  "rx7_7_rx14_7_to_rx1_19|7-7,14-7|1-19"
  "rx8_8_rx14_7_to_rx1_19|8-8,14-7|1-19"
  "rx1_1_rx1_19_to_rx14_7|1-1,1-19|14-7"
  "rx1_1_rx7_7_to_rx14_7|1-1,7-7|14-7"
  "rx1_1_rx8_8_to_rx14_7|1-1,8-8|14-7"
  "rx1_19_rx7_7_to_rx14_7|1-19,7-7|14-7"
  "rx1_19_rx8_8_to_rx14_7|1-19,8-8|14-7"
  "rx7_7_rx8_8_to_rx14_7|7-7,8-8|14-7"
)

echo "[RIEI-TABLE3] root=${REPO_ROOT} rows=${#TABLE3_ROWS[@]} gpu_ids=${GPU_IDS} dry_run=${DRY_RUN}"
status=0
for row in "${TABLE3_ROWS[@]}"; do
  IFS='|' read -r combo_id train_rxs test_rxs <<< "${row}"
  child_args=(
    "${REPO_ROOT}/scripts/launchers/run_cvs_baseline_queue.sh"
    --methods riei_fd
    --wisig-protocol riei_original
    --gpu-ids "${GPU_IDS}"
    --wisig-pkl "${WISIG_PKL}"
    --run-root "${RUN_ROOT}/${combo_id}"
    --log-root "${LOG_ROOT}/${combo_id}"
    --seed "${SEED}"
  )
  if [ -n "${PYTHON_BIN}" ]; then
    child_args+=(--python "${PYTHON_BIN}")
  fi
  if [ "${DRY_RUN}" = "1" ]; then
    child_args+=(--dry-run)
  fi
  echo "[RIEI-TABLE3][${combo_id}] train_rxs=${train_rxs} test_rxs=${test_rxs}"
  if TRAIN_RXS="${train_rxs}" \
      TEST_RXS="${test_rxs}" \
      RIEI_PAPER_EVAL_LAST_N=10 \
      bash "${child_args[@]}"; then
    :
  else
    rc=$?
    status=${rc}
    echo "[RIEI-TABLE3][${combo_id}] failed rc=${rc}" >&2
    if [ "${DRY_RUN}" != "1" ]; then
      exit "${rc}"
    fi
  fi
done

echo "[RIEI-TABLE3] finished status=${status}"
exit "${status}"
