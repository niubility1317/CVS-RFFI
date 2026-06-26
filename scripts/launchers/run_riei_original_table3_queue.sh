#!/usr/bin/env bash
set -euo pipefail

# RIEI original paper Table III WiSig protocol launcher.
# It expands the 12 receiver-combination rows from the paper and delegates
# each row to run_wisig_paper_scope_queue.sh with METHODS=riei_fd and
# WISIG_PROTOCOL=riei_original. RA-Collab and DRIFT are intentionally excluded.

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WISIG_PKL="${WISIG_PKL:-$ROOT/Dataset_WigSig/ManySig.pkl}"
RUN_ROOT="${RUN_ROOT:-$ROOT/runs/riei_original_table3_seed1337}"
LOG_ROOT="${LOG_ROOT:-$ROOT/logs/riei_original_table3_seed1337}"
GPU_IDS="${GPU_IDS:-0}"
PYTHON_BIN="${PYTHON_BIN:-}"
DRY_RUN="${DRY_RUN:-1}"

usage() {
  cat <<'EOF'
Options:
  --gpu-ids CSV     GPU ids forwarded to run_wisig_paper_scope_queue.sh
  --wisig-pkl PATH  Dataset_WigSig/ManySig.pkl path
  --python PATH     Python executable forwarded to the child launcher
  --run-root PATH   Root for run outputs
  --log-root PATH   Root for logs
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

echo "[RIEI-TABLE3] root=$ROOT rows=${#TABLE3_ROWS[@]} gpu_ids=$GPU_IDS dry_run=$DRY_RUN"
status=0
for row in "${TABLE3_ROWS[@]}"; do
  IFS='|' read -r combo_id train_rxs test_rxs <<< "$row"
  child_args=(
    "$ROOT/run_wisig_paper_scope_queue.sh"
    --methods riei_fd
    --wisig-protocol riei_original
    --gpu-ids "$GPU_IDS"
    --wisig-pkl "$WISIG_PKL"
    --run-root "$RUN_ROOT/$combo_id"
    --log-root "$LOG_ROOT/$combo_id"
  )
  if [ -n "$PYTHON_BIN" ]; then
    child_args+=(--python "$PYTHON_BIN")
  fi
  if [ "$DRY_RUN" = "1" ]; then
    child_args+=(--dry-run)
  fi
  echo "[RIEI-TABLE3][$combo_id] train_rxs=$train_rxs test_rxs=$test_rxs"
  if METHODS=riei_fd \
      WISIG_PROTOCOL=riei_original \
      TRAIN_RXS="$train_rxs" \
      TEST_RXS="$test_rxs" \
      RIEI_PAPER_EVAL_LAST_N=10 \
      DRY_RUN="$DRY_RUN" \
      bash "${child_args[@]}"; then
    :
  else
    rc=$?
    status=$rc
    echo "[RIEI-TABLE3][$combo_id] failed rc=$rc" >&2
    if [ "$DRY_RUN" != "1" ]; then
      exit "$rc"
    fi
  fi
done

echo "[RIEI-TABLE3] finished status=$status"
exit "$status"
