#!/usr/bin/env bash
set -euo pipefail

cd ~/2510044040/CV-SincNet
mkdir -p logs finalist_runs

# ============================================================
# CVS-RFFI SGC experiments only
# Uses root train.py / model_dual_cvsincnet.py / sgc_adapter.py
# Does NOT use baselines/*
# GPUs: 0,1,2,3
# ============================================================

if [ ! -f "train.py" ]; then
  echo "[ERROR] train.py not found. Please run this script inside ~/2510044040/CV-SincNet." >&2
  exit 1
fi

if [ ! -f "run_final_best_sgc_queue.sh" ]; then
  echo "[ERROR] run_final_best_sgc_queue.sh not found in $(pwd)." >&2
  echo "[HINT] Copy this launcher to the same CV-SincNet directory that contains run_final_best_sgc_queue.sh." >&2
  exit 1
fi

export PHASES="${PHASES:-E}"
export GPU_IDS="${GPU_IDS:-0,1,2,3}"
export GLOBAL_SEED="${GLOBAL_SEED:-1337}"
export SEED="${SEED:-1337}"

# LEO-only satellite evaluation.
# These three scenarios exist in the current and older training code.
export SAT_EVAL_ON="${SAT_EVAL_ON:-all}"
export SAT_EVAL_SCENARIOS="${SAT_EVAL_SCENARIOS:-clear_leo,low_elev_leo,rain_leo}"
export SAT_EVAL_MAX_BATCHES="${SAT_EVAL_MAX_BATCHES:--1}"

# SGC train-time satellite view: hard pure-LEO default.
export SAT_SCENARIO="${SAT_SCENARIO:-low_elev_leo}"

# Keep SGC focused on the primary source checkpoint unless explicitly overridden.
export RUN_SGC_EXTENDED="${RUN_SGC_EXTENDED:-0}"

echo "[START] CVS-RFFI SGC only"
echo "[INFO] cwd=$(pwd)"
echo "[INFO] PHASES=${PHASES}"
echo "[INFO] GPU_IDS=${GPU_IDS}"
echo "[INFO] GLOBAL_SEED=${GLOBAL_SEED}"
echo "[INFO] SAT_SCENARIO=${SAT_SCENARIO}"
echo "[INFO] SAT_EVAL_ON=${SAT_EVAL_ON}"
echo "[INFO] SAT_EVAL_SCENARIOS=${SAT_EVAL_SCENARIOS}"
echo "[INFO] SAT_EVAL_MAX_BATCHES=${SAT_EVAL_MAX_BATCHES}"
echo "[INFO] launcher=run_final_best_sgc_queue.sh"

bash run_final_best_sgc_queue.sh
