#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-riei_drift_current_sat_supervised_r010_stage2b_targetold_support_head_normproto_k2050_${STAMP}}"
RUN_ROOT="${RUN_ROOT:-runs/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-logs/${RUN_ID}}"
SEED="${SEED:-1337}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/code:${PYTHONPATH:-}"

cat > "${RUN_ROOT}/launch_manifest.json" <<JSON
{
  "run_id": "${RUN_ID}",
  "run_root": "${RUN_ROOT}",
  "log_root": "${LOG_ROOT}",
  "seed": ${SEED},
  "stage": "Stage2-B",
  "phase2_adapter": "SupportHead-CDA",
  "methods": ["RIEI", "DRIFT"],
  "k_shot_grid": [20, 50],
  "training_origin": "current_satellite_supervised_only_checkpoint",
  "source_checkpoint_run_id": "riei_drift_no_unlabeled_r010_sat_20260707_191029",
  "source_train_ratio": 0.1,
  "source_training_satellite_augmentation": true,
  "target_support_satellite_augmentation": true,
  "target_query_satellite_augmentation": true,
  "target_receiver_labels": ["20-1", "3-19", "7-14", "7-7", "8-8"],
  "target_old_only": true,
  "target_new_enabled": false,
  "unknown_rejection_enabled": false,
  "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
  "adaptation_mode": "support_head_finetune",
  "support_amount_diagnostic": true,
  "support_finetune_normalize": true,
  "support_head_init": "prototype",
  "support_head_temperature": 10.0,
  "support_finetune_steps": 80,
  "support_finetune_lr": 0.01,
  "support_finetune_weight_decay": 0.001,
  "cvs_extension": true
}
JSON

launch_one() {
  local name="$1"
  local cfg="$2"
  local gpu="$3"
  local run_dir="${RUN_ROOT}/${name}_seed${SEED}"
  local log_file="${LOG_ROOT}/${name}_seed${SEED}.out"
  mkdir -p "${run_dir}"
  local cmd=(
    env CUDA_VISIBLE_DEVICES="${gpu}"
    "${PYTHON}" -m paper_reproduction.cvs_aligned.evaluate
    --config "${cfg}"
    --run-dir "${run_dir}"
    --device cuda:0
    --seed "${SEED}"
    --formal
  )
  printf "[TARGETOLD-SUPPORT-HEAD-NORMPROTO-K2050-CMD] "
  printf "%q " "${cmd[@]}"
  printf "\n"
  if [[ "${DRY_RUN}" == "1" ]]; then
    "${PYTHON}" -m paper_reproduction.cvs_aligned.evaluate --config "${cfg}" --dry-run --formal
    return 0
  fi
  "${cmd[@]}" > "${log_file}" 2>&1 &
  LAST_PID="$!"
  echo "[TARGETOLD-SUPPORT-HEAD-NORMPROTO-K2050-LAUNCHED] name=${name} pid=${LAST_PID} gpu=${gpu} log=${log_file} run_dir=${run_dir}"
}

echo "[TARGETOLD-SUPPORT-HEAD-NORMPROTO-K2050] run_id=${RUN_ID} dry_run=${DRY_RUN}"
echo "[TARGETOLD-SUPPORT-HEAD-NORMPROTO-K2050] protocol=Stage2-B old_only target_new=false unknown_rejection=false source_train_ratio=0.1"

launch_one "riei_fd_current_sat_k20_leo_support_head_normproto" "paper_reproduction/configs/riei_fd_current_sat_supervised_r010_cvs_stage2b_k20_leo_support_head_normproto_n607.json" 0
p1="${LAST_PID:-}"
launch_one "drift_current_sat_k20_leo_support_head_normproto" "paper_reproduction/configs/drift_current_sat_supervised_r010_cvs_stage2b_k20_leo_support_head_normproto_n607.json" 1
p2="${LAST_PID:-}"
launch_one "riei_fd_current_sat_k50_leo_support_head_normproto" "paper_reproduction/configs/riei_fd_current_sat_supervised_r010_cvs_stage2b_k50_leo_support_head_normproto_n607.json" 2
p3="${LAST_PID:-}"
launch_one "drift_current_sat_k50_leo_support_head_normproto" "paper_reproduction/configs/drift_current_sat_supervised_r010_cvs_stage2b_k50_leo_support_head_normproto_n607.json" 3
p4="${LAST_PID:-}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[TARGETOLD-SUPPORT-HEAD-NORMPROTO-K2050-DRY-RUN-DONE] run_root=${RUN_ROOT} log_root=${LOG_ROOT}"
  exit 0
fi

wait "${p1}" "${p2}" "${p3}" "${p4}"

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
