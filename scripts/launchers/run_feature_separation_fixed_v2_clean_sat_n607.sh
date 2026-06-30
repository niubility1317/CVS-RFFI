#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/feature_separation_fixed_v2_clean_sat_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/feature_separation_fixed_v2_clean_sat_${STAMP}}"
SEEDS="${SEEDS:-1337 2027 3407}"
SEEDS="${SEEDS//,/ }"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/code:${PYTHONPATH:-}"

cat > "${RUN_ROOT}/launch_manifest.json" <<JSON
{
  "run_root": "${RUN_ROOT}",
  "log_root": "${LOG_ROOT}",
  "baseline": "feature_separation_crossrx",
  "uses_cen51": false,
  "training_origin": "paper_baseline_random_init",
  "fix": "cross_branch_entropy_v2",
  "adaptation_mode": "support_head_finetune",
  "satellite_train_augmentation": {
    "enabled_for_satellite_line": true,
    "train_channel_view": "satellite/LEO",
    "train_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
  },
  "seeds": "${SEEDS}",
  "channel_lines": ["clean_all", "satellite_all"]
}
JSON

launch_one() {
  local name="$1"
  local cfg="$2"
  local seed="$3"
  local gpu="$4"
  local run_dir="${RUN_ROOT}/${name}_seed${seed}"
  local log_file="${LOG_ROOT}/${name}_seed${seed}.out"
  mkdir -p "${run_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m paper_reproduction.cvs_aligned.evaluate \
    --config "${cfg}" \
    --run-dir "${run_dir}" \
    --device cuda:0 \
    --seed "${seed}" \
    > "${log_file}" 2>&1 &
  LAST_PID="$!"
}

for seed in ${SEEDS}; do
  echo "WAVE_BEGIN seed=${seed}"
  launch_one "feature_separation_clean_fixed_v2" "paper_reproduction/configs/feature_separation_cvs_stage2c_clean_fixed_v2_n607.json" "${seed}" 2
  p1="${LAST_PID}"
  launch_one "feature_separation_satellite_fixed_v2" "paper_reproduction/configs/feature_separation_cvs_stage2c_satellite_fixed_v2_n607.json" "${seed}" 3
  p2="${LAST_PID}"
  echo "SEED=${seed} FEATURE_CLEAN_PID=${p1} FEATURE_SAT_PID=${p2}"
  wait "${p1}" "${p2}"
  echo "WAVE_END seed=${seed}"
done

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
