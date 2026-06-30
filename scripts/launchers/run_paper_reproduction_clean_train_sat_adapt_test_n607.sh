#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/paper_reproduction_clean_train_sat_adapt_test_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/paper_reproduction_clean_train_sat_adapt_test_${STAMP}}"
SEEDS="${SEEDS:-1337 2027 3407}"
SEEDS="${SEEDS//,/ }"
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/code:${PYTHONPATH:-}"

cat > "${RUN_ROOT}/launch_manifest.json" <<JSON
{
  "run_root": "${RUN_ROOT}",
  "log_root": "${LOG_ROOT}",
  "uses_cen51": false,
  "training_origin": "paper_baseline_random_init",
  "stage": "Stage2-C",
  "channel_line": "clean_train_satellite_adapt_test",
  "source_train_satellite_augmentation": false,
  "target_support_satellite_augmentation": true,
  "target_query_satellite_augmentation": true,
  "source_train_channel_scenarios": ["clean"],
  "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"],
  "adaptation_mode": "support_head_finetune",
  "seeds": "${SEEDS}",
  "baselines": ["protonet_cda", "feature_separation_crossrx"],
  "max_concurrent_per_gpu": 1
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
  launch_one "protonet_cda_clean_train_sat_adapt_test" "paper_reproduction/configs/protonet_cda_cvs_stage2c_clean_train_sat_adapt_test_n607.json" "${seed}" 0
  p1="${LAST_PID}"
  launch_one "feature_separation_clean_train_sat_adapt_test" "paper_reproduction/configs/feature_separation_cvs_stage2c_clean_train_sat_adapt_test_n607.json" "${seed}" 1
  p2="${LAST_PID}"
  echo "SEED=${seed} PROTONET_PID=${p1} FEATURE_SEPARATION_PID=${p2}"
  wait "${p1}" "${p2}"
  echo "WAVE_END seed=${seed}"
done

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
