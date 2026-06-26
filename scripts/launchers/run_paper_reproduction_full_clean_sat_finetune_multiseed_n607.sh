#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(pwd)"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/paper_reproduction_full_clean_sat_finetune_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/paper_reproduction_full_clean_sat_finetune_${STAMP}}"
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
  "adaptation_mode": "support_head_finetune",
  "prototype_metric_for_protonet_source_training": "euclidean",
  "seeds": "${SEEDS}",
  "channel_lines": ["clean_all", "satellite_all"],
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
  launch_one "protonet_cda_clean" "paper_reproduction/configs/protonet_cda_cvs_stage2c_clean_finetune_n607.json" "${seed}" 0
  p1="${LAST_PID}"
  launch_one "protonet_cda_satellite" "paper_reproduction/configs/protonet_cda_cvs_stage2c_satellite_finetune_n607.json" "${seed}" 1
  p2="${LAST_PID}"
  launch_one "feature_separation_clean" "paper_reproduction/configs/feature_separation_cvs_stage2c_clean_finetune_n607.json" "${seed}" 2
  p3="${LAST_PID}"
  launch_one "feature_separation_satellite" "paper_reproduction/configs/feature_separation_cvs_stage2c_satellite_finetune_n607.json" "${seed}" 3
  p4="${LAST_PID}"
  echo "SEED=${seed} PROTO_CLEAN_PID=${p1} PROTO_SAT_PID=${p2} FEATURE_CLEAN_PID=${p3} FEATURE_SAT_PID=${p4}"
  wait "${p1}" "${p2}" "${p3}" "${p4}"
  echo "WAVE_END seed=${seed}"
done

echo "RUN_ROOT=${RUN_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
