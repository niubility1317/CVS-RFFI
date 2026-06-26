#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/paper_reproduction_full_clean_sat_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/paper_reproduction_full_clean_sat_${STAMP}}"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
export PYTHONPATH="$ROOT:$ROOT/code:${PYTHONPATH:-}"

launch_one() {
  local gpu="$1"
  local name="$2"
  local config="$3"
  mkdir -p "$RUN_ROOT/$name"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m paper_reproduction.cvs_aligned.evaluate \
    --config "$config" \
    --device cuda:0 \
    --run-dir "$RUN_ROOT/$name" \
    > "$LOG_ROOT/$name.out" 2>&1 &
  echo "$!"
}

P_PROTO_CLEAN=$(launch_one 0 protonet_cda_clean_seed1337 paper_reproduction/configs/protonet_cda_cvs_stage2c_clean_full_n607.json)
P_PROTO_SAT=$(launch_one 1 protonet_cda_satellite_seed1337 paper_reproduction/configs/protonet_cda_cvs_stage2c_satellite_full_n607.json)
P_FEAT_CLEAN=$(launch_one 2 feature_separation_clean_seed1337 paper_reproduction/configs/feature_separation_cvs_stage2c_clean_full_n607.json)
P_FEAT_SAT=$(launch_one 3 feature_separation_satellite_seed1337 paper_reproduction/configs/feature_separation_cvs_stage2c_satellite_full_n607.json)

cat > "$RUN_ROOT/launch_manifest.json" <<JSON
{
  "run_root": "$RUN_ROOT",
  "log_root": "$LOG_ROOT",
  "cvs_extension": true,
  "uses_cen51": false,
  "training_origin": "paper_baseline_random_init",
  "adaptation_mode": "support_prototype_registration",
  "branches": {
    "protonet_cda_clean_seed1337": {"pid": $P_PROTO_CLEAN, "gpu": 0, "channel_line": "clean_all"},
    "protonet_cda_satellite_seed1337": {"pid": $P_PROTO_SAT, "gpu": 1, "channel_line": "satellite_all"},
    "feature_separation_clean_seed1337": {"pid": $P_FEAT_CLEAN, "gpu": 2, "channel_line": "clean_all"},
    "feature_separation_satellite_seed1337": {"pid": $P_FEAT_SAT, "gpu": 3, "channel_line": "satellite_all"}
  }
}
JSON

echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "PROTO_CLEAN_PID=$P_PROTO_CLEAN"
echo "PROTO_SAT_PID=$P_PROTO_SAT"
echo "FEATURE_CLEAN_PID=$P_FEAT_CLEAN"
echo "FEATURE_SAT_PID=$P_FEAT_SAT"
