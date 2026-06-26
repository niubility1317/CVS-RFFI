#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/szu2070436088/2510044040/CV-SincNet}"
PYTHON="${PYTHON:-/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/paper_reproduction_cvs_stage2c_smoke_${STAMP}}"
LOG_ROOT="${LOG_ROOT:-logs/paper_reproduction_cvs_stage2c_smoke_${STAMP}}"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
export PYTHONPATH="$ROOT:$ROOT/code:${PYTHONPATH:-}"

nohup env CUDA_VISIBLE_DEVICES=0 "$PYTHON" -m paper_reproduction.cvs_aligned.evaluate \
  --config paper_reproduction/configs/protonet_cda_cvs_stage2c_wisig_n607_smoke.json \
  --device cuda:0 \
  --run-dir "$RUN_ROOT/protonet_cda_stage2c_seed1337" \
  > "$LOG_ROOT/protonet_cda_stage2c.out" 2>&1 &
PROTO_PID=$!

nohup env CUDA_VISIBLE_DEVICES=1 "$PYTHON" -m paper_reproduction.cvs_aligned.evaluate \
  --config paper_reproduction/configs/feature_separation_cvs_stage2c_wisig_n607_smoke.json \
  --device cuda:0 \
  --run-dir "$RUN_ROOT/feature_separation_stage2c_seed1337" \
  > "$LOG_ROOT/feature_separation_stage2c.out" 2>&1 &
FEATURE_PID=$!

cat > "$RUN_ROOT/launch_manifest.json" <<JSON
{
  "run_root": "$RUN_ROOT",
  "log_root": "$LOG_ROOT",
  "protonet_pid": $PROTO_PID,
  "feature_separation_pid": $FEATURE_PID,
  "stage": "Stage2-C",
  "cvs_extension": true,
  "target_channel_view": "satellite/LEO",
  "target_channel_scenarios": ["leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"]
}
JSON

echo "RUN_ROOT=$RUN_ROOT"
echo "LOG_ROOT=$LOG_ROOT"
echo "PROTO_PID=$PROTO_PID"
echo "FEATURE_PID=$FEATURE_PID"
