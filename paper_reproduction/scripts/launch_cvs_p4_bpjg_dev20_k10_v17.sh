#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/szu2070436088/2510044040/CV-SincNet
PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
EXPERIMENT_ID=qknnv42_p4_bpjg_dev20_k10_20260715_v17
CONFIG="$ROOT/paper_reproduction/configs/cvs_qknnv42_p4_bpjg_dev20_k10_20260715_n607.json"
CHECKPOINT="$ROOT/runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth"
GROUND_STATE="$ROOT/runs/qknn_ground_adapt_layer_loss_ablation_20260715_v16/p4_r16_e8_k1/adapter_fp16.pt"
GROUND_SHA=95f9a8bac7880d42f705db7f16523c37cf4ce5ff8438ac2c500c7550a38de446
OUT_ROOT="$ROOT/runs/$EXPERIMENT_ID"
LOG_ROOT="$ROOT/logs/$EXPERIMENT_ID"
ADAPTER_ROOT="$OUT_ROOT/adapters"
RESULT_ROOT="$OUT_ROOT/results"

for required in "$PYTHON" "$CONFIG" "$CHECKPOINT" "$GROUND_STATE"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 2
  fi
done
if [[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" != "2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98" ]]; then
  echo "ADV3B02 checkpoint SHA256 drift" >&2
  exit 3
fi
if [[ "$(sha256sum "$GROUND_STATE" | awk '{print $1}')" != "$GROUND_SHA" ]]; then
  echo "P4 ground adapter SHA256 drift" >&2
  exit 4
fi
if [[ -e "$OUT_ROOT" || -e "$LOG_ROOT" ]]; then
  echo "refusing to overwrite existing BP-JG development output" >&2
  exit 5
fi

export PYTHONPATH="$ROOT/code:$ROOT:${PYTHONPATH:-}"
"$PYTHON" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from paper_reproduction.scripts.train_export_cvs_support_lora_adapter import (
    validate_bp_jg_qknn_config,
)

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig"))
validate_bp_jg_qknn_config(config)
if config.get("resource_diagnostic_only") is not True:
    raise SystemExit("development config must remain diagnostic-only")
if config.get("formal_claim_authority") is not False:
    raise SystemExit("development config must not carry formal claim authority")
if len(config.get("target_new_tx_labels", [])) != 20 or int(config.get("k_shot", 0)) != 10:
    raise SystemExit("v17 requires the registered 20-new K10 development cell")
for path in config["feature_npz_by_scenario"].values():
    if not Path(path).is_file():
        raise SystemExit(f"missing target LEO feature/raw-IQ cache: {path}")
PY

mkdir -p "$ADAPTER_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
arms=(P4_IDENTITY P4_JP_R8 P4_JG_R8 P4_JG_R16)
objectives=(p4_identity bp_jg bp_jg bp_jg)
scopes=(joint_gate joint_projection joint_gate joint_gate)
ranks=(8 8 8 16)
gpus=(0 1 2 3)
manifest="$OUT_ROOT/launch_manifest.tsv"
printf 'arm\tobjective\tscope\trank\tgpu\tpid\tlog\n' > "$manifest"

for index in "${!arms[@]}"; do
  arm="${arms[$index]}"
  objective="${objectives[$index]}"
  scope="${scopes[$index]}"
  rank="${ranks[$index]}"
  gpu="${gpus[$index]}"
  log="$LOG_ROOT/$arm.log"
  if [[ "$objective" == p4_identity ]]; then
    run_name="support_p4_${GROUND_SHA:0:12}_identity_rx_8-8_new_20_seed_713101_k_10"
    target_args=(--epochs 0 --optimizer sgd --max_optimizer_steps 0)
  else
    run_name="support_p4_${GROUND_SHA:0:12}_bp_jg_${scope}_r${rank}_rx_8-8_new_20_seed_713101_k_10"
    target_args=(
      --scope "$scope" --rank "$rank" --alpha "$rank"
      --epochs 5 --optimizer sgd --max_optimizer_steps 50
      --learning_rate 0.005 --weight_decay 0.0001 --temperature 18
      --grad_clip 1 --view_sampling_mode shot_index
    )
  fi
  adapter_dir="$ADAPTER_ROOT/$run_name"
  eval_dir="$RESULT_ROOT/$arm/cvs_qknnv42"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    "$PYTHON" -u -m paper_reproduction.scripts.train_export_cvs_support_lora_adapter \
      --config "$CONFIG" --ckpt "$CHECKPOINT" --out_root "$ADAPTER_ROOT" \
      --receiver 8-8 --new_count 20 --seed 713101 --k_shot 10 \
      --adapt_objective "$objective" --adapter_type lora \
      --ground_adapter_state "$GROUND_STATE" \
      --ground_adapter_sha256 "$GROUND_SHA" \
      --ground_adapter_scope projection_feature \
      --ground_adapter_rank 16 --ground_adapter_alpha 16 \
      --support_view_policy formal_scenario_cycle \
      --device cuda:0 "${target_args[@]}"
    resolved="$adapter_dir/resolved_qknn_config.json"
    training_manifest="$adapter_dir/training_manifest.json"
    test -s "$resolved"
    test -s "$training_manifest"
    "$PYTHON" -u -m paper_reproduction.cvs_aligned.cvs_method_runner \
      --config "$resolved" --run-dir "$eval_dir" --method cvs_qknnv42 \
      --target-receiver 8-8 --seed 713101 --split-seed 713101 --k-shot 10 \
      --experiment-id "${EXPERIMENT_ID}_${arm}" --device cpu
    "$PYTHON" - "$eval_dir/metrics.json" "$training_manifest" "$adapter_dir/resource_audit.json" <<'PY'
import json
import sys
from pathlib import Path

metrics_path, manifest_path, output_path = map(Path, sys.argv[1:])
metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
states = []

def visit(value):
    if isinstance(value, dict):
        if "persistent_state_bytes" in value:
            states.append(
                int(value["persistent_state_bytes"])
                + int(value.get("aux_persistent_state_bytes", 0))
            )
        for child in value.values():
            visit(child)
    elif isinstance(value, list):
        for child in value:
            visit(child)

visit(metrics)
if not states:
    raise SystemExit("evaluation did not expose runner persistent_state_bytes")
resources = dict(manifest["resources"])
runner_state = max(states)
ground_file = int(resources["ground_adapter_serialized_file_bytes"])
target_file = int(resources["target_adapter_serialized_file_bytes"])
combined = runner_state + ground_file + target_file + 24
audit = {
    "runner_persistent_state_bytes_max": runner_state,
    "ground_adapter_serialized_file_bytes": ground_file,
    "target_adapter_serialized_file_bytes": target_file,
    "adaptive_tta_threshold_bytes": 24,
    "combined_actual_artifact_and_runner_state_bytes": combined,
    "cap_bytes": 256 * 1024,
    "within_cap": combined <= 256 * 1024,
}
output_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
if not audit["within_cap"]:
    raise SystemExit(f"deployment state cap exceeded: {audit}")
PY
  ) > "$log" 2>&1 &
  pid=$!
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$arm" "$objective" "$scope" "$rank" "$gpu" "$pid" "$log" >> "$manifest"
done

printf '[P4-BPJG-V17-LAUNCHED] manifest=%s\n' "$manifest"
