from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from post_stage_common import build_baseline_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Real-checkpoint, source-shaped, no-query DAOT smoke.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    model_args = dict(payload.get("baseline_args") or payload.get("args") or {})
    state = payload["model"]
    if "num_classes" not in model_args:
        classifier_keys = [key for key in state if key.endswith("cls_head.weight")]
        if not classifier_keys:
            raise ValueError("checkpoint does not expose num_classes")
        model_args["num_classes"] = int(state[classifier_keys[0]].shape[0])
    model_args.setdefault("num_domains", 1)
    model_args.setdefault("input_len", 256)
    model_args["use_daot_nuisance_head"] = True
    model_args["daot_nuisance_dim"] = 9
    device = torch.device(args.device)
    model = build_baseline_model(SimpleNamespace(**model_args), device)
    incompatible = model.load_state_dict(state, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    missing_non_daot = [
        key for key in incompatible.missing_keys if not key.startswith("daot_nuisance_head.")
    ]
    if unexpected or missing_non_daot:
        raise RuntimeError(
            f"checkpoint compatibility failure: unexpected={unexpected} missing_non_daot={missing_non_daot}"
        )
    torch.manual_seed(713104)
    source_shaped_iq = torch.randn(2, 2, int(model_args["input_len"]), device=device)
    model.eval()
    with torch.no_grad():
        output = model(source_shaped_iq, y_tx=None, return_aux=True, domain_labels=None)
    if output["z_id"].shape[0] != 2 or output["daot_nuisance_mean"].shape != (2, 9):
        raise RuntimeError("DAOT checkpoint smoke output shape mismatch")
    result = {
        "schema": "cvs.phase1.adv3b02_daot_real_checkpoint_no_query_smoke.v1",
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "input_role": "source_shaped_synthetic_smoke_only",
        "query_inputs": 0,
        "target_inputs": 0,
        "batch_size": 2,
        "z_id_shape": list(output["z_id"].shape),
        "nuisance_mean_shape": list(output["daot_nuisance_mean"].shape),
        "expected_missing_daot_keys": list(incompatible.missing_keys),
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
