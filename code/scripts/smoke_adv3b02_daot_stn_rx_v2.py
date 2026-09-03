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
from cvsrffi.daot_training import compute_daot_batch_objective
from cvsrffi.orbit_teacher import TensorTemporalOrbitMemory
from cvsrffi.receiver_style_bank import OnlineReceiverStyleBank


def _load_model(checkpoint_path: Path, device: torch.device):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
        raise ValueError("checkpoint must contain a model state dictionary")
    state = payload["model"]
    model_args = dict(payload.get("baseline_args") or payload.get("args") or {})
    if "num_classes" not in model_args:
        keys = [key for key in state if key.endswith("cls_head.weight")]
        if not keys:
            raise ValueError("checkpoint does not expose num_classes")
        model_args["num_classes"] = int(state[keys[0]].shape[0])
    model_args.setdefault("num_domains", 1)
    model_args.setdefault("input_len", 256)
    model_args["use_daot_nuisance_head"] = True
    model_args["daot_nuisance_dim"] = 9
    model = build_baseline_model(SimpleNamespace(**model_args), device)
    incompatible = model.load_state_dict(state, strict=False)
    missing_non_daot = [key for key in incompatible.missing_keys if not key.startswith("daot_nuisance_head.")]
    if incompatible.unexpected_keys or missing_non_daot:
        raise RuntimeError(
            f"checkpoint compatibility failure: unexpected={list(incompatible.unexpected_keys)} "
            f"missing_non_daot={missing_non_daot}"
        )
    return payload, model, model_args, list(incompatible.missing_keys)


def main() -> int:
    parser = argparse.ArgumentParser(description="ADV3B02 DAOT-STN RX-V2 real-checkpoint no-query smoke.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device(args.device)
    payload, model, model_args, missing = _load_model(checkpoint_path, device)
    torch.manual_seed(392005)
    batch_size = 4
    clean = torch.randn(batch_size, 2, int(model_args["input_len"]), device=device)
    receivers = torch.tensor([1, 1, 3, 3], device=device)
    style_bank = OnlineReceiverStyleBank(rank=2, min_receivers=2)
    style_bank.update(clean, receiver_ids=receivers, role="source")
    fresh = style_bank.apply_sampled(clean, seed=392005)
    fresh = fresh + 0.01 * torch.roll(fresh, shifts=1, dims=2)

    model.eval()
    with torch.no_grad():
        teacher_clean = model(clean, y_tx=None, return_aux=True, domain_labels=None)
        teacher_fresh = model(fresh, y_tx=None, return_aux=True, domain_labels=None)
        student_fresh = model(fresh, y_tx=None, return_aux=True, domain_labels=None)
    feature_dim = int(teacher_clean["z_id"].shape[1])
    memory = TensorTemporalOrbitMemory(feature_dim=feature_dim, capacity=16, ttl=8)
    keys = torch.arange(batch_size, device=device)
    memory.update(
        keys=keys,
        features=teacher_fresh["z_id"],
        reliability=torch.ones(batch_size, device=device),
        scenario_bin=torch.ones(batch_size, device=device, dtype=torch.long),
        receiver_bin=receivers,
        step=1,
    )
    memory_features, found, _ = memory.lookup(keys, step=2)
    teacher_memory = {"z_id": memory_features, "tx_logits": teacher_fresh["tx_logits"]}
    result = compute_daot_batch_objective(
        student_clean=teacher_clean,
        student_channel=student_fresh,
        teacher_views=[teacher_clean, teacher_fresh, teacher_memory],
        reliability=torch.ones(batch_size, 3, device=device),
        importance=torch.ones(batch_size, 3, device=device),
        recoverability=torch.ones(batch_size, device=device),
        orbit_scale=1.0,
        tangent_scale=0.0,
        weights={"orbit_z": 0.4, "orbit_logit": 0.075, "clean_anchor": 0.025},
        coverage_floor=0.15,
        huber_beta_min=0.30,
        temperature=3.0,
        rx_v2=True,
        teacher_prior=torch.tensor([0.5, 0.3, 0.2], device=device),
        component_scales={"orbit_z": 1.0, "orbit_logit": 1.0, "clean_anchor": 1.0},
    )
    if not bool(torch.isfinite(result["loss"]).item()) or not bool(found.all()):
        raise RuntimeError("RX-V2 no-query smoke did not produce finite closed artifacts")
    output = {
        "schema": "cvs.phase1.adv3b02_daot_stn_rx_v2_real_checkpoint_no_query_smoke.v1",
        "status": "PASS",
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "input_role": "source_shaped_synthetic_smoke_only",
        "query_inputs": 0,
        "target_inputs": 0,
        "fresh_teacher_forwards": 2,
        "temporal_memory_views": 1,
        "teacher_view_count": 3,
        "memory_hit_rate": float(found.float().mean().item()),
        "receiver_style_ready": style_bank.ready,
        "loss": float(result["loss"].item()),
        "z_id_shape": list(teacher_clean["z_id"].shape),
        "expected_missing_daot_keys": missing,
        "evaluated_scenarios": [],
    }
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
