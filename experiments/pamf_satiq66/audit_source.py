"""Read-only scoring of stopped supervised models on the original L/V records."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_satiq66.model import build_model, normalized
from experiments.pamf_satiq66.wide_model import build_wide_model
from experiments.pamf_satiq66.wide_run import batch, describe, read_supervised


@torch.inference_mode()
def score(model, role, device, short, satellites):
    pieces = []
    for first in range(0, len(role[0]), 256):
        raw, coarse = batch(role, slice(first, first + 256), device)
        logits = model(normalized(raw[:, :, 384:640])) if short else model(raw, coarse)
        pieces.append(np.from_dlpack(logits.cpu()))
    logits = np.concatenate(pieces)
    labels = role[2]["y"]
    result = describe(logits, labels, satellites)
    in_satellite = satellites[None, :] == satellites[labels, None]
    result["beam_acc_if_true_satellite_given"] = float((np.where(in_satellite, logits, -np.inf).argmax(1) == labels).mean())
    result["distinct_top1_labels"] = int(np.unique(logits.argmax(1)).size)
    return result


def main(root, source, device):
    contract, roles = read_supervised(source)
    satellites = np.array([sat for sat, _ in contract["classes"]])
    for arm in ("B0", "L0", "W0", "W1"):
        out = root / arm
        cfg = json.loads((out / "resolved_config.json").read_text())
        if cfg["source"] != str(source) or cfg["classes"] != contract["classes"] or cfg["initialization"] != "scratch":
            raise ValueError(f"Invalid source provenance for {arm}")
        checkpoint_path = max((out / "checkpoints").glob("e*.pt"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        short = arm == "B0"
        model = (build_model(cfg["runtime"], cfg["model_seed"], 0) if short
                 else build_wide_model(cfg["runtime"], cfg["model_seed"], cfg["raw_branch"]))
        model.load_state_dict(checkpoint["model"], strict=True)
        model = model.to(device).eval()
        scores = {name: score(model, role, device, short, satellites) for name, role in roles.items()}
        recorded = checkpoint["validation"]
        if scores["V"]["accuracy"] != recorded["accuracy"] or abs(scores["V"]["nll"] - recorded["nll"]) > 1e-5:
            raise ValueError(f"V readback mismatch for {arm}")
        print(json.dumps(dict(arm=arm, epoch=checkpoint["epoch"], checkpoint=str(checkpoint_path), **scores)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    main(args.root, args.source, torch.device(args.device))
