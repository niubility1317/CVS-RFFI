"""Compare frozen heads and reference readouts on the original visible source roles."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.pamf_satiq66.model import build_model, capture, normalized
from experiments.pamf_satiq66.wide_model import build_wide_model
from experiments.pamf_satiq66.wide_run import batch, read_supervised


ARMS = (("B0", 50), ("L0", 50), ("W0", 50), ("W1", 50), ("W0", 125))
BRANCHES = ("time", "frequency", "pa", "sequence", "response")


@torch.no_grad()
def extract(model, role, device, short):
    raw_encoder = None if short else model.raw_encoder
    native = model if short else model.native
    pieces = {}
    residual_ratios = []
    for start in range(0, len(role[0]), 256):
        raw, coarse = batch(role, slice(start, start + 256), device)
        logits, main, branches = capture(native, normalized(raw[:, :, 384:640]) if short else coarse)
        views = dict(zip(BRANCHES, branches))
        views.update(fused=main, head=logits)
        if raw_encoder is not None:
            residual = raw_encoder(raw)
            fused = main + residual
            views.update(fused=fused, head=native.classify_features(fused), native=main,
                         native_head=logits, raw_residual=residual)
            residual_ratios.extend((residual.norm(dim=1) / main.norm(dim=1).clamp_min(1e-8)).cpu().tolist())
        for name, f in views.items():
            pieces.setdefault(name, []).append(np.from_dlpack(f.cpu()))
    data = {name: np.concatenate(values) for name, values in pieces.items()}
    assert all(np.isfinite(x).all() for x in data.values())
    return data, float(np.median(residual_ratios)) if residual_ratios else None


def score_readout(logits, truth, sats):
    positive = logits[np.arange(len(truth)), truth]
    true_class = np.arange(len(sats)) == truth[:, None]
    same_sat = sats == sats[truth, None]
    wins = (positive[:, None] > logits) + 0.5 * (positive[:, None] == logits)
    guess = logits.argmax(1)
    return dict(top1=float((guess == truth).mean()),
                satellite=float((sats[guess] == sats[truth]).mean()),
                known_sat_top1=float((np.where(same_sat, logits, -np.inf).argmax(1) == truth).mean()),
                within_sat_pair_win=float(wins[same_sat & ~true_class].mean()),
                across_sat_pair_win=float(wins[~same_sat].mean()))


def mean_cosine_scores(train, test, anchors):
    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)

    return unit(test) @ unit(train)[anchors].mean(1).T


def run(root, source, device):
    contract, roles = read_supervised(source)
    sats = np.asarray([sat for sat, _ in contract["classes"]])
    train = roles["L"][2]
    truth = roles["V"][2]["y"]
    rng = np.random.default_rng(2026111904)
    anchors = {
        "K16": np.array([np.concatenate([
            rng.choice(np.flatnonzero((train["y"] == label) & (train["day"] == day)), 2, replace=False)
            for day in range(8)]) for label in range(66)]),
        "K120": np.array([np.flatnonzero(train["y"] == label) for label in range(66)]),
    }
    for arm, epoch in ARMS:
        out = root / arm
        cfg = json.loads((out / "resolved_config.json").read_text())
        if cfg["source"] != str(source) or cfg["classes"] != contract["classes"] or cfg["initialization"] != "scratch":
            raise ValueError(f"Mismatched checkpoint data contract for {arm}")
        ckpt = torch.load(out / "checkpoints" / f"e{epoch:08d}.pt", map_location="cpu", weights_only=True)
        short = arm == "B0"
        model = (build_model(cfg["runtime"], cfg["model_seed"], 0) if short
                 else build_wide_model(cfg["runtime"], cfg["model_seed"], cfg["raw_branch"]))
        model.load_state_dict(ckpt["model"], strict=True)
        model = model.to(device).eval()
        learned, ratio_l = extract(model, roles["L"], device, short)
        heldout, ratio_v = extract(model, roles["V"], device, short)
        scores = {name: score_readout(x, truth, sats) for name, x in heldout.items() if name.endswith("head")}
        if scores["head"]["top1"] != ckpt["validation"]["accuracy"]:
            raise ValueError(f"Native head did not reproduce recorded validation: {arm} E{epoch}")
        for mode, index in anchors.items():
            for name, x in heldout.items():
                if name.endswith("head"):
                    continue
                ref_scores = mean_cosine_scores(learned[name], x, index)
                scores[f"{name}_{mode}"] = score_readout(ref_scores, truth, sats)
        print(json.dumps(dict(arm=arm, epoch=epoch, train_records=len(train["y"]), validation_records=len(truth),
                              raw_to_native_l2_median_L=ratio_l, raw_to_native_l2_median_V=ratio_v, **scores)),
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    run(args.root, args.source, torch.device("cuda:0"))
