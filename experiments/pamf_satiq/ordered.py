"""Matched ordered Iridium UW inputs, from Oxford training to frozen Thun scoring."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .full_data import HEADER, Headers, paired_uk
from .prepare import WIDTH, write_json
from .run import build_model, load_weights, read, source_contract
from .score import metrics, score


ARMS = {
    "uw11": (0, 11000, 0, 11),
    "uw8": (0, 8000, 0, 8),
    "short256": (4500, 4756, 4, 1),
}
SYMBOL_SAMPLES = 1000
LOCAL_STARTS = (0, 256, 512, 744)
TOKEN_COUNT = 11
FEATURES = 160
TOKEN_FEATURES = 32


def tensor(array):
    return torch.utils.dlpack.from_dlpack(array)


def normalize(raw):
    """Equivalent to inverse min/span scaling, then complex DC removal and RMS."""
    centered = raw - raw.mean(dim=1, keepdim=True)
    rms = centered.square().sum(dim=-1).mean(dim=1).sqrt()
    return centered / rms[:, None, None].clamp_min(1e-8)


def patch_view(raw, arm):
    start, stop, _, count = ARMS[arm]
    if raw.ndim != 3 or raw.shape[1:] != (HEADER, 2):
        raise ValueError("Expected complete 11000-point raw packet rows")
    x = normalize(raw[:, start:stop])
    if arm == "short256":
        return x.transpose(1, 2)[:, None, None]
    symbols = x.reshape(len(x), count, SYMBOL_SAMPLES, 2)
    return torch.stack([symbols[:, :, pos:pos + WIDTH].transpose(2, 3)
                        for pos in LOCAL_STARTS], dim=2)


class OrderedPAMF(nn.Module):
    def __init__(self, backbone, seed):
        super().__init__()
        self.backbone = backbone
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed + 1)
            self.projection = nn.Linear(FEATURES, TOKEN_FEATURES, bias=False)
            self.temporal = nn.Conv1d(TOKEN_FEATURES, TOKEN_FEATURES, 3, padding=1, bias=False)
            self.fusion = nn.Linear(TOKEN_COUNT * TOKEN_FEATURES, FEATURES, bias=False)
            self.norm = nn.LayerNorm(FEATURES)
            self.gain = nn.Parameter(torch.tensor(.05))

    def forward(self, raw, arm):
        patch = patch_view(raw, arm)
        _, _, offset, count = ARMS[arm]
        features = self.backbone.features(patch.reshape(-1, 2, WIDTH))
        patches = F.normalize(features, dim=-1).reshape(len(raw), count, patch.shape[2], FEATURES)
        tokens = patches.mean(dim=2)
        local = tokens.mean(dim=1)
        positions = F.gelu(self.projection(tokens))
        positions = positions + F.gelu(self.temporal(positions.transpose(1, 2)).transpose(1, 2))
        ordered = F.pad(positions, (0, 0, offset, TOKEN_COUNT - count - offset)).flatten(1)
        fused = local + self.gain * self.norm(self.fusion(ordered))
        return self.backbone.core.id_backbone.cls_head.classify(fused)


def make_model(cfg, classes, device):
    backbone = build_model(cfg["runtime"], cfg["model_seed"], classes, "pamf_ce")
    return OrderedPAMF(backbone, cfg["model_seed"]).to(device)


@torch.no_grad()
def validate(model, data, arm, batch, classes):
    model.eval()
    x, y = data
    logits = torch.cat([model(v, arm) for v in x.split(batch)])
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Non-finite Oxford validation scores")
    actual = np.from_dlpack(y.cpu())
    predicted = np.from_dlpack(logits.argmax(1).cpu())
    result = metrics(actual, predicted, classes)
    result["ce"] = float(F.cross_entropy(logits, y))
    return result


def train(args, cfg):
    contract = source_contract(args.compact / "source")
    paired = paired_uk(args.full, args.compact)
    if {r: len(y) for r, (_, y) in paired.items()} != contract["source_counts"]:
        raise ValueError("Oxford physical sample counts changed")
    classes = len(contract["classes"])
    device = torch.device("cuda:0")
    data = {role: (tensor(x).to(device), tensor(y).to(device)) for role, (x, y) in paired.items()}
    del paired
    args.out.mkdir(exist_ok=False)
    torch.manual_seed(cfg["model_seed"])
    model = make_model(cfg, classes, device)
    initial = args.out / "scratch_e00.pt"
    torch.save(model.state_dict(), initial)
    load_weights(model, initial)
    smoke = validate(model, tuple(a[:2] for a in data["train"]), args.arm, 2, classes)
    print(json.dumps(dict(arm=args.arm, event="no_query_checkpoint_smoke", status="PASS",
                          ce=smoke["ce"])), flush=True)
    write_json(args.out / "resolved_config.json", dict(
        cfg, arm=args.arm, view=ARMS[args.arm], classes=classes, initialization="scratch",
        parameters=sum(p.numel() for p in model.parameters()), normalization="view-local complex DC then RMS"))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    shuffle = torch.Generator().manual_seed(cfg["shuffle_seed"])
    x, y = data["train"]
    batch = cfg["batch_size"]
    selected, steps = [], 0
    start_time = time.monotonic()
    with (args.out / "epochs.jsonl").open("x", encoding="utf-8") as f:
        for epoch in range(1, cfg["candidate_epochs"][-1] + 1):
            model.train()
            ce = 0.0
            for indices in torch.randperm(len(y), generator=shuffle).split(batch):
                indices = indices.to(device)
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(x[indices], args.arm), y[indices])
                if not bool(loss.isfinite()):
                    raise ValueError("Non-finite Oxford training loss")
                loss.backward()
                opt.step()
                ce += loss.item() * len(indices)
                steps += 1
            val = validate(model, data["val"], args.arm, batch, classes)
            row = dict(arm=args.arm, epoch=epoch, updates=steps, train_ce=ce / len(y),
                       uk_validation={k: val[k] for k in ("ce", "accuracy", "macro_f1",
                                                          "balanced_accuracy", "predicted_class_count")},
                       seconds=time.monotonic() - start_time,
                       peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            line = json.dumps(row, allow_nan=False)
            f.write(line + "\n")
            f.flush()
            print(line, flush=True)
            if epoch in cfg["candidate_epochs"]:
                name = f"e{epoch:03d}.pt"
                torch.save(model.state_dict(), args.out / name)
                selected.append(dict(checkpoint=name, epoch=epoch, updates=steps, validation=val))
    best = max(selected, key=lambda r: r["validation"]["macro_f1"])
    write_json(args.out / "selection.json", dict(selected=best, candidates=selected))


@torch.no_grad()
def predict(args, cfg):
    contract = source_contract(args.compact / "source")
    target = Headers(args.full, "target")
    classes = len(contract["classes"])
    if len(target.ids) != contract["target_physical_records"]:
        raise ValueError("Swiss public prediction population changed")
    selection = read(args.out / "selection.json")["selected"]
    device = torch.device("cuda:0")
    model = make_model(cfg, classes, device)
    load_weights(model, args.out / selection["checkpoint"])
    model.eval()
    logits = np.empty((len(target.ids), classes), dtype=np.float32)
    batch = cfg["prediction_batch_size"]
    for a in range(0, len(target.ids), batch):
        b = min(a + batch, len(target.ids))
        raw = tensor(target.rows(np.arange(a, b))).to(device)
        logits[a:b] = np.from_dlpack(model(raw, args.arm).cpu())
    if not np.isfinite(logits).all():
        raise ValueError("Non-finite Swiss predictions")
    with (args.out / "predictions.npz").open("xb") as f:
        np.savez(f, ids=target.ids, logits=logits)
    print(json.dumps(dict(event="prediction_fixed", arm=args.arm, records=len(target.ids))), flush=True)


def launch(args):
    args.out.mkdir(exist_ok=False)
    command = [sys.executable, "-B", "-u", "-m", __spec__.name]
    inputs = [str(args.config), str(args.compact), str(args.full)]
    for arm in ARMS:
        subprocess.run([*command, "train", *inputs, str(args.out / arm), "--arm", arm], check=True)
    uk = {arm: read(args.out / arm / "selection.json") for arm in ARMS}
    write_json(args.out / "uk_comparison.json", uk)
    print(json.dumps(dict(event="uk_models_frozen",
                          epochs={a: r["selected"]["epoch"] for a, r in uk.items()})), flush=True)
    for arm in ARMS:
        subprocess.run([*command, "predict", *inputs, str(args.out / arm), "--arm", arm], check=True)
    subprocess.run([*command, "score", *inputs, str(args.out)], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("launch", "train", "predict", "score"))
    parser.add_argument("config", type=Path)
    parser.add_argument("compact", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "launch":
        launch(args)
    elif args.mode == "score":
        score(args.compact / "source/contract.json", args.full / "target/query/ids.npy",
              args.full / "target/scorer/truth.npz", args.out, tuple(ARMS))
    else:
        if args.arm is None:
            parser.error("train and predict require --arm")
        (train if args.mode == "train" else predict)(args, read(args.config))
