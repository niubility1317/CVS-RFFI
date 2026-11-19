"""Paired Oxford crop and full-header fits followed by frozen Thun inference."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from . import full_data
from .prepare import write_json
from .run import build_model, evaluate, load_weights, packet_logits, read, source_contract, validate
from .score import metrics


ARMS = {
    "backbone_crop": ("backbone_ce", "crop"),
    "backbone_full": ("backbone_ce", "full"),
    "pamf_crop": ("pamf_ce", "crop"),
    "pamf_full": ("pamf_ce", "full"),
}


def tensor(a):
    return torch.utils.dlpack.from_dlpack(a)


def uk_summary(model, x, y, device, batch, classes):
    logits = evaluate(model, np.from_dlpack(x), device, batch)
    result = metrics(np.from_dlpack(y), logits.argmax(axis=1), classes)
    result["ce"] = float(F.cross_entropy(tensor(logits), y))
    return result


def train(arm, compact, full, cfg, out):
    architecture, mode = ARMS[arm]
    contract = source_contract(compact / "source")
    paired = full_data.paired_uk(full, compact)
    if {role: len(y) for role, (_, y) in paired.items()} != contract["source_counts"]:
        raise ValueError("UK pair count differs from the frozen crop contract")
    nclasses = len(contract["classes"])
    data = {role: (tensor(full_data.views(raw, mode)), tensor(y))
            for role, (raw, y) in paired.items()}
    train_x, train_y = data["train"]
    val_x, val_y = data["val"]
    out.mkdir(exist_ok=False)
    write_json(out / "resolved_config.json",
               dict(cfg, arm=arm, architecture=architecture, representation=mode,
                    windows=full_data.VIEWS[mode], classes=nclasses, initialization="scratch",
                    output=str(out), source_full=str(full / "source"), source_contract=str(compact / "source")))
    device = torch.device("cuda:0")
    torch.manual_seed(cfg["model_seed"])
    model = build_model(cfg["runtime"], cfg["model_seed"], nclasses, architecture).to(device)
    initial = out / "scratch_e00.pt"
    torch.save(model.state_dict(), initial)
    load_weights(model, initial)
    validation = validate(model, train_x[:2], train_y[:2], device, cfg["batch_size"])
    print(json.dumps(dict(event="no_query_checkpoint_smoke", arm=arm, status="PASS",
                          uk_only=True, metrics=validation)), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    shuffle = torch.Generator().manual_seed(cfg["shuffle_seed"])
    steps = 0
    started = time.monotonic()
    batch = cfg["batch_size"]
    with (out / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            ce = 0.0
            for ii in torch.randperm(len(train_y), generator=shuffle).split(batch):
                optimizer.zero_grad(set_to_none=True)
                logits = packet_logits(model, train_x[ii].to(device))
                loss = F.cross_entropy(logits, train_y[ii].to(device))
                if not loss.isfinite():
                    raise ValueError(f"{arm}: non-finite source fitting loss")
                loss.backward()
                optimizer.step()
                steps += 1
                ce += loss.item() * len(ii)
            row = dict(arm=arm, epoch=epoch, updates=steps, uk_train_ce=ce / len(train_y),
                       uk_val=validate(model, val_x, val_y, device, batch),
                       seconds=time.monotonic() - started,
                       peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    torch.save(model.state_dict(), out / "source_final.pt")
    uk = {role: uk_summary(model, x, y, device, batch, nclasses)
          for role, (x, y) in data.items()}
    write_json(out / "uk.json", uk)


@torch.no_grad()
def predict(arm, compact, full, cfg, out):
    architecture, mode = ARMS[arm]
    contract = source_contract(compact / "source")
    target = full_data.Headers(full, "target")
    if len(target.ids) != contract["target_physical_records"]:
        raise ValueError("Wrong Swiss test population")
    nclasses = len(contract["classes"])
    device = torch.device("cuda:0")
    model = build_model(cfg["runtime"], cfg["model_seed"], nclasses, architecture)
    load_weights(model, out / "source_final.pt")
    model = model.to(device).eval()
    result = np.empty((len(target.ids), nclasses), dtype=np.float32)
    batch = cfg["prediction_batch_size"]
    for offset in range(0, len(target.ids), batch):
        stop = min(offset + batch, len(target.ids))
        raw = target.rows(np.arange(offset, stop))
        output = packet_logits(model, tensor(full_data.views(raw, mode)).to(device)).cpu()
        result[offset:stop] = np.from_dlpack(output)
    if not np.isfinite(result).all():
        raise ValueError(f"{arm}: non-finite Swiss predictions")
    with (out / "predictions.npz").open("xb") as f:
        np.savez(f, ids=target.ids, logits=result)
    print(json.dumps(dict(event="prediction_fixed", arm=arm, records=len(target.ids))), flush=True)


def worker(mode, cfg, compact, full, sub, arm):
    subprocess.run([sys.executable, "-B", "-u", "-m", __spec__.name, mode,
                    str(cfg), str(compact), str(full), str(sub), "--arm", arm], check=True)


def launch(cfg, compact, full, out):
    out.mkdir(exist_ok=False)
    for arm in ARMS:
        worker("train", cfg, compact, full, out / arm, arm)
    uk = {arm: read(out / arm / "uk.json") for arm in ARMS}
    write_json(out / "uk_comparison.json", uk)
    print(json.dumps(dict(event="uk_comparison_fixed",
                          accuracy={a: rows["val"]["accuracy"] for a, rows in uk.items()})), flush=True)
    index = full / "target/query/index.json"
    deadline = time.monotonic() + 7200
    if not index.is_file():
        print("Training outputs frozen; waiting for verified Swiss full-header archive.", flush=True)
    while not index.is_file():
        if time.monotonic() > deadline:
            raise TimeoutError("Swiss full-header archive is still unverified")
        time.sleep(10)
    for arm in ARMS:
        worker("predict", cfg, compact, full, out / arm, arm)
    subprocess.run([sys.executable, "-B", "-u", "-m", "experiments.pamf_satiq.score",
                    str(compact / "source/contract.json"), str(full / "target/query/ids.npy"),
                    str(compact / "target/scorer/truth.npz"), str(out), "--arms", *ARMS], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "predict", "launch"))
    parser.add_argument("config", type=Path)
    parser.add_argument("compact", type=Path)
    parser.add_argument("full", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    args = parser.parse_args()
    if args.mode == "launch":
        launch(args.config, args.compact, args.full, args.out)
    else:
        if args.arm is None:
            parser.error("train and predict require --arm")
        (train if args.mode == "train" else predict)(
            args.arm, args.compact, args.full, read(args.config), args.out)
