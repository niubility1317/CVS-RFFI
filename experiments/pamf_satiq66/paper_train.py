"""Source-only supervised comparison under the reconstructed CCS 2023 data split."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, write_json
from .paper_baseline import PaperSatIQ, angular_triplet, paper_loss
from .paper_eval import embeddings, metrics, normalize_packet, panel
from .paper_pamf import PaperPAMF
from .readout import gpu_ready
from .train import cuda_device, device_array


ARMS = ("SATIQ", "PAMF_CE", "PAMF_METRIC")


class SourceBatches:
    def __init__(self, labels, class_counts, seed):
        ordered = np.argsort(labels, kind="stable")
        boundaries = np.r_[0, np.cumsum(class_counts)]
        self.groups = [ordered[boundaries[k]:boundaries[k + 1]] for k, n in enumerate(class_counts) if n >= 4]
        self.weights = np.array([len(g) // 4 for g in self.groups], dtype=np.float64)
        self.weights /= self.weights.sum()
        self.rng = np.random.default_rng(seed)

    def next(self):
        chosen = self.rng.choice(len(self.groups), size=8, replace=False, p=self.weights)
        return self.rng.permutation(np.concatenate([self.rng.choice(self.groups[k], 4, replace=False) for k in chosen]))


def source_inputs(source, cache):
    contract = read_json(source / "contract.json")
    if contract["schema"] != "satiq_oxford_ccs2023_supervised_reindexed_v1" or contract["training_mode"] != "supervised_only":
        raise ValueError("Expected the aligned fully supervised SatIQ contract")
    counts = np.asarray(contract["training_class_counts"])
    eligible = counts >= 4
    remap = np.full(len(counts), -1, dtype=np.int32)
    remap[eligible] = np.arange(eligible.sum())
    roles = {}
    for role in ("L", "V"):
        with np.load(source / f"{role}_index.npz", allow_pickle=False) as manifest:
            original = manifest["y"]
        raw = np.load(cache / f"{role}_raw.npy", mmap_mode="r", allow_pickle=False)
        if raw.shape != (contract["records"][role], 2, 11_000) or raw.dtype != np.float32:
            raise ValueError(f"Expected full-length float32 {role} cache")
        labels = np.full(len(original), -1, dtype=np.int32)
        known = original >= 0
        labels[known] = remap[original[known]]
        roles[role] = raw, labels, original
    return np.asarray(contract["class_pairs"])[eligible], counts, roles


def make_model(config, arm, classes):
    torch.manual_seed(config["model_seed"])
    if arm == "SATIQ":
        return PaperSatIQ()
    return PaperPAMF(config["runtime"], config["model_seed"], len(classes))


@torch.no_grad()
def validate(model, arm, raw, labels, val_panel, device):
    feature = embeddings(model, raw, device)
    result = metrics(feature, val_panel, device)
    if arm != "SATIQ":
        mask = labels >= 0
        target = torch.as_tensor(labels[mask].astype(np.int64), device=device)
        known_features = feature[mask]
        weight = F.normalize(model.head.weight, dim=1)
        block_losses, correct = [], 0
        for start in range(0, len(target), 256):
            end = start + 256
            vec = device_array(known_features[start:end], device)
            scores = 30 * F.linear(vec, weight)
            y = target[start:end]
            block_losses.append(float(F.cross_entropy(scores, y, reduction="sum")))
            correct += int((scores.argmax(1) == y).sum())
        result.update(known_class_records=len(target), head_top1=correct / len(target),
                      head_ce=sum(block_losses) / len(target))
    return result


def save(path, model, optimizer, sampler, step, validation):
    temporary = path.with_suffix(".partial")
    with temporary.open("xb") as writer:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=step,
                        sampling_rng=sampler.rng.bit_generator.state, validation=validation), writer)
    temporary.rename(path)


def train(config, source, cache, out, arm, smoke=False):
    classes, counts, roles = source_inputs(source, cache)
    device = cuda_device()
    model = make_model(config, arm, classes).to(device)
    lr = 1e-5 if arm == "SATIQ" else config["pamf_lr"]
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, eps=1e-7)
    raw, labels, original = roles["L"]
    sampler = SourceBatches(original, counts, config["sampling_seed"])
    milestones = () if smoke else config["milestones"]
    total = 2 if smoke else milestones[-1]
    val_panel = None if smoke else panel(roles["V"][1], classes, config["evaluation_seed"])
    out.mkdir(exist_ok=False)
    write_json(out / "resolved_config.json", dict(config, arm=arm, trained_classes=len(classes), lr=lr,
                                                 source=str(source), cache=str(cache), supervised=True,
                                                 source_roles=["L", "V"], test_access=False, cost_unit="optimizer_updates"))
    start = time.monotonic()
    window = {}
    n = 0
    with (out / "steps.jsonl").open("x", encoding="utf-8") as journal:
        for step in range(1, total + 1):
            model.train()
            idx = sampler.next()
            x = normalize_packet(device_array(raw[idx], device))
            y = device_array(labels[idx].astype(np.int64), device)
            optimizer.zero_grad(set_to_none=True)
            if arm == "SATIQ":
                feature, reconstructed = model(x)
                loss, stats = paper_loss(feature, reconstructed, x, y)
            else:
                feature, logits = model(x)
                ce = F.cross_entropy(logits, y)
                stats = {"ce": ce.detach()}
                loss = ce
                if arm == "PAMF_METRIC":
                    triplet = angular_triplet(feature, y, margin=1.0)
                    loss = ce * config["pamf_metric_ce_weight"] + triplet
                    stats["triplet_loss"] = triplet.detach()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {arm} training objective")
            loss.backward()
            optimizer.step()
            n += 1
            for key, value in {**stats, "loss": loss.detach()}.items():
                window[key] = window.get(key, 0) + value
            if step in milestones or step % config["log_every"] == 0 or smoke:
                info = {"updates": step, "seconds": time.monotonic() - start,
                        **{k: float(v) / n for k, v in window.items()}}
                if step in milestones:
                    info["source_validation"] = validate(model, arm, *roles["V"][:2], val_panel, device)
                    save(out / f"s{step:07d}.pt", model, optimizer, sampler, step, info["source_validation"])
                message = json.dumps(info, allow_nan=False)
                print(message, flush=True)
                journal.write(message + "\n")
                journal.flush()
                window, n = {}, 0
    write_json(out / "complete.json", {"arm": arm, "updates": total, "mode": "smoke" if smoke else "supervised"})


def launch(config, source, cache, out, gpus):
    out.mkdir(exist_ok=False)
    logs = out / "logs"
    logs.mkdir()
    write_json(out / "owner.json", {"pid": os.getpid(), "matrix": ARMS, "physical_gpus": gpus})

    def start(mode, arm, directory, gpu):
        command = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.paper_train", mode, str(config),
                   str(source), str(cache), str(directory), "--arm", arm]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        with (logs / f"{directory.name}.log").open("xb") as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(event="started", arm=arm, mode=mode, gpu=gpu, pid=child.pid)), flush=True)
        return child

    available = []
    while not available:
        available = [gpu for gpu in gpus if gpu_ready(gpu)]
        if not available:
            time.sleep(15)
    for arm in ("SATIQ", "PAMF_METRIC"):
        if start("smoke", arm, out / f"smoke_{arm}", available[0]).wait():
            raise RuntimeError(f"Real source smoke failed for {arm}")

    remaining, active, outcomes = list(ARMS), {}, {}
    while remaining or active:
        for gpu, (arm, child) in list(active.items()):
            result = child.poll()
            if result is not None:
                outcomes[arm] = result
                del active[gpu]
                print(json.dumps(dict(event="exited", arm=arm, returncode=result)), flush=True)
        for gpu in gpus:
            if not remaining:
                break
            if gpu not in active and gpu_ready(gpu):
                arm = remaining.pop(0)
                active[gpu] = arm, start("train", arm, out / arm, gpu)
        if remaining or active:
            time.sleep(15)
    write_json(out / "matrix_outcomes.json", outcomes)
    if any(outcomes.values()):
        raise RuntimeError("A supervised row failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "train", "smoke"))
    for arg in ("config", "source", "cache", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--gpus", nargs="+", type=int)
    args = parser.parse_args()
    if args.mode == "launch":
        if not args.gpus:
            parser.error("launch requires physical GPUs")
        launch(args.config, args.source, args.cache, args.output, args.gpus)
    else:
        if not args.arm:
            parser.error("train/smoke requires a matrix arm")
        train(read_json(args.config), args.source, args.cache, args.output, args.arm, args.mode == "smoke")
