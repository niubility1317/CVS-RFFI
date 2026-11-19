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
from .paper_pamf import PaperPAMF, reconstruction_target, supervised_loss
from .readout import gpu_ready
from .train import cuda_device, device_array


ARMS = ("SATIQ", "PAMF_SUP", "PAMF_METRIC")


class SourceBatches:
    def __init__(self, labels, class_counts, seed, satellites=None):
        ordered = np.argsort(labels, kind="stable")
        boundaries = np.r_[0, np.cumsum(class_counts)]
        self.groups = [ordered[boundaries[k]:boundaries[k + 1]] for k, n in enumerate(class_counts) if n >= 4]
        self.counts = np.array([len(g) // 4 for g in self.groups])
        self.steps_per_epoch = int(self.counts.sum() // 8)
        self.rng = np.random.default_rng(seed)
        self.satellites = satellites

    def group_by_satellite(self, order, labels):
        satellites = self.satellites[labels[order]]
        partition = np.argsort(satellites, kind="stable")
        cut = np.flatnonzero(np.diff(satellites[partition])) + 1
        blocks, leftovers = [], []
        for rows in np.split(order[partition], cut):
            size = len(rows) // 4 * 4
            blocks.append(rows[:size].reshape(-1, 4))
            leftovers.append(rows[size:])
        blocks = np.concatenate(blocks)
        leftovers = np.concatenate(leftovers)
        return np.r_[blocks[self.rng.permutation(len(blocks))].ravel(), self.rng.permutation(leftovers)]

    def epoch(self):
        groups = np.concatenate([self.rng.permutation(g)[:4 * n].reshape(-1, 4)
                                 for g, n in zip(self.groups, self.counts, strict=True)])
        labels = np.repeat(np.arange(len(self.groups)), self.counts)
        order = self.rng.permutation(len(groups))
        if self.satellites is not None:
            # Half-batch satellite groups increase in-batch wrong-beam coverage.
            # The usual duplicate repair below keeps eight distinct beams.
            order = self.group_by_satellite(order, labels)
        usable = self.steps_per_epoch * 8
        for a in range(0, usable, 8):
            for b in range(a + 1, a + 8):
                label = labels[order[b]]
                if label not in labels[order[a:b]]:
                    continue
                taken = set(labels[order[a:a + 8]])
                swap = next((k for k in range(a + 8, len(order)) if labels[order[k]] not in taken), None)
                if swap is None:
                    swap = next((k for k in range(a) if labels[order[k]] not in taken
                                 and label not in labels[order[k - k % 8:k - k % 8 + 8]]), None)
                if swap is None:
                    raise ValueError("Unable to form a complete batch of distinct beam identities")
                order[b], order[swap] = order[swap], order[b]
        return groups[order[:usable]].reshape(self.steps_per_epoch, 32)


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
    return PaperPAMF(config["runtime"], config["model_seed"], len(classes), config["raw_time_bins"],
                     feature_batch_norm=config.get("pamf_feature_batch_norm", False),
                     full_reconstruction=config.get("full_reconstruction_weight", 0) > 0)


@torch.no_grad()
def validate(model, arm, raw, labels, val_panel, device):
    feature = embeddings(model, raw, device)
    result = metrics(feature, val_panel, device)
    if arm != "SATIQ":
        mask = labels >= 0
        target = device_array(labels[mask].astype(np.int64), device)
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


def save(path, model, optimizer, sampler, step, epoch, validation):
    temporary = path.with_suffix(".partial")
    with temporary.open("xb") as writer:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=step, epoch=epoch,
                        sampling_rng=sampler.rng.bit_generator.state, validation=validation), writer)
    temporary.rename(path)


def train(config, source, cache, out, arm, smoke=False):
    classes, counts, roles = source_inputs(source, cache)
    device = cuda_device()
    model = make_model(config, arm, classes).to(device)
    reconstruction_weight = config.get("full_reconstruction_weight", 0)
    lr = 1e-5 if arm == "SATIQ" else config["pamf_lr"]
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, eps=1e-7)
    raw, labels, original = roles["L"]
    grouped_satellites = classes[:, 0] if config.get("group_same_satellite_batches", False) else None
    sampler = SourceBatches(original, counts, config["sampling_seed"], satellites=grouped_satellites)
    epochs = 1 if smoke else config["epochs"]
    total = 2 if smoke else epochs * sampler.steps_per_epoch
    if not smoke and "diagnostic_updates" in config:
        limit = config["diagnostic_updates"]
        if not 0 < limit <= total or limit not in config["early_validation_updates"]:
            raise ValueError("Diagnostic budget must be within the epoch budget and end at a source validation")
        total = limit
    val_panel = None if smoke else panel(roles["V"][1], classes, config["evaluation_seed"])
    satellites = device_array(classes[:, 0].astype(np.int64), device)
    out.mkdir(exist_ok=False)
    write_json(out / "resolved_config.json", dict(config, arm=arm, trained_classes=len(classes), lr=lr,
                                                 input_samples=11000, feature_dimensions=512 if arm == "SATIQ" else 320,
                                                 source=str(source), cache=str(cache), supervised=True,
                                                 source_roles=["L", "V"], test_access=False,
                                                 steps_per_epoch=sampler.steps_per_epoch, total_updates=total))
    start = time.monotonic()
    window = {}
    n = 0
    step = 0
    batches = ((epoch, i, index) for epoch in range(1, epochs + 1)
               for i, index in enumerate(sampler.epoch(), 1))
    with (out / "steps.jsonl").open("x", encoding="utf-8") as journal:
        for epoch, epoch_step, idx in batches:
            step += 1
            model.train()
            x = normalize_packet(device_array(raw[idx], device))
            y = device_array(labels[idx].astype(np.int64), device)
            optimizer.zero_grad(set_to_none=True)
            if arm == "SATIQ":
                feature, reconstructed = model(x)
                loss, stats = paper_loss(feature, reconstructed, x, y)
            else:
                outputs = model(x, view_logits=True, reconstruct=reconstruction_weight > 0)
                feature, logits, common, full = outputs[:4]
                loss, stats = supervised_loss(logits, common, full, y, satellites, config["pamf_sup"])
                if arm == "PAMF_METRIC":
                    triplet = angular_triplet(feature, y, margin=1.0)
                    loss = loss + config["pamf_triplet_weight"] * triplet
                    stats["triplet_loss"] = triplet.detach()
                if reconstruction_weight > 0:
                    reconstruction = F.mse_loss(outputs[4], reconstruction_target(x))
                    loss = loss + reconstruction_weight * reconstruction
                    stats["full_reconstruction_loss"] = reconstruction.detach()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {arm} training objective")
            loss.backward()
            optimizer.step()
            n += 1
            for key, value in {**stats, "loss": loss.detach()}.items():
                window[key] = window.get(key, 0) + value
            epoch_end = epoch_step == sampler.steps_per_epoch
            validate_now = not smoke and (step in config["early_validation_updates"]
                                          or (epoch_end and epoch in config["validation_epochs"]))
            if validate_now or epoch_end or step % config["log_every"] == 0 or smoke:
                info = {"updates": step, "epoch": epoch, "epoch_step": epoch_step,
                        "seconds": time.monotonic() - start,
                        **{k: float(v) / n for k, v in window.items()}}
                if validate_now:
                    info["source_validation"] = validate(model, arm, *roles["V"][:2], val_panel, device)
                    save(out / f"s{step:07d}.pt", model, optimizer, sampler, step, epoch, info["source_validation"])
                message = json.dumps(info, allow_nan=False)
                print(message, flush=True)
                journal.write(message + "\n")
                journal.flush()
                window, n = {}, 0
            if step >= total:
                break
    result = {"arm": arm, "updates": step, "mode": "smoke" if smoke else "supervised",
              "epochs": epochs if smoke else step // sampler.steps_per_epoch}
    if smoke:
        v_raw, v_labels, _ = roles["V"]
        eligible = np.flatnonzero(np.bincount(v_labels[v_labels >= 0]) > 16)[:4]
        rows = np.concatenate([np.flatnonzero(v_labels == label)[:18] for label in eligible])
        smoke_panel = panel(v_labels[rows], classes, config["evaluation_seed"])
        result["source_validation"] = validate(model, arm, v_raw[rows], v_labels[rows], smoke_panel, device)
    write_json(out / "complete.json", result)


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

    for arm in ("SATIQ", "PAMF_METRIC"):
        available = [gpu for gpu in gpus if gpu_ready(gpu)]
        while not available:
            time.sleep(15)
            available = [gpu for gpu in gpus if gpu_ready(gpu)]
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
