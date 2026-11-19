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
from .paper_eval import embeddings, metrics, model_input, panel
from .paper_pamf import PaperPAMF, branch_prototype_loss, reconstruction_target, supervised_loss
from .paper_native_pamf import EXTRA_WIDTH, NATIVE_PATHS, PaperNativePAMF, native_supervised_loss
from .wide_model import FEATURES
from .readout import gpu_ready
from .train import cuda_device, device_array


ARMS = ("SATIQ", "PAMF_SUP", "PAMF_METRIC")


class SourceBatches:
    def __init__(self, labels, class_counts, seed, satellites=None, *, pair_same_satellite=False):
        ordered = np.argsort(labels, kind="stable")
        boundaries = np.r_[0, np.cumsum(class_counts)]
        self.groups = [ordered[boundaries[k]:boundaries[k + 1]] for k, n in enumerate(class_counts) if n >= 4]
        self.counts = np.array([len(g) // 4 for g in self.groups])
        self.steps_per_epoch = int(self.counts.sum() // 8)
        self.rng = np.random.default_rng(seed)
        self.satellites = satellites
        self.pair_same_satellite = pair_same_satellite
        if pair_same_satellite and satellites is None:
            raise ValueError("Same-satellite pairing requires the source satellite map")

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

    def add_satellite_pairs(self, order, labels):
        for start in range(0, len(order) - 8, 8):
            end = start + 8
            current = labels[order[start:end]]
            satellites = self.satellites[current]
            if np.unique(satellites).size < 8:
                continue
            future = labels[order[end:end + 16 * 8]].reshape(-1, 8)
            donor_ok = (future != current[-1]).all(axis=1, keepdims=True)
            can_join = np.isin(self.satellites[future], satellites[:-1]) & ~np.isin(future, current)
            candidates = np.flatnonzero(donor_ok & can_join)
            if len(candidates):
                donor = end + candidates[0]
                order[end - 1], order[donor] = order[donor], order[end - 1]

    def epoch(self):
        groups = np.concatenate([self.rng.permutation(g)[:4 * n].reshape(-1, 4)
                                 for g, n in zip(self.groups, self.counts, strict=True)])
        labels = np.repeat(np.arange(len(self.groups)), self.counts)
        order = self.rng.permutation(len(groups))
        if self.satellites is not None and not self.pair_same_satellite:
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
        if self.pair_same_satellite:
            self.add_satellite_pairs(order[:usable], labels)
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
    if "native_expansion" in config:
        if arm != "PAMF_SUP":
            raise ValueError("The native expansion experiment requires the PAMF-SUP arm")
        model = PaperNativePAMF(config["runtime"], config["model_seed"], len(classes), config["native_expansion"],
                                satiq_sinc_init=config.get("satiq_sinc_init", False),
                                center_fusion=config.get("center_fusion", False),
                                waveform_samples=config.get("native_waveform_samples", 8000),
                                align_reference_time=config.get("align_reference_time", False))
    elif arm == "SATIQ":
        model = PaperSatIQ()
    else:
        model = PaperPAMF(config["runtime"], config["model_seed"], len(classes), config["raw_time_bins"],
                          feature_batch_norm=config.get("pamf_feature_batch_norm", False),
                          full_reconstruction=config.get("full_reconstruction_weight", 0) > 0,
                          phase_relative_raw=config.get("phase_relative_raw", False))
    model.signed_input = config.get("signed_input", False)
    return model


@torch.no_grad()
def validate(model, arm, raw, labels, val_panel, device, class_log_prior=None):
    feature = embeddings(model, raw, device)
    result = metrics(feature, val_panel, device)
    if arm != "SATIQ":
        mask = labels >= 0
        target = device_array(labels[mask].astype(np.int64), device)
        known_features = feature[mask]
        weight = F.normalize(model.head.weight, dim=1)
        block_losses, correct = [], 0
        if class_log_prior is not None:
            score_sum = torch.zeros_like(class_log_prior)
            predicted_counts = torch.zeros_like(class_log_prior, dtype=torch.int64)
            prior_losses, prior_correct = [], 0
        for start in range(0, len(target), 256):
            end = start + 256
            vec = device_array(known_features[start:end], device)
            scores = 30 * F.linear(vec, weight)
            y = target[start:end]
            block_losses.append(float(F.cross_entropy(scores, y, reduction="sum")))
            prediction = scores.argmax(1)
            correct += int((prediction == y).sum())
            if class_log_prior is not None:
                adjusted = scores + class_log_prior
                prior_losses.append(float(F.cross_entropy(adjusted, y, reduction="sum")))
                prior_correct += int((adjusted.argmax(1) == y).sum())
                predicted_counts += torch.bincount(prediction, minlength=len(class_log_prior))
                score_sum += scores.sum(dim=0)
        result.update(known_class_records=len(target), head_top1=correct / len(target),
                      head_ce=sum(block_losses) / len(target))
        if class_log_prior is not None:
            mean_scores = score_sum / len(target)
            correlation = F.cosine_similarity(mean_scores - mean_scores.mean(),
                                              class_log_prior - class_log_prior.mean(), dim=0)
            result.update(head_prior_adjusted_ce=sum(prior_losses) / len(target),
                          head_prior_adjusted_top1=prior_correct / len(target),
                          head_predicted_classes=int((predicted_counts > 0).sum()),
                          head_mode_share=int(predicted_counts.max()) / len(target),
                          head_mean_logit_ce=float(-mean_scores.log_softmax(0)[target].mean()),
                          head_mean_logit_log_prior_correlation=float(correlation))
    return result


def save(path, model, optimizer, sampler, step, epoch, validation):
    temporary = path.with_suffix(".partial")
    with temporary.open("xb") as writer:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), updates=step, epoch=epoch,
                        sampling_rng=sampler.rng.bit_generator.state, validation=validation,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state() if torch.cuda.is_initialized() else None), writer)
    temporary.rename(path)


def restore_completed_epoch(checkpoint, config, model, optimizer, sampler):
    parent = checkpoint.parent
    previous = read_json(parent / "resolved_config.json")
    complete = read_json(parent / "complete.json")
    epoch = complete["epochs"]
    step = epoch * sampler.steps_per_epoch
    if (complete["mode"] != "supervised" or not 0 < epoch < config["epochs"]
            or epoch != previous["epochs"] or complete["updates"] != step or previous["total_updates"] != step
            or checkpoint.name != f"s{step:07d}.pt" or complete["arm"] != config["arm"]):
        raise ValueError("Resume requires a completed shorter supervised source budget")
    budget_fields = {"epochs", "early_validation_updates", "validation_epochs", "total_updates", "resume"}
    if ({k: v for k, v in previous.items() if k not in budget_fields}
            != {k: v for k, v in config.items() if k not in budget_fields}
            or previous["source_roles"] != ["L", "V"] or previous["test_access"]):
        raise ValueError("The previous source data or model contract differs")
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if frozen["epoch"] != epoch or frozen["updates"] != step or not frozen["validation"]:
        raise ValueError("Checkpoint does not match the completed source epoch")
    model.load_state_dict(frozen["model"], strict=True)
    optimizer.load_state_dict(frozen["optimizer"])
    sampler.rng.bit_generator.state = frozen["sampling_rng"]
    restart_seed = None
    if "torch_rng" in frozen:
        torch.set_rng_state(frozen["torch_rng"])
        if frozen["cuda_rng"] is not None:
            torch.cuda.set_rng_state(frozen["cuda_rng"])
    else:
        # The original 100-epoch runs stored NumPy batching RNG but no Torch RNG.
        restart_seed = config["model_seed"] + epoch
        torch.manual_seed(restart_seed)
    return dict(checkpoint=str(checkpoint.resolve()), epoch=epoch, updates=step, restart_torch_seed=restart_seed)


def train(config, source, cache, out, arm, smoke=False, resume=None):
    classes, counts, roles = source_inputs(source, cache)
    device = cuda_device()
    model = make_model(config, arm, classes).to(device)
    native_only = isinstance(model, PaperNativePAMF)
    input_samples = model.waveform_samples if native_only else 11000
    feature_dimensions = getattr(model, "feature_dimensions", 512 if arm == "SATIQ" else 320)
    reconstruction_weight = config.get("full_reconstruction_weight", 0)
    prototype_weight = config.get("branch_prototype_weight", 0)
    prototype_path_weights = None
    if "branch_prototype_weights" in config:
        prototype_path_weights = torch.tensor(config["branch_prototype_weights"], dtype=torch.float32, device=device)
        if (not native_only or prototype_weight <= 0 or prototype_path_weights.shape != (len(NATIVE_PATHS),)
                or not torch.isfinite(prototype_path_weights).all() or (prototype_path_weights <= 0).any()):
            raise ValueError("Expected positive weights for the five native branch prototypes")
        prototype_path_weights = prototype_path_weights / prototype_path_weights.sum()
    expansion_prototype_weight = config.get("expansion_prototype_weight", 0)
    if expansion_prototype_weight > 0 and (not native_only or feature_dimensions <= FEATURES):
        raise ValueError("Expansion prototype supervision requires a widened native PAMF-SUP embedding")
    lr = 1e-5 if arm == "SATIQ" else config["pamf_lr"]
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=lr, eps=1e-7)
    raw, labels, original = roles["L"]
    grouped = config.get("group_same_satellite_batches", False)
    paired = config.get("pair_same_satellite_batches", False)
    if grouped and paired:
        raise ValueError("Select one same-satellite batching strategy")
    satellites_by_beam = classes[:, 0] if grouped or paired else None
    sampler = SourceBatches(original, counts, config["sampling_seed"], satellites=satellites_by_beam,
                            pair_same_satellite=paired)
    class_log_prior = None
    if config.get("prior_adjusted_sup", False):
        if arm != "PAMF_SUP":
            raise ValueError("Class-frequency adjustment is enabled only for the PAMF-SUP diagnostic")
        log_prior = np.log(sampler.counts) - np.log(sampler.counts.sum())
        class_log_prior = device_array(log_prior.astype(np.float32), device)
    epochs = 1 if smoke else config["epochs"]
    total = 2 if smoke else epochs * sampler.steps_per_epoch
    if not smoke and "diagnostic_updates" in config:
        limit = config["diagnostic_updates"]
        if not 0 < limit <= total or limit not in config["early_validation_updates"]:
            raise ValueError("Diagnostic budget must be within the epoch budget and end at a source validation")
        total = limit
    val_panel = None if smoke else panel(roles["V"][1], classes, config["evaluation_seed"])
    satellites = device_array(classes[:, 0].astype(np.int64), device)
    resolved = dict(config, arm=arm, trained_classes=len(classes), lr=lr, input_samples=input_samples,
                    feature_dimensions=feature_dimensions, source=str(source.resolve()), cache=str(cache.resolve()),
                    supervised=True, source_roles=["L", "V"], test_access=False,
                    steps_per_epoch=sampler.steps_per_epoch, total_updates=total)
    start_epoch, step = 0, 0
    if resume is not None:
        if smoke or "diagnostic_updates" in config:
            raise ValueError("Epoch continuation requires the full supervised source budget")
        resumed = restore_completed_epoch(resume, resolved, model, optimizer, sampler)
        resolved["resume"] = resumed
        start_epoch, step = resumed["epoch"], resumed["updates"]
    out.mkdir(exist_ok=False)
    write_json(out / "resolved_config.json", resolved)
    start = time.monotonic()
    window = {}
    n = 0
    batches = ((epoch, i, index) for epoch in range(start_epoch + 1, epochs + 1)
               for i, index in enumerate(sampler.epoch(), 1))
    with (out / "steps.jsonl").open("x", encoding="utf-8") as journal:
        for epoch, epoch_step, idx in batches:
            step += 1
            model.train()
            x = model_input(model, device_array(raw[idx, :, :input_samples], device))
            y = device_array(labels[idx].astype(np.int64), device)
            optimizer.zero_grad(set_to_none=True)
            if arm == "SATIQ":
                feature, reconstructed = model(x)
                loss, stats = paper_loss(feature, reconstructed, x, y)
            elif native_only:
                feature, logits, branch_logits, branches = model(x, view_logits=True)
                loss, stats = native_supervised_loss(logits, branch_logits, y, satellites, config["pamf_sup"])
                if prototype_weight > 0:
                    prototype = branch_prototype_loss(branches, prototype_path_weights)
                    loss = loss + prototype_weight * prototype
                    stats["branch_prototype_ce"] = prototype.detach()
                if expansion_prototype_weight > 0:
                    expansion = feature[:, FEATURES:].unsqueeze(0)
                    prototype = branch_prototype_loss(expansion)
                    loss = loss + expansion_prototype_weight * prototype
                    stats["expansion_prototype_ce"] = prototype.detach()
            else:
                outputs = model(x, view_logits=True, reconstruct=reconstruction_weight > 0,
                                capture_branches=prototype_weight > 0)
                feature, logits, common, full = outputs[:4]
                loss, stats = supervised_loss(logits, common, full, y, satellites, config["pamf_sup"],
                                              class_log_prior)
                if arm == "PAMF_METRIC":
                    triplet = angular_triplet(feature, y, margin=1.0)
                    loss = loss + config["pamf_triplet_weight"] * triplet
                    stats["triplet_loss"] = triplet.detach()
                if reconstruction_weight > 0:
                    reconstruction = F.mse_loss(outputs[4], reconstruction_target(x))
                    loss = loss + reconstruction_weight * reconstruction
                    stats["full_reconstruction_loss"] = reconstruction.detach()
                if prototype_weight > 0:
                    prototype = branch_prototype_loss(outputs[-1])
                    loss = loss + prototype_weight * prototype
                    stats["branch_prototype_ce"] = prototype.detach()
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
                    info["source_validation"] = validate(model, arm, *roles["V"][:2], val_panel, device,
                                                         class_log_prior)
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
        result["source_validation"] = validate(model, arm, v_raw[rows], v_labels[rows], smoke_panel, device,
                                               class_log_prior)
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
    parser.add_argument("--native-expansion", choices=tuple(EXTRA_WIDTH))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--final-epoch", type=int)
    args = parser.parse_args()
    if (args.resume is None) != (args.final_epoch is None) or (args.resume and args.mode != "train"):
        parser.error("--resume and --final-epoch must be used together in train mode")
    if args.mode == "launch":
        if not args.gpus:
            parser.error("launch requires physical GPUs")
        if args.native_expansion:
            parser.error("native expansion needs a single PAMF-SUP train or smoke row")
        launch(args.config, args.source, args.cache, args.output, args.gpus)
    else:
        if not args.arm:
            parser.error("train/smoke requires a matrix arm")
        config = read_json(args.config)
        if args.native_expansion:
            config["native_expansion"] = args.native_expansion
        if args.resume:
            if args.final_epoch <= config["epochs"]:
                parser.error("--final-epoch must be greater than the existing budget")
            checks = range((config["epochs"] // 10 + 1) * 10, args.final_epoch + 1, 10)
            config["validation_epochs"] = sorted(set(config["validation_epochs"]) | set(checks) | {args.final_epoch})
            config["epochs"] = args.final_epoch
        train(config, args.source, args.cache, args.output, args.arm, args.mode == "smoke", args.resume)
