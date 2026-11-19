"""Matched supervised experiments on the 8,000-sample SatIQ common prefix."""

import argparse
from collections import Counter
from itertools import count
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .data import read_json, write_json
from .model import build_model, normalized
from .objectives import supervised
from .readout import gpu_ready
from .train import Batches, cuda_device, device_array, metrics, strength
from .wide_model import build_wide_model, wide_views


ROWS = ("W0", "B0", "W1", "L0")
WIDE_SCHEMA = "satiq_oxford_66_sat_cell_common_uw8000_v1"
COUNTS = {"L": 7920, "V": 2376}
L_PER_CLASS_DAY = 15
V_PER_CLASS = 36


def read_supervised(source):
    contract = read_json(source / "contract.json")
    required = {"schema": WIDE_SCHEMA, "counts": COUNTS,
                "sample_rates_hz": [25000000, 800000], "prefix_samples": 8000,
                "split_seed": 2026111901, "label_key": ["ra_sat", "ra_cell"],
                "l_per_class_day": L_PER_CLASS_DAY, "v_per_class": V_PER_CLASS}
    if any(contract.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected common-prefix SatIQ source contract")
    label_map = contract["classes"]
    if (len(label_map) != 66 or any(len(pair) != 2 or any(type(x) is not int for x in pair)
                                   or not 0 <= pair[0] < 128 or not 0 <= pair[1] < 64 for pair in label_map)
            or len({tuple(pair) for pair in label_map}) != 66
            or list(Counter(pair[0] for pair in label_map).values()) != [3] * 22):
        raise ValueError("Label map must identify 66 satellite and beam pairs, three per satellite")
    roles = {}
    for role, n in COUNTS.items():
        with np.load(source / f"{role}_meta.npz", allow_pickle=False) as rows:
            if set(rows.files) != {"ids", "day", "y"}:
                raise ValueError(f"{role} must contain only IDs, days and visible source labels")
            meta = {key: rows[key] for key in rows.files}
        raw = np.load(source / f"{role}_raw.npy", mmap_mode="r", allow_pickle=False)
        coarse = np.load(source / f"{role}_coarse.npy", mmap_mode="r", allow_pickle=False)
        valid_days = range(8) if role == "L" else range(8, 11)
        if (raw.shape != (n, 2, 8000) or coarse.shape != (n, 2, 256)
                or raw.dtype != np.float32 or coarse.dtype != np.float32
                or any(values.shape != (n,) for values in meta.values())
                or meta["y"].dtype != np.int64 or meta["day"].dtype != np.int64
                or len(set(meta["ids"])) != n or not np.isin(meta["day"], valid_days).all()
                or not np.isin(meta["y"], range(66)).all()):
            raise ValueError(f"Invalid supervised common-prefix role {role}")
        if role == "V" and not np.all(np.bincount(meta["y"], minlength=66) == V_PER_CLASS):
            raise ValueError("V must contain an equal number of records per beam transmitter")
        roles[role] = raw, coarse, meta
    if set(roles["L"][2]["ids"]) & set(roles["V"][2]["ids"]):
        raise ValueError("L and V overlap")
    return contract, roles


def normalize_pair(raw, coarse):
    gain = raw.square().sum(1, keepdim=True).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
    return raw / gain, coarse / gain


def batch(role, indices, device):
    raw, coarse, _ = role
    return normalize_pair(device_array(raw[indices], device), device_array(coarse[indices], device))


def describe(logits, labels, satellites):
    if logits.shape != (len(labels), 66) or not np.isfinite(logits).all():
        raise ValueError("Validation requires one finite 66-class row per source V packet")
    predicted = logits.argmax(axis=1)
    matrix = np.bincount(66 * labels + predicted, minlength=66**2).reshape(66, 66)
    scores = metrics(matrix)
    scores["top5"] = float((np.argpartition(logits, -5, axis=1)[:, -5:] == labels[:, None]).any(1).mean())
    correct_satellite = satellites[predicted] == satellites[labels]
    same_satellite_count = int(correct_satellite.sum())
    scores["satellite_accuracy"] = float(correct_satellite.mean())
    scores["same_satellite_wrong_beam"] = int((correct_satellite & (predicted != labels)).sum())
    scores["beam_accuracy_given_correct_satellite"] = (
        float((predicted == labels).sum() / same_satellite_count) if same_satellite_count else None)
    offset = logits.max(axis=1, keepdims=True)
    scores["nll"] = float((np.log(np.exp(logits - offset).sum(1)) + offset[:, 0]
                           - logits[np.arange(len(labels)), labels]).mean())
    return scores


@torch.no_grad()
def validate(model, role, device, short, satellites):
    model.eval()
    data = []
    for start in range(0, len(role[0]), 256):
        raw, coarse = batch(role, slice(start, start + 256), device)
        logits = model(normalized(raw[:, :, 384:640])) if short else model(raw, coarse)
        data.append(np.from_dlpack(logits.cpu()))
    logits = np.concatenate(data)
    result = describe(logits, role[2]["y"], satellites)
    result["days"] = {}
    for day in range(8, 11):
        subset = role[2]["day"] == day
        result["days"][str(day)] = describe(logits[subset], role[2]["y"][subset], satellites)
    return result


def snapshot(folder, model, optimizer, sampler, augment, device, report):
    path = folder / f"e{report['epoch']:08d}.pt"
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as writer:
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                        sampler=sampler.rng.bit_generator.state, augmentation=augment.get_state(),
                        torch_rng=torch.random.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                        epoch=report["epoch"], updates=report["updates"], validation=report.get("v")), writer)
    temporary.rename(path)


def train(config, source, out, arm, smoke=False):
    if config["source_v_accuracy_target"] != .98 or config["labels"] != ["ra_sat", "ra_cell"]:
        raise ValueError("Expected 98% supervised source validation target")
    contract, data = read_supervised(source)
    satellites = np.asarray([pair[0] for pair in contract["classes"]])
    device = cuda_device()
    torch.manual_seed(config["model_seed"])
    short, use_raw, hybrid = arm == "B0", arm in {"W0", "W1"}, arm == "W1"
    network = (build_model(config["runtime"], config["model_seed"], 0) if short
               else build_wide_model(config["runtime"], config["model_seed"], use_raw)).to(device)
    out.mkdir(parents=True, exist_ok=False)
    checkpoints = out / "checkpoints"
    checkpoints.mkdir()
    write_json(out / "resolved_config.json",
               dict(config, role="smoke" if smoke else "wide_supervised", arm=arm, raw_branch=use_raw,
                    hybrid=hybrid, source=str(source), output=str(out), epochs=None, initialization="scratch",
                    uses_u=False, input_schema=WIDE_SCHEMA, source_counts=COUNTS, classes=contract["classes"],
                    input_samples=256 if short else 8000,
                    parameters=sum(p.numel() for p in network.parameters())))
    initial = out / "scratch_e00.pt"
    torch.save(network.state_dict(), initial)
    network.load_state_dict(torch.load(initial, map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(network.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    sampler = Batches(data["L"][2], config["data_seed"], L_PER_CLASS_DAY)
    augment = torch.Generator(device=device).manual_seed(config["augmentation_seed"])
    updates_per_epoch = 1 if smoke else config["steps_per_epoch"]
    start = time.monotonic()
    with (out / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in count(1):
            network.train()
            totals = {}
            physical_strength = max(strength(epoch, 1), config["physical_strength_floor"])
            for _ in range(updates_per_epoch):
                ids = sampler.next()
                raw, coarse = batch(data["L"], ids, device)
                visible = data["L"][2]
                target, day = [device_array(visible[key][ids], device) for key in ("y", "day")]
                packet_ids = device_array(ids, device)
                phase_raw, phase_coarse, phys_raw, phys_coarse = wide_views(raw, coarse, augment, physical_strength)
                optimizer.zero_grad(set_to_none=True)
                if short:
                    raw, phase_raw, phys_raw = (x[:, :, 384:640] for x in (raw, phase_raw, phys_raw))
                    gain = raw.square().sum(1, keepdim=True).mean(-1, keepdim=True).clamp_min(1e-8).sqrt()
                    raw, phase_raw, phys_raw = (x / gain for x in (raw, phase_raw, phys_raw))
                    primary, ordinary, physical = (network(x) for x in (raw, phase_raw, phys_raw))
                else:
                    primary = network(raw, coarse, capture=hybrid)
                    ordinary = network(phase_raw, phase_coarse)
                    physical = network(phys_raw, phys_coarse, capture=hybrid)
                if not hybrid:
                    primary, physical = (primary, None, None), (physical, None, None)
                delta = (phys_raw - raw).square().mean(dim=(1, 2))
                loss, measures = supervised(primary, ordinary, physical, target, day, packet_ids,
                                             hybrid, delta > 0)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite supervised wide loss")
                loss.backward()
                if smoke:
                    if not hybrid or any(int(measures[k]) != len(ids)
                                         for k in ("packet_anchors", "day_anchors", "prototype_anchors")):
                        raise ValueError("All supervised Hybrid losses must be enabled in the wide smoke")
                    if any(p.grad is not None and not p.grad.isfinite().all() for p in network.parameters()):
                        raise ValueError("Nonfinite gradient in the wide smoke")
                    if not any(p.grad is not None and bool((p.grad != 0).any())
                               for p in network.raw_encoder.parameters()):
                        raise ValueError("The original-rate path did not receive a supervised gradient")
                optimizer.step()
                measures.update(total_loss=loss.detach(), l_seen=len(ids),
                                physical_rms=delta.sqrt().mean().detach())
                for key, val in measures.items():
                    totals[key] = totals.get(key, 0) + val
            report = dict(epoch=epoch, updates=epoch * updates_per_epoch,
                          train={k: float(v) if k.endswith(("seen", "anchors")) else float(v) / updates_per_epoch
                                 for k, v in totals.items()}, physical_strength=physical_strength,
                          seconds=time.monotonic() - start, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            reached = False
            if not smoke and (epoch == 1 or epoch % config["validate_every"] == 0):
                report["v"] = validate(network, data["V"], device, short, satellites)
                reached = report["v"]["accuracy"] >= config["source_v_accuracy_target"]
                report["target_reached"] = reached
            if not smoke and (epoch % config["checkpoint_every"] == 0 or reached):
                snapshot(checkpoints, network, optimizer, sampler, augment, device, report)
            entry = json.dumps(report, allow_nan=False)
            log.write(entry + "\n")
            log.flush()
            print(entry, flush=True)
            if smoke or reached:
                break
    torch.save(network.state_dict(), out / "source_final.pt")
    write_json(out / "complete.json", report)


def launch(config_path, source, out, gpus):
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "logs"
    logs.mkdir()
    write_json(out / "owner.json", dict(pid=os.getpid(), eligible_physical_gpus=gpus, row_order=list(ROWS)))
    print(json.dumps(dict(event="queue_registered", rows=ROWS, eligible_physical_gpus=gpus)), flush=True)

    def start(mode, arm, name, gpu):
        cmd = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.wide_run", mode,
               str(config_path), str(source), str(out / name), "--arm", arm]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        with (logs / f"{name}.log").open("xb") as log:
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(event="started", row=name, pid=child.pid, gpu=gpu, command=cmd)), flush=True)
        return child

    while True:
        available = [gpu for gpu in gpus if gpu_ready(gpu)]
        if available:
            break
        time.sleep(20)
    if start("smoke", "W1", "smoke_source", available[0]).wait():
        raise RuntimeError("Real-source wide smoke failed")

    pending, workers, outcomes = list(ROWS), {}, {}
    while pending or workers:
        for arm, (child, gpu) in list(workers.items()):
            exit_code = child.poll()
            if exit_code is None:
                continue
            del workers[arm]
            completed = out / arm / "complete.json"
            status = "TECHNICAL_FAILURE"
            if exit_code == 0 and completed.is_file() and read_json(completed).get("target_reached"):
                status = "SOURCE_V_TARGET_REACHED"
            outcomes[arm] = dict(status=status, returncode=exit_code)
            print(json.dumps(dict(event="exited", row=arm, gpu=gpu, **outcomes[arm])), flush=True)
        reserved = {gpu for _, gpu in workers.values()}
        for gpu in gpus:
            if not pending:
                break
            if gpu not in reserved and gpu_ready(gpu):
                arm = pending.pop(0)
                workers[arm] = (start("train", arm, arm, gpu), gpu)
        if pending or workers:
            time.sleep(15)
    write_json(out / "matrix_outcomes.json", outcomes)
    if any(row["status"] == "TECHNICAL_FAILURE" for row in outcomes.values()):
        raise RuntimeError("One or more wide rows ended with a technical failure")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=("train", "smoke", "launch"))
    for name in ("config", "source", "out"):
        cli.add_argument(name, type=Path)
    cli.add_argument("--arm", choices=ROWS)
    cli.add_argument("--gpus", type=int, nargs="+")
    args = cli.parse_args()
    if args.mode == "launch":
        if not args.gpus or len(set(args.gpus)) != len(args.gpus):
            cli.error("Launch requires a unique set of eligible physical GPUs")
        launch(args.config.resolve(), args.source.resolve(), args.out.resolve(), args.gpus)
    else:
        if args.arm is None:
            cli.error("Train or smoke needs --arm")
        train(read_json(args.config), args.source.resolve(), args.out.resolve(), args.arm, args.mode == "smoke")
