"""Compare the two matched SatIQ models after a scheduled source validation epoch."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import read_json
from .paper_final_dispatch import wait_for_gpu
from .paper_single_packet import ARMS
from .paper_single_packet_dispatch import exit_status, training_controller_active


SOURCE_STEPS_PER_EPOCH = 28225


def checkpoint_paths(training, source, cache, epoch=5):
    updates = epoch * SOURCE_STEPS_PER_EPOCH
    paths = []
    for arm in ARMS:
        row = training / arm
        config = read_json(row / "resolved_config.json")
        if (config["arm"] != arm or config["steps_per_epoch"] != SOURCE_STEPS_PER_EPOCH
                or epoch not in config["validation_epochs"]
                or config["trained_classes"] != 3103 or config["input_samples"] != 11000
                or not config["signed_input"] or config["augmentation"] != "none"
                or config["source_roles"] != ["L", "V"] or config["test_access"]
                or Path(config["source"]).resolve() != source or Path(config["cache"]).resolve() != cache
                or config["evaluation_seed"] != 2026111907):
            raise ValueError(f"Invalid matched source configuration for {arm}")
        paths.append(row / f"s{updates:07d}.pt")
    return paths


def dispatch(source, cache, training, output, owner, gpus, epoch=5):
    if read_json(training / "owner.json")["pid"] != owner:
        raise ValueError("Training owner differs from the registered matched run")
    checkpoints = checkpoint_paths(training, source, cache, epoch)
    updates = epoch * SOURCE_STEPS_PER_EPOCH
    output.mkdir(exist_ok=False)
    print(f"Waiting for matched source E{epoch} checkpoints from PID {owner}", flush=True)
    while not all(checkpoint.is_file() for checkpoint in checkpoints):
        statuses = [exit_status(training, arm) for arm in ARMS]
        if any(status not in ("", "0") for status in statuses) or not training_controller_active(training, owner):
            raise RuntimeError("Matched training ended without both requested source checkpoints")
        time.sleep(30)

    gpu = wait_for_gpu(gpus)
    scores = output / f"source_e{epoch}_closed_set.json"
    pending = scores.with_suffix(".pending")
    argv = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.paper_sup_prototype_readout",
            str(source), str(cache), str(pending), *map(str, checkpoints),
            "--seed", "2026111907", "--per-class", "16", "--device", "cuda"]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
    with (output / "readout.log").open("xb") as stream:
        worker = subprocess.Popen(argv, env=env, stdout=stream, stderr=subprocess.STDOUT)
        print(f"Scoring both E{epoch} models on physical GPU {gpu}, PID {worker.pid}", flush=True)
        if status := worker.wait():
            raise subprocess.CalledProcessError(status, argv)
    result = read_json(pending)
    paths = [Path(row["checkpoint"]) for row in result["results"]]
    if (result["classes"] != 3103 or result["known_validation_records"] != 49814
            or paths != checkpoints or any(row["updates"] != updates for row in result["results"])):
        raise ValueError(f"Expected both exact E{epoch} source comparisons on the shared cohort")
    pending.rename(scores)
    print(f"Completed matched source E{epoch} comparison", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "cache", "training", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--owner", required=True, type=int)
    parser.add_argument("--gpus", required=True, type=int, nargs="+")
    parser.add_argument("--epoch", type=int, default=5)
    args = parser.parse_args()
    dispatch(args.source.resolve(), args.cache.resolve(), args.training.resolve(), args.output.resolve(),
             args.owner, args.gpus, args.epoch)
