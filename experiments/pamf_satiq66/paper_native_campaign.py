"""Run the four matched native PAMF-SUP width rows on separate N607 GPUs."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import write_json
from .paper_native_pamf import EXTRA_WIDTH
from .wide_direct import gpu_inventory


def launch(config, source, cache, out, gpus):
    if len(gpus) != len(EXTRA_WIDTH) or len(set(gpus)) != len(gpus):
        raise ValueError("The native width matrix requires four distinct physical GPUs")
    inventory = gpu_inventory()
    if any(inventory[gpu][0] < 4096 or inventory[gpu][1] > 2 for gpu in gpus):
        raise RuntimeError("A requested GPU lacks 4 GiB free or already has three compute jobs")
    out.mkdir(exist_ok=False)
    assignments = dict(zip(EXTRA_WIDTH, gpus, strict=True))
    write_json(out / "owner.json", dict(pid=os.getpid(), assignments=assignments, inventory=inventory))
    logs = out / "logs"
    logs.mkdir()

    def start(expansion, gpu, mode):
        row = expansion if mode == "train" else f"smoke_{expansion}"
        argv = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.paper_train", mode,
                str(config), str(source), str(cache), str(out / row), "--arm", "PAMF_SUP",
                "--native-expansion", expansion]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        with (logs / f"{row}.log").open("xb") as log:
            child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(event="started", expansion=expansion, mode=mode, gpu=gpu, pid=child.pid)), flush=True)
        return child

    workers, outcome = {}, {}
    for expansion, gpu in assignments.items():
        free, count = gpu_inventory()[gpu]
        if free < 4096 or count > 2:
            outcome[expansion] = {"stage": "preflight", "memory_free_mib": free, "compute_processes": count}
            continue
        code = start(expansion, gpu, "smoke").wait()
        if code:
            outcome[expansion] = {"stage": "smoke", "returncode": code}
        else:
            workers[expansion] = start(expansion, gpu, "train")

    while workers:
        for expansion, worker in list(workers.items()):
            code = worker.poll()
            if code is not None:
                outcome[expansion] = {"stage": "train", "returncode": code}
                del workers[expansion]
        if workers:
            time.sleep(15)
    write_json(out / "matrix_outcomes.json", outcome)
    if any(row.get("returncode") != 0 for row in outcome.values()):
        raise RuntimeError("A native width row did not complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "source", "cache", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--gpus", nargs=4, type=int, required=True)
    args = parser.parse_args()
    launch(args.config, args.source, args.cache, args.output, args.gpus)
