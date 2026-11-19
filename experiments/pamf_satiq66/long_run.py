"""Train both supervised SatIQ66 rows to 98%, then check hidden-U readiness."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import read_json, write_json
from .readout import gpu_ready


ARMS = ("S0", "S1")


def run(config, source, out, gpus):
    cfg = read_json(config)
    if cfg.get("long_budget", {}).get("source_v_accuracy_target") != .98:
        raise ValueError("This launch requires the registered 98% source V accuracy target")
    out.mkdir(parents=True, exist_ok=False)
    logs = out / "logs"
    logs.mkdir()
    mapping = dict(zip(ARMS, gpus, strict=True))
    write_json(out / "owner.json", {"pid": os.getpid(), "physical_gpus": mapping})

    def start(mode, arm, gpu, name, row_out):
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.train", mode, str(config),
               str(source), str(row_out), "--arm", arm, "--representation", "0"]
        with (logs / f"{name}.log").open("xb") as log:
            task = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(json.dumps(dict(row=name, pid=task.pid, gpu=gpu, command=cmd)), flush=True)
        return task

    while not gpu_ready(mapping["S1"]):
        time.sleep(15)
    if start("smoke", "S1", mapping["S1"], "smoke_source", out / "smoke_source").wait():
        raise RuntimeError("Real-source scratch smoke failed before long-run dispatch")

    pending = mapping.copy()
    workers = {}
    outcomes = {}
    while pending or workers:
        for arm, gpu in list(pending.items()):
            if gpu_ready(gpu):
                workers[arm] = start("train", arm, gpu, arm, out / arm)
                del pending[arm]
        for arm, task in list(workers.items()):
            code = task.poll()
            if code is None:
                continue
            if code == 0:
                completion = read_json(out / arm / "complete.json")
                if completion["v"]["accuracy"] < .98:
                    raise RuntimeError(f"{arm} exited below the registered threshold")
                outcomes[arm] = dict(status="VALIDATION_TARGET_REACHED", **completion)
            else:
                outcomes[arm] = {"status": "FAILED", "returncode": code}
            print(json.dumps(dict(row=arm, outcome=outcomes[arm])), flush=True)
            del workers[arm]
        if pending or workers:
            time.sleep(10)
    write_json(out / "matrix_outcomes.json", outcomes)
    if any(row["status"] == "FAILED" for row in outcomes.values()):
        raise RuntimeError("One or more rows ended with a technical failure")
    while not gpu_ready(mapping["S1"]):
        time.sleep(15)
    if start("probe-source-u", "S1", mapping["S1"], "unlabeled_readiness", out).wait():
        raise RuntimeError("Frozen supervised hidden-U readiness probe failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "source", "out"):
        parser.add_argument(arg, type=Path)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2:
        parser.error("This long run uses one task worker per distinct physical GPU")
    run(args.config.resolve(), args.source.resolve(), args.out.resolve(), args.gpus)
