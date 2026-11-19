"""Dispatch the two pending PAMF rows after taking ownership from the queue."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from experiments.pamf_satiq66.wide_direct import await_state, controller_state, gpu_inventory


def run(release, data, root, gpus):
    config = release / "configs" / "pamf_satiq_ccs2023_supervised_20261119.json"
    assigned = dict(zip(("PAMF_SUP", "PAMF_METRIC"), gpus, strict=True))
    owner = json.loads((root / "owner.json").read_text())
    pid = owner["pid"]
    proc = Path(f"/proc/{pid}")
    expected = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.paper_campaign",
                str(config), str(data), str(root)]
    argv = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if proc.stat().st_uid != os.getuid() or argv[:len(expected)] != expected or (proc / "cwd").readlink() != release:
        raise RuntimeError("Original supervised dispatcher identity did not match")
    os.kill(pid, signal.SIGSTOP)
    try:
        await_state(pid, {"T"})
        if any((root / name).exists() or (root / "logs" / f"{name}.log").exists() for name in assigned):
            raise RuntimeError("One or more PAMF rows were already dispatched")
        if not (root / "SATIQ" / "resolved_config.json").is_file():
            raise RuntimeError("Expected the original SATIQ training worker to have started")
        inventory = gpu_inventory()
        if any(inventory[gpu][0] < 4096 or inventory[gpu][1] > 2 for gpu in assigned.values()):
            raise RuntimeError("A selected GPU has less than 4 GiB free or already has three workers")
        with (root / "owner_direct.json").open("x", encoding="utf-8") as marker:
            json.dump(dict(pid=os.getpid(), previous_dispatcher=pid, assignments=assigned,
                           inventory=inventory, retained_rows=["SATIQ"]), marker, indent=2)
            marker.write("\n")
        os.kill(pid, signal.SIGTERM)
    finally:
        if controller_state(pid) == "T":
            os.kill(pid, signal.SIGCONT)
    await_state(pid, {"X", "Z"})
    children = {}
    for arm, gpu in assigned.items():
        command = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.paper_train", "train", str(config),
                   str(data / "manifest" / "source"), str(data / "cache_source"), str(root / arm), "--arm", arm]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        with (root / "logs" / f"{arm}.log").open("xb") as log:
            child = subprocess.Popen(command, cwd=release, env=env, stdout=log, stderr=subprocess.STDOUT)
        children[arm] = child
        print(json.dumps(dict(event="started", arm=arm, gpu=gpu, pid=child.pid, command=command)), flush=True)
    result = {}
    while children:
        for arm, child in list(children.items()):
            value = child.poll()
            if value is not None:
                result[arm] = value
                del children[arm]
                print(json.dumps(dict(event="exited", arm=arm, returncode=value)), flush=True)
        if children:
            time.sleep(20)
    with (root / "pamf_outcomes.json").open("x", encoding="utf-8") as out:
        json.dump(result, out)
    if any(result.values()):
        raise RuntimeError("One or more PAMF rows failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "data", "out"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    a = parser.parse_args()
    if len(set(a.gpus)) != 2:
        parser.error("PAMF rows must use different physical GPUs")
    run(a.release.resolve(), a.data.resolve(), a.out.resolve(), a.gpus)
