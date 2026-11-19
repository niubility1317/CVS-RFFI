"""Queue a fixed SatIQ budget extension after the original source runs finish."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import read_json, write_json
from .readout import gpu_ready


def parent_running(parent, pid):
    try:
        argv = (Path("/proc") / str(pid) / "cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return os.fsencode(parent) in argv


def queue(config, out):
    final_epoch = config["final_epoch"]
    pending = {row["name"]: row for row in config["rows"]}
    if len(pending) != len(config["rows"]) or any(Path(name).name != name or name == ".." for name in pending):
        raise ValueError("Expected distinct row directory names")
    out.mkdir(exist_ok=False)
    write_json(out / "owner.json", dict(pid=os.getpid(), final_epoch=final_epoch, rows=config["rows"]))
    outcomes, active = {}, {}

    def record(kind, name, **details):
        print(json.dumps(dict(event=kind, row=name, **details)), flush=True)

    for name, row in pending.items():
        record("awaiting_parent", name, pid=row["parent_pid"], source_run=row["parent"])
    while pending or active:
        for name, child in list(active.items()):
            code = child.poll()
            if code is None:
                continue
            result = {"returncode": code, "pid": child.pid}
            if code == 0:
                try:
                    final = read_json(out / name / "complete.json")
                    if final["epochs"] != final_epoch:
                        raise ValueError("Worker ended before the requested epoch")
                    result["complete"] = final
                except (OSError, ValueError, KeyError) as exc:
                    result["error"] = str(exc)
            outcomes[name] = result
            record("exited", name, **result)
            del active[name]
        for name, row in list(pending.items()):
            parent = Path(row["parent"])
            if parent_running(parent, row["parent_pid"]):
                continue
            try:
                resolved = parent / "resolved_config.json"
                completed = read_json(parent / "complete.json")
                checkpoint = parent / f"s{completed['updates']:07d}.pt"
                if completed["epochs"] != config["parent_epoch"] or not checkpoint.is_file():
                    raise ValueError("The predecessor did not complete its registered source budget")
            except (OSError, ValueError, KeyError) as exc:
                outcomes[name] = {"error": str(exc)}
                record("parent_failed", name, **outcomes[name])
                del pending[name]
                continue
            gpu = row["gpu"]
            try:
                available = gpu_ready(gpu)
            except subprocess.CalledProcessError as exc:
                record("gpu_inventory_failed", name, gpu=gpu, returncode=exc.returncode)
                continue
            if not available:
                continue
            command = [
                sys.executable, "-u", "-m", "experiments.pamf_satiq66.paper_train", "train", str(resolved),
                config["source"], config["cache"], str(out / name), "--arm", row["arm"],
                "--resume", str(checkpoint), "--final-epoch", str(final_epoch),
            ]
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
            with (out / f"{name}.log").open("xb") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            active[name] = child
            record("started", name, pid=child.pid, gpu=gpu, command=command)
            del pending[name]
        if pending or active:
            time.sleep(30)
    write_json(out / "matrix_outcomes.json", outcomes)
    if any("complete" not in result for result in outcomes.values()):
        raise RuntimeError("A queued source continuation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    queue(read_json(args.config), args.output.resolve())
