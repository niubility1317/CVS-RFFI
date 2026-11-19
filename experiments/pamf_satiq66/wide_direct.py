"""Transfer an existing Beam66 queue and dispatch its remaining supervised rows."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def gpu_inventory():
    def query(kind, fields):
        output = subprocess.check_output(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"], text=True
        )
        return [tuple(part.strip() for part in line.split(",")) for line in output.splitlines()]

    active = query("compute-apps", "gpu_uuid,pid")
    return {
        int(index): (int(free), sum(gpu == uuid for gpu, _ in active))
        for index, uuid, free in query("gpu", "index,uuid,memory.free")
    }


def controller_state(pid):
    status = Path(f"/proc/{pid}/status")
    try:
        return next(line.split()[1] for line in status.read_text().splitlines() if line.startswith("State:"))
    except FileNotFoundError:
        return "X"


def await_state(pid, states):
    for _ in range(100):
        if controller_state(pid) in states:
            return
        time.sleep(0.05)
    raise RuntimeError(f"Dispatcher PID {pid} did not enter {sorted(states)}")


def take_over(release, root, assignments):
    config = release / "configs/pamf_satiq_beam66_wide_sup_20261119.json"
    source = release / "source"
    cmd = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.wide_run"]
    original_pid = json.loads((root / "owner.json").read_text())["pid"]
    proc = Path(f"/proc/{original_pid}")
    expected = cmd + ["launch", str(config), str(source), str(root)]
    actual = (proc / "cmdline").read_bytes().rstrip(b"\0").decode().split("\0")
    if (proc.stat().st_uid != os.getuid() or actual[:len(expected)] != expected
            or (proc / "cwd").readlink() != release):
        raise RuntimeError("Original matrix dispatcher identity did not match")

    arm_logs = {arm: root / "logs" / f"{arm}.log" for arm in assignments}
    owner = root / "owner_direct.json"
    os.kill(original_pid, signal.SIGSTOP)
    try:
        await_state(original_pid, {"T"})
        for arm, log in arm_logs.items():
            if log.exists() or (root / arm).exists():
                raise RuntimeError(f"{arm} already has a worker or output")
        inventory = gpu_inventory()
        if any(inventory[gpu][0] < 4096 or inventory[gpu][1] > 2 for gpu in assignments.values()):
            raise RuntimeError("Direct dispatch requires at least 4 GiB free and at most two incumbents per GPU")
        with owner.open("x", encoding="utf-8") as marker:
            json.dump(dict(pid=os.getpid(), previous_dispatcher=original_pid, assignments=assignments,
                           inventory=inventory, retained_rows=["W0"]), marker, indent=2)
            marker.write("\n")
        os.kill(original_pid, signal.SIGTERM)
    finally:
        if controller_state(original_pid) == "T":
            os.kill(original_pid, signal.SIGCONT)
    await_state(original_pid, {"X", "Z"})

    workers = {}
    for arm, gpu in assignments.items():
        worker = cmd + ["train", str(config), str(source), str(root / arm), "--arm", arm]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        with arm_logs[arm].open("xb") as log:
            child = subprocess.Popen(worker, env=env, cwd=release, stdout=log, stderr=subprocess.STDOUT)
        workers[arm] = child
        print(json.dumps(dict(event="started", row=arm, pid=child.pid, gpu=gpu, command=worker)), flush=True)

    while workers:
        for arm, child in list(workers.items()):
            result = child.poll()
            if result is not None:
                complete = root / arm / "complete.json"
                reached = result == 0 and complete.is_file() and json.loads(complete.read_text()).get("target_reached")
                print(json.dumps(dict(event="exited", row=arm, returncode=result, target_reached=bool(reached))),
                      flush=True)
                del workers[arm]
        if workers:
            time.sleep(20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("release", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpus", nargs=3, type=int, required=True, metavar=("B0", "W1", "L0"))
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error("B0, W1 and L0 must use different physical GPUs")
    take_over(args.release.resolve(), args.output.resolve(), dict(zip(("B0", "W1", "L0"), args.gpus)))
