"""Replace a failed Hybrid-SSL row and score it with the healthy frozen arms."""

import argparse
import json
import os
from pathlib import Path
from shutil import copy2
import subprocess
import sys
import time

from .data import read_json, write_json
from .readout import gpu_ready, run_readout


def recover(release, old_run, new_run, query, truth, gpus):
    prior = read_json(old_run / "supervised_validated.json")
    representation = read_json(old_run / "representation.json")["selected"]
    previous_owner = read_json(old_run / "owner.json")["pid"]
    if not all(prior[arm]["records"] == 19800 for arm in ("S0", "S1")):
        raise ValueError("Missing frozen supervised validation")
    if (old_run / "U1" / "source_final.pt").exists():
        raise ValueError("Recovery requires a failed U1 without a frozen endpoint")
    new_run.mkdir(parents=True, exist_ok=False)
    logs = new_run / "logs"
    logs.mkdir()
    write_json(new_run / "owner.json", {"pid": os.getpid(), "gpus": gpus})
    write_json(new_run / "replaces.json", {"run": str(old_run), "replaced": "U1",
                                           "unchanged_endpoints": ["S0", "S1", "U0"]})
    config = release / "configs/pamf_satiq66_20261119.json"
    source = release / "source"
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpus[1]))
    for mode, out in (("ssl-smoke", "U1_smoke"), ("train", "U1")):
        while not gpu_ready(gpus[1]):
            time.sleep(15)
        cmd = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.train", mode, str(config),
               str(source), str(new_run / out), "--arm", "U1", "--representation", str(representation)]
        with (logs / f"{out}.log").open("xb") as f:
            child = subprocess.Popen(cmd, cwd=release, env=env, stdout=f, stderr=subprocess.STDOUT)
            print(f"{out}: PID={child.pid} GPU={gpus[1]}", flush=True)
            if child.wait():
                raise RuntimeError(f"{out} failed")
    while not (old_run / "U0" / "complete.json").is_file():
        try:
            state = Path(f"/proc/{previous_owner}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            state = "Z"
        if state == "Z" and not (old_run / "U0" / "complete.json").is_file():
            raise RuntimeError("Original owner exited without a healthy U0 fixed endpoint")
        time.sleep(15)
    for arm in ("S0", "S1", "U0"):
        origin = old_run / arm
        actual = read_json(origin / "resolved_config.json")
        if (actual["representation"] != representation or actual["arm"] != arm or actual["role"] != "formal"
                or read_json(origin / "complete.json")["updates"] != 5000):
            raise ValueError(f"Cannot use incomplete or incompatible fixed endpoint {arm}")
        dest = new_run / arm
        dest.mkdir()
        for name in ("source_final.pt", "resolved_config.json", "complete.json", "epochs.jsonl"):
            copy2(origin / name, dest / name)
    recovered = read_json(new_run / "U1" / "complete.json")
    if recovered["updates"] != 5000:
        raise ValueError("Replacement did not produce the fixed U1 endpoint")
    write_json(new_run / "source_complete.json", {"representation": representation, "updates_per_row": 5000,
                                                   "rows": ["S0", "S1", "U0", "U1"]})
    run_readout(release, new_run, query, truth, gpus)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "old_run", "new_run", "query", "truth"):
        p.add_argument(name, type=Path)
    p.add_argument("--gpus", nargs=2, type=int, required=True)
    a = p.parse_args()
    recover(a.release, a.old_run, a.new_run, a.query, a.truth, a.gpus)
