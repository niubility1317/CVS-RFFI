"""Wait for frozen source models, then predict and score the isolated future set."""

import argparse
import json
from pathlib import Path
import os
import subprocess
import sys
import time

def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wait_for_source(run, owner):
    final = run / "source_complete.json"
    while not final.is_file():
        try:
            state = Path(f"/proc/{owner}/stat").read_text().rsplit(")", 1)[1].split()[0]
        except FileNotFoundError:
            state = "Z"
        if state == "Z" and not final.is_file():
            raise RuntimeError(f"Source owner {owner} exited without frozen source models")
        time.sleep(15)
    return read_json(final)


def gpu_ready(gpu):
    result = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    tasks = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return int(result.strip()) >= 4096 and sum(line.strip().isdigit() for line in tasks.splitlines()) < 2


def run_readout(release, run, query, truth, gpus):
    owner = read_json(run / "owner.json")["pid"]
    final = wait_for_source(run, owner)
    if final["rows"] != ["S0", "S1", "U0", "U1"] or final["updates_per_row"] != 5000:
        raise ValueError("The frozen source matrix does not match the four preregistered arms")
    config = release / "configs/pamf_satiq66_20261119.json"
    source = release / "source"
    args = [sys.executable, "-u", "-m"]
    for rows in (("S0", "S1"), ("U0", "U1")):
        while not all(gpu_ready(gpu) for gpu in gpus):
            time.sleep(15)
        workers = []
        for arm, gpu in zip(rows, gpus, strict=True):
            cmd = args + ["experiments.pamf_satiq66.train", "predict", str(config), str(source),
                          str(run / arm), "--query", str(query)]
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
            with (run / "logs" / f"{arm}_query.log").open("xb") as log:
                worker = subprocess.Popen(cmd, cwd=release, env=env, stdout=log, stderr=subprocess.STDOUT)
            print(f"{arm}: future prediction PID={worker.pid}, GPU={gpu}", flush=True)
            workers.append(worker)
        exits = [p.wait() for p in workers]
        if any(exits):
            raise RuntimeError(f"Future prediction failed for {rows}: {exits}")
    with (run / "predictions_complete.json").open("x", encoding="utf-8") as f:
        json.dump({"rows": final["rows"], "readout_pid": os.getpid()}, f)
    scorer = args + ["experiments.pamf_satiq66.score", str(run), str(truth), str(source / "contract.json")]
    with (run / "logs" / "truth_last.log").open("xb") as log:
        subprocess.run(scorer, cwd=release, stdout=log, stderr=subprocess.STDOUT, check=True)
    print("Four frozen future prediction sets scored", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("release", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("query", type=Path)
    p.add_argument("truth", type=Path)
    p.add_argument("--gpus", nargs=2, type=int, required=True)
    a = p.parse_args()
    run_readout(a.release, a.run, a.query, a.truth, a.gpus)
