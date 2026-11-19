"""Run the paired source-only pilots, supervision, then the two SSL arms."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .data import read_json, write_json


def controller(config, source, run, gpus):
    run.mkdir(parents=True, exist_ok=False)
    logs = run / "logs"
    logs.mkdir()
    write_json(run / "owner.json", {"pid": os.getpid(), "gpus": list(gpus)})

    def parallel(stage, rows):
        processes = []
        for gpu, (arm, representation, directory) in zip(gpus, rows, strict=True):
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
            cmd = [sys.executable, "-u", "-m", "experiments.pamf_satiq66.train",
                   stage, str(config), str(source), str(run / directory),
                   "--arm", arm, "--representation", str(representation)]
            with (logs / f"{directory}.log").open("xb") as log:
                child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            processes.append((directory, child))
            print(json.dumps({"action": "started", "stage": stage, "row": directory,
                              "pid": child.pid, "gpu": gpu, "command": cmd}), flush=True)
        results = {name: task.wait() for name, task in processes}
        if any(results.values()):
            raise RuntimeError(f"{stage} subprocess failure: {results}")
        return [read_json(run / name / "complete.json") for name, _ in processes]

    parallel("smoke", [("S1", 0, "smoke_raw"), ("S1", 1, "smoke_narrow")])
    pilots = parallel("pilot", [("S0", 0, "pilot_raw"), ("S0", 1, "pilot_narrow")])
    scores = [p["v"]["macro_f1"] for p in pilots]
    selected = int(scores[1] > scores[0])
    write_json(run / "representation.json", {"selected": selected, "source_v_macro_f1": scores})
    s0, s1 = parallel("train", [("S0", selected, "S0"), ("S1", selected, "S1")])
    budget = read_json(config)["epochs"] * read_json(config)["steps_per_epoch"]
    for arm, result in (("S0", s0), ("S1", s1)):
        c = read_json(run / arm / "resolved_config.json")
        if result["updates"] != budget or result["v"]["records"] != 19800 or c["uses_u"]:
            raise ValueError(f"Incomplete supervised source validation for {arm}")
    traces = [json.loads(line) for line in (run / "S1" / "epochs.jsonl").read_text(encoding="utf-8").splitlines()]
    active = {k: sum(t["train"][k] for t in traces)
              for k in ("packet_anchors", "day_anchors", "prototype_anchors")}
    if min(active.values()) <= 0:
        raise ValueError("Supervised Hybrid mechanism never activated")
    write_json(run / "supervised_validated.json", {"S0": s0["v"], "S1": s1["v"], "S1_anchors": active})
    parallel("train", [("U0", selected, "U0"), ("U1", selected, "U1")])
    write_json(run / "source_complete.json", {"representation": selected, "updates_per_row": budget,
                                               "rows": ["S0", "S1", "U0", "U1"]})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path)
    p.add_argument("source", type=Path)
    p.add_argument("run", type=Path)
    p.add_argument("--gpus", type=int, nargs=2, required=True)
    a = p.parse_args()
    if a.gpus[0] == a.gpus[1]:
        p.error("The two GPU workers must use distinct physical cards")
    controller(a.config.resolve(), a.source.resolve(), a.run.resolve(), a.gpus)
