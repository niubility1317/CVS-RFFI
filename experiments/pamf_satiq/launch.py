"""Run source training, frozen prediction and independent truth-last scoring."""

import argparse
from pathlib import Path
import subprocess
import sys


ARMS = ("backbone_ce", "pamf_ce")


def run(command):
    print("RUN " + " ".join(map(str, command)), flush=True)
    subprocess.run([sys.executable, "-B", "-u", "-m", *map(str, command)], check=True)


def launch(root, config, data):
    root.mkdir(exist_ok=False)
    source = data / "source"
    query = data / "target/query"
    for arm in ARMS:
        run(("experiments.pamf_satiq.run", "train", config, source, root / arm, arm))
    for arm in ARMS:
        run(("experiments.pamf_satiq.run", "predict", config, source, root / arm, arm,
             "--query", query))
    run(("experiments.pamf_satiq.score", source / "contract.json", query / "ids.npy",
         data / "target/scorer/truth.npz", root))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    launch(args.root, args.config, args.data)
