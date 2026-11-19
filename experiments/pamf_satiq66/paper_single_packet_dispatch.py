"""Wait for the matched SatIQ and PAMF checkpoints and score them in order."""

import argparse
import json
from pathlib import Path
import time

from .data import read_json
from .paper_final_dispatch import run_phase, wait_for_gpu
from .paper_readout import class_map
from .paper_single_packet import ARMS, completed_checkpoints


def training_controller_active(training, owner):
    try:
        command = (Path(f"/proc/{owner}") / "cmdline").read_bytes().strip(b"\0").split(b"\0")
        return len(command) == 4 and command[1] == b"-c" and json.loads(command[-1]).get("root") == str(training)
    except (FileNotFoundError, ValueError, AttributeError):
        return False


def exit_status(training, arm):
    path = training / f"{arm}.exit"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def dispatch(data, training, output, owner, gpus):
    if read_json(training / "owner.json")["pid"] != owner:
        raise ValueError("The matched training owner differs from the registered process")
    output.mkdir(exist_ok=False)
    print(f"Waiting for matched E200 training PID {owner}", flush=True)
    while True:
        statuses = [exit_status(training, arm) for arm in ARMS]
        if any(state not in ("", "0") for state in statuses):
            raise RuntimeError("A matched training row failed")
        if all(state == "0" for state in statuses):
            break
        if not training_controller_active(training, owner):
            raise RuntimeError("Matched training owner ended before both training rows completed")
        time.sleep(30)

    source, cache = data / "manifest" / "source", data / "cache_source"
    classes, _, _ = class_map(source)
    completed_checkpoints(source, cache, training, len(classes))
    gpu = wait_for_gpu(gpus)
    predictions = output / "frozen_predictions"
    print(f"Starting matched single-packet prediction on GPU {gpu}", flush=True)
    run_phase("predict", (source, cache, training, data / "manifest" / "heldout_iq", data / "cache_test", predictions),
              output / "prediction.log", str(gpu), module="paper_single_packet")
    print("Both prediction cohorts frozen. Starting CPU score", flush=True)
    truth = data / "manifest" / "scorer_only" / "T_truth.npz"
    run_phase("score", (source, predictions, truth, output / "single_packet_scores.json"),
              output / "scoring.log", "", module="paper_single_packet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "training", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--owner", required=True, type=int)
    parser.add_argument("--gpus", required=True, type=int, nargs="+")
    args = parser.parse_args()
    dispatch(args.data.resolve(), args.training.resolve(), args.output.resolve(), args.owner, args.gpus)
