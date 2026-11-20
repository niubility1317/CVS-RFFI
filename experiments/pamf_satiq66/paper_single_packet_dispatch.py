"""Wait for the matched SatIQ and PAMF checkpoints and score them in order."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from .data import read_json
from .paper_final_dispatch import run_phase, wait_for_gpu
from .paper_readout import class_map
from .paper_single_packet import (ARMS, FINAL_EPOCH, REDUCED_EPOCH, STEPS_PER_EPOCH,
                                  completed_checkpoints, source_milestone)


def training_controller_active(training, owner):
    try:
        command = (Path(f"/proc/{owner}") / "cmdline").read_bytes().strip(b"\0").split(b"\0")
        return len(command) == 4 and command[1] == b"-c" and json.loads(command[-1]).get("root") == str(training)
    except (FileNotFoundError, ValueError, AttributeError):
        return False


def exit_status(training, arm):
    path = training / f"{arm}.exit"
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def stop_registered_trainer(training, owner, arm, command):
    pid = int((training / f"{arm}.pid").read_text(encoding="utf-8"))
    sent = subprocess.check_output(["/usr/bin/python3", "-m", "experiments.pamf_satiq66.paper_budget_signal",
                                    str(pid), str(owner), command], text=True)
    return json.loads(sent)


def wait_for_reduced_budget(training, output, owner, roles):
    signalled, frozen = set(), {}
    while len(frozen) != len(ARMS):
        for arm in ARMS:
            if arm in frozen:
                continue
            run = training / arm
            checkpoint = source_milestone(run, REDUCED_EPOCH)
            status = exit_status(training, arm)
            if status and (checkpoint is None or status not in ("0", "-15")
                           or status == "-15" and arm not in signalled):
                raise RuntimeError(f"{arm} exited before completing the registered E100 cap")
            if checkpoint is not None and status:
                frozen[arm] = {"checkpoint": str(checkpoint), "updates": REDUCED_EPOCH * STEPS_PER_EPOCH,
                               "trainer_exit": int(status)}
            elif checkpoint is not None and arm not in signalled:
                if read_json(run / "resolved_config.json")["epochs"] > REDUCED_EPOCH:
                    if stop_registered_trainer(training, owner, arm, roles[arm]["cmd"]):
                        signalled.add(arm)
                        print(f"Validated E100 for {arm}; requested stop of the registered trainer", flush=True)
        if len(frozen) != len(ARMS):
            if not training_controller_active(training, owner):
                if all(exit_status(training, arm) for arm in ARMS):
                    continue
                raise RuntimeError("Matched training owner ended before both budget results were recorded")
            time.sleep(1)
    with (output / "budget_stops.json").open("x", encoding="utf-8") as log:
        json.dump(frozen, log, indent=2)


def dispatch(data, training, output, owner, gpus, epoch=FINAL_EPOCH):
    metadata = read_json(training / "owner.json")
    if metadata["pid"] != owner:
        raise ValueError("The matched training owner differs from the registered process")
    output.mkdir(exist_ok=False)
    print(f"Waiting for matched E{epoch} training PID {owner}", flush=True)
    if epoch == REDUCED_EPOCH:
        wait_for_reduced_budget(training, output, owner, {row["name"]: row for row in metadata["roles"]})
    else:
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
    completed_checkpoints(source, cache, training, len(classes), epoch)
    gpu = wait_for_gpu(gpus)
    predictions = output / "frozen_predictions"
    print(f"Starting matched single-packet prediction on GPU {gpu}", flush=True)
    run_phase("predict", (source, cache, training, data / "manifest" / "heldout_iq", data / "cache_test",
                          predictions, "--epoch", epoch),
              output / "prediction.log", str(gpu), module="paper_single_packet")
    print("Both prediction cohorts frozen. Starting CPU score", flush=True)
    truth = data / "manifest" / "scorer_only" / "T_truth.npz"
    run_phase("score", (source, predictions, truth, output / "single_packet_scores.json", "--epoch", epoch),
              output / "scoring.log", "", module="paper_single_packet")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "training", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--owner", required=True, type=int)
    parser.add_argument("--gpus", required=True, type=int, nargs="+")
    parser.add_argument("--epoch", type=int, choices=(REDUCED_EPOCH, FINAL_EPOCH), default=FINAL_EPOCH)
    args = parser.parse_args()
    dispatch(args.data.resolve(), args.training.resolve(), args.output.resolve(), args.owner, args.gpus, args.epoch)
