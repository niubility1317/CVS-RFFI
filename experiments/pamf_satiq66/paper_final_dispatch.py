"""Wait for the registered E200 matrix, then run separate prediction and scoring jobs."""

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from .data import read_json
from .paper_extend import parent_running
from .paper_final_readout import final_checkpoints
from .readout import gpu_ready


def wait_for_gpu(gpus):
    while True:
        for gpu in gpus:
            try:
                if gpu_ready(gpu):
                    return gpu
            except subprocess.CalledProcessError as exc:
                print(f"GPU {gpu} inventory failed: {exc.returncode}", flush=True)
        time.sleep(30)


def run_phase(mode, inputs, log, visible_devices, module="paper_final_readout"):
    command = [sys.executable, "-u", "-m", f"experiments.pamf_satiq66.{module}", mode]
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=visible_devices)
    with log.open("xb") as stream:
        subprocess.run([*command, *map(str, inputs)], env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


def dispatch(data, training, archive_times, output, owner, gpus):
    if read_json(training / "owner.json")["pid"] != owner:
        raise ValueError("The E200 training owner does not match the registered queue")
    output.mkdir(exist_ok=False)
    source, cache = data / "manifest" / "source", data / "cache_source"
    print(f"Waiting for E200 queue PID {owner}", flush=True)
    while parent_running(training, owner):
        time.sleep(30)
    final_checkpoints(source, cache, training)
    print("Verified four frozen E200 source checkpoints", flush=True)

    gpu = wait_for_gpu(gpus)
    iq_index = data / "manifest" / "heldout_iq"
    predictions = output / "frozen_predictions"
    print(f"Starting label-free matrix prediction on physical GPU {gpu}", flush=True)
    run_phase("predict", (source, cache, iq_index, data / "cache_test", training, predictions),
              output / "prediction.log", str(gpu))

    print("Prediction completed. Starting the independent CPU scorer", flush=True)
    truth = data / "manifest" / "scorer_only" / "T_truth.npz"
    run_phase("score", (source, predictions, iq_index, archive_times, truth, output / "scores"),
              output / "scoring.log", "")
    print("Scored the frozen SatIQ E200 matrix", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "training", "archive_times", "output"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--owner", required=True, type=int)
    parser.add_argument("--gpus", required=True, type=int, nargs="+")
    args = parser.parse_args()
    dispatch(args.data.resolve(), args.training.resolve(), args.archive_times.resolve(),
             args.output.resolve(), args.owner, args.gpus)
