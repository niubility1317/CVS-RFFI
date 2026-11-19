"""Run the fixed 200-epoch SatIQ comparison with a separate truth-last command."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import read_json, write_json
from .paper_readout import class_map, export_calibration, fit_thresholds, predict, score, stage_dates


FINAL_EPOCH = 200
FINAL_UPDATES = 5_645_000
METHODS = {
    "SATIQ": ("SATIQ", None),
    "NATIVE160": ("PAMF_SUP", "native160"),
    "FUSED256": ("PAMF_SUP", "fused256"),
    "PROTO320": ("PAMF_SUP", "fused320"),
}


def stage_archive_times(source, scan, out):
    names = read_json(source / "contract.json")["archive_map"]
    out.mkdir(exist_ok=False)
    for name in names:
        segments = read_json(scan / f"{name}.json")["segments"]
        with np.load(scan / f"{name}.npz", allow_pickle=False) as archive:
            times = archive["timestamp_global"].astype("datetime64[ns]")
        if (times.ndim != 1 or len(segments) != (len(times) + 9999) // 10000
                or len(set(segments)) != len(segments)):
            raise ValueError(f"Inconsistent source scan segments for {name}")
        write_json(out / f"{name}.json", {"segments": segments})
        np.savez(out / f"{name}.npz", timestamp_global=times)


def final_checkpoints(source, source_cache, training):
    outcomes = read_json(training / "matrix_outcomes.json")
    if set(outcomes) != set(METHODS):
        raise ValueError("Expected all four registered SatIQ continuations")
    checkpoints = {}
    for name, (arm, expansion) in METHODS.items():
        outcome = outcomes[name]
        if outcome.get("returncode") != 0:
            raise ValueError(f"{name} did not complete its registered source budget")
        run = training / name
        complete = read_json(run / "complete.json")
        config = read_json(run / "resolved_config.json")
        checkpoint = run / f"s{FINAL_UPDATES:07d}.pt"
        if (outcome.get("complete") != complete
                or complete != dict(arm=arm, mode="supervised", epochs=FINAL_EPOCH, updates=FINAL_UPDATES)
                or config["arm"] != arm or config.get("native_expansion") != expansion
                or config["epochs"] != FINAL_EPOCH or config["total_updates"] != FINAL_UPDATES
                or Path(config["source"]).resolve() != source.resolve()
                or Path(config["cache"]).resolve() != source_cache.resolve()
                or config["source_roles"] != ["L", "V"] or config["test_access"]
                or config["augmentation"] != "none"
                or (name == "PROTO320" and config.get("expansion_prototype_weight") != 2.0)
                or not checkpoint.is_file()):
            raise ValueError(f"{name} is missing its registered completed source checkpoint")
        checkpoints[name] = checkpoint
    return checkpoints


def predict_matrix(source, source_cache, iq_index, iq_cache, training, out):
    checkpoints = final_checkpoints(source, source_cache, training)
    out.mkdir(exist_ok=False)
    for name, checkpoint in checkpoints.items():
        row = out / name
        row.mkdir()
        calibration = row / "whitening.pt"
        export_calibration(source, source_cache, checkpoint, calibration)
        fit_thresholds(source, source_cache, checkpoint, row / "thresholds.json", calibration)
        print(f"Fitted source readout for {name}", flush=True)
    for name, checkpoint in checkpoints.items():
        row = out / name
        predict(source, iq_index, iq_cache, checkpoint, row / "predictions", row / "whitening.pt")
        print(f"Frozen label-free predictions for {name}", flush=True)


def score_matrix(source, predictions, iq_index, scan, truth, out):
    _, _, records = class_map(source)
    evaluation_seeds = set()
    for name, (arm, _) in METHODS.items():
        row = predictions / name
        cohort = row / "predictions"
        complete = read_json(cohort / "complete.json")
        if (complete["arm"] != arm or complete["updates"] != FINAL_UPDATES
                or complete["records"] != records or not (cohort / "predictions.npz").is_file()
                or not (row / "thresholds.json").is_file()):
            raise ValueError(f"All four frozen prediction sets are required; incomplete {name}")
        evaluation_seeds.add(complete["evaluation_seed"])
    if len(evaluation_seeds) != 1:
        raise ValueError("All rows must share the same registered test panel seed")
    out.mkdir(exist_ok=False)
    dates = out / "test_dates.npz"
    stage_dates(iq_index, scan, dates)
    results = {}
    for name in METHODS:
        row = predictions / name
        scores = out / f"{name}.json"
        score(source, row / "predictions", truth, scores, chronology=dates, thresholds=row / "thresholds.json")
        results[name] = read_json(scores)
        print(f"Scored independent predictions for {name}", flush=True)
    write_json(out / "matrix_scores.json", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="mode", required=True)
    fields = {
        "predict": ("source", "source_cache", "iq_index", "iq_cache", "training", "output"),
        "score": ("source", "predictions", "iq_index", "scan", "truth", "output"),
        "stage_scan": ("source", "scan", "output"),
    }
    for mode, names in fields.items():
        sub = actions.add_parser(mode)
        for name in names:
            sub.add_argument(name, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    action = {"predict": predict_matrix, "score": score_matrix, "stage_scan": stage_archive_times}[args.mode]
    action(*(getattr(args, key).resolve() for key in fields[args.mode]))
