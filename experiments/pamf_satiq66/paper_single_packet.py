"""Freeze full-candidate single-packet decisions before reading SatIQ test truth."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, write_json
from .paper_calibrated_readout import encode
from .paper_final_readout import FINAL_EPOCH, FINAL_UPDATES
from .paper_readout import class_map, load_source_model
from .paper_sup_prototype_readout import enroll_rows, prototype_weights
from .train import cuda_device, device_array


ARMS = ("SATIQ", "PAMF_SUP")
STEPS_PER_EPOCH = FINAL_UPDATES // FINAL_EPOCH
REDUCED_EPOCH = 100


def source_milestone(run, epoch):
    updates = epoch * STEPS_PER_EPOCH
    checkpoint = run / f"s{updates:07d}.pt"
    if not checkpoint.is_file():
        return None
    with (run / "steps.jsonl").open(encoding="utf-8") as journal:
        for line in journal:
            if not line.endswith("\n"):
                break
            record = json.loads(line)
            if record["updates"] == updates:
                if (record["epoch"] != epoch or record["epoch_step"] != STEPS_PER_EPOCH
                        or not record.get("source_validation")):
                    raise ValueError(f"Unvalidated E{epoch} checkpoint in {run}")
                return checkpoint
            if record["updates"] > updates:
                raise ValueError(f"Missing E{epoch} validation record in {run}")
    return None


def completed_checkpoints(source, cache, training, class_count, epoch=FINAL_EPOCH):
    if epoch not in (REDUCED_EPOCH, FINAL_EPOCH):
        raise ValueError("Expected a registered E100 or E200 budget")
    updates = epoch * STEPS_PER_EPOCH
    checkpoints, seeds = {}, set()
    for arm in ARMS:
        run = training / arm
        contract = read_json(run / "resolved_config.json")
        if (contract["arm"] != arm or contract["trained_classes"] != class_count
                or contract["input_samples"] != 11000 or contract["signed_input"] is not True
                or contract["source_roles"] != ["L", "V"] or contract["test_access"]
                or contract["augmentation"] != "none" or contract["steps_per_epoch"] != STEPS_PER_EPOCH
                or Path(contract["source"]).resolve() != source.resolve()
                or Path(contract["cache"]).resolve() != cache.resolve()):
            raise ValueError(f"{arm} does not match the registered full-header source data")
        if epoch == FINAL_EPOCH:
            completion = read_json(run / "complete.json")
            checkpoint = run / f"s{updates:07d}.pt"
            if (completion != dict(arm=arm, mode="supervised", epochs=epoch, updates=updates)
                    or contract["epochs"] != epoch or contract["total_updates"] != updates
                    or not checkpoint.is_file()):
                raise ValueError(f"{arm} requires its exact completed E{epoch} checkpoint")
        else:
            checkpoint = source_milestone(run, epoch)
            if (checkpoint is None or contract["epochs"] not in (epoch, FINAL_EPOCH)
                    or contract["total_updates"] != contract["epochs"] * STEPS_PER_EPOCH
                    or epoch not in contract["validation_epochs"]):
                raise ValueError(f"{arm} requires its fixed source-validated E{epoch} checkpoint")
        seeds.add(contract["evaluation_seed"])
        checkpoints[arm] = checkpoint
    if len(seeds) != 1:
        raise ValueError("Both methods must share the same enrollment seed")
    return checkpoints, seeds.pop()


def argmax_classes(features, weights, device):
    batches = [F.linear(block.to(device), weights).argmax(1).cpu() for block in features.split(256)]
    return np.from_dlpack(torch.cat(batches))


@torch.no_grad()
def predict(source, source_cache, training, iq_manifest, iq_cache, output, epoch=FINAL_EPOCH):
    classes, remap, expected = class_map(source)
    checkpoints, seed = completed_checkpoints(source, source_cache, training, len(classes), epoch)
    with np.load(source / "L_index.npz", allow_pickle=False) as manifest:
        labels = remap[manifest["y"]]
    training_iq = np.load(source_cache / "L_raw.npy", mmap_mode="r", allow_pickle=False)
    if training_iq.shape != (len(labels), 2, 11000) or training_iq.dtype != np.float32:
        raise ValueError("Source L waveform cache and manifest disagree")
    refs = enroll_rows(labels, len(classes), 16, seed)
    if len(np.unique(labels[refs])) != len(classes):
        raise ValueError("Every training-eligible beam must have a source reference")
    with np.load(iq_manifest / "T_index.npz", allow_pickle=False) as manifest:
        if set(manifest.files) != {"archive", "segment", "row", "ids"}:
            raise ValueError("Expected the label-free T manifest")
        ids = manifest["ids"]
    waveforms = np.load(iq_cache / "T_raw.npy", mmap_mode="r", allow_pickle=False)
    if ids.shape != (expected,) or waveforms.shape != (expected, 2, 11000) or waveforms.dtype != np.float32:
        raise ValueError("Expected every registered physical test packet")

    device = cuda_device()
    output.mkdir(exist_ok=False)
    ref_labels = device_array(labels[refs].astype(np.int64), torch.device("cpu"))
    for arm, checkpoint in checkpoints.items():
        model, _, step = load_source_model(source, checkpoint, classes, device)
        if step != epoch * STEPS_PER_EPOCH:
            raise ValueError(f"Stored checkpoint step does not match its frozen E{epoch} path")
        (ref_features,) = encode(model, arm, training_iq, refs, device)
        weights = prototype_weights(F.normalize(ref_features, dim=1), ref_labels, len(classes)).to(device)
        (query_features,) = encode(model, arm, waveforms, np.arange(expected), device)
        query_features = F.normalize(query_features, dim=1)
        decisions = dict(prototype=argmax_classes(query_features, weights, device))
        if arm == "PAMF_SUP":
            head = F.normalize(model.head.weight, dim=1)
            decisions["native_head"] = argmax_classes(query_features, head, device)
        row = output / arm
        row.mkdir()
        np.savez(row / "predictions.npz", ids=ids, **decisions)
        write_json(row / "complete.json", dict(arm=arm, updates=step, records=expected, classes=len(classes),
                   checkpoint=str(checkpoint.resolve()), source_references=len(refs), evaluation_seed=seed))
        print(f"Frozen {expected} single-packet decisions for {arm}", flush=True)


def score(source, predictions, truth, output, epoch=FINAL_EPOCH):
    if epoch not in (REDUCED_EPOCH, FINAL_EPOCH):
        raise ValueError("Expected a registered E100 or E200 budget")
    classes, remap, expected = class_map(source)
    rows, cohort, enrollment = {}, None, set()
    for arm in ARMS:
        row = predictions / arm
        metadata = read_json(row / "complete.json")
        if (metadata["arm"], metadata["updates"], metadata["records"], metadata["classes"]) != (
                arm, epoch * STEPS_PER_EPOCH, expected, len(classes)):
            raise ValueError(f"Both methods must finish predictions for the fixed E{epoch} test cohort")
        fields = ("prototype", "native_head") if arm == "PAMF_SUP" else ("prototype",)
        with np.load(row / "predictions.npz", allow_pickle=False) as artifact:
            ids = artifact["ids"]
            if ids.shape != (expected,) or (cohort is not None and not np.array_equal(cohort, ids)):
                raise ValueError("The two methods must predict the same packets in the same order")
            cohort = ids
            decisions = {name: artifact[name] for name in fields}
        if any(x.shape != (expected,) or not np.issubdtype(x.dtype, np.integer)
               or (x < 0).any() or (x >= len(classes)).any() for x in decisions.values()):
            raise ValueError("A prediction is missing or does not name an eligible beam")
        enrollment.add((metadata["source_references"], metadata["evaluation_seed"]))
        rows[arm] = (metadata, decisions)
    if len(enrollment) != 1:
        raise ValueError("Methods must share the source gallery budget and seed")

    with np.load(truth, allow_pickle=False) as artifact:
        if not np.array_equal(cohort, artifact["ids"]):
            raise ValueError("Frozen prediction IDs do not match test truth")
        original = artifact["y"]
    if original.shape != (expected,) or (original >= len(remap)).any():
        raise ValueError("Test labels do not match the registered class map")
    labels = np.full(expected, -1, dtype=np.int32)
    present = original >= 0
    labels[present] = remap[original[present]]
    known = int((labels >= 0).sum())
    if known == 0:
        raise ValueError("No test packets have training-eligible beam labels")
    result = {}
    for arm, (metadata, predictions_by_method) in rows.items():
        metrics = {}
        for name, guesses in predictions_by_method.items():
            correct = int((guesses == labels).sum())
            metrics[name] = dict(correct=correct, top1_known=correct / known, top1_all=correct / expected)
        result[arm] = dict(metadata, metrics=metrics)
    write_json(output, dict(records=expected, known_records=known, coverage=known / expected, methods=result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    fields = dict(predict=("source", "source_cache", "training", "iq_manifest", "iq_cache", "output"),
                  score=("source", "predictions", "truth", "output"))
    for mode, keys in fields.items():
        command = modes.add_parser(mode)
        for key in keys:
            command.add_argument(key, type=Path)
        command.add_argument("--epoch", type=int, choices=(REDUCED_EPOCH, FINAL_EPOCH), default=FINAL_EPOCH)
    args = parser.parse_args()
    torch.set_num_threads(4)
    action = predict if args.mode == "predict" else score
    action(*(getattr(args, field).resolve() for field in fields[args.mode]), epoch=args.epoch)
