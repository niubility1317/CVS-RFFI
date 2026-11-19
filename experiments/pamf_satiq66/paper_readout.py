"""Freeze independent SatIQ predictions before joining held-out truth."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, write_json
from .paper_eval import embeddings, metrics, panel
from .paper_train import ARMS, make_model
from .train import cuda_device, device_array


def class_map(source):
    contract = read_json(source / "contract.json")
    if contract["schema"] != "satiq_oxford_ccs2023_supervised_reindexed_v1":
        raise ValueError("Expected the registered SatIQ supervised split")
    eligible = np.asarray(contract["training_class_counts"]) >= 4
    remap = np.full(len(eligible), -1, dtype=np.int32)
    remap[eligible] = np.arange(eligible.sum())
    return np.asarray(contract["class_pairs"])[eligible], remap, contract["records"]["T"]


@torch.no_grad()
def predict(source, iq_index, iq_cache, checkpoint, out):
    classes, _, n = class_map(source)
    config = read_json(checkpoint.parent / "resolved_config.json")
    arm = config["arm"]
    if arm not in ARMS or config["source"] != str(source) or config["test_access"]:
        raise ValueError("Prediction requires a checkpoint trained only on the registered source")
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not 0 < frozen["updates"] <= config["total_updates"] or not frozen["validation"]:
        raise ValueError("Expected frozen state with completed source validation")
    device = cuda_device()
    model = make_model(config, arm, classes).to(device)
    model.load_state_dict(frozen["model"], strict=True)
    cache = np.load(iq_cache / "T_raw.npy", mmap_mode="r", allow_pickle=False)
    if cache.shape != (n, 2, 11_000) or cache.dtype != np.float32:
        raise ValueError("Held-out IQ cache does not match the class contract")
    with np.load(iq_index / "T_index.npz", allow_pickle=False) as f:
        if set(f.files) != {"archive", "segment", "row", "ids"} or len(f["ids"]) != n:
            raise ValueError("Expected a label-free test IQ index")
        ids = f["ids"]
    feature = embeddings(model, cache, device)
    results = {"ids": ids, "features": feature}
    if arm != "SATIQ":
        weight = F.normalize(model.head.weight, dim=1)
        predictions = []
        for start in range(0, n, 256):
            block = device_array(feature[start:start + 256], device)
            predictions.append(np.from_dlpack(F.linear(block, weight).argmax(1).cpu()))
        results["head_predictions"] = np.concatenate(predictions)
    out.mkdir(exist_ok=False)
    np.savez(out / "predictions.npz", **results)
    write_json(out / "complete.json", {"arm": arm, "updates": frozen["updates"], "checkpoint": str(checkpoint),
                                     "records": n, "evaluation_seed": config["evaluation_seed"]})


def score(source, predictions, truth_path, output):
    classes, remap, n = class_map(source)
    complete = read_json(predictions / "complete.json")
    if complete["records"] != n:
        raise ValueError("Incomplete prediction cohort")
    with (np.load(predictions / "predictions.npz", allow_pickle=False) as pred,
          np.load(truth_path, allow_pickle=False) as target):
        if len(target["ids"]) != n or not np.array_equal(pred["ids"], target["ids"]):
            raise ValueError("Prediction and independently stored truth IDs differ")
        original = target["y"]
        labels = np.full(len(original), -1, dtype=np.int32)
        seen = original >= 0
        labels[seen] = remap[original[seen]]
        eligible = labels >= 0
        data_panel = panel(labels, classes, complete["evaluation_seed"])
        result = metrics(pred["features"], data_panel, torch.device("cpu"))
        if "head_predictions" in pred.files:
            result["head_top1"] = float((pred["head_predictions"][eligible] == labels[eligible]).mean())
        result["eligible_known_records"] = int(eligible.sum())
    write_json(output, dict(complete, **result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    args_by_mode = {
        "predict": ("source", "iq_index", "iq_cache", "checkpoint", "output"),
        "score": ("source", "predictions", "truth", "output"),
    }
    for mode, names in args_by_mode.items():
        action = sub.add_parser(mode)
        for name in names:
            action.add_argument(name, type=Path)
    options = parser.parse_args()
    arguments = [getattr(options, name).resolve() for name in args_by_mode[options.mode]]
    (predict if options.mode == "predict" else score)(*arguments)
