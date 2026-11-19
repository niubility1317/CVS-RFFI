"""Freeze independent SatIQ predictions before joining held-out truth."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import read_json, write_json
from .paper_cache import archive_requests, load_group
from .paper_calibrated_readout import apply_calibration, encode, fit_calibration
from .paper_eval import embeddings, metrics, panel
from .paper_signal_readout import source_date_cutoff, source_panels
from .paper_train import ARMS, make_model, source_inputs
from .train import cuda_device, device_array


def class_map(source):
    contract = read_json(source / "contract.json")
    if contract["schema"] != "satiq_oxford_ccs2023_supervised_reindexed_v1":
        raise ValueError("Expected the registered SatIQ supervised split")
    eligible = np.asarray(contract["training_class_counts"]) >= 4
    remap = np.full(len(eligible), -1, dtype=np.int32)
    remap[eligible] = np.arange(eligible.sum())
    return np.asarray(contract["class_pairs"])[eligible], remap, contract["records"]["T"]


def load_source_model(source, checkpoint, classes, device):
    config = read_json(checkpoint.parent / "resolved_config.json")
    arm = config["arm"]
    if arm not in ARMS or Path(config["source"]).resolve() != source.resolve() or config["test_access"]:
        raise ValueError("Prediction requires a checkpoint trained only on the registered source")
    frozen = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not 0 < frozen["updates"] <= config["total_updates"] or not frozen["validation"]:
        raise ValueError("Expected frozen state with completed source validation")
    model = make_model(config, arm, classes).to(device).eval()
    model.load_state_dict(frozen["model"], strict=True)
    return model, config, frozen["updates"]


@torch.no_grad()
def export_calibration(source, source_cache, checkpoint, out, fit_count=8192, seed=2026111907):
    classes, _, _ = class_map(source)
    device = cuda_device()
    model, config, updates = load_source_model(source, checkpoint, classes, device)
    if Path(config["cache"]).resolve() != source_cache.resolve():
        raise ValueError("Calibration cache must match the checkpoint source")
    raw = np.load(source_cache / "L_raw.npy", mmap_mode="r", allow_pickle=False)
    expected = read_json(source / "contract.json")["records"]["L"]
    if raw.shape != (expected, 2, 11_000) or raw.dtype != np.float32:
        raise ValueError("Calibration requires the registered source L cache")
    ids = np.sort(np.random.default_rng(seed).choice(len(raw), fit_count, replace=False))
    features = encode(model, config["arm"], raw, ids, device)
    artifact = dict(schema="satiq_partial_whiten_v1", source=str(source.resolve()),
                    checkpoint=str(checkpoint.resolve()), updates=updates, arm=config["arm"],
                    fit_count=fit_count, fit_seed=seed, stats=[fit_calibration(v) for v in features])
    with out.open("xb") as f:
        torch.save(artifact, f)


def load_calibration(source, checkpoint, updates, arm, calibration):
    if calibration is None:
        return None
    artifact = torch.load(calibration, map_location="cpu", weights_only=True)
    if (artifact["schema"], artifact["source"], artifact["checkpoint"], artifact["updates"], artifact["arm"]) != (
            "satiq_partial_whiten_v1", str(source.resolve()), str(checkpoint.resolve()), updates, arm):
        raise ValueError("Calibration must belong to the exact frozen source checkpoint")
    return artifact["stats"]


def readout_features(model, arm, iq, device, stats):
    if stats is None:
        return {"features": embeddings(model, iq, device)}
    paths = encode(model, arm, iq, np.arange(len(iq)), device)
    calibrated = [F.normalize(apply_calibration(x, s), dim=1) for x, s in zip(paths, stats, strict=True)]
    return {"features": np.from_dlpack(F.normalize(torch.cat(paths, dim=1), dim=1)),
            "calibrated_features": np.from_dlpack(F.normalize(torch.cat(calibrated, dim=1), dim=1))}


@torch.no_grad()
def predict(source, iq_index, iq_cache, checkpoint, out, calibration=None):
    classes, _, n = class_map(source)
    device = cuda_device()
    model, config, updates = load_source_model(source, checkpoint, classes, device)
    arm = config["arm"]
    stats = load_calibration(source, checkpoint, updates, arm, calibration)
    cache = np.load(iq_cache / "T_raw.npy", mmap_mode="r", allow_pickle=False)
    if cache.shape != (n, 2, 11_000) or cache.dtype != np.float32:
        raise ValueError("Held-out IQ cache does not match the class contract")
    with np.load(iq_index / "T_index.npz", allow_pickle=False) as f:
        if set(f.files) != {"archive", "segment", "row", "ids"} or len(f["ids"]) != n:
            raise ValueError("Expected a label-free test IQ index")
        ids = f["ids"]
    results = {"ids": ids, **readout_features(model, arm, cache, device, stats)}
    feature = results["features"]
    if arm != "SATIQ":
        weight = F.normalize(model.head.weight, dim=1)
        predictions = []
        for start in range(0, n, 256):
            block = device_array(feature[start:start + 256], device)
            predictions.append(np.from_dlpack(F.linear(block, weight).argmax(1).cpu()))
        results["head_predictions"] = np.concatenate(predictions)
    out.mkdir(exist_ok=False)
    np.savez(out / "predictions.npz", **results)
    complete = {"arm": arm, "updates": updates, "checkpoint": str(checkpoint.resolve()),
                "records": n, "evaluation_seed": config["evaluation_seed"]}
    if calibration is not None:
        complete["calibration"] = str(calibration.resolve())
    write_json(out / "complete.json", complete)


@torch.no_grad()
def fit_thresholds(source, source_cache, checkpoint, output, calibration=None):
    classes, _, roles = source_inputs(source, source_cache)
    raw, labels, _ = roles["V"]
    device = cuda_device()
    model, config, updates = load_source_model(source, checkpoint, classes, device)
    if Path(config["cache"]).resolve() != source_cache.resolve():
        raise ValueError("Threshold cache must match the checkpoint source")
    arm, seed = config["arm"], config["evaluation_seed"]
    stats = load_calibration(source, checkpoint, updates, arm, calibration)
    panels, cutoff = source_panels(source, labels, classes, seed)
    views = readout_features(model, arm, raw, device, stats)
    thresholds = {}
    for feature_name, feature in views.items():
        thresholds[feature_name] = {}
        for panel_name, data_panel in panels.items():
            scores = metrics(feature, data_panel, torch.device("cpu"), include_eer_threshold=True)
            thresholds[feature_name][panel_name] = {
                k: {group: score["eer_threshold"] for group, score in scores[k].items()
                    if group != "top1" and score is not None}
                for k in ("K1", "K16")
            }
    write_json(output, dict(schema="satiq_source_verification_thresholds_v1", source=str(source.resolve()),
                            checkpoint=str(checkpoint.resolve()), arm=arm, updates=updates, evaluation_seed=seed,
                            calibration=str(calibration.resolve()) if calibration is not None else None,
                            cutoff_utc=cutoff, thresholds=thresholds))


def score_panel(pred, data_panel, thresholds, name):
    device = torch.device("cpu")
    raw = thresholds["features"][name] if thresholds is not None else None
    result = metrics(pred["features"], data_panel, device, thresholds=raw)
    if "calibrated_features" in pred.files:
        white = thresholds["calibrated_features"][name] if thresholds is not None else None
        result["partial_whiten"] = metrics(pred["calibrated_features"], data_panel, device, thresholds=white)
    return result


def stage_dates(heldout_manifest, scan, output):
    """Recover dates by archive coordinate from the existing metadata scan."""
    names, indices, _, _ = load_group("heldout", heldout_manifest)
    ids = indices["T"]["ids"]
    dates = np.empty(len(ids), dtype=np.int64)
    for archive_number, segments in archive_requests(indices).items():
        name = names[archive_number]
        metadata = read_json(scan / f"{name}.json")
        order = {int(segment): i for i, segment in enumerate(metadata["segments"])}
        with np.load(scan / f"{name}.npz", allow_pickle=False) as archive:
            times = archive["timestamp_global"].astype("datetime64[ns]")
        if times.ndim != 1 or len(order) != (len(times) + 9999) // 10000:
            raise ValueError(f"Timestamp scan has inconsistent segments for {name}")
        for segment, roles in segments.items():
            if segment not in order:
                raise ValueError(f"Timestamp scan is missing {name}/{segment}")
            rows, positions = roles["T"]
            start = order[segment] * 10000
            selected = times[start:start + 10000][rows]
            if np.isnat(selected).any():
                raise ValueError(f"Missing timestamp in {name}/{segment}")
            dates[positions] = selected.astype(np.int64)
    with output.open("xb") as file:
        np.savez(file, ids=ids, timestamp_ns=dates)


def score(source, predictions, truth_path, output, chronology=None, thresholds=None):
    classes, remap, n = class_map(source)
    complete = read_json(predictions / "complete.json")
    if complete["records"] != n:
        raise ValueError("Incomplete prediction cohort")
    threshold_values = None
    if thresholds is not None:
        cutoff_artifact = read_json(thresholds)
        if (cutoff_artifact["schema"] != "satiq_source_verification_thresholds_v1"
                or cutoff_artifact["source"] != str(source.resolve())
                or any(cutoff_artifact.get(key) != complete.get(key)
                       for key in ("checkpoint", "arm", "updates", "evaluation_seed", "calibration"))):
            raise ValueError("Thresholds must belong to the frozen checkpoint and its readout")
        threshold_values = cutoff_artifact["thresholds"]
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
        result = score_panel(pred, data_panel, threshold_values, "paper")
        if "head_predictions" in pred.files:
            result["head_top1"] = float((pred["head_predictions"][eligible] == labels[eligible]).mean())
        result["eligible_known_records"] = int(eligible.sum())
        if chronology is not None:
            with np.load(chronology, allow_pickle=False) as dates:
                if set(dates.files) != {"ids", "timestamp_ns"} or not np.array_equal(dates["ids"], pred["ids"]):
                    raise ValueError("Held-out timestamps must match all prediction IDs in order")
                times = dates["timestamp_ns"]
            if times.shape != (n,) or times.dtype != np.int64 or np.isnat(times.view("datetime64[ns]")).any():
                raise ValueError("Invalid held-out timestamps")
            cutoff = source_date_cutoff(source)
            cutoff_utc = str(np.datetime64(int(cutoff), "ns"))
            if thresholds is not None and cutoff_artifact["cutoff_utc"] != cutoff_utc:
                raise ValueError("Thresholds use a different source date cutoff")
            dated_panel = panel(labels, classes, complete["evaluation_seed"], reference_period=times < cutoff)
            dated = score_panel(pred, dated_panel, threshold_values, "chronological")
            result["chronological"] = dict(cutoff_utc=cutoff_utc, **dated)
    if thresholds is not None:
        complete["source_thresholds"] = str(thresholds)
    write_json(output, dict(complete, **result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    args_by_mode = {
        "predict": ("source", "iq_index", "iq_cache", "checkpoint", "output"),
        "score": ("source", "predictions", "truth", "output"),
        "fit_calibration": ("source", "source_cache", "checkpoint", "output"),
        "fit_thresholds": ("source", "source_cache", "checkpoint", "output"),
        "stage_dates": ("heldout_manifest", "scan", "output"),
    }
    for mode, names in args_by_mode.items():
        action = sub.add_parser(mode)
        for name in names:
            action.add_argument(name, type=Path)
        if mode in ("predict", "fit_thresholds"):
            action.add_argument("--calibration", type=Path)
        if mode == "score":
            action.add_argument("--chronology", type=Path)
            action.add_argument("--thresholds", type=Path)
        if mode == "fit_calibration":
            action.add_argument("--fit-count", type=int, default=8192)
            action.add_argument("--seed", type=int, default=2026111907)
    options = parser.parse_args()
    torch.set_num_threads(4)
    arguments = [getattr(options, name).resolve() for name in args_by_mode[options.mode]]
    if options.mode == "fit_calibration":
        export_calibration(*arguments, fit_count=options.fit_count, seed=options.seed)
    elif options.mode == "fit_thresholds":
        fit_thresholds(*arguments, calibration=options.calibration)
    elif options.mode == "predict":
        predict(*arguments, calibration=options.calibration)
    elif options.mode == "stage_dates":
        stage_dates(*arguments)
    else:
        score(*arguments, chronology=options.chronology, thresholds=options.thresholds)
