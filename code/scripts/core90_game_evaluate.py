"""Frozen SOURCE V evaluation: write all four scene predictions before scoring.

This module provides no target-data loader or target-confirmation switch.
Source V is read-only. Results are source development evidence, not a claim of
unseen-target generalization or a substitute for the Phase2 scoring protocol.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset, DataLoader

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from cvsrffi.game_tracking.config import SCENES, validate
from cvsrffi.game_tracking.state import RNGState, clone_buffers, restore_buffers

EVAL_SCENES = ("clean",) + SCENES


class PredictionInputs(Dataset):
    """Do not propagate truth or TX-bearing capture metadata to prediction."""
    def __init__(self, validation):
        self.validation = validation

    def __len__(self):
        return len(self.validation)

    def __getitem__(self, index):
        x, _, _, meta = self.validation[index]
        if meta.get("role") != "val":
            raise ValueError("Only read-only source V may enter source evaluation")
        return x, {key: meta[key] for key in ("sample_id", "rx_i", "day_i")}


def source_truth_records(source):
    """Scoring-side iterator, invoked only after every prediction file closes."""
    for index in range(len(source.val)):
        _, truth, _, meta = source.val[index]
        if meta.get("role") != "val":
            raise ValueError("Source scoring requires source V")
        yield {"sample_id": str(meta["sample_id"]), "truth": int(truth),
               "rx_i": int(meta["rx_i"]), "day_i": int(meta["day_i"])}


def _json_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _classification_metrics(pairs):
    """Sparse confusion counts; macro metrics explicitly use observed classes."""
    total = sum(pairs.values())
    actual, predicted, correct = defaultdict(int), defaultdict(int), defaultdict(int)
    for (truth, prediction), count in pairs.items():
        actual[truth] += count
        predicted[prediction] += count
        if truth == prediction:
            correct[truth] += count
    classes = sorted(actual)
    recall = {str(label): correct[label] / actual[label] for label in classes}
    f1 = [2 * correct[label] / (actual[label] + predicted[label]) for label in classes]
    return {"count": total, "accuracy": sum(correct.values()) / total if total else None,
            "macro_recall": sum(recall.values()) / len(classes) if classes else None,
            "macro_f1": sum(f1) / len(classes) if classes else None,
            "worst_tx_accuracy": min(recall.values()) if classes else None,
            "observed_classes": classes, "per_tx_accuracy": recall,
            "per_tx_count": {str(k): actual[k] for k in classes}}


def score_source_predictions(predictions_path, truth_records, *, expected_samples, num_classes,
                             scenes=EVAL_SCENES):
    """Independent artifact scorer: no model, optimizer, or data adaptation."""
    if expected_samples <= 0 or num_classes <= 1:
        raise ValueError("Scoring needs a nonempty multi-class registered problem")
    if tuple(scenes) != EVAL_SCENES:
        raise ValueError("Source scoring requires clean and all three prescribed LEO scenes")
    truth = {}
    records = _json_rows(truth_records) if isinstance(truth_records, (str, Path)) else truth_records
    for record in records:
        key = str(record["sample_id"])
        if key in truth or not 0 <= int(record["truth"]) < num_classes:
            raise ValueError("Duplicate source truth ID or unregistered truth class")
        truth[key] = record
    if len(truth) != expected_samples:
        raise ValueError("Source truth artifact count is incomplete")
    seen = {scene: set() for scene in scenes}
    aggregate = {scene: defaultdict(int) for scene in scenes}
    by_rx = {scene: defaultdict(lambda: defaultdict(int)) for scene in scenes}
    by_day = {scene: defaultdict(lambda: defaultdict(int)) for scene in scenes}
    for record in _json_rows(predictions_path):
        key, scene = str(record["sample_id"]), record["scene"]
        if "truth" in record or "y" in record or "capture_group" in record:
            raise ValueError("Prediction artifact contains scoring-only metadata")
        if scene not in seen or key not in truth or key in seen[scene]:
            raise ValueError("Unexpected scene/ID or duplicate prediction")
        target = truth[key]
        if any(int(record[k]) != int(target[k]) for k in ("rx_i", "day_i")):
            raise ValueError("Prediction/source metadata mismatch")
        prediction = int(record["prediction"])
        confidence = float(record["confidence"])
        if not 0 <= prediction < num_classes or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Unregistered prediction or invalid confidence")
        pair = (int(target["truth"]), prediction)
        seen[scene].add(key)
        aggregate[scene][pair] += 1
        by_rx[scene][int(target["rx_i"])][pair] += 1
        by_day[scene][int(target["day_i"])][pair] += 1
    if any(len(ids) != expected_samples for ids in seen.values()):
        raise ValueError("Not all source predictions are complete in all required scenes")
    results = {}
    for scene in scenes:
        result = _classification_metrics(aggregate[scene])
        result["per_rx"] = {str(rx): _classification_metrics(counts) for rx, counts in sorted(by_rx[scene].items())}
        result["per_day"] = {str(day): _classification_metrics(counts) for day, counts in sorted(by_day[scene].items())}
        result["macro_rx_accuracy"] = sum(row["accuracy"] for row in result["per_rx"].values()) / len(result["per_rx"])
        result["worst_rx_accuracy"] = min(row["accuracy"] for row in result["per_rx"].values())
        result["missing_registered_classes"] = sorted(set(range(num_classes)) - set(result["observed_classes"]))
        results[scene] = result
    return {"schema": "core90_game_source_scores_v1", "source_only": True,
            "complete": True, "samples_per_scene": expected_samples,
            "prediction_count": expected_samples * len(scenes), "num_classes": num_classes,
            "metric_units": "fractions_0_to_1", "macro_scope": "observed_truth_classes",
            "scenes": results,
            "leo_mean_accuracy": sum(results[s]["accuracy"] for s in SCENES) / len(SCENES),
            "leo_worst_accuracy": min(results[s]["accuracy"] for s in SCENES)}


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _rss_bytes():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def evaluate_source(model, source, args, output_dir, *, apply_scene=None):
    """Freeze four complete top-1 prediction sets, then score source V once."""
    if len(source.val) < 1:
        raise ValueError("Empty source validation data")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions_path, truth_path = output / "source_predictions.jsonl", output / "source_truth.jsonl"
    paths = (predictions_path, truth_path, output / "source_prediction_manifest.json", output / "source_scores.json")
    if any(path.exists() for path in paths):
        raise FileExistsError("Evaluation artifacts already exist; choose a fresh output directory")
    if apply_scene is None:
        from cvsrffi.eval import apply_sat_channel_for_scenario
        apply_scene = lambda x, scene, generator: apply_sat_channel_for_scenario(x, scene, args, gen=generator)[0]
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started, rng, buffers = time.perf_counter(), RNGState.capture(), clone_buffers(model)
    modes = [(module, module.training) for module in model.modules()]
    rss_max = _rss_bytes()
    counts = {}
    try:
        model.eval()
        with predictions_path.open("x", encoding="utf-8", newline="\n") as stream, torch.inference_mode():
            for index, scene in enumerate(EVAL_SCENES):
                generator = torch.Generator(device=device).manual_seed(int(args.game_split_seed) + 100003 * index)
                loader = DataLoader(PredictionInputs(source.val), batch_size=int(args.eval_batch_size),
                                    shuffle=False, drop_last=False, num_workers=0,
                                    generator=torch.Generator().manual_seed(int(args.game_split_seed)))
                counts[scene] = 0
                for x, meta in loader:
                    x = x.to(device)
                    if scene != "clean":
                        x = apply_scene(x, scene, generator)
                    output_model = model(x, return_aux=True)
                    logits = output_model["tx_logits"] if isinstance(output_model, dict) else output_model
                    if logits.shape != (len(x), int(args.num_classes)) or not torch.isfinite(logits).all():
                        raise ValueError("Prediction logits must cover all registered classes and be finite")
                    confidence, prediction = logits.float().softmax(-1).max(-1)
                    for offset in range(len(x)):
                        record = {"sample_id": str(meta["sample_id"][offset]), "scene": scene,
                                  "rx_i": int(meta["rx_i"][offset]), "day_i": int(meta["day_i"][offset]),
                                  "prediction": int(prediction[offset]), "confidence": float(confidence[offset])}
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    counts[scene] += len(x)
                    rss = _rss_bytes()
                    if rss is not None:
                        rss_max = max(rss_max or 0, rss)
        # All predictions are now closed before any truth iterator is consumed.
        if any(count != len(source.val) for count in counts.values()):
            raise ValueError("Source prediction pass was incomplete")
        if any(not torch.equal(value, dict(model.named_buffers())[key]) for key, value in buffers.items()):
            raise RuntimeError("Read-only source evaluation changed model buffers")
        _write_json(output / "source_prediction_manifest.json",
                    {"schema": "core90_game_source_predictions_v1", "complete": True,
                     "source_only": True, "counts": counts, "num_classes": int(args.num_classes),
                     "prediction_scope": "all_registered_classes_argmax", "truth_used_for_prediction": False,
                     "prediction_file_closed_before_scoring": True,
                     "augmentation_seed": int(args.game_split_seed), "synthetic": bool(args.game_synthetic)})
        prediction_seconds = time.perf_counter() - started
        with truth_path.open("x", encoding="utf-8", newline="\n") as stream:
            for record in source_truth_records(source):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        scores = score_source_predictions(predictions_path, truth_path, expected_samples=len(source.val),
                                          num_classes=int(args.num_classes))
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        scores["resources"] = {"prediction_wall_seconds": prediction_seconds,
                               "total_wall_seconds": time.perf_counter() - started,
                               "sampled_peak_process_rss_bytes": rss_max,
                               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                               "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None}
        scores["synthetic"] = bool(args.game_synthetic)
        _write_json(output / "source_scores.json", scores)
        return scores
    finally:
        restore_buffers(model, buffers)
        for module, mode in modes:
            module.training = mode
        rng.restore()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    a = parser.parse_args(argv)
    # Training checkpoints include Python/RNG state; only load a trusted local
    # checkpoint created by the new trainer, never an arbitrary downloaded file.
    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if not all(key in checkpoint for key in ("model", "args", "source_info")):
        raise ValueError("Checkpoint is not the CORE90 game-tracking schema")
    args = SimpleNamespace(**checkpoint["args"])
    validate(args)
    if a.device is not None:
        args.device = a.device
    from cvsrffi.game_tracking.data import build_source
    from cvsrffi.game_tracking.runtime import build_model
    source = build_source(args)
    previous = checkpoint["source_info"]
    if previous.get("target_access_before_freeze") is not False or previous.get("checkpoint_init") != "scratch_only":
        raise ValueError("Checkpoint source provenance or target-contact boundary is unverified")
    for key in ("source_rxs", "source_days", "domain_map", "role_ids", "num_classes", "counts", "split_rule", "split_seed"):
        if previous.get(key) != source.info.get(key):
            raise ValueError(f"Checkpoint source-data contract mismatch: {key}")
    model = build_model(args, len(source.domains), torch.device(args.device))
    model.load_state_dict(checkpoint["model"], strict=True)
    scores = evaluate_source(model, source, args, a.output_dir)
    print(json.dumps({"complete": scores["complete"], "source_only": True,
                      "prediction_count": scores["prediction_count"], "output_dir": str(Path(a.output_dir).resolve())}))


if __name__ == "__main__":
    main()
