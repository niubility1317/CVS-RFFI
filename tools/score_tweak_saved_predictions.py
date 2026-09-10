"""Read-only independent NumPy Algorithm-2 scorer for saved Tweak predictions."""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("log", type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "results.json").read_text(encoding="utf-8"))
    log_text = args.log.read_text(encoding="utf-8")
    events = [json.loads(line) for line in log_text.splitlines() if line.strip()]
    epochs = [r for r in events if r.get("event") == "epoch"]
    probes = [r for r in events if r.get("event") == "learning_rate_probe"]
    assert [r["epoch"] for r in epochs] == list(range(1, 101))
    assert len(probes) == 5 and events[-1]["event"] == "complete"
    for rows, key in ((epochs, "epoch_rows"), (probes, "learning_rate_probes")):
        assert [{k: v for k, v in row.items() if k != "event"} for row in rows] == result["training"][key]
        assert all(np.isfinite(v) for row in rows for v in row.values() if isinstance(v, (int, float)))
    best = max(epochs, key=lambda r: r["source_monitor_accuracy"])
    checkpoint = torch.load(args.run / "best_checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["training"] == result["training"]
    assert best["epoch"] == result["training"]["best_epoch"]
    assert all(torch.isfinite(v).all() for v in checkpoint["state_dict"].values())
    verified = []
    expected = {(r["calibration_configuration"], r["test_configuration"]): r for r in result["single_domain_calibration_4x4"]}
    expected.update({("multi", r["test_configuration"]): r for r in result["multiple_configuration_calibration"]})
    paths = sorted(p for p in args.run.glob("predictions_*.pt") if not p.name.endswith(".truth.pt"))
    assert len(paths) == len(expected) == 20
    for path in paths:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        points, centers, radii = [payload[k].double().numpy() for k in ("points", "centroids", "radii")]
        assert all(np.isfinite(a).all() for a in (points, centers, radii)) and (radii >= 0).all()
        distances = np.sqrt(np.square(points[:, None, :] - centers[None, :, :]).sum(axis=2))
        inside = distances < radii[None, :]
        positions = np.where(inside.any(axis=1), np.where(inside, distances, np.inf).argmin(axis=1), (distances-radii).argmin(axis=1))
        computed = payload["registered_labels"].numpy()[positions]
        saved = payload["predictions"].numpy()
        assert np.array_equal(computed, saved), path
        # Only after reproducing the immutable predictions, connect truth by opaque ID.
        truth = torch.load(path.with_suffix(".truth.pt"), map_location="cpu", weights_only=True)
        qids, tids = payload["query_ids"].tolist(), truth["query_ids"].tolist()
        assert len(set(qids)) == len(qids) == 39060 and set(qids) == set(tids) and len(set(tids)) == len(tids)
        lookup = dict(zip(tids, truth["labels"].tolist()))
        labels = np.array([lookup[q] for q in qids])
        correct = int((saved == labels).sum())
        key = tuple(path.stem.removeprefix("predictions_").split("_"))
        reference = expected.pop(key)
        assert correct == reference["correct_decisions"] and len(saved) == reference["decisions"]
        assert abs(correct / len(saved) - reference["accuracy"]) < 5e-8
        verified.append({"calibration": key[0], "test": key[1], "correct": correct, "decisions": len(saved), "accuracy": correct / len(saved),
                         "prediction_mismatches": 0, "per_class_accuracy": {str(c): float((saved[labels == c] == c).mean()) for c in np.unique(labels)},
                         "prediction_counts": {str(c): int(n) for c, n in zip(*np.unique(saved, return_counts=True))},
                         "mean_radius": float(radii.mean()), "inside_any_fraction": float(inside.any(axis=1).mean())})
    assert not expected
    print(json.dumps({"status": "VERIFIED", "scorer": "NumPy float64 explicit difference; no production predictor import", "events": dict(Counter(r["event"] for r in events)), "best_epoch": best, "first_epoch": epochs[0], "last_epoch": epochs[-1], "total_active_batches": sum(r["active_batches"] for r in epochs+probes), "source_monitor_range": [min(r["source_monitor_accuracy"] for r in epochs), max(r["source_monitor_accuracy"] for r in epochs)], "rows": verified}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
