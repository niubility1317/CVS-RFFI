"""Join all frozen Oxford SatIQ predictions to future-only scoring metadata."""

import argparse
from pathlib import Path

import numpy as np

from .data import CLASSES, read_json, write_json
from .train import ROWS, metrics


def calculate(y, pred):
    return metrics(np.bincount(66 * y + pred, minlength=66**2).reshape(66, 66))


def score(root, truth_path, source_contract):
    if not (root / "source_complete.json").is_file():
        raise ValueError("All four source models must be frozen before scoring")
    guesses = {}
    for arm in ROWS:
        with np.load(root / arm / "predictions.npz", allow_pickle=False) as z:
            ids, logits = z["ids"], z["logits"]
        if (logits.shape != (len(ids), 66) or not np.isfinite(logits).all()
                or len(set(ids.tolist())) != len(ids)):
            raise ValueError(f"Incomplete predictions for {arm}")
        guesses[arm] = ids, logits.argmax(1)
    with np.load(truth_path, allow_pickle=False) as z:
        qids, y, days, magnitude = (z[k] for k in ("ids", "y", "day", "magnitude"))
    low = magnitude <= read_json(source_contract)["low_quality_threshold"]
    result = {}
    predictions = {}
    for arm, (ids, classes) in guesses.items():
        index = dict(zip(ids.tolist(), classes.tolist(), strict=True))
        if set(qids.tolist()) != set(index):
            raise ValueError(f"Prediction and truth coverage mismatch: {arm}")
        pred = np.array([index[key] for key in qids])
        predictions[arm] = pred
        entry = calculate(y, pred)
        entry["day"] = {str(24 + int(d)): calculate(y[days == d], pred[days == d])
                        for d in np.unique(days)}
        entry["low_detect_quality"] = calculate(y[low], pred[low])
        result[arm] = entry
    for arm, base in (("S1", "S0"), ("U0", "S1"), ("U1", "U0")):
        old = predictions[base] == y
        new = predictions[arm] == y
        result[f"{arm}_versus_{base}"] = {
            "macro_f1_delta": result[arm]["macro_f1"] - result[base]["macro_f1"],
            "wrong_to_correct": int((~old & new).sum()),
            "correct_to_wrong": int((old & ~new).sum())}
    result["satellite_support"] = {str(sat): int(n) for sat, n in
                                   zip(CLASSES, np.bincount(y, minlength=66), strict=True)}
    write_json(root / "truth_last_metrics.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("truth", type=Path)
    p.add_argument("source_contract", type=Path)
    a = p.parse_args()
    score(a.root, a.truth, a.source_contract)
