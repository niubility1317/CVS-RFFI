"""Score only after both arms have emitted complete predictions keyed by opaque IDs."""

import argparse
import json
from pathlib import Path

import numpy as np

from .prepare import write_json


ARMS = ("backbone_ce", "pamf_ce")


def metrics(y, pred, class_count):
    cm = np.bincount(class_count * y + pred, minlength=class_count**2).reshape(
        class_count, class_count)
    support, predicted = cm.sum(axis=1), cm.sum(axis=0)
    tp = cm.diagonal()
    observed = support > 0
    recall = np.divide(tp, support, out=np.zeros(class_count), where=observed)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(class_count),
                   where=support + predicted > 0)
    return {
        "messages": len(y),
        "observed_classes": int(observed.sum()),
        "accuracy": float(tp.sum() / len(y)) if len(y) else None,
        "balanced_accuracy": float(recall[observed].mean()) if len(y) else None,
        "macro_f1": float(f1[observed].mean()) if len(y) else None,
        "predicted_class_count": int((predicted > 0).sum()),
        "support": support.tolist(),
        "per_class_f1": [float(f1[i]) if observed[i] else None for i in range(class_count)],
        "confusion_matrix": cm.tolist(),
    }


def score(source_contract, query_ids, truth, out):
    contract = json.loads(source_contract.read_text(encoding="utf-8"))
    class_map = {(c["satellite_id"], c["beam_id"]): c["class_id"] for c in contract["classes"]}
    n = contract["target_physical_records"]
    ids = np.load(query_ids, allow_pickle=False)
    if len(ids) != n or len(set(ids.tolist())) != n:
        raise ValueError("Swiss public IDs are incomplete or duplicated")
    predictions = {}
    for arm in ARMS:
        with np.load(out / arm / "predictions.npz", allow_pickle=False) as d:
            logits = d["logits"]
            if (not np.array_equal(d["ids"], ids) or logits.shape != (n, len(class_map))
                    or not np.isfinite(logits).all()):
                raise ValueError(f"{arm}: incomplete, misaligned or invalid predictions")
            predictions[arm] = logits.argmax(axis=1)
    with np.load(truth, allow_pickle=False) as d:
        order = {ident: i for i, ident in enumerate(d["ids"].tolist())}
        if len(order) != n or len(d["ids"]) != n or order.keys() != set(ids.tolist()):
            raise ValueError("Private truth IDs differ from the complete public query")
        joined = np.asarray([order[ident] for ident in ids])
        sat, beam = d["satellite_id"][joined], d["beam_id"][joined]
        y = np.asarray([class_map.get((s, b), -1) for s, b in zip(sat, beam)], dtype=np.int64)
        confidence, split = d["confidence"][joined], d["original_split"][joined]
    known = y >= 0
    most_common = max(contract["classes"], key=lambda c: c["uk_train"])["class_id"]
    predictions["uk_train_majority"] = np.full(n, most_common)
    masks = {
        "swiss_all_known": known,
        "swiss_confidence_ge90_known": known & (confidence >= 90),
        "swiss_original_test_shard_known": known & (split == "test"),
    }
    result = {
        "swiss_predictions_per_arm": n, "uk_registered_classes": len(class_map),
        "swiss_known": int(known.sum()), "swiss_out_of_scope": int((~known).sum()),
        "chance_accuracy_uniform": 1 / len(class_map),
        "metrics": {
            arm: {name: metrics(y[mask], pred[mask], len(class_map)) for name, mask in masks.items()}
            for arm, pred in predictions.items()
        },
        "class_map": contract["classes"],
    }
    write_json(out / "scores.json", result)
    print(json.dumps({key: result[key] for key in ("swiss_known", "swiss_out_of_scope")}))
    for arm, values in result["metrics"].items():
        print(arm, {k: v for k, v in values["swiss_all_known"].items()
                    if k in {"accuracy", "macro_f1", "balanced_accuracy", "observed_classes"}})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_contract", type=Path)
    parser.add_argument("query_ids", type=Path)
    parser.add_argument("truth", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    score(args.source_contract, args.query_ids, args.truth, args.out)
