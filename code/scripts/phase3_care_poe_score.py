#!/usr/bin/env python
"""Score already-sealed CARE-PoE predictions with an isolated truth sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase3_care_poe import canonical_json, read_jsonl, score_predictions, sha256_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--prediction-manifest", required=True)
    parser.add_argument("--truth-sidecar", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prediction_path = Path(args.predictions)
    manifest_path = Path(args.prediction_manifest)
    predictions = read_jsonl(prediction_path)
    prediction_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    prediction_sha256 = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    if prediction_manifest.get("prediction_sha256") != prediction_sha256:
        raise ValueError("prediction file does not match prediction_manifest hash")
    if int(prediction_manifest.get("prediction_rows", -1)) != len(predictions):
        raise ValueError("prediction row count does not match prediction_manifest")
    if prediction_manifest.get("truth_sidecar_opened") is not False:
        raise ValueError("prediction_manifest must prove truth_sidecar_opened=false")
    budgets = prediction_manifest.get("budgets")
    expected_budgets = [1, 2, 3, 4, 5]
    if budgets != expected_budgets:
        raise ValueError("prediction_manifest budgets must be exactly [1, 2, 3, 4, 5]")
    truth = read_jsonl(args.truth_sidecar)
    metrics = score_predictions(
        predictions,
        truth,
        expected_arms=("A", "B", "C", "D"),
        expected_budgets=expected_budgets,
    )
    metrics["prediction_root"] = sha256_json(predictions)
    metrics["prediction_sha256"] = prediction_sha256
    metrics["prediction_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    metrics["truth_sidecar_root"] = sha256_json(truth)
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(metrics) + "\n")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
