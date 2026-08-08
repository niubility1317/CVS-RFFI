#!/usr/bin/env python
"""Score already-sealed CARE-PoE predictions with an isolated truth sidecar."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--truth-sidecar", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    truth = read_jsonl(args.truth_sidecar)
    metrics = score_predictions(predictions, truth)
    metrics["prediction_root"] = sha256_json(predictions)
    metrics["truth_sidecar_root"] = sha256_json(truth)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(metrics) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

