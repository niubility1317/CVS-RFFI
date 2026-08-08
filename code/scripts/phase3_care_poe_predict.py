#!/usr/bin/env python
"""Run the truth-free CARE-PoE A/B/C/D predictor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from cvsrffi.phase3_care_poe import FusionConfig, canonical_json, read_jsonl, run_abcd_matrix, sha256_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-evidence", required=True)
    parser.add_argument("--new-evidence", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--node-roster", default="SAT-01,SAT-02,SAT-03,SAT-04,SAT-05")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    base = read_jsonl(args.base_evidence)
    new = read_jsonl(args.new_evidence)
    roster = [value for value in args.node_roster.split(",") if value]
    config = FusionConfig()
    predictions = run_abcd_matrix(base, new, config, node_roster=roster)
    prediction_path = output / "predictions.jsonl"
    write_jsonl(prediction_path, predictions)
    payload = prediction_path.read_bytes()
    manifest = {
        "evidence_level": "TECHNICAL_SYNTHETIC_NO_PERFORMANCE_RESULT",
        "truth_sidecar_opened": False,
        "node_roster": roster,
        "budgets": [1, 2, 3, 4, 5],
        "prediction_rows": len(predictions),
        "prediction_sha256": __import__("hashlib").sha256(payload).hexdigest(),
        "config": config.__dict__,
        "input_evidence_root": sha256_json({"base": base, "new": new}),
    }
    (output / "prediction_manifest.json").write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

