#!/usr/bin/env python
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import torch

from cvsrffi.gate_metrics import binary_reject_metrics, summarize_gate_decisions


def summarize_decision_csv(path: str | Path) -> dict[str, Any]:
    rows = []
    y_unknown = []
    reject_scores = []
    accepted = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
            if "is_unknown" in row:
                y_unknown.append(str(row.get("is_unknown", "")).lower() in ("1", "true", "yes"))
                reject_scores.append(float(row.get("reject_score", 0.0) or 0.0))
                accepted.append(str(row.get("decision", "")).startswith("ACCEPT"))
    summary = summarize_gate_decisions(rows)
    if y_unknown:
        summary.update(binary_reject_metrics(torch.tensor(y_unknown), torch.tensor(reject_scores), torch.tensor(accepted)))
    else:
        summary.update({"unknown_FAR": None, "unknown_reject_rate": None, "FPR95": None, "AUROC_unknown": None})
    summary["source_note"] = "real unknown metrics are null unless per-sample unknown labels are present"
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Stage2 local-gate per-sample decisions.")
    parser.add_argument("--decisions_csv", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    metrics = summarize_decision_csv(args.decisions_csv)
    payload = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
