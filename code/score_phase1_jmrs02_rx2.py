#!/usr/bin/env python3
"""Independent truth-last scorer for the focused JMRS02 RX2 repair."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cvsrffi.jmrs02_rx2 import RX2_ROWS
from cvsrffi.jmrs02_rx2_scoring import SCENARIOS, score_rx2_records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"{path}:{line_number} is not an object")
                rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite RX2 scorer output: {output}")
    predictions = _read_jsonl(Path(args.predictions).resolve())
    truths = _read_jsonl(Path(args.truth).resolve())
    if {str(row["row"]) for row in predictions} != set(RX2_ROWS):
        raise ValueError("focused RX2 rows are incomplete or unexpected")
    if {str(row["scenario"]) for row in predictions} != set(SCENARIOS):
        raise ValueError("focused RX2 scenarios are incomplete or unexpected")
    result = score_rx2_records(predictions, truths)
    if not _finite(result):
        raise FloatingPointError("non-finite RX2 score")
    output.mkdir(parents=True, exist_ok=False)
    payloads = {
        "jmrs02_rx2_metrics.json": result["metrics"],
        "jmrs02_rx2_decision.json": result["decision"],
        "jmrs02_rx2_scope.json": {
            "role": result["role"],
            "target_dg_claim_authorized": result["target_dg_claim_authorized"],
            "prediction_count": len(predictions),
        },
    }
    for name, value in payloads.items():
        (output / name).write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps({"status": "ANALYZED", "decision": result["decision"]}), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score focused JMRS02 RX2 predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_arg_parser().parse_args()))
