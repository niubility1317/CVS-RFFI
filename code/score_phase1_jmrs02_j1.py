#!/usr/bin/env python3
"""Independent truth-last scorer for JMRS02 J1."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cvsrffi.jmrs02_j1 import J1_ROWS
from cvsrffi.jmrs02_j1_scoring import LEO_SCENARIOS, score_j1_records


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
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def run(args: argparse.Namespace) -> int:
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite scorer output: {output}")
    predictions = _read_jsonl(Path(args.predictions).resolve())
    truths = _read_jsonl(Path(args.truth).resolve())
    observed_rows = tuple(dict.fromkeys(str(row["row"]) for row in predictions))
    if set(observed_rows) != set(J1_ROWS):
        raise ValueError(f"formal J1 rows missing or unexpected: {observed_rows}")
    observed_scenarios = {str(row["scenario"]) for row in predictions}
    required = {"clean", *LEO_SCENARIOS}
    if observed_scenarios != required:
        raise ValueError(f"formal J1 scenarios must be {sorted(required)}, got {sorted(observed_scenarios)}")
    result = score_j1_records(predictions, truths)
    if not _finite(result):
        raise FloatingPointError("non-finite J1 score")
    output.mkdir(parents=True, exist_ok=False)
    payloads = {
        "jmrs02_j1_metrics.json": result["metrics"],
        "jmrs02_j1_gate_metrics.json": {
            row: {scenario: {key: value for key, value in metric.items() if key in (
                "gate_coverage", "rescue_precision", "rescue_recall", "harm_per_1000_selected",
                "gate_selected_rescue_count", "gate_selected_harm_count",
            )} for scenario, metric in scenarios.items()}
            for row, scenarios in result["metrics"].items()
        },
        "jmrs02_j1_nuisance.json": result["nuisance"],
        "jmrs02_j1_decision.json": result["decision"],
    }
    for name, value in payloads.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ANALYZED", "prediction_count": len(predictions), "artifact_count": len(payloads), "decision": result["decision"]}), flush=True)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score JMRS02 J1 closed predictions")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> int:
    return run(build_arg_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
