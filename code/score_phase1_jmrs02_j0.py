#!/usr/bin/env python3
"""Independent offline JMRS02-J0 scorer over immutable JMRS01 streams."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

from cvsrffi.jmrs02_j0 import analyze_j0_rows


REQUIRED_ARTIFACTS = (
    "jmrs02_j0_semantic_audit.json",
    "jmrs02_j0_joint_rescue.json",
    "jmrs02_j0_identity_geometry.json",
    "jmrs02_j0_cost_scope.json",
    "jmrs02_j0_decision.json",
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _join_truth(
    predictions: Sequence[dict[str, Any]], truths: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    truth_by_id: dict[str, int] = {}
    for row in truths:
        sample_id = str(row["sample_id"])
        if sample_id in truth_by_id:
            raise ValueError(f"duplicate truth sample_id: {sample_id}")
        truth_by_id[sample_id] = int(row["true_class"])
    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in predictions:
        sample_id = str(row["sample_id"])
        if sample_id in seen:
            raise ValueError(f"duplicate prediction sample_id: {sample_id}")
        if sample_id not in truth_by_id:
            raise ValueError(f"prediction has no truth record: {sample_id}")
        seen.add(sample_id)
        joined.append({**row, "true_class": truth_by_id[sample_id]})
    if seen != set(truth_by_id):
        raise ValueError("prediction/truth closure mismatch")
    return joined


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite J0 result at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def score_j0_prediction_streams(
    prediction_path: Path,
    truth_path: Path,
    output_dir: Path,
    *,
    bootstrap_resamples: int = 2000,
    seed: int = 20260826,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite J0 output root: {output}")
    predictions = _read_jsonl(Path(prediction_path))
    truths = _read_jsonl(Path(truth_path))
    joined = _join_truth(predictions, truths)
    audit = analyze_j0_rows(
        joined,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    _assert_finite(audit)
    artifacts = {
        "jmrs02_j0_semantic_audit.json": audit["semantic_audit"],
        "jmrs02_j0_joint_rescue.json": audit["joint_rescue"],
        "jmrs02_j0_identity_geometry.json": audit["identity_geometry"],
        "jmrs02_j0_cost_scope.json": audit["cost_scope"],
        "jmrs02_j0_decision.json": audit["decision"],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in artifacts.items():
        _write_json(output / name, value)
    return {
        "status": "ANALYZED",
        "prediction_count": len(predictions),
        "truth_count": len(truths),
        "artifact_count": len(artifacts),
        "decision": audit["decision"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run immutable JMRS02-J0 offline audit")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--truth", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstrap_resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = score_j0_prediction_streams(
        Path(args.predictions),
        Path(args.truth),
        Path(args.output_dir),
        bootstrap_resamples=args.bootstrap_resamples,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
