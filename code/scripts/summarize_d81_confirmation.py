#!/usr/bin/env python3
"""Summarize one matched D81/D62 independent-confirmation cell."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load summary helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


d81 = _load("d81_confirmation_d81_summary", SCRIPT_DIR / "summarize_d81_performance.py")
d62 = _load("d81_confirmation_d62_summary", SCRIPT_DIR / "summarize_d62_performance.py")
base, quant = d81.base, d81.quant
TARGET, FP32 = d81.TARGET, d81.FP32


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": base.sha256(path),
    }


def _gate(comparison: dict[str, Any]) -> dict[str, Any]:
    aggregate = comparison["aggregate_mean_delta"]
    class_floor = comparison["aggregate_class_floor_delta"]
    scenes = comparison["by_scene_mean_delta"]
    scene_floors = comparison["by_scene_class_floor_delta"]
    confusion = comparison["confusion_delta"]
    checks = {
        "aggregate_after_old_nonregression": aggregate["after_old"] >= 0.0,
        "aggregate_seen_new_nonregression": aggregate["seen_new"] >= 0.0,
        "aggregate_h_nonregression": aggregate["H_old_new"] >= 0.0,
        "aggregate_forgetting_nonregression": aggregate["forgetting"] <= 0.0,
        "aggregate_joint_floor_nonregression": aggregate["joint_floor"] >= 0.0,
        "aggregate_row_floors_nonregression": all(
            aggregate[key] >= 0.0
            for key in ("before_floor", "after_floor", "new_floor")
        ),
        "aggregate_class_floors_nonregression": all(
            value >= 0.0 for value in class_floor.values()
        ),
        "all_scene_means_and_row_floors_nonregression": all(
            values[key] >= 0.0
            for values in scenes.values()
            for key in (
                "before_old", "after_old", "seen_new", "H_old_new",
                "joint_floor", "before_floor", "after_floor", "new_floor",
            )
        ) and all(
            values["forgetting"] <= 0.0 for values in scenes.values()
        ),
        "all_scene_class_floors_nonregression": all(
            value >= 0.0
            for values in scene_floors.values()
            for value in values.values()
        ),
        "confusions_nonregression": all(value <= 0 for value in confusion.values()),
    }
    strict = {
        "after_old": aggregate["after_old"] > 0.0,
        "H_old_new": aggregate["H_old_new"] > 0.0,
        "forgetting": aggregate["forgetting"] < 0.0,
        "old_to_new_confusion": confusion["final_argmax_old_to_new_count"] < 0,
        "new_to_old_confusion": confusion["final_argmax_new_to_old_count"] < 0,
        "new_to_new_confusion": confusion["final_argmax_new_to_new_count"] < 0,
    }
    return {
        "nonregression_checks": checks,
        "strict_gain_checks": strict,
        "passes_nonregression": all(checks.values()),
        "has_strict_gain": any(strict.values()),
        "passes_joint_gate": all(checks.values()) and any(strict.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d81-log", required=True, type=Path)
    parser.add_argument("--d62-log", required=True, type=Path)
    parser.add_argument("--d81-metadata", required=True, type=Path)
    parser.add_argument("--d62-metadata", required=True, type=Path)
    parser.add_argument("--d81-receipt", required=True, type=Path)
    parser.add_argument("--d62-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    logs = {
        "D81": base.load_jsonl(args.d81_log),
        "D62": base.load_jsonl(args.d62_log),
    }
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = {
        name: [row for row in rows if row["candidate_id"] == FP32]
        for name, rows in logs.items()
    }
    if any(len(rows) != 15 for rows in [*target.values(), *fp32.values()]):
        raise ValueError("expected 15 INT8 and 15 FP32 target rows per run")

    metadata = {
        "D81": json.loads(args.d81_metadata.read_text(encoding="utf-8-sig")),
        "D62": json.loads(args.d62_metadata.read_text(encoding="utf-8-sig")),
    }
    receipts = {
        "D81": json.loads(args.d81_receipt.read_text(encoding="utf-8-sig")),
        "D62": json.loads(args.d62_receipt.read_text(encoding="utf-8-sig")),
    }
    identity = tuple(
        receipts["D81"].get(key) for key in ("receiver", "seed", "k_shot", "new_class_count")
    )
    if identity != tuple(
        receipts["D62"].get(key) for key in ("receiver", "seed", "k_shot", "new_class_count")
    ) or identity != ("20-1", 713102, 10, 5):
        raise ValueError("D81/D62 confirmation cell identity mismatch")
    if any(receipt.get("training_log_row_count") != 105 for receipt in receipts.values()):
        raise ValueError("receipt training row count drift")
    if any(receipt.get("query_opened") is not False for receipt in receipts.values()):
        raise ValueError("query-opened receipt is not diagnostic-safe")

    comparison = base.matched_delta(target["D81"], target["D62"])
    comparison = {
        "D81": comparison.pop("D49"),
        "D62": comparison.pop("D45"),
        **comparison,
    }
    result = {
        "schema": "cvs.phase2.d81.independent_confirmation_summary.v1",
        "cell": {"receiver": identity[0], "seed": identity[1], "k_shot": 10, "new_count": 5},
        "inputs": {
            "D81_log": _artifact(args.d81_log),
            "D62_log": _artifact(args.d62_log),
            "D81_metadata": _artifact(args.d81_metadata),
            "D62_metadata": _artifact(args.d62_metadata),
            "D81_receipt": _artifact(args.d81_receipt),
            "D62_receipt": _artifact(args.d62_receipt),
        },
        "protocol": {
            "query_opened": False,
            "ground_component_formal_phase2_eligible": False,
            "claim_scope": "development_diagnostic_independent_confirmation",
        },
        "D81_INT8": {
            "aggregate": base.aggregate(target["D81"]),
            "by_scene": {
                scene: base.aggregate(rows)
                for scene, rows in base.scene_groups(target["D81"]).items()
            },
            "classes": base.class_summary(target["D81"]),
            "outer_rows": base.detailed_rows(target["D81"]),
            "mechanism": d81.mechanism(target["D81"], metadata["D81"]),
            "training": d81.training(target["D81"]),
            "quantization": quant(target["D81"]),
            "resources": d81.resources(target["D81"]),
        },
        "D81_FP32_MATCHED": {"aggregate": base.aggregate(fp32["D81"])},
        "D62_INT8": {
            "aggregate": base.aggregate(target["D62"]),
            "by_scene": {
                scene: base.aggregate(rows)
                for scene, rows in base.scene_groups(target["D62"]).items()
            },
            "classes": base.class_summary(target["D62"]),
            "outer_rows": base.detailed_rows(target["D62"]),
            "mechanism": d62.mechanism(target["D62"]),
            "training": base.training_summary(target["D62"]),
            "quantization": quant(target["D62"]),
            "resources": d62.resources(target["D62"]),
            "fp32_centering_audit": metadata["D62"]["fp32_centering_audit"],
        },
        "D62_FP32_MATCHED": {"aggregate": base.aggregate(fp32["D62"])},
        "D81_vs_D62": comparison,
        "confirmation_gate": _gate(comparison),
        "receipts": receipts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": base.sha256(args.output),
        "D81": result["D81_INT8"]["aggregate"],
        "D62": result["D62_INT8"]["aggregate"],
        "delta": comparison["aggregate_mean_delta"],
        "confusion_delta": comparison["confusion_delta"],
        "confirmation_gate": result["confirmation_gate"],
        "D62_centering": metadata["D62"]["fp32_centering_audit"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
