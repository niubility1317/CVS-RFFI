#!/usr/bin/env python3
"""Create the complete D79 centered-ground-tangent performance ledger."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
D78_SUMMARY = SCRIPT_DIR / "summarize_d78_performance.py"
SPEC = importlib.util.spec_from_file_location("d79_summary_helper", D78_SUMMARY)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load D78 summary helper")
h = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(h)
base, quant = h.base, h.quant
TARGET, FP32 = "D42-USLDA-INT8", "D42-USLDA-FP32-MATCHED"


def _as_d78(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    translated = copy.deepcopy(rows)
    for row in translated:
        if row["candidate_id"] not in (TARGET, FP32):
            continue
        geometry = row["geometry_summary"]
        for key in list(geometry):
            if key.startswith("d79_"):
                geometry["d78_" + key[4:]] = geometry.pop(key)
        resource = row["resource"]
        for key in list(resource):
            if key.startswith("d79_"):
                resource["d78_" + key[4:]] = resource.pop(key)
        for item in row["training_trace"]:
            if item.get("phase") == "stage2c_centered_ground_tangent_worstclass_top2_margin":
                item["phase"] = "stage2c_ground_tangent_worstclass_top2_margin"
    return translated


def centered_mechanism(rows: list[dict[str, Any]]) -> dict[str, Any]:
    translated = _as_d78(rows)
    result = h.mechanism(translated)
    audits = [row["geometry_summary"]["d79_worstclass_margin_audit"] for row in rows]
    result["support_centering_enabled"] = True
    result["centered_support_mean_max_abs"] = base.stats(
        audit["centered_support_mean_max_abs"] for audit in audits
    )
    result["bias_residual_frobenius"] = base.stats(
        audit["bias_residual_frobenius"] for audit in audits
    )
    result["residual_logit_at_support_center_max_abs"] = base.stats(
        audit["residual_logit_at_support_center_max_abs"] for audit in audits
    )
    result["unique_support_center_sha256"] = len(
        {audit["support_center_sha256"] for audit in audits}
    )
    result["unique_bias_residual_sha256"] = len(
        {audit["bias_residual_sha256"] for audit in audits}
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    names = ("d79", "d78", "d77", "d66", "d62")
    for name in names:
        parser.add_argument(f"--{name}-log", required=True, type=Path)
    parser.add_argument("--stdout", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = {name.upper(): getattr(args, f"{name}_log") for name in names}
    logs = {name: base.load_jsonl(path) for name, path in paths.items()}
    if any(len(rows) != 105 for rows in logs.values()):
        raise ValueError("expected 105 rows per log")
    target = {
        name: [row for row in rows if row["candidate_id"] == TARGET]
        for name, rows in logs.items()
    }
    fp32 = [row for row in logs["D79"] if row["candidate_id"] == FP32]
    if any(len(rows) != 15 for rows in [*target.values(), fp32]):
        raise ValueError("expected 15 target rows")
    translated_target = _as_d78(target["D79"])
    stdout_text = args.stdout.read_text(encoding="utf-8-sig")
    stderr_text = args.stderr.read_text(encoding="utf-8-sig")
    metadata = json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    markers = ("Traceback", "RuntimeError", "KeyError", "OOM", "OutOfMemory", "Killed", "NaN", "Inf")
    result: dict[str, Any] = {
        "schema": "cvs.phase2.d79.full_performance_summary.v1",
        "input": {
            name: {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": base.sha256(path),
                "rows": len(logs[name]),
            }
            for name, path in paths.items()
        },
        "stdout_stderr": {
            "stdout_path": str(args.stdout),
            "stdout_size": args.stdout.stat().st_size,
            "stdout_sha256": base.sha256(args.stdout),
            "stdout_full_text_length": len(stdout_text),
            "stderr_path": str(args.stderr),
            "stderr_size": args.stderr.stat().st_size,
            "stderr_sha256": base.sha256(args.stderr),
            "stderr_full_text_length": len(stderr_text),
            "error_marker_counts": {
                marker: stdout_text.count(marker) + stderr_text.count(marker)
                for marker in markers
            },
        },
        "metadata": metadata,
        "all_candidates": base.candidate_summary(logs["D79"]),
        "D79_INT8": {
            "aggregate": base.aggregate(target["D79"]),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(target["D79"]).items()
            },
            "classes": base.class_summary(target["D79"]),
            "outer_rows": base.detailed_rows(target["D79"]),
            "mechanism": centered_mechanism(target["D79"]),
            "training": h.training(translated_target),
            "quantization": quant(target["D79"]),
            "resources": {
                **h.resources(translated_target),
                "d79_bias_compile_mac_equivalents": base.stats(
                    row["resource"]["d79_bias_compile_mac_equivalents"]
                    for row in target["D79"]
                ),
            },
        },
        "D79_FP32_MATCHED": {
            "aggregate": base.aggregate(fp32),
            "by_scene": {
                scene: base.aggregate(group)
                for scene, group in base.scene_groups(fp32).items()
            },
        },
    }
    for baseline in ("D78", "D77", "D66", "D62"):
        comparison = base.matched_delta(target["D79"], target[baseline])
        result[f"D79_vs_{baseline}"] = {
            "D79": comparison.pop("D49"),
            baseline: comparison.pop("D45"),
            **comparison,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": base.sha256(args.output),
                "D79_INT8": result["D79_INT8"]["aggregate"],
                "D79_vs_D62": result["D79_vs_D62"],
                "D79_vs_D78": result["D79_vs_D78"],
                "mechanism": result["D79_INT8"]["mechanism"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
