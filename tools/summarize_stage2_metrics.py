"""Summarize Stage2 metric JSON files without reading feature arrays."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _finite(value):
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean(rows, key: str):
    vals = [_finite(row.get(key)) for row in rows]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _vals(rows, key: str):
    return [v for v in (_finite(row.get(key)) for row in rows) if v is not None]


def summarize(run_root: Path) -> dict:
    rows = []
    for path in sorted(run_root.glob("*/metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
        adapter = (
            data.get("telemetry", {})
            .get("oa_mse_onboard_adaptation", {})
            .get("target_adapter", {})
        )
        if not isinstance(adapter, dict):
            adapter = {}
        rows.append(
            {
                "id": path.parent.name,
                "old": metrics.get("old_class_accuracy"),
                "seen": metrics.get("new_class_accuracy"),
                "rej": metrics.get("unknown_rejection_rate"),
                "far": metrics.get("unknown_false_accept_rate"),
                "full": metrics.get("full_accuracy"),
                "cov": metrics.get("coverage"),
                "alpha": adapter.get("selected_alpha"),
                "policy": adapter.get("adapter_selection_policy"),
                "loss0": data.get("loss_initial"),
                "loss1": data.get("loss_final"),
            }
        )
    summary = {
        "rows": len(rows),
        "old_mean": _mean(rows, "old"),
        "old_max": max(_vals(rows, "old") or [-1.0]),
        "seen_mean": _mean(rows, "seen"),
        "seen_max": max(_vals(rows, "seen") or [-1.0]),
        "rej_mean": _mean(rows, "rej"),
        "rej_max": max(_vals(rows, "rej") or [-1.0]),
        "far_mean": _mean(rows, "far"),
        "far_min": min(_vals(rows, "far") or [-1.0]),
        "full_mean": _mean(rows, "full"),
        "cov_mean": _mean(rows, "cov"),
        "loss_initial_mean": _mean(rows, "loss0"),
        "loss_final_mean": _mean(rows, "loss1"),
        "old_ge_95": sum(1 for row in rows if (row.get("old") or -1) >= 0.95),
        "seen_ge_80": sum(1 for row in rows if (row.get("seen") or -1) >= 0.80),
        "rej_ge_95": sum(1 for row in rows if (row.get("rej") or -1) >= 0.95),
        "all_three": sum(
            1
            for row in rows
            if (row.get("old") or -1) >= 0.95
            and (row.get("seen") or -1) >= 0.80
            and (row.get("rej") or -1) >= 0.95
        ),
        "alpha_counts": {},
        "policies": sorted({str(row.get("policy")) for row in rows}),
    }
    for row in rows:
        key = str(row.get("alpha"))
        summary["alpha_counts"][key] = summary["alpha_counts"].get(key, 0) + 1
    return {
        "summary": summary,
        "top_old": sorted(rows, key=lambda row: row.get("old") or -1, reverse=True)[:10],
        "top_seen": sorted(rows, key=lambda row: row.get("seen") or -1, reverse=True)[:10],
        "top_rej": sorted(rows, key=lambda row: row.get("rej") or -1, reverse=True)[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    out = summarize(args.run_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
