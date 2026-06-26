from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


MARKERS = (
    "[WISIG-SPLIT]",
    "[CONFIG-PAPER]",
    "[CONFIG-OPT]",
    "[CONFIG-DOMAINS]",
    "[PAPER-EVAL]",
    "[PAPER-EVAL-SUMMARY]",
    "[FINAL-TEST]",
    "[FINAL-TEST-NAMED]",
    "Traceback",
    "RuntimeError",
    "CUDA out of memory",
    "Killed",
    "nan",
    "NaN",
)


def read_text_lossy(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(read_text_lossy(path))


def nested_get(obj: Any, keys: list[str], default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def infer_method(run_dir: Path, metrics: dict[str, Any]) -> str:
    name = run_dir.name.lower()
    text = str(run_dir).lower()
    if name.startswith("riei_fd") or name.startswith("riei"):
        return "riei_fd"
    if name.startswith("drift"):
        return "drift"
    if name.startswith("cen_a31") or name.startswith("cvsrffi") or "cen_a31" in name:
        return "cvsrffi_cen_a31"
    if name.startswith("cvcnn"):
        return "cvcnn_ce"
    if name.startswith("ra_collab"):
        return "ra_collab"
    if "riei_fd" in text or "riei_original" in text:
        return "riei_fd"
    if "/drift_" in text.replace("\\", "/"):
        return "drift"
    if "cen_a31" in text or "cvsrffi" in text:
        return "cvsrffi_cen_a31"
    return "unknown"


def infer_protocol(run_dir: Path, log_lines: list[str]) -> str:
    text = str(run_dir).lower()
    if "riei_original" in text or "riei_table3" in text:
        return "riei_original"
    if "drift_day1" in text:
        return "drift_day1"
    for line in log_lines:
        m = re.search(r"protocol=([A-Za-z0-9_]+)", line)
        if m:
            return m.group(1)
    return "unknown"


def collect_log_lines(log_paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for path in log_paths:
        try:
            text = read_text_lossy(path)
        except OSError:
            continue
        for line in text.splitlines():
            if any(marker in line for marker in MARKERS):
                lines.append(f"{path.name}: {line.strip()}")
    return lines


def find_related_logs(run_dir: Path, log_root: Path | None) -> list[Path]:
    if log_root is None or not log_root.exists():
        return []
    run_name = run_dir.name
    combo_name = run_dir.parent.name
    candidates: list[Path] = []
    for path in log_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".log", ".out"}:
            continue
        path_text = str(path)
        if run_name in path_text or combo_name in path_text:
            candidates.append(path)
    return sorted(candidates)


def primary_named_acc(named: Any) -> tuple[str, float | None]:
    if not isinstance(named, dict) or not named:
        return "", None
    preferred = [
        "test_seen_day_unseen_rx",
        "test_unseen_rx_day1",
        "test_unseen_day_unseen_rx",
        "main",
    ]
    for key in preferred:
        if key in named:
            return key, safe_float(nested_get(named[key], ["tx_acc"]))
    if len(named) == 1:
        key = next(iter(named))
        return key, safe_float(nested_get(named[key], ["tx_acc"]))
    return "", None


def summarize_run(metrics_path: Path, log_root: Path | None) -> dict[str, Any]:
    run_dir = metrics_path.parent
    metrics = load_json(metrics_path)
    related_logs = find_related_logs(run_dir, log_root)
    log_lines = collect_log_lines(related_logs)
    final = metrics.get("final") if isinstance(metrics.get("final"), dict) else {}
    best = metrics.get("best") if isinstance(metrics.get("best"), dict) else {}
    epochs = metrics.get("epochs") if isinstance(metrics.get("epochs"), list) else []
    primary_name, final_primary = primary_named_acc(final.get("test_named"))
    best_primary_name, best_primary = primary_named_acc(best.get("test_named"))
    paper_window = nested_get(final, ["paper_eval_window", "test_overall_tx_acc"], {})
    return {
        "run_dir": str(run_dir),
        "metrics_path": str(metrics_path),
        "method": infer_method(run_dir, metrics),
        "protocol": infer_protocol(run_dir, log_lines),
        "status": "complete" if final else ("partial" if epochs else "metrics_without_epochs"),
        "last_epoch": nested_get(final, ["epoch"], epochs[-1].get("epoch") if epochs else None),
        "best_epoch": best.get("epoch"),
        "final_overall_tx_acc": safe_float(nested_get(final, ["test_overall", "tx_acc"])),
        "final_primary_name": primary_name,
        "final_primary_tx_acc": final_primary,
        "best_overall_tx_acc": safe_float(nested_get(best, ["test_overall", "tx_acc"])),
        "best_primary_name": best_primary_name,
        "best_primary_tx_acc": best_primary,
        "paper_eval_window_mean": safe_float(paper_window.get("mean") if isinstance(paper_window, dict) else None),
        "paper_eval_window_std": safe_float(paper_window.get("std") if isinstance(paper_window, dict) else None),
        "paper_eval_window_n": paper_window.get("n") if isinstance(paper_window, dict) else None,
        "log_paths": [str(p) for p in related_logs],
        "config_lines": [line for line in log_lines if "[CONFIG-" in line or "[WISIG-SPLIT]" in line],
        "final_lines": [line for line in log_lines if "[FINAL-" in line or "[PAPER-EVAL-SUMMARY]" in line],
        "warning_lines": [
            line
            for line in log_lines
            if any(marker in line for marker in ("Traceback", "RuntimeError", "CUDA out of memory", "Killed", "nan", "NaN"))
        ],
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "method",
        "protocol",
        "status",
        "last_epoch",
        "best_epoch",
        "final_overall_tx_acc",
        "final_primary_name",
        "final_primary_tx_acc",
        "best_overall_tx_acc",
        "best_primary_name",
        "best_primary_tx_acc",
        "paper_eval_window_mean",
        "paper_eval_window_std",
        "paper_eval_window_n",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fields})


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def write_markdown(path: Path, records: list[dict[str, Any]], run_root: Path, log_root: Path | None) -> None:
    lines = [
        "# Paper Reproduction Summary",
        "",
        f"- run_root: `{run_root}`",
        f"- log_root: `{log_root}`" if log_root else "- log_root: not provided",
        f"- records: `{len(records)}`",
        "",
        "| method | protocol | status | last_epoch | best_epoch | final_overall | final_primary | best_overall | paper_window_mean | run_dir |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for record in records:
        final_primary = f"{record.get('final_primary_name') or ''}:{fmt(record.get('final_primary_tx_acc'))}"
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(record.get("method")),
                    fmt(record.get("protocol")),
                    fmt(record.get("status")),
                    fmt(record.get("last_epoch")),
                    fmt(record.get("best_epoch")),
                    fmt(record.get("final_overall_tx_acc")),
                    final_primary,
                    fmt(record.get("best_overall_tx_acc")),
                    fmt(record.get("paper_eval_window_mean")),
                    fmt(record.get("run_dir")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Config Evidence", ""])
    for record in records:
        lines.append(f"### {record.get('method')} / {record.get('protocol')}")
        lines.append("")
        lines.append(f"- run_dir: `{record.get('run_dir')}`")
        lines.append(f"- logs: `{len(record.get('log_paths', []))}`")
        for line in record.get("config_lines", [])[:12]:
            lines.append(f"- `{line}`")
        warnings = record.get("warning_lines", [])
        if warnings:
            lines.append("- warnings/errors:")
            for line in warnings[:10]:
                lines.append(f"  - `{line}`")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect paper reproduction metrics/log evidence.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--log-root", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    log_root = Path(args.log_root) if args.log_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = sorted(run_root.rglob("metrics.json")) if run_root.exists() else []
    records = [summarize_run(path, log_root) for path in metrics_paths]
    summary = {
        "run_id": args.run_id,
        "run_root": str(run_root),
        "log_root": str(log_root) if log_root else "",
        "record_count": len(records),
        "records": records,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(out_dir / "summary.csv", records)
    write_markdown(out_dir / "summary.md", records, run_root, log_root)
    print(json.dumps({"record_count": len(records), "summary_dir": str(out_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
