from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from post_stage_eval import summarize_epoch_records


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def find_metric_files(paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(sorted(path.rglob("metrics_epoch.csv")))
    return files


def summarize_metric_file(
    path: Path,
    *,
    proxy_key: str,
    test_key: str,
    source_key: str,
) -> Dict[str, Any]:
    records = _read_csv(path)
    summary = summarize_epoch_records(records, proxy_key=proxy_key, test_key=test_key, source_key=source_key)
    exp_id = path.parent.name
    if records:
        exp_id = str(records[-1].get("exp_id") or exp_id)
    final = records[-1] if records else {}

    def as_float(key: str, default: float = 0.0) -> float:
        try:
            return float(final.get(key, default))
        except Exception:
            return default

    proxy_safe_score = (
        as_float("proxy_sat_mid_safe_acc", as_float("safe_sat_mid_acc", as_float(proxy_key)))
        - 2.0 * as_float("proxy_sat_mid_harm", as_float("safe_sat_mid_harm"))
        + as_float("proxy_sat_mid_rescue", as_float("safe_sat_mid_rescue"))
        - as_float("clean_harm", as_float("safe_clean_harm", as_float("harm_rate")))
        - 0.5 * as_float("clean_acc_drop")
        - 0.5 * max(0.0, as_float("rho_mean") - 0.25)
    )
    warnings = []
    if as_float("rho_mean") > 0.25:
        warnings.append("rho_mean_high")
    if as_float("gate_easy") > 0.20:
        warnings.append("gate_easy_high")
    if as_float("safe_clean_acc", 1.0) < as_float("base_clean_acc") - 0.003:
        warnings.append("safe_clean_acc_drop")
    return {
        "exp_id": exp_id,
        "metrics_file": str(path),
        "proxy_safe_score": proxy_safe_score,
        "warnings": ",".join(warnings),
        "final_UDU": summary.get("final_test"),
        "best_proxy_UDU": summary.get("best_proxy_test"),
        "best_test_UDU": summary.get("best_test_value"),
        "final_best_delta": summary.get("final_minus_best_test"),
        "proxy_test_rank_corr": summary.get("proxy_test_rank_corr"),
        "source_test_rank_corr": summary.get("source_test_rank_corr"),
        "harm_rate": summary.get("harm_rate"),
        "rescue_rate": summary.get("rescue_rate"),
        "net_gain_rate": summary.get("net_gain_rate"),
        "changed_pred_rate": summary.get("changed_pred_rate"),
        "final_epoch": summary.get("final_epoch"),
        "best_source_epoch": summary.get("best_source_epoch"),
        "best_proxy_epoch": summary.get("best_proxy_epoch"),
        "best_test_epoch": summary.get("best_test_epoch"),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize FJMP v2 experiment metrics without selecting on test.")
    parser.add_argument("paths", nargs="+", help="Experiment directories or metrics_epoch.csv files.")
    parser.add_argument("--proxy_key", type=str, default="proxy_val_rx_day")
    parser.add_argument("--test_key", type=str, default="unseen_day_unseen_rx")
    parser.add_argument("--source_key", type=str, default="val_source")
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    files = find_metric_files(args.paths)
    rows = [
        summarize_metric_file(path, proxy_key=args.proxy_key, test_key=args.test_key, source_key=args.source_key)
        for path in files
    ]
    rows.sort(key=lambda row: str(row.get("exp_id", "")))
    if args.out_csv:
        _write_csv(Path(args.out_csv), rows)
    if args.json or not args.out_csv:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
