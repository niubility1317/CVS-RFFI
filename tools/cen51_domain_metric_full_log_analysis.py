#!/usr/bin/env python
"""Parse the full CEN51 domain-metric controller logs.

The scheduler log directory contains non-candidate stdout files, so this parser
uses the matrix JSON as the authoritative candidate list and then reads each
matching stdout log completely.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?|nan"
SAT_SCENARIOS = ["clear_leo", "low_elev_leo", "rain_leo", "storm_mp", "mixed_orbit"]
ERROR_RE = re.compile(r"Traceback|CUDA out of memory|out of memory|unrecognized arguments|Killed|BLOCKED_PATH|RuntimeError", re.I)


def fnum(text: str | None) -> float | None:
    if text is None:
        return None
    if text.lower() == "nan":
        return math.nan
    return float(text)


def parse_kv_line(line: str) -> dict[str, str]:
    data: dict[str, str] = {}
    for key, value in re.findall(r"([A-Za-z0-9_]+)=([^ |\]]+)", line):
        data[key] = value.rstrip(",")
    return data


def pct_after(prefix: str, line: str) -> float | None:
    match = re.search(re.escape(prefix) + rf"=({FLOAT})%", line)
    return fnum(match.group(1)) if match else None


def epoch_from_token(token: str | None) -> int | None:
    if not token:
        return None
    match = re.search(r"E(-?\d+)", token)
    return int(match.group(1)) if match else None


def read_matrix(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def meta_int(meta: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        if meta.get(key) is not None:
            return int(meta[key])
    return int(default)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not math.isfinite(number):
        return "nan"
    return f"{number:.2f}"


@dataclass
class EpochRow:
    run_name: str
    cid: str
    shot: int
    gpu: int
    epoch: int
    declared_epochs: int | None = None
    lr: float | None = None
    aux_scale: float | None = None
    phase: str | None = None
    loss_total_raw: float | None = None
    loss_cls_raw: float | None = None
    loss_dom_raw: float | None = None
    loss_adv_raw: float | None = None
    loss_cons_raw: float | None = None
    loss_group_ce_raw: float | None = None
    loss_cls_w: float | None = None
    loss_dom_w: float | None = None
    loss_adv_w: float | None = None
    loss_cons_w: float | None = None
    loss_group_ce_w: float | None = None
    proto_raw: float | None = None
    supcon_raw: float | None = None
    fishr_raw: float | None = None
    feature_norm_raw: float | None = None
    proto_w: float | None = None
    supcon_w: float | None = None
    fishr_w: float | None = None
    feature_norm_w: float | None = None
    weight_dom: float | None = None
    weight_adv: float | None = None
    weight_cons: float | None = None
    weight_group_ce: float | None = None
    train_tx: float | None = None
    train_dom: float | None = None
    val_tx: float | None = None
    val_dom: float | None = None
    test_overall: float | None = None
    test_udu: float | None = None
    time_train: float | None = None
    time_val: float | None = None
    time_test: float | None = None
    time_sat_test: float | None = None


def parse_epoch_rows(text: str, run_name: str, cid: str, shot: int, gpu: int) -> list[EpochRow]:
    rows: list[EpochRow] = []
    current: EpochRow | None = None
    for line in text.splitlines():
        match = re.search(r"\[EPOCH-BEGIN\]\s+E(\d+)/(\d+).*lr=([0-9.eE+-]+).*aux_scale=([0-9.eE+-]+)", line)
        if match:
            current = EpochRow(
                run_name=run_name,
                cid=cid,
                shot=shot,
                gpu=gpu,
                epoch=int(match.group(1)),
                declared_epochs=int(match.group(2)),
                lr=float(match.group(3)),
                aux_scale=float(match.group(4)),
            )
            rows.append(current)
            continue
        if current is None:
            continue
        if line.startswith("[TIME]"):
            for key, attr in [("train", "time_train"), ("val", "time_val"), ("test", "time_test"), ("sat_test", "time_sat_test")]:
                match = re.search(rf"{key}=({FLOAT})s", line)
                if match:
                    setattr(current, attr, fnum(match.group(1)))
        elif line.startswith("[STAGE]"):
            match = re.search(r"phase=([^ |]+)", line)
            if match:
                current.phase = match.group(1)
        elif line.startswith("[LOSS-CORE-RAW]"):
            kv = parse_kv_line(line)
            current.loss_total_raw = fnum(kv.get("total"))
            current.loss_cls_raw = fnum(kv.get("cls"))
            current.loss_dom_raw = fnum(kv.get("dom"))
            current.loss_adv_raw = fnum(kv.get("adv"))
            current.loss_cons_raw = fnum(kv.get("cons"))
            current.loss_group_ce_raw = fnum(kv.get("group_ce"))
        elif line.startswith("[LOSS-CORE-W]"):
            kv = parse_kv_line(line)
            current.loss_cls_w = fnum(kv.get("cls"))
            current.loss_dom_w = fnum(kv.get("dom"))
            current.loss_adv_w = fnum(kv.get("adv"))
            current.loss_cons_w = fnum(kv.get("cons"))
            current.loss_group_ce_w = fnum(kv.get("group_ce"))
        elif line.startswith("[LOSS-DG-RAW]"):
            kv = parse_kv_line(line)
            current.proto_raw = fnum(kv.get("proto"))
            current.supcon_raw = fnum(kv.get("supcon"))
            current.fishr_raw = fnum(kv.get("fishr"))
            current.feature_norm_raw = fnum(kv.get("feature_norm"))
        elif line.startswith("[LOSS-DG-W]"):
            kv = parse_kv_line(line)
            current.proto_w = fnum(kv.get("proto"))
            current.supcon_w = fnum(kv.get("supcon"))
            current.fishr_w = fnum(kv.get("fishr"))
            current.feature_norm_w = fnum(kv.get("feature_norm"))
        elif line.startswith("[LOSS-WEIGHT]"):
            kv = parse_kv_line(line)
            current.weight_dom = fnum(kv.get("dom"))
            current.weight_adv = fnum(kv.get("adv"))
            current.weight_cons = fnum(kv.get("cons"))
            current.weight_group_ce = fnum(kv.get("group_ce"))
        elif line.startswith("[TRAIN]"):
            current.train_tx = pct_after("tx", line)
            current.train_dom = pct_after("dom", line)
        elif line.startswith("[VAL]"):
            current.val_tx = pct_after("tx", line)
            current.val_dom = pct_after("dom", line)
        elif line.startswith("[TEST]  overall_tx="):
            current.test_overall = pct_after("overall_tx", line)
        elif "unseen_day_unseen_rx" in line and "tx=" in line and current.test_udu is None:
            current.test_udu = pct_after("tx", line)
    return rows


def parse_final_sections(lines: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    section = None
    avg_mode = None
    for line in lines:
        if line.startswith("[FINAL-BEST] val_tx="):
            section = "best"
            avg_mode = None
            out["final_best_val_tx"] = pct_after("val_tx", line)
            out["final_best_test_overall"] = pct_after("test_overall_tx", line)
        elif line.startswith("[FINAL-PRIMARY] val_tx="):
            section = "primary"
            avg_mode = None
            out["final_primary_val_tx"] = pct_after("val_tx", line)
            out["final_primary_overall"] = pct_after("test_overall_tx", line)
            out["final_primary_strict"] = pct_after("strict_udu", line)
            match = re.search(r"score=([0-9.]+)", line)
            out["final_primary_score"] = fnum(match.group(1)) if match else None
        elif line.startswith("[FINAL-AVG] mode="):
            section = "avg"
            mode = re.search(r"mode=([A-Z]+)", line)
            avg_mode = mode.group(1).lower() if mode else "avg"
            prefix = f"final_{avg_mode}"
            out[f"{prefix}_val_tx"] = pct_after("val_tx", line)
            out[f"{prefix}_overall"] = pct_after("test_overall", line)
            out[f"{prefix}_strict"] = pct_after("strict_udu", line)
            out[f"{prefix}_worst_rx"] = pct_after("worst_rx", line)
            score = re.search(r"score=([0-9.]+)", line)
            out[f"{prefix}_score"] = fnum(score.group(1)) if score else None
        elif line.startswith("[FINAL-BEST] [SAT-TEST]"):
            scenario = re.search(r"scenario=([^ ]+)", line)
            if scenario:
                out[f"final_best_sat_{scenario.group(1)}"] = pct_after("strict_udu", line)
        elif line.startswith("[FINAL-PRIMARY] [SAT-TEST]"):
            scenario = re.search(r"scenario=([^ ]+)", line)
            if scenario:
                out[f"final_primary_sat_{scenario.group(1)}"] = pct_after("strict_udu", line)
        elif line.startswith("[FINAL-AVG][") and "[SAT-TEST]" in line:
            mode = re.search(r"\[FINAL-AVG\]\[([A-Z]+)\]", line)
            scenario = re.search(r"scenario=([^ ]+)", line)
            if mode and scenario:
                out[f"final_{mode.group(1).lower()}_sat_{scenario.group(1)}"] = pct_after("strict_udu", line)
        elif section in {"best", "primary"} and line.startswith(f"[FINAL-{section.upper()}] unseen_day_unseen_rx"):
            out[f"final_{section}_strict"] = pct_after("tx", line)
        elif section in {"best", "primary"} and re.search(r"\[FINAL-[A-Z]+\] rx=\d+ on unseen_days", line):
            rx_match = re.search(r"rx=(\d+)", line)
            if rx_match:
                out[f"final_{section}_rx{rx_match.group(1)}_unseen"] = pct_after("tx", line)
        elif line.startswith("[FINAL-AVG][") and "rx=" in line and "on unseen_days" in line:
            mode = re.search(r"\[FINAL-AVG\]\[([A-Z]+)\]", line)
            rx_match = re.search(r"rx=(\d+)", line)
            if mode and rx_match:
                out[f"final_{mode.group(1).lower()}_rx{rx_match.group(1)}_unseen"] = pct_after("tx", line)
    for prefix in ["final_best", "final_primary", "final_ema", "final_swad"]:
        sat_values = [out.get(f"{prefix}_sat_{s}") for s in SAT_SCENARIOS if out.get(f"{prefix}_sat_{s}") is not None]
        rx_values = [out.get(f"{prefix}_rx{r}_unseen") for r in [7, 8, 9, 10, 11] if out.get(f"{prefix}_rx{r}_unseen") is not None]
        if sat_values:
            out[f"{prefix}_sat_mean"] = mean(sat_values)
            out[f"{prefix}_sat_floor"] = min(sat_values)
        if rx_values:
            out[f"{prefix}_rx_floor"] = min(rx_values)
            out[f"{prefix}_rx_mean"] = mean(rx_values)
    return out


def parse_log(path: Path, meta: dict[str, Any]) -> tuple[dict[str, Any], list[EpochRow]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    shot = meta_int(meta, "shot", "k_cap")
    row: dict[str, Any] = {
        "cid": meta["cid"],
        "run_name": meta["run_name"],
        "shot": shot,
        "k_cap": meta.get("k_cap", shot),
        "ratio": meta.get("ratio"),
        "ratio_tag": meta.get("ratio_tag"),
        "strategy": meta.get("strategy"),
        "n_eff_nominal": meta.get("n_eff_nominal"),
        "cap_fraction": meta.get("cap_fraction"),
        "gpu": int(meta["gpu"]),
        "seed": int(meta["seed"]),
        "axis": meta["axis"],
        "action": meta["action"],
        "anchor_name": meta.get("anchor_name"),
        "target_strict": meta.get("target_strict"),
        "target_overall": meta.get("target_overall"),
        "hypothesis": meta.get("hypothesis"),
        "success_gate": meta.get("success_gate"),
        "log_file": str(path),
        "log_lines": len(lines),
        "log_bytes": path.stat().st_size,
        "error_count": sum(1 for line in lines if ERROR_RE.search(line)),
        "warnings_unsafe_backward": sum(1 for line in lines if "[WARN]" in line and "unsafe backward" in line),
        "finished": any("Training finished." in line for line in lines),
    }
    for line in lines:
        if line.startswith("[CONFIG-DATA]"):
            kv = parse_kv_line(line)
            for key in ["train_ratio", "val_ratio", "batch", "eval_batch", "workers", "num_classes", "num_domains"]:
                row[f"config_data_{key}"] = fnum(kv.get(key)) if key in ["train_ratio", "val_ratio"] else kv.get(key)
        elif line.startswith("[CONFIG-OPT]"):
            kv = parse_kv_line(line)
            for key in ["lr", "lr_min", "wd", "epochs", "label_smoothing", "clip_backbone", "clip_aux", "clip_domain"]:
                row[f"config_opt_{key}"] = fnum(kv.get(key))
        elif line.startswith("[CONFIG-LOSS]"):
            kv = parse_kv_line(line)
            for key in [
                "lambda_dom",
                "lambda_adv",
                "lambda_orth",
                "lambda_cons",
                "lambda_group_ce",
                "group_mode",
                "group_ce_min_domains",
                "lambda_proto",
                "lambda_supcon_id",
                "lambda_fishr",
                "fishr_min_domains",
                "lambda_feature_norm_guard",
                "feature_norm_guard_mode",
                "feature_norm_guard_target",
            ]:
                value = kv.get(key)
                row[f"config_loss_{key}"] = fnum(value) if value and re.match(r"^[0-9.eE+-]+$", value) else value
        elif line.startswith("[CONFIG-MIXSTYLE]"):
            kv = parse_kv_line(line)
            for key in ["enabled", "p", "strength", "alpha", "layers", "mix", "late_start", "late_ramp", "late_min_p", "late_min_strength"]:
                value = kv.get(key)
                row[f"config_mixstyle_{key}"] = fnum(value) if value and re.match(r"^[0-9.eE+-]+$", value) else value
        elif line.startswith("[CONFIG-SAT]"):
            kv = parse_kv_line(line)
            for key in ["train_enabled", "train_scenario", "train_cycle", "lambda_sat_cls", "lambda_sat_cons", "start_epoch", "eval_enabled", "eval_scenarios", "eval_on", "eval_max_batches"]:
                value = kv.get(key)
                row[f"config_sat_{key}"] = fnum(value) if value and re.match(r"^[0-9.eE+-]+$", value) else value
        elif line.startswith("[CONFIG-CONCAT-SAT]"):
            kv = parse_kv_line(line)
            for key in ["enabled", "start_epoch", "view_prob", "seed"]:
                row[f"config_concat_sat_{key}"] = fnum(kv.get(key)) if kv.get(key) is not None else None
            row["config_concat_sat_line"] = line
        elif line.startswith("[WISIG] split_info="):
            match = re.search(r"'max_samples_per_combo_train': ([0-9]+|None)", line)
            row["split_max_samples_per_combo_train"] = None if not match or match.group(1) == "None" else int(match.group(1))
            match = re.search(r"'train_size': ([0-9]+)", line)
            row["split_train_size"] = int(match.group(1)) if match else None
            match = re.search(r"'val_size': ([0-9]+)", line)
            row["split_val_size"] = int(match.group(1)) if match else None
        elif line.startswith("[BEST-TEST]"):
            match = re.search(r"overall=([0-9.]+)% @ (E-?\d+).*unseen_day_unseen_rx=([0-9.]+)% @ (E-?\d+).*unseen_day_seen_rx=([0-9.]+)% @ (E-?\d+).*seen_day_unseen_rx=([0-9.]+)% @ (E-?\d+)", line)
            if match:
                row["best_overall"] = fnum(match.group(1))
                row["best_overall_epoch"] = epoch_from_token(match.group(2))
                row["best_strict"] = fnum(match.group(3))
                row["best_strict_epoch"] = epoch_from_token(match.group(4))
                row["best_unseen_day_seen_rx"] = fnum(match.group(5))
                row["best_unseen_day_seen_rx_epoch"] = epoch_from_token(match.group(6))
                row["best_seen_day_unseen_rx"] = fnum(match.group(7))
                row["best_seen_day_unseen_rx_epoch"] = epoch_from_token(match.group(8))
        elif line.startswith("[BEST-PRIMARY]"):
            match = re.search(r"score=([0-9.]+) @ (E-?\d+).*overall=([0-9.]+)% strict_udu=([0-9.]+)%", line)
            if match:
                row["best_primary_score"] = fnum(match.group(1))
                row["best_primary_epoch"] = epoch_from_token(match.group(2))
                row["best_primary_overall"] = fnum(match.group(3))
                row["best_primary_strict"] = fnum(match.group(4))
        elif line.startswith("[BEST-WORST-RX]"):
            match = re.search(r"worst_rx=([0-9.]+)% \(([^)]*)\) @ (E-?\d+)", line)
            if match:
                row["best_worst_rx"] = fnum(match.group(1))
                row["best_worst_rx_name"] = match.group(2)
                row["best_worst_rx_epoch"] = epoch_from_token(match.group(3))
        elif line.startswith("Training finished. best_joint_val_tx_acc="):
            row["finished_joint_val_tx"] = pct_after("best_joint_val_tx_acc", line)
        elif line.startswith("Training finished. best_test_overall_tx_acc="):
            row["finished_best_overall"] = pct_after("best_test_overall_tx_acc", line)
            match = re.search(r"at epoch (\d+)", line)
            row["finished_best_overall_epoch"] = int(match.group(1)) if match else None
        elif line.startswith("Training finished. best_primary_ood_score="):
            match = re.search(r"best_primary_ood_score=([0-9.]+) at epoch (\d+)", line)
            if match:
                row["finished_best_primary_score"] = fnum(match.group(1))
                row["finished_best_primary_epoch"] = int(match.group(2))
        elif line.startswith("Training finished. best_unseen_day_unseen_rx_tx_acc="):
            row["finished_best_strict"] = pct_after("best_unseen_day_unseen_rx_tx_acc", line)
            match = re.search(r"at epoch (\d+)", line)
            row["finished_best_strict_epoch"] = int(match.group(1)) if match else None
        elif line.startswith("Training finished. best_worst_rx_tx_acc="):
            match = re.search(r"best_worst_rx_tx_acc=([0-9.]+)% \(([^)]*)\) at epoch (\d+)", line)
            if match:
                row["finished_best_worst_rx"] = fnum(match.group(1))
                row["finished_best_worst_rx_name"] = match.group(2)
                row["finished_best_worst_rx_epoch"] = int(match.group(3))
        elif line.startswith("Training finished. skipped_backward_batches="):
            match = re.search(r"skipped_backward_batches=(\d+)", line)
            row["skipped_backward_batches"] = int(match.group(1)) if match else None
    row.update(parse_final_sections(lines))
    for prefix in ["best", "primary"]:
        strict = row.get(f"final_{prefix}_strict")
        overall = row.get(f"final_{prefix}_test_overall") or row.get(f"final_{prefix}_overall")
        if strict is not None and overall is not None:
            row[f"final_{prefix}_clean_gap_overall_minus_strict"] = overall - strict
    target_strict = maybe_float(row.get("target_strict"))
    target_overall = maybe_float(row.get("target_overall"))
    row["strict_gap_vs_target"] = None if target_strict is None else (row.get("best_strict") or 0) - target_strict
    row["overall_gap_vs_target"] = None if target_overall is None else (row.get("best_overall") or 0) - target_overall
    if row.get("best_strict") is not None and row.get("final_primary_strict") is not None:
        row["primary_gap_vs_best_strict"] = row["final_primary_strict"] - row["best_strict"]
    epoch_rows = parse_epoch_rows(text, row["run_name"], row["cid"], int(row["shot"]), int(row["gpu"]))
    if epoch_rows:
        row["epoch_rows"] = len(epoch_rows)
        row["max_epoch"] = max(e.epoch for e in epoch_rows)
        vals = [e.val_tx for e in epoch_rows if e.val_tx is not None]
        if vals:
            row["best_val_tx"] = max(vals)
            row["final_val_tx_epoch_log"] = vals[-1]
            row["val_drop_from_best"] = max(vals) - vals[-1]
            tail = vals[-30:]
            row["late_val_std"] = pstdev(tail) if len(tail) > 1 else 0.0
        tests = [e.test_udu for e in epoch_rows if e.test_udu is not None]
        if tests:
            row["latest_eval_strict"] = tests[-1]
            row["best_eval_strict"] = max(tests)
            row["latest_strict_drop_from_eval_best"] = max(tests) - tests[-1]
    return row, epoch_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    out["n"] = len(rows)
    out["by_shot"] = dict(sorted(Counter(int(r["shot"]) for r in rows).items()))
    if any(r.get("ratio") is not None for r in rows):
        out["by_ratio"] = dict(sorted(Counter(str(r.get("ratio")) for r in rows).items()))
    if any(r.get("strategy") is not None for r in rows):
        out["by_strategy"] = dict(sorted(Counter(str(r.get("strategy")) for r in rows).items()))
    out["by_axis"] = dict(sorted(Counter(r["axis"] for r in rows).items()))
    out["finished"] = sum(1 for r in rows if r.get("finished"))
    out["errors"] = sum(int(r.get("error_count") or 0) for r in rows)
    for metric in ["best_strict", "best_overall", "best_primary_score", "best_worst_rx", "final_primary_sat_floor", "final_primary_rx_floor"]:
        vals = [r.get(metric) for r in rows if r.get(metric) is not None]
        if vals:
            out[f"{metric}_max"] = max(vals)
            out[f"{metric}_mean"] = mean(vals)
    return out


def rank_rows(rows: list[dict[str, Any]], metric: str, limit: int = 12) -> list[dict[str, Any]]:
    filtered = [r for r in rows if r.get(metric) is not None]
    return sorted(filtered, key=lambda r: (r.get(metric) if r.get(metric) is not None else -999), reverse=True)[:limit]


def group_summary(rows: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)
    out = []
    for key, items in grouped.items():
        rec: dict[str, Any] = {group_key: key, "n": len(items)}
        for metric in ["best_strict", "best_overall", "best_primary_score", "best_worst_rx", "final_primary_sat_floor", "final_primary_rx_floor", "strict_gap_vs_target"]:
            vals = [r.get(metric) for r in items if r.get(metric) is not None]
            if vals:
                rec[f"{metric}_mean"] = mean(vals)
                rec[f"{metric}_max"] = max(vals)
                best = max(items, key=lambda r: r.get(metric) if r.get(metric) is not None else -999)
                rec[f"{metric}_best_run"] = best["cid"]
        out.append(rec)
    return sorted(out, key=lambda r: str(r[group_key]))


def render_report(out_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    best_by_shot = []
    for shot in sorted(set(int(r["shot"]) for r in rows)):
        shot_rows = [r for r in rows if int(r["shot"]) == shot]
        best = max(shot_rows, key=lambda r: r.get("best_strict") if r.get("best_strict") is not None else -999)
        best_by_shot.append(best)
    lines = [
        "# CEN51 Domain Metric Controller Full Log Analysis",
        "",
        "## Evidence scope",
        "",
        f"- Parsed candidate logs: {summary['n']}",
        f"- Finished runs: {summary['finished']}",
        f"- Full stdout-derived error markers: {summary['errors']}",
        f"- Outputs: `{out_dir}`",
        "",
        "## Best strict by shot",
        "",
        "| K | cid | axis/action | best strict | best overall | primary score | worst rx | primary sat floor | target strict | strict gap |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_by_shot:
        lines.append(
            f"| {row['shot']} | `{row['cid']}` | {row['axis']}/{row['action']} | "
            f"{fmt(row.get('best_strict'))} | {fmt(row.get('best_overall'))} | "
            f"{fmt(row.get('best_primary_score'))} | {fmt(row.get('best_worst_rx'))} | "
            f"{fmt(row.get('final_primary_sat_floor'))} | {fmt(row.get('target_strict'))} | "
            f"{fmt(row.get('strict_gap_vs_target'))} |"
        )
    lines.extend(["", "## Top 12 by strict UDU", "", "| rank | K | cid | axis/action | strict | overall | primary score | rx floor | sat floor |", "|---:|---:|---|---|---:|---:|---:|---:|---:|"])
    for i, row in enumerate(rank_rows(rows, "best_strict", 12), 1):
        lines.append(
            f"| {i} | {row['shot']} | `{row['cid']}` | {row['axis']}/{row['action']} | "
            f"{fmt(row.get('best_strict'))} | {fmt(row.get('best_overall'))} | "
            f"{fmt(row.get('best_primary_score'))} | {fmt(row.get('final_primary_rx_floor'))} | "
            f"{fmt(row.get('final_primary_sat_floor'))} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-json", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    matrix = read_matrix(args.matrix_json)
    rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    missing = []
    for meta in matrix:
        path = args.log_dir / f"{meta['run_name']}.out"
        if not path.exists():
            missing.append(meta["run_name"])
            continue
        row, epochs = parse_log(path, meta)
        rows.append(row)
        epoch_rows.extend([vars(e) for e in epochs])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "run_summary.csv", rows)
    write_csv(args.out_dir / "epoch_metrics.csv", epoch_rows)
    write_csv(args.out_dir / "shot_summary.csv", group_summary(rows, "shot"))
    if any(r.get("ratio") is not None for r in rows):
        write_csv(args.out_dir / "ratio_summary.csv", group_summary(rows, "ratio"))
    if any(r.get("strategy") is not None for r in rows):
        write_csv(args.out_dir / "strategy_summary.csv", group_summary(rows, "strategy"))
    write_csv(args.out_dir / "axis_summary.csv", group_summary(rows, "axis"))
    write_csv(args.out_dir / "action_summary.csv", group_summary(rows, "action"))
    write_csv(args.out_dir / "top_strict.csv", rank_rows(rows, "best_strict", 30))
    write_csv(args.out_dir / "top_primary_score.csv", rank_rows(rows, "best_primary_score", 30))
    write_csv(args.out_dir / "top_sat_floor.csv", rank_rows(rows, "final_primary_sat_floor", 30))
    summary = summarize(rows)
    summary["missing_logs"] = missing
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.out_dir / "analysis_report.md").write_text(render_report(args.out_dir, rows, summary), encoding="utf-8")
    print(json.dumps({"parsed": len(rows), "epoch_rows": len(epoch_rows), "missing": missing, "out_dir": str(args.out_dir)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
