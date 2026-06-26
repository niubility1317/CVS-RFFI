"""Organize CV-SincNet training logs by experiment semantics.

The tool is intentionally read-only for source logs. It builds a catalog that
uses training route, model/config, objective, split, and metrics as primary
keys instead of treating collection date as the organizing axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shlex
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping


FIELDNAMES = [
    "source_path",
    "source_kind",
    "training_route",
    "family",
    "model_key",
    "experiment_id",
    "config",
    "run_name",
    "objective",
    "train_mode",
    "client_key",
    "train_ratio",
    "domain",
    "dataset",
    "model_variant",
    "exp_group",
    "fl_sat_aug_mode",
    "fedprox_mu",
    "lambda_fishr",
    "lambda_sat_cls",
    "seed",
    "status",
    "records_parsed",
    "last_step",
    "best_step",
    "best_val_acc",
    "latest_val_acc",
    "best_overall_acc",
    "latest_overall_acc",
    "best_strict_udu_acc",
    "best_strict_udu_step",
    "latest_strict_udu_acc",
    "sat_clear_leo_strict_udu",
    "sat_low_elev_leo_strict_udu",
    "sat_rain_leo_strict_udu",
    "sat_storm_mp_strict_udu",
    "sat_mixed_orbit_strict_udu",
    "warnings",
    "run_dir",
    "cmd",
]

GROUP_FIELDNAMES = [
    "training_route",
    "family",
    "model_key",
    "objective",
    "train_mode",
    "client_key",
    "train_ratio",
    "domain",
    "model_variant",
    "exp_group",
    "fl_sat_aug_mode",
    "source_count",
    "best_strict_udu_acc",
    "best_overall_acc",
    "latest_strict_udu_acc",
    "best_step",
    "last_step",
    "sat_clear_leo_strict_udu",
    "sat_low_elev_leo_strict_udu",
    "sat_rain_leo_strict_udu",
    "sat_storm_mp_strict_udu",
    "sat_mixed_orbit_strict_udu",
    "best_source_path",
    "all_source_paths",
]

LOG_ROOTS = [
    "logs",
    "code/logs",
    "server_log_backups",
    "baselines/baseline_runs",
    "5.14-logs",
    "5.15logs",
    "5.16logs",
    "5.17logs",
    "5.18logs",
    "5.19logs",
    "5.20-adapt-logs",
]

SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "analysis",
    "analysis_tmp",
    "conversation_index",
    "paper",
    "paper_reading_work",
    "tmp",
    "code/snapshots",
}

TIMESTAMP_SUFFIX_RE = re.compile(r"_(?:20\d{6})(?:_\d{6})?$")
PERCENT_RE = r"([-+]?\d+(?:\.\d+)?)%"
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _blank_record() -> dict[str, str]:
    return {name: "" for name in FIELDNAMES}


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return str(value)
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_number(text: str | None) -> str:
    if not text:
        return ""
    match = re.search(NUMBER_RE, text)
    return match.group(0) if match else ""


def _strip_timestamp(stem: str) -> str:
    previous = None
    current = stem
    while previous != current:
        previous = current
        current = TIMESTAMP_SUFFIX_RE.sub("", current)
    return current


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_key_values(text: str) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_\-]*)=(.*)$", line.strip())
        if match:
            kv[match.group(1).replace("-", "_")] = match.group(2).strip()
    return kv


def _parse_flags(cmd: str) -> dict[str, str]:
    if not cmd:
        return {}
    try:
        tokens = shlex.split(cmd, posix=True)
    except ValueError:
        tokens = cmd.split()
    flags: dict[str, str] = {}
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.startswith("--"):
            key = token[2:].replace("-", "_")
            value = "true"
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                value = tokens[i + 1]
                i += 1
            flags[key] = value.replace("\\,", ",")
        i += 1
    return flags


def _route_from_context(path: Path, flags: Mapping[str, str], text: str = "") -> str:
    joined = f"{path.as_posix()} {path.stem} {text[:5000]}".lower()
    train_mode = flags.get("train_mode", "").lower()
    stem = path.stem.lower()
    if "scheduler" in stem:
        return "scheduler"
    if path.name == "logs.jsonl" or train_mode.startswith("fed") or "fedavg" in joined or "fedprox" in joined or "federated" in joined:
        return "federated"
    rel = path.as_posix().lower()
    stem = path.stem.lower()
    baseline_stems = ("cvcnn", "riei", "drift", "tifs2025", "receiver_agnostic", "ra_collab")
    baseline_dirs = ("baseline_runs", "baseline_supervised_rx_curriculum", "baseline_pseudo_rx_curriculum")
    if any(part in rel for part in baseline_dirs) or stem.startswith(baseline_stems):
        return "comparison_baseline"
    if train_mode == "centralized" or " train.py " in f" {joined} " or "python3 -u train.py" in joined:
        return "centralized"
    if path.suffix.lower() == ".out" and stem.startswith("nohup"):
        return "scheduler"
    if path.suffix.lower() == ".log":
        return "centralized"
    if path.suffix.lower() == ".out":
        return "centralized"
    return "unknown"


def _family(model_key: str, path: Path, objective: str = "") -> str:
    model = model_key.lower()
    text = f"{model_key} {path.as_posix()} {objective}".lower()
    if "fl82" in text:
        return "FL82"
    if "fsdg" in text or "fed_fewshot" in text:
        return "FSDG"
    if "fedcvs" in text or "vmb" in text:
        return "FedCVS/VMB"
    if "split_bex02" in text:
        return "split_bex02"
    if "bex02" in text:
        return "BEX02"
    if re.search(r"\bbex\d+", text) or re.search(r"^bbf\d+", model):
        return "BEX"
    if "sat" in text and "baseline" not in text:
        return "satellite_ablation"
    if model.startswith("cvcnn"):
        return "cvcnn"
    if model.startswith("riei"):
        return "riei"
    if model.startswith("drift"):
        return "drift"
    if model.startswith("tifs2025"):
        return "tifs2025"
    if model.startswith("receiver_agnostic") or model.startswith("ra_collab"):
        return "receiver_agnostic"
    return "other"


def _model_from_path(path: Path, kv: Mapping[str, str], flags: Mapping[str, str]) -> str:
    for key in ["EXP_ID", "CONFIG"]:
        if kv.get(key):
            return kv[key]
    if flags.get("run_name"):
        return flags["run_name"]
    stem = _strip_timestamp(path.stem)
    if "_seed" in stem:
        stem = stem.split("_seed", 1)[0]
    return stem


def _status_from_text(text: str) -> tuple[str, str]:
    warnings = []
    lowered = text.lower()
    if "traceback" in lowered:
        warnings.append("traceback")
    if "out of memory" in lowered or "cuda oom" in lowered:
        warnings.append("oom")
    if re.search(r"\bkilled\b", lowered):
        warnings.append("killed")
    if "nan" in lowered:
        warnings.append("nan")
    if "training finished" in lowered:
        status = "finished"
    elif warnings:
        status = "warning"
    else:
        status = "parsed"
    return status, ",".join(sorted(set(warnings)))


def _last_match(pattern: str, text: str, flags: int = 0) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def _best_round(rows: Iterable[Mapping[str, Any]], key: str) -> tuple[float | None, Mapping[str, Any] | None]:
    best_value: float | None = None
    best_row: Mapping[str, Any] | None = None
    for row in rows:
        value = _float(row.get(key))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_row = row
    return best_value, best_row


def parse_stdout_log(path: Path, project_root: Path) -> dict[str, str]:
    text = _read_text(path)
    kv = _parse_key_values(text)
    cmd = kv.get("CMD", "")
    flags = _parse_flags(cmd)

    record = _blank_record()
    record["source_path"] = _rel(path, project_root)
    record["source_kind"] = "stdout_log"
    record["training_route"] = _route_from_context(path, flags, text)
    record["model_key"] = _model_from_path(path, kv, flags)
    record["experiment_id"] = kv.get("EXP_ID", "")
    record["config"] = kv.get("CONFIG", "")
    record["run_name"] = flags.get("run_name", "")
    record["objective"] = flags.get("fl_local_objective", "")
    record["train_mode"] = flags.get("train_mode", "centralized" if record["training_route"] == "centralized" else "")
    record["client_key"] = flags.get("fl_client_key", "")
    record["train_ratio"] = flags.get("wisig_train_ratio", "")
    record["domain"] = flags.get("wisig_domain", "")
    record["dataset"] = flags.get("dataset", "")
    record["model_variant"] = flags.get("model_variant", "")
    record["exp_group"] = flags.get("exp_group", "")
    record["fl_sat_aug_mode"] = flags.get("fl_sat_aug_mode", "")
    record["fedprox_mu"] = flags.get("fedprox_mu", "")
    record["lambda_fishr"] = flags.get("lambda_fishr", "")
    record["lambda_sat_cls"] = flags.get("lambda_sat_cls", "")
    record["seed"] = flags.get("seed", "")
    record["run_dir"] = kv.get("RUN_DIR", flags.get("output_dir", ""))
    record["cmd"] = cmd
    record["family"] = _family(record["model_key"], path, record["objective"])
    record["status"], record["warnings"] = _status_from_text(text)

    epoch_matches = list(re.finditer(r"\[EPOCH-END\]\s+E(\d+)/(\d+)", text))
    record["records_parsed"] = _fmt(len(epoch_matches))
    if epoch_matches:
        record["last_step"] = epoch_matches[-1].group(1)

    match = _last_match(rf"best_joint_val_tx_acc={PERCENT_RE}", text)
    if match:
        record["best_val_acc"] = match.group(1)
    match = _last_match(rf"\[FINAL-(?:BEST|PRIMARY)\]\s+val_tx={PERCENT_RE}", text)
    if match:
        record["latest_val_acc"] = match.group(1)

    match = _last_match(rf"best_test_overall_tx_acc={PERCENT_RE}\s+at epoch\s+(\d+)", text)
    if match:
        record["best_overall_acc"] = match.group(1)
        record["best_step"] = match.group(2)
    match = _last_match(rf"\[FINAL-(?:BEST|PRIMARY)\].*?test_overall_tx={PERCENT_RE}", text)
    if match:
        record["latest_overall_acc"] = match.group(1)

    match = _last_match(rf"best_unseen_day_unseen_rx_tx_acc={PERCENT_RE}\s+at epoch\s+(\d+)", text)
    if match:
        record["best_strict_udu_acc"] = match.group(1)
        record["best_strict_udu_step"] = match.group(2)
    match = _last_match(rf"\[FINAL-(?:BEST|PRIMARY)\].*?strict_udu={PERCENT_RE}", text)
    if match:
        record["latest_strict_udu_acc"] = match.group(1)

    for sat_match in re.finditer(rf"\[SAT-TEST\]\s+scenario=([a-zA-Z0-9_]+).*?strict_udu={PERCENT_RE}", text):
        scenario = sat_match.group(1)
        key = f"sat_{scenario}_strict_udu"
        if key in record:
            record[key] = sat_match.group(2)

    return record


def parse_federated_jsonl(path: Path, project_root: Path) -> dict[str, str]:
    rows: list[dict[str, Any]] = []
    config: dict[str, Any] = {}
    bad_lines = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if row.get("event") == "fed_config":
                config = row
            if isinstance(row.get("round"), int):
                rows.append(row)
    rows.sort(key=lambda item: int(item.get("round", 0)))
    first = rows[0] if rows else {}
    latest = rows[-1] if rows else {}
    best_strict, best_strict_row = _best_round(rows, "global_strict_udu_acc")
    best_overall, best_overall_row = _best_round(rows, "global_test_overall_acc")
    best_val, best_val_row = _best_round(rows, "global_eval_acc")

    parent = path.parent
    model_key = parent.name
    record = _blank_record()
    record["source_path"] = _rel(path, project_root)
    record["source_kind"] = "logs_jsonl"
    record["training_route"] = "federated"
    record["model_key"] = model_key
    record["family"] = _family(model_key, path, _fmt(first.get("fl_local_objective")))
    record["objective"] = _fmt(latest.get("fl_local_objective") or first.get("fl_local_objective"))
    record["train_mode"] = _fmt(latest.get("train_mode") or first.get("train_mode"))
    record["client_key"] = _fmt(
        (config.get("federated") or {}).get("fl_client_key")
        or latest.get("fl_client_key")
        or first.get("fl_client_key")
    )
    record["fl_sat_aug_mode"] = _fmt(latest.get("fl_sat_aug_mode") or first.get("fl_sat_aug_mode"))
    record["fedprox_mu"] = _fmt(latest.get("fedprox_mu") or first.get("fedprox_mu"))
    record["train_ratio"] = _fmt(
        latest.get("train_ratio")
        or first.get("train_ratio")
        or (config.get("data") or {}).get("wisig_train_ratio")
    )
    record["status"] = "parsed"
    record["warnings"] = "json_decode_errors" if bad_lines else ""
    record["records_parsed"] = _fmt(len(rows))
    record["last_step"] = _fmt(latest.get("round"))
    record["best_val_acc"] = _fmt(best_val)
    record["best_step"] = _fmt((best_overall_row or {}).get("round"))
    record["best_overall_acc"] = _fmt(best_overall)
    record["latest_overall_acc"] = _fmt(latest.get("global_test_overall_acc"))
    record["best_strict_udu_acc"] = _fmt(best_strict)
    record["best_strict_udu_step"] = _fmt((best_strict_row or {}).get("round"))
    record["latest_strict_udu_acc"] = _fmt(latest.get("global_strict_udu_acc"))
    record["latest_val_acc"] = _fmt(latest.get("global_eval_acc"))
    record["run_dir"] = str(parent)

    for row in rows:
        extra = row.get("global_extra_tests") or row.get("extra_tests") or {}
        if not isinstance(extra, Mapping):
            continue
        sat = extra.get("sat_channel") if isinstance(extra.get("sat_channel"), Mapping) else extra
        if not isinstance(sat, Mapping):
            continue
        for scenario, value in sat.items():
            key = f"sat_{scenario}_strict_udu"
            if key in record and isinstance(value, Mapping):
                record[key] = _fmt(value.get("strict_udu") or value.get("udu"))

    return record


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def parse_baseline_metrics_json(path: Path, project_root: Path) -> dict[str, str]:
    data = json.loads(_read_text(path))
    best = data.get("best") if isinstance(data.get("best"), Mapping) else {}
    final = data.get("final") if isinstance(data.get("final"), Mapping) else {}
    model_key = path.parent.name
    if "_seed" in model_key:
        model_key = model_key.split("_seed", 1)[0]

    record = _blank_record()
    record["source_path"] = _rel(path, project_root)
    record["source_kind"] = "metrics_json"
    record["training_route"] = "comparison_baseline"
    record["model_key"] = model_key
    record["family"] = _family(model_key, path)
    record["status"] = "parsed"
    record["records_parsed"] = _fmt(len(data.get("epochs", [])) if isinstance(data.get("epochs"), list) else "")
    record["best_step"] = _fmt(best.get("epoch"))
    record["last_step"] = _fmt(final.get("epoch") or best.get("epoch"))
    record["best_overall_acc"] = _fmt(
        _nested(best, "test", "overall", "tx_acc")
        or _nested(best, "named_tests", "overall", "tx_acc")
        or _nested(best, "test_overall", "tx_acc")
    )
    record["latest_overall_acc"] = _fmt(
        _nested(final, "test", "overall", "tx_acc")
        or _nested(final, "named_tests", "overall", "tx_acc")
        or _nested(final, "test_overall", "tx_acc")
    )
    record["best_strict_udu_acc"] = _fmt(
        _nested(best, "test", "named", "test_unseen_day_unseen_rx", "tx_acc")
        or _nested(best, "named_tests", "test_unseen_day_unseen_rx", "tx_acc")
    )
    record["latest_strict_udu_acc"] = _fmt(
        _nested(final, "test", "named", "test_unseen_day_unseen_rx", "tx_acc")
        or _nested(final, "named_tests", "test_unseen_day_unseen_rx", "tx_acc")
    )
    sat = _nested(best, "extra_tests", "sat_channel")
    if isinstance(sat, Mapping):
        for scenario, value in sat.items():
            key = f"sat_{scenario}_strict_udu"
            if key in record and isinstance(value, Mapping):
                record[key] = _fmt(value.get("strict_udu"))
    return record


def discover_sources(project_root: Path, roots: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for root_name in roots:
        root = project_root / root_name
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel_parts = set(_rel(path, project_root).split("/")[:-1])
            rel_joined = _rel(path, project_root)
            if rel_parts & SKIP_PARTS or any(skip in rel_joined for skip in SKIP_PARTS):
                continue
            if path.name in {"logs.jsonl", "metrics.json"} or path.suffix.lower() in {".log", ".out"}:
                paths.append(path)
    return sorted(set(paths), key=lambda p: _rel(p, project_root))


def parse_source(path: Path, project_root: Path) -> dict[str, str] | None:
    if path.name == "logs.jsonl":
        return parse_federated_jsonl(path, project_root)
    if path.name == "metrics.json":
        return parse_baseline_metrics_json(path, project_root)
    if path.suffix.lower() in {".log", ".out"}:
        return parse_stdout_log(path, project_root)
    return None


def parse_all(project_root: Path, roots: Iterable[str]) -> list[dict[str, str]]:
    records = []
    for path in discover_sources(project_root, roots):
        try:
            record = parse_source(path, project_root)
        except Exception as exc:  # noqa: BLE001 - cataloging should continue.
            record = _blank_record()
            record["source_path"] = _rel(path, project_root)
            record["source_kind"] = path.name if path.name in {"logs.jsonl", "metrics.json"} else path.suffix.lstrip(".")
            record["training_route"] = "parse_error"
            record["status"] = "parse_error"
            record["warnings"] = f"{type(exc).__name__}: {exc}"
        if record is not None:
            records.append(record)
    return records


def write_csv(path: Path, records: Iterable[Mapping[str, str]], fieldnames: list[str] | None = None) -> None:
    fieldnames = fieldnames or FIELDNAMES
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def summarize_groups(records: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    group_fields = [
        "training_route",
        "family",
        "model_key",
        "objective",
        "train_mode",
        "client_key",
        "train_ratio",
        "domain",
        "model_variant",
        "exp_group",
        "fl_sat_aug_mode",
    ]
    buckets: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        key = tuple(record.get(field, "") for field in group_fields)
        buckets[key].append(record)

    grouped: list[dict[str, str]] = []
    for key, rows in buckets.items():
        best = sorted(
            rows,
            key=lambda row: (
                _float(row.get("best_strict_udu_acc")) is not None,
                _float(row.get("best_strict_udu_acc")) or -1.0,
                _float(row.get("best_overall_acc")) or -1.0,
            ),
            reverse=True,
        )[0]
        out = {name: "" for name in GROUP_FIELDNAMES}
        for field, value in zip(group_fields, key):
            out[field] = value
        out["source_count"] = _fmt(len(rows))
        for field in [
            "best_strict_udu_acc",
            "best_overall_acc",
            "latest_strict_udu_acc",
            "best_step",
            "last_step",
            "sat_clear_leo_strict_udu",
            "sat_low_elev_leo_strict_udu",
            "sat_rain_leo_strict_udu",
            "sat_storm_mp_strict_udu",
            "sat_mixed_orbit_strict_udu",
        ]:
            out[field] = best.get(field, "")
        out["best_source_path"] = best.get("source_path", "")
        out["all_source_paths"] = ";".join(sorted(row.get("source_path", "") for row in rows if row.get("source_path")))
        grouped.append(out)
    return _sort_for_leaderboard(grouped)


def _sort_for_leaderboard(records: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    def score(row: Mapping[str, str]) -> tuple[bool, float, float]:
        strict = _float(row.get("best_strict_udu_acc"))
        overall = _float(row.get("best_overall_acc"))
        return (strict is not None or overall is not None, strict if strict is not None else -1.0, overall if overall is not None else -1.0)

    return sorted(
        records,
        key=score,
        reverse=True,
    )


def _md_table(records: list[Mapping[str, str]], columns: list[str], limit: int = 20) -> list[str]:
    if not records:
        return ["No records."]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in records[:limit]:
        values = [str(row.get(col, "")).replace("|", "/") for col in columns]
        lines.append("| " + " | ".join(values) + " |")
    return lines


def write_markdown_report(path: Path, records: list[dict[str, str]], groups: list[dict[str, str]]) -> None:
    route_counts = Counter(record["training_route"] for record in records)
    family_counts = Counter(record["family"] for record in records)
    group_route_counts = Counter(record["training_route"] for record in groups)
    by_route: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in groups:
        by_route[record["training_route"]].append(record)

    lines: list[str] = [
        "# Training Log Catalog",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Primary organization key: training_route -> family -> model_key/objective -> metrics.",
        "Dates remain metadata only; they are not the first sorting axis.",
        "",
        "## Route Counts",
        "",
        "| training_route | records |",
        "| --- | --- |",
    ]
    for route, count in sorted(route_counts.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(["", "## Semantic Group Counts", "", "| training_route | groups |", "| --- | --- |"])
    for route, count in sorted(group_route_counts.items()):
        lines.append(f"| {route} | {count} |")

    lines.extend(
        [
            "",
            "## Family Counts",
            "",
            "| family | records |",
            "| --- | --- |",
        ]
    )
    for family, count in sorted(family_counts.items()):
        lines.append(f"| {family} | {count} |")

    leaderboard_columns = [
        "model_key",
        "family",
        "objective",
        "train_ratio",
        "client_key",
        "best_strict_udu_acc",
        "best_overall_acc",
        "latest_strict_udu_acc",
        "last_step",
        "source_count",
        "best_source_path",
    ]
    for route in ["centralized", "federated", "comparison_baseline"]:
        lines.extend(["", f"## {route} Leaderboard", ""])
        lines.extend(_md_table(_sort_for_leaderboard(by_route.get(route, [])), leaderboard_columns, limit=25))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `log_catalog.csv`: all parsed records.",
            "- `centralized_runs.csv`: centralized CV-SincNet/satellite runs.",
            "- `federated_runs.csv`: FL/FedAvg/FedProx/FedCVS records from structured JSONL or stdout.",
            "- `comparison_model_runs.csv`: baseline comparison models such as CVCNN, RIEI, DRIFT, TIFS2025, receiver-agnostic.",
            "- `model_group_summary.csv`: duplicate-collapsed semantic groups keyed by route/model/objective/split.",
            "",
            "## Column Notes",
            "",
            "- `training_route` separates centralized, federated, comparison baselines, schedulers, and parse errors.",
            "- `family` groups BEX/FL82/FSDG/FedCVS/baseline families independent of date.",
            "- `model_key` is derived from `EXP_ID`, `CONFIG`, `--run_name`, or the run directory.",
            "- `best_strict_udu_acc` is the main cross-domain strict unseen-day/unseen-RX metric when present.",
            "- `sat_*_strict_udu` columns keep satellite-channel scenario results separate from clean metrics.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(project_root: Path, out_dir: Path, records: list[dict[str, str]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = summarize_groups(records)
    write_csv(out_dir / "log_catalog.csv", records)
    route_to_file = {
        "centralized": "centralized_runs.csv",
        "federated": "federated_runs.csv",
        "comparison_baseline": "comparison_model_runs.csv",
    }
    for route, filename in route_to_file.items():
        write_csv(out_dir / filename, [row for row in records if row.get("training_route") == route])
        write_csv(
            out_dir / filename.replace("_runs.csv", "_model_groups.csv").replace("comparison_model_model_groups", "comparison_model_groups"),
            [row for row in groups if row.get("training_route") == route],
            GROUP_FIELDNAMES,
        )
    write_csv(out_dir / "model_group_summary.csv", groups, GROUP_FIELDNAMES)
    write_markdown_report(out_dir / "grouped_index.md", records, groups)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build semantic CV-SincNet training-log catalogs.")
    parser.add_argument("--project-root", default=".", help="Project root to scan.")
    parser.add_argument("--out-dir", default="analysis/training_log_catalog", help="Directory for generated CSV/Markdown outputs.")
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Relative root to scan. Can be supplied multiple times. Defaults to known log roots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    roots = args.roots if args.roots else LOG_ROOTS
    out_dir = (project_root / args.out_dir).resolve()
    records = parse_all(project_root, roots)
    write_outputs(project_root, out_dir, records)
    route_counts = Counter(record["training_route"] for record in records)
    print(f"parsed_records={len(records)}")
    for route, count in sorted(route_counts.items()):
        print(f"{route}={count}")
    print(f"out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
