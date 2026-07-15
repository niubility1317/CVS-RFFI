#!/usr/bin/env python3
"""Validate and summarize a completed strict effective8 matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCENARIOS = {"leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak"}
MEAN_METRICS = (
    "old_acc_before_increment",
    "old_acc_after_increment",
    "direct_adv3b02_old_acc",
    "seen_new_acc_after_increment",
    "H_old_new_after_increment",
    "candidate_average_forgetting",
    "identity_old_acc_after_increment",
    "shared_view_count_mean",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(float(value) for value in values))


def _json_cell(value: Mapping[str, float]) -> str:
    return json.dumps(dict(sorted(value.items())), ensure_ascii=False, sort_keys=True)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path.name}")
    fields = list(rows[0])
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _group(rows: list[dict[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for identity, members in sorted(grouped.items()):
        item = {key: value for key, value in zip(keys, identity)}
        item["cell_count"] = len(members)
        for metric in MEAN_METRICS:
            item[f"mean_{metric}"] = _mean(row[metric] for row in members)
        item["mean_delta_before_vs_direct"] = _mean(
            row["delta_before_vs_direct"] for row in members
        )
        item["mean_delta_after_vs_direct"] = _mean(
            row["delta_after_vs_direct"] for row in members
        )
        item["min_old_class_acc_after_global"] = min(
            row["min_old_class_acc_after_global"] for row in members
        )
        item["max_candidate_average_forgetting"] = max(
            row["candidate_average_forgetting"] for row in members
        )
        result.append(item)
    return result


def summarize(run_root: Path, output_dir: Path, *, expected_cells: int = 300) -> dict[str, Any]:
    run_root = run_root.resolve(strict=True)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite summary directory: {output_dir}")
    cell_dirs = sorted(path for path in (run_root / "cells").iterdir() if path.is_dir())
    if len(cell_dirs) != expected_cells:
        raise ValueError(f"strict cell directory count drift: {len(cell_dirs)} != {expected_cells}")

    cells: list[dict[str, Any]] = []
    scenarios: list[dict[str, Any]] = []
    for cell_dir in cell_dirs:
        receipt = _read_json(cell_dir / "cell_receipt.json")
        rows_payload = _read_json(cell_dir / "scoring_output" / "formal_rows.json")
        resource = _read_json(
            cell_dir / "predictor_output" / "predictor_resource_receipt.json"
        )
        rows = rows_payload.get("rows") if isinstance(rows_payload, dict) else None
        if receipt.get("status") != "PROTOCOL_VALID" or receipt.get("cell_id") != cell_dir.name:
            raise ValueError(f"cell receipt binding drift: {cell_dir.name}")
        if not isinstance(rows, list) or len(rows) != 3:
            raise ValueError(f"formal row count drift: {cell_dir.name}")
        if {row.get("scenario") for row in rows} != SCENARIOS:
            raise ValueError(f"formal scenario coverage drift: {cell_dir.name}")
        if any(row.get("row_id") != cell_dir.name for row in rows):
            raise ValueError(f"formal row/cell binding drift: {cell_dir.name}")
        if any(int(row.get("k_shot", -1)) != int(receipt["k_shot"]) for row in rows):
            raise ValueError(f"formal K binding drift: {cell_dir.name}")
        if any(row.get("receiver_label") != receipt["receiver"] for row in rows):
            raise ValueError(f"formal receiver binding drift: {cell_dir.name}")
        if resource.get("schema") != "cvs.phase2.predictor_resource_receipt.v2":
            raise ValueError(f"resource receipt schema drift: {cell_dir.name}")

        old_after_labels = sorted(rows[0]["candidate_old_class_acc_after_increment"])
        if any(
            sorted(row["candidate_old_class_acc_after_increment"]) != old_after_labels
            for row in rows
        ):
            raise ValueError(f"old-class identity drift: {cell_dir.name}")
        old_after_mean = {
            label: _mean(row["candidate_old_class_acc_after_increment"][label] for row in rows)
            for label in old_after_labels
        }
        old_before_mean = {
            label: _mean(row["candidate_old_class_acc_before_increment"][label] for row in rows)
            for label in old_after_labels
        }
        forgetting_mean = {
            label: _mean(row["candidate_old_class_forgetting"][label] for row in rows)
            for label in old_after_labels
        }
        cell = {
            "cell_id": cell_dir.name,
            "receiver": str(receipt["receiver"]),
            "seed": int(receipt["seed"]),
            "new_class_count": int(receipt["new_class_count"]),
            "k_shot": int(receipt["k_shot"]),
        }
        for metric in MEAN_METRICS:
            cell[metric] = _mean(row[metric] for row in rows)
        cell["delta_before_vs_direct"] = (
            cell["old_acc_before_increment"] - cell["direct_adv3b02_old_acc"]
        )
        cell["delta_after_vs_direct"] = (
            cell["old_acc_after_increment"] - cell["direct_adv3b02_old_acc"]
        )
        cell["delta_after_vs_identity"] = (
            cell["old_acc_after_increment"] - cell["identity_old_acc_after_increment"]
        )
        cell["min_old_class_acc_after_global"] = min(
            value
            for row in rows
            for value in row["candidate_old_class_acc_after_increment"].values()
        )
        cell["old_class_acc_before_mean_json"] = _json_cell(old_before_mean)
        cell["old_class_acc_after_mean_json"] = _json_cell(old_after_mean)
        cell["old_class_forgetting_mean_json"] = _json_cell(forgetting_mean)
        for name in (
            "trainable_parameters",
            "adapt_epochs",
            "persistent_state_bytes",
            "peak_cuda_memory_bytes",
            "candidate_query_latency_ms",
            "mean_backbone_forwards",
            "p95_backbone_forwards",
            "view1_rate",
            "view3_rate",
            "view5_rate",
        ):
            cell[name] = resource[name]
        cells.append(cell)
        for row in rows:
            scenarios.append(
                {
                    "cell_id": cell_dir.name,
                    "receiver": str(receipt["receiver"]),
                    "seed": int(receipt["seed"]),
                    "new_class_count": int(receipt["new_class_count"]),
                    "k_shot": int(receipt["k_shot"]),
                    **row,
                }
            )

    if len(scenarios) != expected_cells * 3:
        raise ValueError("strict formal scenario row count drift")
    identities = {
        (row["receiver"], row["seed"], row["new_class_count"], row["k_shot"])
        for row in cells
    }
    if len(identities) != expected_cells:
        raise ValueError("strict cell identity duplication")

    by_new_k = _group(cells, ("new_class_count", "k_shot"))
    by_receiver_new_k = _group(cells, ("receiver", "new_class_count", "k_shot"))
    output_dir.mkdir(parents=True)
    paths = {
        "cell_summary_json": output_dir / "cell_summary.json",
        "cell_summary_csv": output_dir / "cell_summary.csv",
        "scenario_rows_json": output_dir / "scenario_rows.json",
        "by_new_k_json": output_dir / "by_new_k.json",
        "by_new_k_csv": output_dir / "by_new_k.csv",
        "by_receiver_new_k_json": output_dir / "by_receiver_new_k.json",
        "by_receiver_new_k_csv": output_dir / "by_receiver_new_k.csv",
        "cell_summary_md": output_dir / "cell_summary.md",
    }
    _write_json(paths["cell_summary_json"], {"schema": "cvs.effective8.strict_cell_summary.v1", "rows": cells})
    _write_csv(paths["cell_summary_csv"], cells)
    _write_json(paths["scenario_rows_json"], {"schema": "cvs.effective8.strict_scenario_rows.v1", "rows": scenarios})
    _write_json(paths["by_new_k_json"], {"schema": "cvs.effective8.strict_by_new_k.v1", "rows": by_new_k})
    _write_csv(paths["by_new_k_csv"], by_new_k)
    _write_json(paths["by_receiver_new_k_json"], {"schema": "cvs.effective8.strict_by_receiver_new_k.v1", "rows": by_receiver_new_k})
    _write_csv(paths["by_receiver_new_k_csv"], by_receiver_new_k)
    with paths["cell_summary_md"].open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("# effective8 strict v13 300-cell result table\n\n")
        handle.write("| cell | receiver | seed | new | K | old before | old after | seen-new | H | floor | forgetting | delta after vs direct | views | verdict |\n")
        handle.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in cells:
            verdict = "joint-negative" if row["delta_after_vs_direct"] < 0 else "joint-nonnegative"
            handle.write(
                f"| {row['cell_id']} | {row['receiver']} | {row['seed']} | {row['new_class_count']} | {row['k_shot']} | "
                f"{row['old_acc_before_increment']:.4f} | {row['old_acc_after_increment']:.4f} | "
                f"{row['seen_new_acc_after_increment']:.4f} | {row['H_old_new_after_increment']:.4f} | "
                f"{row['min_old_class_acc_after_global']:.4f} | {row['candidate_average_forgetting']:.4f} | "
                f"{row['delta_after_vs_direct']:.4f} | {row['mean_backbone_forwards']:.3f} | {verdict} |\n"
            )

    hashes = {name: _sha256(path) for name, path in paths.items()}
    audit = {
        "schema": "cvs.effective8.strict_matrix_summary_audit.v1",
        "status": "PASS",
        "run_root": str(run_root),
        "cell_count": len(cells),
        "formal_scenario_row_count": len(scenarios),
        "by_new_k_count": len(by_new_k),
        "by_receiver_new_k_count": len(by_receiver_new_k),
        "output_sha256": hashes,
    }
    _write_json(output_dir / "audit.json", audit)
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-cells", type=int, default=300)
    args = parser.parse_args(argv)
    print(json.dumps(summarize(args.run_root, args.output_dir, expected_cells=args.expected_cells), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
