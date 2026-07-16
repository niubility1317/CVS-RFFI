#!/usr/bin/env python
"""Audit and summarize a completed strict ADV3B02 class-incremental matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SCENARIOS = ("leo_clear_weak", "leo_low_elev_weak", "leo_rain_weak")
METRICS = (
    "old_acc_before_increment",
    "old_acc_after_increment",
    "seen_new_acc",
    "H_old_new",
    "candidate_average_forgetting",
    "direct_adv3b02_old_acc",
    "identity_old_acc_after_increment",
    "old_after_minus_direct_adv3b02",
    "old_after_minus_identity",
    "min_old_class_acc",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    return float(statistics.fmean(values))


def _std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _group_metrics(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        item = {key: value for key, value in zip(keys, group_key)}
        item["scenario_rows"] = len(group_rows)
        item["cells"] = len({row["cell_id"] for row in group_rows})
        for metric in METRICS:
            values = [float(row[metric]) for row in group_rows]
            item[f"{metric}_mean"] = _mean(values)
            item[f"{metric}_std"] = _std(values)
        result.append(item)
    return result


def _group_resources(
    rows: list[dict[str, Any]], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for group_key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        item = {key: value for key, value in zip(keys, group_key)}
        item["cells"] = len(group_rows)
        for field in (
            "trainable_parameters",
            "persistent_state_bytes",
            "optimizer_steps_total",
            "adaptation_wall_seconds_total",
            "peak_cuda_memory_bytes",
            "support_backbone_forward_samples",
            "query_backbone_forward_samples",
            "query_view_count",
        ):
            values = [float(row[field]) for row in group_rows]
            item[f"{field}_mean"] = _mean(values)
            item[f"{field}_max"] = max(values)
        result.append(item)
    return result


def summarize(plan_path: Path, output_root: Path) -> dict[str, Any]:
    plan_path = plan_path.resolve(strict=True)
    plan = _load(plan_path)
    if plan.get("schema") != "cvs.phase2.adv3b02_ci_strict_plan.v1":
        raise ValueError("strict plan schema drift")
    if plan.get("launch_authority") is not True:
        raise ValueError("summary requires a smoke-authorized plan")
    cells = plan.get("cells")
    if not isinstance(cells, list) or len(cells) != 900:
        raise ValueError("strict plan must contain exactly 900 cells")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite summary root: {output_root}")

    expected_cell_ids = {str(cell["cell_id"]) for cell in cells}
    if len(expected_cell_ids) != 900:
        raise ValueError("cell IDs are not unique")
    formal_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    receipt_hashes: dict[str, str] = {}
    protocol_failures: list[str] = []

    for cell in cells:
        cell_id = str(cell["cell_id"])
        cell_root = Path(cell["output_root"])
        cell_receipt_path = cell_root / "cell_receipt.json"
        metric_path = cell_root / "scoring" / "formal_rows.json"
        predictor_receipt_path = cell_root / "predictor" / "predictor_receipt.json"
        for required in (cell_receipt_path, metric_path, predictor_receipt_path):
            if not required.is_file():
                raise FileNotFoundError(f"missing completed-cell artifact: {required}")

        cell_receipt = _load(cell_receipt_path)
        if cell_receipt.get("status") != "PROTOCOL_VALID":
            raise ValueError(f"cell is not protocol-valid: {cell_id}")
        if cell_receipt.get("formal_rows_sha256") != _sha256(metric_path):
            raise ValueError(f"formal row SHA mismatch: {cell_id}")
        receipt_hashes[cell_id] = _sha256(cell_receipt_path)

        metric_document = _load(metric_path)
        if metric_document.get("schema") != "cvs.phase2.formal_metric_rows.v1":
            raise ValueError(f"formal row schema drift: {cell_id}")
        rows = metric_document.get("rows")
        if not isinstance(rows, list) or len(rows) != 3:
            raise ValueError(f"cell does not contain three scenario rows: {cell_id}")
        if {row.get("scenario") for row in rows} != set(SCENARIOS):
            raise ValueError(f"scenario coverage drift: {cell_id}")
        for row in rows:
            if row.get("row_id") != cell_id:
                raise ValueError(f"row ID drift: {cell_id}")
            enriched = dict(row)
            enriched.update(
                {
                    "cell_id": cell_id,
                    "method": cell["method"],
                    "receiver": cell["receiver"],
                    "seed": int(cell["seed"]),
                    "new_class_count": int(cell["new_class_count"]),
                    "k_shot": int(cell["k_shot"]),
                }
            )
            enriched["old_after_minus_direct_adv3b02"] = float(
                enriched["old_acc_after_increment"]
            ) - float(enriched["direct_adv3b02_old_acc"])
            enriched["old_after_minus_identity"] = float(
                enriched["old_acc_after_increment"]
            ) - float(enriched["identity_old_acc_after_increment"])
            formal_rows.append(enriched)

        predictor = _load(predictor_receipt_path)
        required_guards = {
            "status": "PROTOCOL_VALID",
            "backbone": "ADV3B02",
            "backbone_frozen": True,
            "clean_sample_access": False,
            "clean_derived_signal_access": False,
            "phase2_clean_dataset_reachable": False,
            "phase2_clean_cache_reachable": False,
            "phase2_clean_control_flow_reachable": False,
            "phase2_query_role_oracle_access": False,
            "phase2_query_true_batch_class_count_access": False,
            "phase2_query_class_quota_access": False,
            "phase2_query_batch_global_assignment": False,
            "dense_query_graph_used": False,
            "query_rows_used_for_training": 0,
            "query_members_opened_before_head_lock": False,
            "query_labels_available_to_predictor": False,
        }
        for field, expected in required_guards.items():
            if predictor.get(field) != expected:
                protocol_failures.append(
                    f"{cell_id}:{field}={predictor.get(field)!r}, expected {expected!r}"
                )
        scenario_resources = predictor.get("candidate_resources_by_scenario", [])
        if len(scenario_resources) != 3:
            raise ValueError(f"resource scenario coverage drift: {cell_id}")
        resource_rows.append(
            {
                "cell_id": cell_id,
                "method": cell["method"],
                "receiver": cell["receiver"],
                "seed": int(cell["seed"]),
                "new_class_count": int(cell["new_class_count"]),
                "k_shot": int(cell["k_shot"]),
                "trainable_parameters": max(
                    int(row["trainable_parameters"]) for row in scenario_resources
                ),
                "persistent_state_bytes": max(
                    int(row["persistent_state_bytes"]) for row in scenario_resources
                ),
                "optimizer_steps_total": sum(
                    int(row["optimizer_steps"]) for row in scenario_resources
                ),
                "adaptation_wall_seconds_total": sum(
                    float(row["adaptation_wall_seconds"]) for row in scenario_resources
                ),
                "peak_cuda_memory_bytes": int(predictor["peak_cuda_memory_bytes"]),
                "support_backbone_forward_samples": int(
                    predictor["support_backbone_forward_samples"]
                ),
                "query_backbone_forward_samples": int(
                    predictor["query_backbone_forward_samples"]
                ),
                "query_view_count": int(predictor["query_view_count"]),
            }
        )

    if len(formal_rows) != 2700:
        raise ValueError("formal scenario-row count is not 2700")
    if protocol_failures:
        raise ValueError("protocol guard failure: " + "; ".join(protocol_failures[:10]))

    group_method_k_new = _group_metrics(
        formal_rows, ("method", "k_shot", "new_class_count")
    )
    group_method_k = _group_metrics(formal_rows, ("method", "k_shot"))
    group_receiver = _group_metrics(formal_rows, ("method", "receiver"))
    resources = _group_resources(
        resource_rows, ("method", "k_shot", "new_class_count")
    )
    best_groups = sorted(
        group_method_k_new,
        key=lambda row: (
            -float(row["H_old_new_mean"]),
            -float(row["old_acc_after_increment_mean"]),
            -float(row["seen_new_acc_mean"]),
        ),
    )

    output_root.mkdir(parents=True, exist_ok=False)
    _write_csv(output_root / "formal_rows.csv", formal_rows)
    _write_csv(output_root / "resource_rows.csv", resource_rows)
    _write_csv(output_root / "summary_by_method_k_new.csv", group_method_k_new)
    _write_csv(output_root / "summary_by_method_k.csv", group_method_k)
    _write_csv(output_root / "summary_by_method_receiver.csv", group_receiver)
    _write_csv(output_root / "resources_by_method_k_new.csv", resources)

    audit = {
        "schema": "cvs.phase2.adv3b02_ci_matrix_summary.v1",
        "status": "PASS",
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "smoke_receipt_sha256": plan["smoke_receipt_sha256"],
        "expected_cell_count": 900,
        "completed_cell_count": len(receipt_hashes),
        "formal_scenario_row_count": len(formal_rows),
        "method_count": len({row["method"] for row in formal_rows}),
        "receiver_count": len({row["receiver"] for row in formal_rows}),
        "seed_count": len({row["seed"] for row in formal_rows}),
        "k_values": sorted({row["k_shot"] for row in formal_rows}),
        "new_class_counts": sorted({row["new_class_count"] for row in formal_rows}),
        "scenarios": sorted({row["scenario"] for row in formal_rows}),
        "protocol_guard_failures": [],
        "result_claim_boundary": (
            "protocol-valid class-incremental matrix; superiority claims require the "
            "matched Stage2-C MRIOR-SDA baseline"
        ),
        "best_group_by_mean_H": best_groups[0],
        "cell_receipt_hashes": receipt_hashes,
    }
    _write_json(output_root / "summary_audit.json", audit)

    def pct(value: Any) -> str:
        return f"{100.0 * float(value):.2f}%"

    md = [
        "# ADV3B02类增量严格矩阵汇总",
        "",
        f"- 覆盖：{len(receipt_hashes)}个cell、{len(formal_rows)}个场景行。",
        "- 协议：LEO weak-only、无clean、无角色Oracle/类别配额/全局分配、逐样本全注册类决策。",
        "- 声明边界：matched Stage2-C MRIOR-SDA完成前，不声明优于域适应基线。",
        "",
        "## 方法×K×新类数",
        "",
        "|方法|K|新类数|old_before|old_after|new_acc|H|遗忘|相对直接ADV3B02旧类|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in group_method_k_new:
        md.append(
            "|{method}|{k_shot}|{new_class_count}|{old_before}|{old_after}|"
            "{new_acc}|{harmonic}|{forgetting}|{delta}|".format(
                method=row["method"],
                k_shot=row["k_shot"],
                new_class_count=row["new_class_count"],
                old_before=pct(row["old_acc_before_increment_mean"]),
                old_after=pct(row["old_acc_after_increment_mean"]),
                new_acc=pct(row["seen_new_acc_mean"]),
                harmonic=pct(row["H_old_new_mean"]),
                forgetting=pct(row["candidate_average_forgetting_mean"]),
                delta=pct(row["old_after_minus_direct_adv3b02_mean"]),
            )
        )
    md.extend(
        [
            "",
            "## 资源口径",
            "",
            "|方法|K|新类数|可训练参数均值|持久状态最大值|三场景optimizer step均值|三场景适配秒数均值|峰值显存最大值|",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in resources:
        md.append(
            "|{method}|{k_shot}|{new_class_count}|{params:.0f}|{state:.0f}B|"
            "{steps:.0f}|{seconds:.3f}|{memory:.0f}B|".format(
                method=row["method"],
                k_shot=row["k_shot"],
                new_class_count=row["new_class_count"],
                params=row["trainable_parameters_mean"],
                state=row["persistent_state_bytes_max"],
                steps=row["optimizer_steps_total_mean"],
                seconds=row["adaptation_wall_seconds_total_mean"],
                memory=row["peak_cuda_memory_bytes_max"],
            )
        )
    (output_root / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.plan, args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
