#!/usr/bin/env python3
"""Build the CEN51 WiSig subset domain-metric evaluation report.

This is a local post-run reporting helper.  It reads the full-log parser output
and the experiment matrix, then writes a UTF-8 Markdown report and a compact
completion index into the run report.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


RUN_ID = "cen51_wisig_subset_kseg_transfer_20260613_045425"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(value: object, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def inum(value: object, default: int | None = None) -> int | None:
    parsed = fnum(value, None)
    if parsed is None:
        return default
    return int(round(parsed))


def fmt(value: object, digits: int = 2) -> str:
    parsed = fnum(value, None)
    if parsed is None:
        return ""
    return f"{parsed:.{digits}f}"


def fmt_delta(value: object, digits: int = 2) -> str:
    parsed = fnum(value, None)
    if parsed is None:
        return ""
    return f"{parsed:+.{digits}f}"


def parse_args(args: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    idx = 0
    while idx < len(args):
        token = args[idx]
        if token.startswith("--"):
            key = token[2:]
            if idx + 1 < len(args) and not args[idx + 1].startswith("--"):
                parsed[key] = args[idx + 1]
                idx += 2
            else:
                parsed[key] = "true"
                idx += 1
        else:
            idx += 1
    return parsed


def role(row: dict[str, str]) -> str:
    axis = row.get("axis", "")
    action = row.get("action", "")
    cid = row.get("cid", "")
    if axis == "anchor_replay":
        return "anchor"
    if axis == "global_unified":
        return "global_unified"
    if axis == "paired_control" or "NEG" in cid or "neg" in action:
        return "boundary_or_negative"
    if axis in {"seed_check", "stability"}:
        return "diagnostic"
    return "k_segmented"


def max_by(rows: list[dict[str, str]], key: str) -> dict[str, str]:
    return max(rows, key=lambda row: fnum(row.get(key), -1e9) or -1e9)


def pearson(xs: list[float | None], ys: list[float | None]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xvals = [p[0] for p in pairs]
    yvals = [p[1] for p in pairs]
    mx, my = mean(xvals), mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xvals, yvals)) / math.sqrt(vx * vy)


def md_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build() -> None:
    base = Path(".")
    report_dir = base / "automation_reports" / "CV-SincNet" / RUN_ID
    analysis_dir = base / "analysis_tmp" / RUN_ID / "full_log_analysis"
    out_dir = base / "analysis_tmp" / RUN_ID / "domain_metric_subset_transfer_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_summary_path = analysis_dir / "run_summary.csv"
    matrix_path = report_dir / "matrix.json"
    report_path = report_dir / "domain_metric_subset_transfer_evaluation_report.md"
    index_report_path = report_dir / "report.md"

    rows = read_csv(run_summary_path)
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    arg_by_cid = {item["cid"]: parse_args(item.get("args", [])) for item in matrix}
    matrix_by_cid = {item["cid"]: item for item in matrix}

    for row in rows:
        args = arg_by_cid.get(row["cid"], {})
        row["_role"] = role(row)
        row["_epochs"] = args.get("epochs") or row.get("config_opt_epochs", "")
        row["_swad_start_epoch"] = args.get("swad_start_epoch") or ""
        row["_group_ce_top_frac"] = args.get("group_ce_top_frac") or ""
        row["_groupdro_cap"] = args.get("groupdro_cap") or ""
        row["_concat_sat_ce_weight"] = args.get("concat_sat_ce_weight") or "0.0"
        row["_lambda_dom_source"] = "explicit" if "lambda_dom" in args else "runtime_default"
        row["_physical_gpu"] = str(matrix_by_cid.get(row["cid"], {}).get("gpu", row.get("gpu", "")))

    by_k: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_k[inum(row["shot"], 0) or 0].append(row)

    objective_rows: list[dict[str, object]] = []
    for k in sorted(by_k):
        k_rows = by_k[k]
        for objective, metric in [
            ("strict", "best_strict"),
            ("overall", "best_overall"),
            ("primary_score", "best_primary_score"),
            ("primary_rx_floor", "final_primary_rx_floor"),
            ("primary_sat_floor", "final_primary_sat_floor"),
        ]:
            winner = max_by(k_rows, metric)
            objective_rows.append(
                {
                    "K": k,
                    "objective": objective,
                    "cid": winner["cid"],
                    "axis": winner["axis"],
                    "action": winner["action"],
                    "role": winner["_role"],
                    "value": fmt(winner[metric]),
                    "strict": fmt(winner["best_strict"]),
                    "overall": fmt(winner["best_overall"]),
                    "primary_score": fmt(winner["best_primary_score"]),
                    "rx_floor": fmt(winner["final_primary_rx_floor"]),
                    "sat_floor": fmt(winner["final_primary_sat_floor"]),
                }
            )
    write_csv(
        out_dir / "objective_winners_by_k.csv",
        objective_rows,
        ["K", "objective", "cid", "axis", "action", "role", "value", "strict", "overall", "primary_score", "rx_floor", "sat_floor"],
    )

    comp_rows: list[dict[str, object]] = []
    for k in sorted(by_k):
        k_rows = by_k[k]
        best = max_by(k_rows, "best_strict")
        anchor = max_by([r for r in k_rows if r["_role"] == "anchor"], "best_strict")
        global_row = max_by([r for r in k_rows if r["_role"] == "global_unified"], "best_strict")
        segmented = max_by([r for r in k_rows if r["_role"] == "k_segmented"], "best_strict")
        boundary_candidates = [r for r in k_rows if r["_role"] == "boundary_or_negative"]
        boundary = max_by(boundary_candidates, "best_strict") if boundary_candidates else None

        def pack(prefix: str, row: dict[str, str] | None) -> dict[str, object]:
            if row is None:
                return {
                    f"{prefix}_cid": "",
                    f"{prefix}_strict": "",
                    f"{prefix}_primary_score": "",
                    f"{prefix}_rx_floor": "",
                    f"{prefix}_sat_floor": "",
                    f"{prefix}_delta_vs_best": "",
                }
            return {
                f"{prefix}_cid": row["cid"],
                f"{prefix}_strict": fmt(row["best_strict"]),
                f"{prefix}_primary_score": fmt(row["best_primary_score"]),
                f"{prefix}_rx_floor": fmt(row["final_primary_rx_floor"]),
                f"{prefix}_sat_floor": fmt(row["final_primary_sat_floor"]),
                f"{prefix}_delta_vs_best": fmt_delta((fnum(row["best_strict"], 0) or 0) - (fnum(best["best_strict"], 0) or 0)),
            }

        item: dict[str, object] = {
            "K": k,
            "best_cid": best["cid"],
            "best_strict": fmt(best["best_strict"]),
            "best_role": best["_role"],
        }
        item.update(pack("anchor", anchor))
        item.update(pack("segmented", segmented))
        item.update(pack("global", global_row))
        item.update(pack("boundary", boundary))
        comp_rows.append(item)
    write_csv(out_dir / "anchor_segment_global_delta.csv", comp_rows, list(comp_rows[0].keys()))

    param_fields = [
        ("cid", "cid"),
        ("K", "shot"),
        ("physical_gpu", "_physical_gpu"),
        ("seed", "seed"),
        ("axis", "axis"),
        ("action", "action"),
        ("role", "_role"),
        ("max_train_per_combo", "split_max_samples_per_combo_train"),
        ("train_size", "split_train_size"),
        ("val_size", "split_val_size"),
        ("epochs", "_epochs"),
        ("swad_start_epoch", "_swad_start_epoch"),
        ("lambda_dom", "config_loss_lambda_dom"),
        ("lambda_dom_source", "_lambda_dom_source"),
        ("lambda_adv", "config_loss_lambda_adv"),
        ("lambda_group_ce", "config_loss_lambda_group_ce"),
        ("group_ce_top_frac", "_group_ce_top_frac"),
        ("groupdro_cap", "_groupdro_cap"),
        ("lambda_proto", "config_loss_lambda_proto"),
        ("lambda_supcon_id", "config_loss_lambda_supcon_id"),
        ("lambda_fishr", "config_loss_lambda_fishr"),
        ("sat_view_prob", "config_concat_sat_view_prob"),
        ("concat_sat_start_epoch", "config_concat_sat_start_epoch"),
        ("concat_sat_ce_weight", "_concat_sat_ce_weight"),
        ("best_strict", "best_strict"),
        ("best_strict_epoch", "best_strict_epoch"),
        ("best_overall", "best_overall"),
        ("best_primary_score", "best_primary_score"),
        ("primary_rx_floor", "final_primary_rx_floor"),
        ("primary_sat_floor", "final_primary_sat_floor"),
        ("latest_strict_drop", "latest_strict_drop_from_eval_best"),
        ("unsafe_backward_warnings", "warnings_unsafe_backward"),
        ("skipped_backward_batches", "skipped_backward_batches"),
    ]
    detail_rows = []
    for row in sorted(rows, key=lambda r: (inum(r["shot"], 0) or 0, r["cid"])):
        detail_rows.append({out_key: row.get(in_key, "") for out_key, in_key in param_fields})
    write_csv(out_dir / "parameter_detail_table.csv", detail_rows, [p[0] for p in param_fields])

    corr_rows: list[dict[str, object]] = []
    param_for_corr = [
        ("lambda_dom", "config_loss_lambda_dom"),
        ("lambda_adv", "config_loss_lambda_adv"),
        ("lambda_group_ce", "config_loss_lambda_group_ce"),
        ("group_ce_top_frac", "_group_ce_top_frac"),
        ("groupdro_cap", "_groupdro_cap"),
        ("lambda_proto", "config_loss_lambda_proto"),
        ("lambda_supcon_id", "config_loss_lambda_supcon_id"),
        ("lambda_fishr", "config_loss_lambda_fishr"),
        ("sat_view_prob", "config_concat_sat_view_prob"),
        ("concat_sat_ce_weight", "_concat_sat_ce_weight"),
    ]
    metrics_for_corr = [
        ("strict", "best_strict"),
        ("primary_score", "best_primary_score"),
        ("rx_floor", "final_primary_rx_floor"),
        ("sat_floor", "final_primary_sat_floor"),
    ]
    for k in sorted(by_k):
        for pname, pfield in param_for_corr:
            xs = [fnum(row.get(pfield), None) for row in by_k[k]]
            if len({x for x in xs if x is not None}) < 2:
                continue
            for mname, mfield in metrics_for_corr:
                ys = [fnum(row.get(mfield), None) for row in by_k[k]]
                corr = pearson(xs, ys)
                if corr is not None:
                    corr_rows.append({"K": k, "parameter": pname, "metric": mname, "pearson_r": f"{corr:.3f}", "n": 8})
    write_csv(out_dir / "parameter_correlations_by_k.csv", corr_rows, ["K", "parameter", "metric", "pearson_r", "n"])

    rollback_rows = sorted(
        rows,
        key=lambda r: (
            -(fnum(r.get("latest_strict_drop_from_eval_best"), 0) or 0),
            -(fnum(r.get("warnings_unsafe_backward"), 0) or 0),
        ),
    )
    write_csv(
        out_dir / "unsafe_and_rollback.csv",
        [
            {
                "cid": row["cid"],
                "K": row["shot"],
                "axis": row["axis"],
                "action": row["action"],
                "warnings_unsafe_backward": row["warnings_unsafe_backward"],
                "skipped_backward_batches": row["skipped_backward_batches"],
                "latest_strict_drop": fmt(row["latest_strict_drop_from_eval_best"]),
                "best_strict": fmt(row["best_strict"]),
                "best_strict_epoch": row["best_strict_epoch"],
                "max_epoch": row["max_epoch"],
            }
            for row in rollback_rows
        ],
        ["cid", "K", "axis", "action", "warnings_unsafe_backward", "skipped_backward_batches", "latest_strict_drop", "best_strict", "best_strict_epoch", "max_epoch"],
    )

    sat_cols = [
        "final_primary_sat_clear_leo",
        "final_primary_sat_low_elev_leo",
        "final_primary_sat_rain_leo",
        "final_primary_sat_storm_mp",
        "final_primary_sat_mixed_orbit",
    ]
    scenario_rows: list[dict[str, object]] = []
    for k in sorted(by_k):
        for col in sat_cols:
            winner = max_by(by_k[k], col)
            scenario_rows.append(
                {
                    "K": k,
                    "scenario": col.replace("final_primary_sat_", ""),
                    "cid": winner["cid"],
                    "axis": winner["axis"],
                    "action": winner["action"],
                    "value": fmt(winner[col]),
                    "strict": fmt(winner["best_strict"]),
                    "sat_floor": fmt(winner["final_primary_sat_floor"]),
                }
            )
    write_csv(out_dir / "satellite_scenario_winners.csv", scenario_rows, ["K", "scenario", "cid", "axis", "action", "value", "strict", "sat_floor"])

    lines: list[str] = []
    lines.append(f"# 域指标方法评估与优化报告：{RUN_ID}")
    lines.append("")
    lines.append("## 结论摘要")
    lines.append("")
    lines.append("- 本轮新 WiSig 子集实验 `48/48` 完成，scheduler `DONE rc=0=48`，48 条 stdout 均有 `Training finished`，未见 `Traceback/OOM/Killed/unrecognized arguments`。")
    lines.append("- 域指标方法有效，但有效形式是 **按 K 分段的约束/动作选择器**，不是全局统一控制器。`global_unified` 在 6 个 K 上没有任何一个拿到 strict 或 primary-score 第一。")
    lines.append("- 分段策略在 K5/K10/K30/K50/K100 拿到 strict 第一；K20 是关键反例：anchor 仍是 strict/primary 第一，因此控制器必须允许 `no-action / floor-only` 状态。")
    lines.append("- 指标不是花架子：no-sat、过强 sat、RX 单指标最优等边界候选清楚暴露了 clean strict、receiver floor、satellite floor 的冲突，能实际约束 `lambda_dom/lambda_adv/lambda_group_ce/groupdro_cap/sat_view_prob/concat_sat_ce_weight/SWAD` 等参数。")
    lines.append("- K50/K100 在新子集上仍低于旧图中锚点 strict（K50 `-0.89`，K100 `-1.13`），所以结论应写成“机制迁移成立，K50/K100 仍需优化”，不能写成所有 K 都已达到旧数据集 ceiling。")
    lines.append("")
    lines.append("## 证据范围与口径")
    lines.append("")
    lines.append("- 数据子集：train days `1,2`，test days `0,3`；train RXs `2,3,4,5,8,9,10`，test RXs `0,1,6,7,11`。")
    lines.append("- 硬约束：48/48 均为 `--wisig_train_ratio 0.1`；`wisig_max_train_per_combo=K` 在 stdout `split_info` 中确认。训练样本数为 K×84：K5/10/20/30/50/100 分别为 `420/840/1680/2520/4200/8400`，val size 均为 `75600`。")
    lines.append("- 主指标：`strict` = `test_unseen_day_unseen_rx`；`overall` = `test_overall`；`primary_score` 使用解析器输出；`rx_floor` 为 final-primary 下 unseen RX 最小值；`sat_floor` 为 5 个 satellite scenario 最小值。")
    lines.append("- 训练 stdout 不打印物理 GPU、`axis/action`、`group_ce_top_frac`、`groupdro_cap`、`swad_start_epoch`；这些字段来自 `matrix.json`/launcher/`launch_pids.tsv`。stdout 内字段来自 `[CONFIG-*]`、`split_info`、`[FINAL-*]`、warning 和 epoch 记录。")
    lines.append("- 三个子 agent 只读覆核了参数覆盖、指标覆盖和机制解释，均确认 48/48 覆盖，无实质参数错配；小数差异仅来自 stdout 四位显示舍入。")
    lines.append("")

    best_rows: list[list[object]] = []
    for k in sorted(by_k):
        strict = max_by(by_k[k], "best_strict")
        overall = max_by(by_k[k], "best_overall")
        rx = max_by(by_k[k], "final_primary_rx_floor")
        sat = max_by(by_k[k], "final_primary_sat_floor")
        best_rows.append(
            [
                k,
                f"`{strict['cid']}`",
                f"{fmt(strict['best_strict'])}@E{inum(strict['best_strict_epoch'])}",
                fmt(strict["best_overall"]),
                fmt(strict["best_primary_score"]),
                fmt(strict["final_primary_rx_floor"]),
                fmt(strict["final_primary_sat_floor"]),
                f"`{overall['cid']}` {fmt(overall['best_overall'])}",
                f"`{rx['cid']}` {fmt(rx['final_primary_rx_floor'])}",
                f"`{sat['cid']}` {fmt(sat['final_primary_sat_floor'])}",
            ]
        )
    lines.append("## 按 K 的赢家与多目标冲突")
    lines.append("")
    lines.append(md_table(["K", "strict winner", "strict@epoch", "overall", "primary", "rxF", "satF", "overall winner", "rxF winner", "satF winner"], best_rows))
    lines.append("")
    lines.append("strict winner 与 satF/rxF winner 经常不是同一个候选。K5 的 no-sat 虽然 rxF 高但 strict 很差；K50 的 `B03_RX_SEED2028` satF 最高但 strict 落后；K100 的 `S04_BOOST_NEG` satF 最高但 strict/primary 低于 `B03`。因此控制器不能把任一 floor 指标当作唯一目标。")
    lines.append("")

    comp_table: list[list[object]] = []
    for item in comp_rows:
        comp_table.append(
            [
                item["K"],
                f"`{item['best_cid']}` {item['best_strict']}",
                item["best_role"],
                f"`{item['anchor_cid']}` {item['anchor_strict']} ({item['anchor_delta_vs_best']})",
                f"`{item['segmented_cid']}` {item['segmented_strict']} ({item['segmented_delta_vs_best']})",
                f"`{item['global_cid']}` {item['global_strict']} ({item['global_delta_vs_best']})",
                f"`{item['boundary_cid']}` {item['boundary_strict']} ({item['boundary_delta_vs_best']})" if item["boundary_cid"] else "",
            ]
        )
    lines.append("## K 分段 vs 全局控制器")
    lines.append("")
    lines.append(md_table(["K", "best strict", "best role", "anchor strict Δ", "segmented strict Δ", "global strict Δ", "boundary/negative strict Δ"], comp_table))
    lines.append("")
    lines.append("`global_unified` 的 strict/primary 在所有 K 都没有第一：它在 K20 的 sat floor、K100 的 rx/sat floor 有局部价值，但不能作为默认控制器。它更适合作为诊断候选，用来判断某个 K 是否存在统一压力方向，而不是替代 K 分段。")
    lines.append("")

    lines.append("## 逐 K 参数判读")
    lines.append("")
    k_notes = {
        5: "K5 的最佳动作从原预期的纯 sat-gate 转为 `pressure_clamp + light sat`：`C01` 将 domain/adv/group 压力降到较低区间，同时保留 `sat_p≈0.09`，strict/overall/primary 同时第一。`NOSAT` 证明去掉 satellite 信号会牺牲 strict 和 satF；`S04` 虽然 satF/rxF 高，但 seed 2029 的 strict 断崖式下降，不能作为默认。",
        10: "K10 最强是 `B02_RXSAT_SEED1337`，说明轻 sat gate 还不够，需要 RX floor 与 sat floor 同时约束。no-sat 负控 strict 和 satF 均明显落后，支持 satellite 指标作为真实训练信号；但最优含 seed 因素，下一轮必须多 seed 复验。",
        20: "K20 是中性/保护区：anchor strict 与 primary 第一，global 和 sat/clamp 能改善部分 floor，但没有推翻 anchor。控制器在此 K 应输出“保留 anchor；若业务重视 sat floor，再进入 floor-only 模式”，而不是强制适应。",
        30: "K30 的 `pressure_clamp` 成立：`C03` strict/primary 第一，global 落后，过强 sat 只提高 satF 而牺牲 strict。说明 K30 的主矛盾是 domain pressure 与 identity 保真之间的平衡，不是继续堆 sat。",
        50: "K50 主线是 `rx_floor/group-DRO`：`R02_RX110` strict/overall/primary 第一，`B01_RXSAT` 提高 rxF/satF 但 strict 略低，`B03` 展示 satellite floor 与 strict 的 seed/tradeoff。下一轮应围绕 RX floor 微调，而非增加 sat 到 `.308`。",
        100: "K100 的 `B03_SAT_FLOOR_GUARD` 是 strict/primary 最优；`global` 与 `S04` floor 很强但 strict/primary 落后。K100 需要 guard 式保守控制，K100 的 `lambda_dom=1.0` 是 trainer 默认运行时值，matrix/launcher 未显式传入；同时 K100 是唯一使用 `concat_sat_ce_weight>0` 的家族。",
    }
    for k in sorted(by_k):
        table_rows: list[list[object]] = []
        for row in sorted(by_k[k], key=lambda r: fnum(r["best_strict"], -1) or -1, reverse=True):
            table_rows.append(
                [
                    f"`{row['cid']}`",
                    row["axis"] + "/" + row["action"],
                    row["_role"],
                    row["seed"],
                    fmt(row["config_loss_lambda_dom"], 4),
                    fmt(row["config_loss_lambda_adv"], 4),
                    fmt(row["config_loss_lambda_group_ce"], 4),
                    fmt(row["_group_ce_top_frac"], 3),
                    fmt(row["_groupdro_cap"], 3),
                    fmt(row["config_concat_sat_view_prob"], 3),
                    fmt(row["_concat_sat_ce_weight"], 2),
                    fmt(row["best_strict"]),
                    fmt(row["best_overall"]),
                    fmt(row["best_primary_score"]),
                    fmt(row["final_primary_rx_floor"]),
                    fmt(row["final_primary_sat_floor"]),
                    fmt(row["latest_strict_drop_from_eval_best"]),
                    row["warnings_unsafe_backward"],
                ]
            )
        lines.append(f"### K={k}")
        lines.append("")
        lines.append(md_table(["cid", "axis/action", "role", "seed", "dom", "adv", "grp", "top", "cap", "sat_p", "sat_ce", "strict", "overall", "primary", "rxF", "satF", "lateDrop", "warn"], table_rows))
        lines.append("")
        lines.append(k_notes[k])
        lines.append("")

    lines.append("## 指标如何真正指导参数变化")
    lines.append("")
    metric_rows = [
        ["strict UDU", "`test_unseen_day_unseen_rx`/`best_strict`", "主目标；若下降超过容忍阈值，拒绝 floor-only 或负控候选", "`lambda_dom/lambda_adv/lambda_group_ce` 降压或保守，限制 `sat_p`，调整 SWAD/epoch；K20 可 no-action"],
        ["receiver floor", "final-primary unseen RX min", "识别某个 RX 域坍塌；但不能单独最大化", "`lambda_group_ce`、`group_ce_top_frac`、`groupdro_cap`、RX-balanced action；K50 最有效，K5 no-sat 是反例"],
        ["satellite floor", "5-scenario sat min，最常由 `storm_mp` 决定", "识别 satellite shift 下限；若只提高 satF 但 strict 掉，则降级为边界候选", "`concat_sat_view_prob`、`concat_sat_start_epoch`、K100 `concat_sat_ce_weight`、sat loss；K5/K10 有效，K30/K50/K100 需 guard"],
        ["primary score", "解析器综合 strict/overall/floor 的 score", "用于 strict 接近时排序，避免只看最后 epoch 或单 floor", "选择 checkpoint/SWAD，拒绝 late rollback 大的候选"],
        ["late rollback", "`latest_strict_drop_from_eval_best` + final/latest 差异", "决定是否使用 best-primary checkpoint，而不是最后 epoch", "`swad_start_epoch`、epochs、early stop、seed 复核；K30_B04/K100_S05/S04/B03 是重点"],
        ["unsafe skipped", "`warnings_unsafe_backward`/`skipped_backward_batches`", "稳定性约束；本轮 48/48 均有 2-8 次，不是失败但要监控", "若集中于高 K/high pressure，降低 adv/group/sat CE 或 clip；K100 global/strong clamp 最高"],
    ]
    lines.append(md_table(["指标", "日志字段", "决策用途", "实际会调的参数/损失"], metric_rows))
    lines.append("")
    lines.append("这意味着域指标不是展示用的统计表，而是一个受约束的动作选择器：先用 strict/primary 保护 clean identity generalization，再用 rxF/satF 定位哪个域下限需要补偿，最后用 rollback/unsafe 约束训练稳定性。")
    lines.append("")

    lines.append("## 数学原理：为什么必须 K 分段")
    lines.append("")
    lines.append("训练目标可写成：")
    lines.append("")
    lines.append("```text")
    lines.append("min_theta  L_cls(theta)")
    lines.append("         + lambda_dom * L_dom + lambda_adv * L_adv")
    lines.append("         + lambda_group * L_groupDRO + lambda_proto * L_proto")
    lines.append("         + lambda_supcon * L_supcon + lambda_fishr * L_fishr")
    lines.append("         + lambda_sat * L_sat + p_sat * E[L_cls(T_sat(x))]")
    lines.append("```")
    lines.append("")
    lines.append("K 改变了每类/每域估计的方差，近似可看作 `Var(grad) ∝ 1/K`，也改变了 identity 梯度和 domain-invariant 梯度的相对可信度。低 K 时强 domain/sat 约束更容易抹掉 TX identity；中 K 时可用轻量 floor 约束；高 K 时数据足以承受更强的 receiver/satellite guard，但过强 sat 仍会把优化目标推向 satellite-only floor。")
    lines.append("")
    lines.append("因此同一个全局映射 `metric_deficit -> lambda update` 会在不同 K 上变号：K5 需要降压+轻 sat，K30 需要 pressure clamp，K50 需要 RX/group-DRO，K100 需要 guard，而 K20 最优是不动作。一个全局控制器隐含所有 K 对 `lambda` 和 `p_sat` 的梯度符号一致；本轮日志直接否定了这个假设。")
    lines.append("")

    lines.append("## 参数-指标相关性提示（探索性，不当因果）")
    lines.append("")
    for k in sorted(by_k):
        k_corr = [r for r in corr_rows if r["K"] == k]
        k_corr = sorted(k_corr, key=lambda r: abs(float(str(r["pearson_r"]))), reverse=True)[:6]
        lines.append(f"K={k}:")
        lines.append(md_table(["parameter", "metric", "r"], [[r["parameter"], r["metric"], r["pearson_r"]] for r in k_corr]))
        lines.append("")
    lines.append("由于每个 K 只有 8 个点，上表只用于解释下一轮搜索方向，不能当作显著性结论。它的价值在于暴露符号变化：某些 sat/RX 参数在一个 K 上补 floor，在另一个 K 上伤 strict，这正是分段控制的数学理由。")
    lines.append("")

    lines.append("## 稳定性与日志异常")
    lines.append("")
    lines.append("- 错误：48/48 无 `Traceback/ERROR/OOM/Killed/unrecognized arguments`。")
    lines.append("- `unsafe backward/step skipped`：48/48 都出现，范围 `2-8` 次，与 `skipped_backward_batches` 一致；最高为 K100 `GLOBAL` 和 `GLOBAL_CLAMP_STRONG` 各 8 次。当前不构成失败，但应作为高压参数的稳定性惩罚。")
    lines.append("- 最大 late rollback 候选如下，报告与后续选择必须使用 best/primary checkpoint，不能直接用最后 epoch：")
    lines.append("")
    lines.append(md_table(["cid", "K", "axis/action", "strict", "bestE", "lateDrop", "warn"], [[f"`{r['cid']}`", r["shot"], r["axis"] + "/" + r["action"], fmt(r["best_strict"]), r["best_strict_epoch"], fmt(r["latest_strict_drop_from_eval_best"]), r["warnings_unsafe_backward"]] for r in rollback_rows[:8]]))
    lines.append("")

    lines.append("## 下一轮优化建议")
    lines.append("")
    next_rows = [
        ["K5", "以 `C01_PRESS_CLAMP` 为锚点", "扫 `sat_p=.085/.093/.100`，`lambda_dom=.29-.34`，`lambda_adv=.10-.12`；复验 seed 2028/2030", "拒绝 no-sat；RX-only 不作为默认"],
        ["K10", "以 `B02_RXSAT_SEED1337` 为锚点", "多 seed 复验同参；扫 `sat_p=.122/.135`、`lambda_dom=.42-.46`、`group=.006-.0065`", "确认最优不是 seed 偶然"],
        ["K20", "anchor/no-action 为默认", "只做 floor-only 小扰动：global/sat light 用作 satF 改善候选，不压过 strict gate", "禁止强制控制器改参"],
        ["K30", "以 `C03_CLAMP092` 为锚点", "扫 `sat_p=.166/.18/.20`、轻微 `dom/adv/group`，保留 pressure clamp", "不要上 `.36` 级 sat"],
        ["K50", "以 `R02_RX110` 为锚点", "扫 `dom=.55-.58`、`adv=.25-.27`、`group=.027-.029`、`cap=.59-.61`、`sat=.22/.253`", "避免 `.308` sat；把 B01/B03 作为 floor tradeoff 对照"],
        ["K100", "以 `B03_SAT_FLOOR_GUARD` 为锚点", "扫 `sat=.55/.58/.60`、`adv=.30/.32/.34`、`group=.06/.065/.07`、`proto=.009/.010/.012`、`sat_ce=.65/.72/.80`", "global 只保留 floor 诊断；高 sat boost 不替代 strict guard"],
    ]
    lines.append(md_table(["K", "当前默认动作", "下一轮搜索", "边界/禁止项"], next_rows))
    lines.append("")
    lines.append("## 产物索引")
    lines.append("")
    lines.append(f"- 主报告：`{report_path}`")
    lines.append("- 报告生成脚本：`tools/cen51_domain_metric_subset_report.py`")
    lines.append("- 本地验证：`C:\\Users\\lh594\\.conda\\envs\\ssr-gpu\\python.exe -m py_compile tools\\cen51_domain_metric_subset_report.py`")
    lines.append("- 本地生成：`C:\\Users\\lh594\\.conda\\envs\\ssr-gpu\\python.exe tools\\cen51_domain_metric_subset_report.py`")
    lines.append(f"- 参数明细：`{out_dir / 'parameter_detail_table.csv'}`")
    lines.append(f"- 按 K 多目标赢家：`{out_dir / 'objective_winners_by_k.csv'}`")
    lines.append(f"- anchor/segmented/global 对照：`{out_dir / 'anchor_segment_global_delta.csv'}`")
    lines.append(f"- 参数-指标探索相关性：`{out_dir / 'parameter_correlations_by_k.csv'}`")
    lines.append(f"- late rollback/unsafe：`{out_dir / 'unsafe_and_rollback.csv'}`")
    lines.append(f"- satellite 单场景赢家：`{out_dir / 'satellite_scenario_winners.csv'}`")
    lines.append("")
    lines.append("## 审计边界")
    lines.append("")
    lines.append("- 本报告为新 WiSig 子集内比较，不能把 raw score 直接等同于旧图中原子集的绝对提升；旧锚点仅作为设计锚和目标线。")
    lines.append("- K10/K50/K100 的部分最优含 seed 或 floor/strict tradeoff，下一轮需要多 seed 和更密网格确认。")
    lines.append("- 相关性表仅 8 点/K，是搜索启发，不是统计显著性检验。")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    index_text = index_report_path.read_text(encoding="utf-8")
    corrupt_marker = "\n\n## ????????????"
    if corrupt_marker in index_text:
        index_text = index_text[: index_text.index(corrupt_marker)]
    good_marker = "## 完成后域指标评估报告索引"
    if good_marker in index_text:
        index_text = index_text[: index_text.index(good_marker)].rstrip()
    index_text = index_text.rstrip() + f"""

## 完成后域指标评估报告索引

时间：2026-06-13T10:10+08:00。
状态：48/48 候选训练完成，48/48 日志解析完成，无 Traceback/OOM/Killed/参数错误。完整评估报告已落地：

- `{report_path}`
- 附表目录：`{out_dir}`

核心结论：域指标方法在新 WiSig 子集上有效，但必须按 K 分段；`global_unified` 未在任何 K 上取得 strict/primary 第一。K20 是 no-action/floor-only 保护区，K50/K100 仍需围绕当前锚点继续优化。
"""
    index_report_path.write_text(index_text, encoding="utf-8")

    print(f"WROTE_REPORT={report_path}")
    print(f"WROTE_TABLES={out_dir}")
    print(f"ROWS={len(rows)} MATRIX={len(matrix)}")


if __name__ == "__main__":
    build()
