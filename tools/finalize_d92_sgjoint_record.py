"""Append the frozen SGJoint repeated-benchmark outcome; never overwrite raw data.

Dry-run by default. --write exclusively creates two artifacts.json files and
appends final evidence to the existing reports. It never touches registry state.
"""
import argparse
import datetime
import json
import math
from pathlib import Path


METHODS = ("D92", "D92-SGJoint-v1")
COHORTS = {"rx3": (7236, 3600), "rx1": (2412, 1200)}
MARKER = "<!-- SGJOINT_FINAL_ANALYSIS_20260928 -->"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return math.fsum(values) / len(values)


def metrics(rows):
    result = []
    for k in (1, 5, 10, 20):
        all_cells = {m: [r for r in rows if r["method"] == m and r["k"] == k] for m in METHODS}
        joint = {m: [r for r in rr if r["new_count"] > 0] for m, rr in all_cells.items()}
        delta = {f: 100 * (mean(joint[METHODS[1]], f) - mean(joint[METHODS[0]], f))
                 for f in ("old_accuracy", "new_accuracy", "harmonic_mean")}
        old = {m: mean(rr, "old_accuracy") for m, rr in all_cells.items()}
        result.append(dict(k=k, joint_cells=len(joint[METHODS[0]]), joint_delta_pp=delta,
                           old_guard_all_cells=dict(cells=len(all_cells[METHODS[0]]),
                               baseline=old[METHODS[0]], candidate=old[METHODS[1]],
                               delta_pp=100 * (old[METHODS[1]] - old[METHODS[0]]),
                               pass_guard=old[METHODS[1]] - old[METHODS[0]] >= -0.01)))
    return result


def table(values):
    lines = ["| K | Δ旧类（联合） | Δ新类 | ΔH | Δ旧类（全部注册任务） |",
             "|---|---:|---:|---:|---:|"]
    for v in values:
        d = v["joint_delta_pp"]
        lines.append(f'| {v["k"]} | {d["old_accuracy"]:+.3f} | {d["new_accuracy"]:+.3f} | '
                     f'{d["harmonic_mean"]:+.3f} | {v["old_guard_all_cells"]["delta_pp"]:+.3f} |')
    return "\n".join(lines)


def prepare(workspace):
    root = workspace / "automation_reports/CV-SincNet"
    combined = root / "20260928-phase2-d92-sgjoint-repeat-rx3-m4-r01/results/combined_rx4"
    interpretation = read(combined / "interpretation_audit.json")
    assert interpretation["status"] == "VERIFIED" and interpretation["records"] == 9648
    pending = []
    all_rows = []
    for cohort, (records, fits) in COHORTS.items():
        run = f"20260928-phase2-d92-sgjoint-repeat-{cohort}-m4-r01"
        folder = root / run
        result = folder / "results"
        output = result / "artifacts.json"
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifact: {output}")
        scores, complete = read(result / "scores.json"), read(result / "complete.json")
        startup, fit = read(result / "startup.json"), read(result / "fit_audit.json")
        arithmetic, summary = read(result / "arithmetic_audit.json"), read(result / "summary/summary.json")
        assert scores["status"] == complete["status"] == "SCORED"
        assert len(scores["results"]) == complete["records"] == summary["rows"] == records
        assert arithmetic["status"] == fit["status"] == "VERIFIED"
        assert fit["total_fits"] == fits == sum(m["fits"] for m in fit["models"])
        assert fit["query_rows_used_for_fit"] == fit["source_rows_read"] == fit["new_ground_statistics_bytes"] == 0
        assert fit["optimizer_convergence_claim"] is False
        assert startup["commit"] == complete["commit"]
        assert startup["spec"]["run_id"] == run
        assert all(m["payload_files"]["numeric_array_bytes"] == 4410 for m in fit["models"])
        assert all(m["payload_files"]["already_deployed"] is False for m in fit["models"])
        assert all(m["payload_files"]["incremental_transfer_bytes"] == m["payload_files"]["total_file_bytes"] for m in fit["models"])
        seconds = math.fsum(m["total_fit_seconds"] for m in fit["models"])
        states = [s for m in fit["models"] for s in m["state_bytes_by_classes"].values()]
        state_ranges = {k: [min(s[k] for s in states), max(s[k] for s in states)]
                        for k in ("head_bytes", "summary_operator_bytes", "persistent_state_bytes", "covariance_matrix_bytes")}
        per_k = metrics(scores["results"])
        report = folder / "report.md"
        if MARKER in report.read_text(encoding="utf-8"):
            raise ValueError(f"Final analysis already appended: {report}")
        for file in (result / "summary/report.md", result / "fit_logs", result / "arithmetic_audit.json"):
            assert file.exists(), file
        remote = "/home/szu2070436088/2510044040/CV-SincNet/runs/" + run
        artifacts = dict(run_id=run, status="ANALYZED", execution_status="SCORED", evidence_status="VERIFIED",
            candidate=METHODS[1], candidate_promoted=False, goal_complete=False,
            release_commit=startup["commit"], records=records, paired_cells=fits,
            raw_scores=dict(local=str(result / "scores.json"), remote=remote + "/scores.json", bytes=(result / "scores.json").stat().st_size),
            raw_predictions=remote + "/<row_id>/sgjoint/predictions.jsonl",
            baseline_predictions=[dict(row_id=r["row_id"], path=r["reuse_row_root"] + "/predictions.jsonl") for r in startup["spec"]["rows"]],
            raw_fit_traces=[m["trace_file"] for m in fit["models"]],
            audits=dict(arithmetic="arithmetic_audit.json", fit="fit_audit.json", interpretation=str(combined / "interpretation_audit.json")),
            summaries=dict(cohort="summary/report.md", combined=str(combined / "report.md")), compact_fit_logs="fit_logs/",
            per_k=per_k, fit_cost=dict(fits=fits, fixed_k1_fits=fit["fixed_k1_fits"], support_cv_fits=fit["support_cv_fits"],
                total_fit_seconds=seconds, mean_fit_seconds=seconds/fits, process_peak_rss_bytes=fit["process_peak_rss_bytes"],
                state_byte_ranges=state_ranges, note="Sum of measured fit_seconds, not wall duration or satellite hardware speed; closed-form head plus support CV, no epoch/gradient convergence claim"),
            payload_byte_ranges=fit["payload_byte_ranges"], payload_delivery_assumption="summary_already_deployed=false: existing summary file package counts toward delivery; no newly fitted ground statistics",
            new_ground_statistics_bytes=0, source_rows_read=0, query_rows_used_for_fit=0,
            old_guard_scope="All paired cells at K, including new_count=0; joint table old/new/H restricts to new_count>0. Explicit supplement corrects summary guard scope without replacing original summary.",
            claim_scope="REPEATED_BENCHMARK_PREVIOUSLY_SCORED_TARGETS", selection_feedback_forbidden=True,
            limitations=["Previously scored target reuse is not independent unexposed generalization evidence", "Shared query and overlapping support draws are not independent model repeats", "Not a comprehensive performance improvement; all-K goal remains unmet"],
            analysis_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        text = (f"\n\n{MARKER}\n\n## 最终结果与完整核验\n\n"
            f"当前结果已完成评分与分析（SCORED / ANALYZED），证据状态 VERIFIED。上文的启动、运行中及待读分说明是历史记录。"
            f"实际发布 commit 为 `{startup['commit']}`。本批次全部 {records} 条评分通过混淆矩阵独立复算，{fits} 次拟合日志完整核验。"
            "D92-SGJoint-v1 未晋级，整体优化目标尚未完成。\n\n"
            "下表均为候选减 D92，单位为百分点。联合列只含有新类的任务；最后一列含 old-only，按预登记用于旧类退化约束，阈值为不低于 −1 个百分点。\n\n"
            + table(per_k) + "\n\n"
            f"本批次累计实际拟合计时 {seconds:.3f} 秒，平均 {seconds/fits:.6f} 秒/任务；K1 固定配置 {fit['fixed_k1_fits']} 次，support 内部交叉验证 {fit['support_cv_fits']} 次。"
            "这是 N607 CPU 运行中的 fit_seconds 累加，包含该拟合函数内的工作，不等同于并行运行墙钟时间或星载硬件速度。分类头采用闭式求解，温度由有界标量求解选择，不声称迭代训练收敛。\n\n"
            f"记录到的单进程峰值 RSS 为 {fit['process_peak_rss_bytes']['min']} 至 {fit['process_peak_rss_bytes']['max']} 字节，不能称为纯模型状态。"
            f"实际分类头状态为 {state_ranges['head_bytes'][0]} 至 {state_ranges['head_bytes'][1]} 字节，固定摘要算子为 {state_ranges['summary_operator_bytes'][0]} 字节，"
            f"持久数值状态合计 {state_ranges['persistent_state_bytes'][0]} 至 {state_ranges['persistent_state_bytes'][1]} 字节。\n\n"
            "既有摘要数值数组为 4,410 字节，注册/schema 数组为 670 字节，数组合计 5,080 字节。压缩 NPZ 与 manifest 合计 8,191 至 8,383 字节/模型。"
            "冻结配置 `summary_already_deployed=false`，因此该文件包仍计入交付字节；本轮新增地面统计为 0 字节，不代表既有摘要交付为 0。"
            "source_rows_read 与 query_rows_used_for_fit 均为 0，Phase1 保持冻结。\n\n"
            "本批次不能单独替代完整联合结论。四个 RX 按真实配对单元等权，两个批次权重为 3∶1。复用数据此前已经评分，"
            "结果只支持透明重复基准比较，不支持新的独立泛化声明；共享 query 的 support 抽样不作为独立模型重复。结果不用于本候选调参或选择性重跑。\n\n"
            "证据：[产物索引](results/artifacts.json)、[本批次完整汇总](results/summary/report.md)、[拟合审计](results/fit_audit.json)。\n")
        pending.append((output, artifacts, report, text))
        all_rows.extend(scores["results"])
    report = combined / "report.md"
    if MARKER in report.read_text(encoding="utf-8"):
        raise ValueError("Combined final analysis already exists")
    for file in ("old_new_h.png", "delta_all_k_new.png", "interpretation_audit.json"):
        assert (combined / file).is_file(), file
    joint = metrics(all_rows)
    assert all(v["old_guard_all_cells"]["pass_guard"] for v in joint)
    for _, a, _, _ in pending:
        a["combined_old_guard_all_cells"] = [dict(k=v["k"], **v["old_guard_all_cells"]) for v in joint]
    extra = (f"\n\n{MARKER}\n\n## 完整解释与旧类验收口径\n\n"
        "完整 9,648 条评分包含 4,800 对任务与 48 条冻结 DG。每 K 的联合新旧类汇总为 rx3 的 720 对加 rx1 的 240 对，权重 3∶1。H 是单元 H 的均值。"
        "独立核验 1,760 个汇总数值，最大绝对差为 2.23×10⁻¹⁶ 以内。\n\n"
        "原联合表中的旧类列仅含 new_count>0；预登记旧类退化约束含 old-only。下表最后一列补齐正式验收口径；所有 K 的旧类约束均通过，"
        "但 K1 的 H 及 K1、K5、K10 的新类指标未通过，整体不晋级。原 summary 文件保留，口径补充见两 run 的 artifacts.json。\n\n"
        + table(joint) + "\n\n"
        "四模型 seed 中，K1 的 H 全部下降，K1、K5、K10 的新类准确率全部下降。K5 的 H 在四个 seed 均上升，但新类下降；"
        "K10 的 H 仅小幅提升且 1/4 seed 下降；K20 的 H 四个 seed 均提升，但新类仅 2/4 seed 提升。不能称为全面明显改善。"
        "rx3 在 K10、K20 的 ΔH 分别为 −0.760、−0.008 个百分点，rx1 对应 +2.797、+2.023，联合改善存在接收机差异。"
        "old-only 的 K20 旧类准确率还下降 0.581 个百分点，完整旧类任务没有被隐藏。\n\n"
        "两批次共 4,800 次拟合，累计 fit_seconds 为 "
        f"{sum(a['fit_cost']['total_fit_seconds'] for _, a, _, _ in pending):.3f} 秒。详细成本、状态字节和摘要交付假设见各 run 的 artifacts.json 与 fit_audit.json。"
        "此处没有迭代头训练收敛结论，也没有独立泛化或统计显著性声明。\n\n"
        "[独立解释核验](interpretation_audit.json)保留每 K、模型 seed、批次和全部 K×新增类规模的结果。\n\n"
        "![各 K 的旧类、新类与 H](old_new_h.png)\n\n![全部 K 与新增类规模的差值](delta_all_k_new.png)\n")
    return pending, report, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("E:/type10-7"))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    pending, combined_report, extra = prepare(args.workspace.resolve())
    if args.write:
        for output, artifact, report, text in pending:
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(artifact, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            with report.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
        with combined_report.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(extra)
        for output, artifact, report, _ in pending:
            assert read(output) == artifact
            assert report.read_text(encoding="utf-8").count(MARKER) == 1
        assert combined_report.read_text(encoding="utf-8").count(MARKER) == 1
    print(json.dumps(dict(status="VERIFIED" if args.write else "PREFLIGHT_VERIFIED",
        artifacts=[str(p[0]) for p in pending], reports=[str(p[2]) for p in pending] + [str(combined_report)],
        old_guard_all_cells=pending[0][1]["combined_old_guard_all_cells"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
