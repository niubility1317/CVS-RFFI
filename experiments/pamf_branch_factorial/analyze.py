"""Post-scoring per-seed and paired summaries; no prediction or truth access."""
from collections import defaultdict
import csv
from pathlib import Path
import statistics

from experiments.cric_dura_comparison.io import read, write
from experiments.practical_full_2x2.plan import VIEWS
from .method import ARMS, SEEDS, TASKS


def pooled(a, b):
    if a["count"] != 126000 or b["count"] != 42000:
        raise ValueError("Target populations differ from preregistration")
    cm = [[a["confusion"][i][j] + b["confusion"][i][j] for j in range(6)] for i in range(6)]
    count = sum(map(sum, cm))
    f1 = []
    for i in range(6):
        den = sum(cm[i]) + sum(row[i] for row in cm)
        f1.append(2 * cm[i][i] / den if den else 0.)
    return sum(cm[i][i] for i in range(6)) / count, statistics.mean(f1)


def run(rt):
    root = Path(rt["run_root"])
    sources = read(root / "sources.json")
    values = {}
    for arm in ARMS:
        for seed in SEEDS:
            row = sources[f"{arm}-s{seed}"]
            done = read(Path(row["scores"]) / "complete.json")
            score = read(Path(row["scores"]) / "scores.json")
            if done["status"] != "SCORED_COMPLETE" or not score["truth_last"] or not score["target_feedback_forbidden"]:
                raise ValueError("Independent scoring incomplete")
            if score["model_seed"] != seed:
                raise ValueError("Paired seed differs")
            for view in VIEWS:
                records = [r for r in score["results"] if r["test"] in TASKS and r["view"] == view
                           and r["readout"] == "raw" and r["dimension"] == "overall" and r["stratum"] == "ALL"]
                by_task = {r["test"]: r for r in records}
                if len(records) != 2 or set(by_task) != set(TASKS):
                    raise ValueError("Two complete target groups required")
                for task in TASKS:
                    rec = by_task[task]
                    if rec["count"] != {"cross_rx": 126000, "cross_both": 42000}[task]:
                        raise ValueError("Target population differs")
                    for metric in ("accuracy", "macro_f1"):
                        values[(arm, seed, task, view, metric)] = 100. * rec[metric]
                for metric, val in zip(("accuracy", "macro_f1"), pooled(by_task["cross_rx"], by_task["cross_both"])):
                    values[(arm, seed, "pooled", view, metric)] = 100. * val
            for task in (*TASKS, "pooled"):
                for metric in ("accuracy", "macro_f1"):
                    values[(arm, seed, task, "six_practical_mean", metric)] = statistics.mean(
                        values[(arm, seed, task, view, metric)] for view in VIEWS if view != "clean")
    per_seed = [dict(arm=arm, model_seed=seed, test=task, view=view, metric=metric, value=value)
                for (arm, seed, task, view, metric), value in values.items()]
    summary = []
    paired = []
    comparisons = [(arm, "base") for arm in ARMS if arm != "base"] + [
        ("no_time", "no_signature"), ("frequency_only", "no_signature_no_pa"),
        ("pa_only", "no_path_no_frequency"), ("time_only", "no_path_time_only")]
    for task in (*TASKS, "pooled"):
        for view in (*VIEWS, "six_practical_mean"):
            for metric in ("accuracy", "macro_f1"):
                for arm in ARMS:
                    nums = [values[(arm, seed, task, view, metric)] for seed in SEEDS]
                    summary.append(dict(arm=arm, test=task, view=view, metric=metric,
                                        mean=statistics.mean(nums), sd=statistics.stdev(nums), seed_n=3))
                for arm, baseline in comparisons:
                    nums = [values[(arm, seed, task, view, metric)] - values[(baseline, seed, task, view, metric)]
                            for seed in SEEDS]
                    paired.append(dict(arm=arm, baseline=baseline, test=task, view=view, metric=metric,
                                       mean=statistics.mean(nums), sd=statistics.stdev(nums),
                                       seed_values=";".join(map(str, nums)), sign="arm_minus_baseline"))
    out = root / "analysis"
    out.mkdir(exist_ok=False)
    for name, items in (("per_seed.csv", per_seed), ("summary.csv", summary), ("paired.csv", paired)):
        with (out / name).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    names = {'base': '完整网络', 'no_signature': '去三视图路径描述',
             'no_signature_no_pa': '去路径描述及功放', 'no_pa': '去功放',
             'strict_no_frequency': '去频域', 'no_time': '去时域及路径描述',
             'time_only': '仅时域与路径描述', 'frequency_only': '仅频域',
             'pa_only': '仅功放', 'no_frequency_with_rho': '去频域但保留圆对称度',
             'no_path_no_frequency': '仅时域与功放', 'no_path_time_only': '仅时域'}
    lines = ['# PAMF 第一阶段分支消融结果', '',
             '本报告只使用分类损失监督训练和直接识别。每个结构使用三个匹配模型种子；'
             '差值为变体减完整网络，负值表示变体更低。', '',
             '跨接收机为 126,000 条、跨日跨接收机为 42,000 条；综合结果合并两组混淆矩阵后计算 Macro-F1。'
             '六种信道场景先在各种子内取平均，再汇总种子。历史对照已暴露过测试结果。', '',
             '## 六种信道场景的平均准确率', '',
             '| 结构 | 跨接收机（%） | 跨日跨接收机（%） | 综合（%） | 综合配对差（百分点） |',
             '|---|---:|---:|---:|---:|']
    def get(items, arm, task):
        return next(x for x in items if x['arm'] == arm and x['test'] == task and
                    x['view'] == 'six_practical_mean' and x['metric'] == 'accuracy' and
                    ('baseline' not in x or x['baseline'] == 'base'))
    for arm in ARMS:
        cells = [get(summary, arm, task) for task in (*TASKS, 'pooled')]
        fmt = [f"{x['mean']:.2f} ± {x['sd']:.2f}" for x in cells]
        delta = '—' if arm == 'base' else f"{get(paired, arm, 'pooled')['mean']:+.2f} ± {get(paired, arm, 'pooled')['sd']:.2f}"
        lines.append('| ' + names[arm] + ' | ' + ' | '.join(fmt + [delta]) + ' |')
    lines += ['', '完整逐种子、逐场景准确率和 Macro-F1 及配对比较见同目录 CSV。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    write(out / "complete.json", dict(status="ANALYZED", logical_models=36, new_models=27,
          reused_models=9, populations=list(TASKS), pooled_count=168000, readout="raw",
          macro_f1_from_summed_confusion=True, truth_last=True, target_feedback_forbidden=True))
