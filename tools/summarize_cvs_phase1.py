"""Summarize frozen Phase1 scores without selecting or relaunching models."""
import argparse
import json
from pathlib import Path
import statistics

VIEWS = ['clean', 'practical_high', 'practical_mid', 'practical_low_urban']
HEAD = '| 项目 | clean | high | mid | low urban |\n|---|---:|---:|---:|---:|'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cvs', required=True, type=Path)
    p.add_argument('--baselines', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    cvs = json.loads(a.cvs.read_text(encoding='utf-8'))['results']
    base = json.loads(a.baselines.read_text(encoding='utf-8'))['results']
    def stats(rows, key='accuracy'):
        values = [r[key] * 100 for r in rows]
        if len(values) != 4:
            raise ValueError('Fresh-seed table must contain exactly four seeds')
        return statistics.mean(values), statistics.stdev(values)
    def formatted(rows, key='accuracy'):
        mean, sd = stats(rows, key)
        return f'{mean:.2f} ± {sd:.2f}'
    fresh = [r for r in cvs if not r['row_id'].endswith('-s392005')]
    text = ['# CVS Phase1目标测试结果（2026-09-28）',
        'VERIFIED：5个固定seed全部完成200轮训练，并完成clean与residual_noeq星地目标测试。使用最后一轮final_ssdg.pth；未按目标结果选seed、选epoch或重训。',
        '评分修复commit：281b92099b76233c1d4dc341372426894006bd84。原队列将Phase1评分错误地绑定到Phase2完整状态；现已解除依赖。全部Phase1预测完成并通过ID/形状/整数类别校验后，独立读取truth。原预测、checkpoint与Phase2失败产物保留。',
        '版本：原始DAOT A1＋FastTrust-RC4；Practical LEO residual/post_sync/noeq。Phase2仍固定D92 E0去RF32（P2-256-FULL），其结果另行完成，不混入本表。',
        '## 逐seed准确率（%）', HEAD]
    for rowid in dict.fromkeys(r['row_id'] for r in cvs):
        cells = [next(r for r in cvs if r['row_id'] == rowid and r['view'] == v and r['receiver'] == 'ALL')['accuracy'] * 100 for v in VIEWS]
        text.append('| ' + rowid.rsplit('-s', 1)[1] + ' | ' + ' | '.join(f'{x:.2f}' for x in cells) + ' |')
    text += ['\n392005属于历史优化seed，单列展示；主统计采用2026092701至2026092704四个预先固定的新seed。',
             '## 新seed均值与样本标准差（%）', HEAD]
    for metric in ['accuracy', 'macro_f1']:
        text.append('| ' + metric + ' | ' + ' | '.join(formatted([r for r in fresh if r['view'] == v and r['receiver'] == 'ALL'], metric) for v in VIEWS) + ' |')
    text += ['\nclean每seed为168000例；high为55998例、mid为56174例、low urban为55828例。三种星地场景覆盖同一目标池的固定子集，clean覆盖全池；比较使用所有方法相同的既有capsule。',
             '## 与现有对比方法比较：新seed准确率（%）', HEAD]
    methods = list(dict.fromkeys(r['row_id'].rsplit('-s', 1)[0] for r in base))
    baseline_means = {}
    for method in methods:
        rows = [r for r in base if r['row_id'].rsplit('-s', 1)[0] == method and not r['row_id'].endswith('-s392005') and r['receiver'] == 'ALL']
        text.append('| ' + method + ' | ' + ' | '.join(formatted([r for r in rows if r['view'] == v]) for v in VIEWS) + ' |')
        baseline_means[method] = {v: stats([r for r in rows if r['view'] == v])[0] for v in VIEWS}
    text.append('| CVS | ' + ' | '.join(formatted([r for r in fresh if r['view'] == v and r['receiver'] == 'ALL']) for v in VIEWS) + ' |')
    text.append('\n相对于每列最高对比均值的差值（百分点）：' + '；'.join(f'{v} +{stats([r for r in fresh if r["view"] == v and r["receiver"] == "ALL"])[0] - max(m[v] for m in baseline_means.values()):.2f}' for v in VIEWS) + '。该差值是描述统计，不代表已完成显著性检验。')
    text += ['\n比较边界：各方法均按200轮和最终权重报告，但每轮更新次数不一致（CVS原生加载器约222步，对比launcher约50步），当前不是等计算量对比。不能据此声称等预算优势。低仰角城市优势较小，应完整报告标准差和后续配对统计。',
             '## 逐接收机新seed准确率（%）', HEAD]
    for rx in sorted({r['receiver'] for r in fresh} - {'ALL'}):
        text.append('| ' + rx + ' | ' + ' | '.join(formatted([r for r in fresh if r['view'] == v and r['receiver'] == rx]) for v in VIEWS) + ' |')
    text += ['\n## 证据与未完成项',
        f'CVS完整160条汇总（含逐seed、接收机、场景、Macro-F1与混淆矩阵）：`{a.cvs.as_posix()}`。',
        f'对比数据：`{a.baselines.as_posix()}`。',
        '远端结果：`/home/szu2070436088/2510044040/CV-SincNet/runs/20260927-phase2-cvs-d92-practical-manytx-m5-r01/phase1_final_results.json`。',
        'Phase2原矩阵3个seed预测完成、2个seed因LDA数值一致性检查失败；尚无完整Phase2评分。修复只使用失败support与source ground，目标truth不参与修复。']
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text('\n\n'.join(text[:5]) + '\n\n' + '\n'.join(text[5:]) + '\n', encoding='utf-8')
    print(a.output)


if __name__ == '__main__':
    main()
