"""Independently recompute all frozen source cells; no target inputs or ranking."""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import statistics
from experiments.cvs_source_sensitivity.audit import RUN, SEEDS, SOURCE_RUNS, TRANSFORMS, STAGES


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')


def table(path, records):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(records[0]));w.writeheader();w.writerows(records)


def analyze(report):
    evidence=report/'evidence'; d=json.loads((evidence/'final_readback.json').read_text(encoding='utf-8'))
    expected={(v,s,t) for v in SOURCE_RUNS for s in SEEDS for t in TRANSFORMS}
    rows=d['rows'];actual={(r['variant'],r['seed'],r['transform']) for r in rows}
    if actual!=expected or len(rows)!=56 or d['process'] or not d['completion']['backend_flags_restored']:
        raise ValueError('Incomplete fixed matrix or live process')
    if set(d['completion']['rows'])!={r['row_id'] for r in rows} or d['completion']['complete_rows']!=56:
        raise ValueError('Completion differs from independently read rows')
    cells=set(itertools.product(range(6),(1,3,4,6,8),(1,2,3)))
    flat=[]; per_seed=[]
    for r in rows:
        if r['count']!=27000 or r['target_access'] or r['model_updated'] or r['optimizer_steps'] or not r['state_unchanged']:
            raise ValueError('Invalid frozen row')
        g=r['groups']
        if len(g)!=90 or {(a['tx'],a['receiver'],a['day']) for a in g}!=cells or any(a['count']!=300 for a in g):
            raise ValueError('Incomplete fixed TX/RX/day coverage')
        flags=dict(d['resolved']['initial_flags'])
        if r['variant']=='equivariant_memory':flags['cudnn_allow_tf32']=False
        if r['effective_flags']!=flags:raise ValueError('Inference policy differs from registered source policy')
        for name,value in r['mean'].items():
            recomputed=math.fsum(a['count']*a['mean'][name] for a in g)/27000
            maximum=max(a['maximum'][name] for a in g)
            if not math.isfinite(value) or not math.isclose(recomputed,value,rel_tol=1e-12,abs_tol=1e-12) or maximum!=r['maximum'][name]:
                raise ValueError('Independent source metric recomputation differs')
        identity=next(a for a in rows if (a['variant'],a['seed'],a['transform'])==(r['variant'],r['seed'],'identity'))
        if r['mean']['reference_correct']!=identity['mean']['correct']:
            raise ValueError('Same source packets/reference differ')
        if r['transform']=='identity' and (r['mean']['agreement']!=1 or any(r['maximum'][n]!=0 for n in ['logit_max_abs_error','features_unit_distance',*[s+'_unit_distance' for s in STAGES]])):
            raise ValueError('Unchanged identity observation differs')
        per_seed.append(dict(variant=r['variant'],seed=r['seed'],transform=r['transform'],count=r['count'],
            accuracy_pct=100*r['mean']['correct'],paired_accuracy_delta_pp=100*(r['mean']['correct']-identity['mean']['correct']),
            agreement_pct=100*r['mean']['agreement'],unit_embedding_distance=r['mean']['features_unit_distance'],
            whole_logit_max_abs_error=r['maximum']['logit_max_abs_error'],
            **{s+'_unit_distance':r['mean'][s+'_unit_distance'] for s in STAGES}))
        for a in g:flat.append(dict(variant=r['variant'],seed=r['seed'],transform=r['transform'],tx=a['tx'],receiver=a['receiver'],day=a['day'],count=a['count'],**a['mean']))
    summary=[]
    names=[k for k in per_seed[0] if k not in ('variant','seed','transform','count')]
    for v in SOURCE_RUNS:
        for t in TRANSFORMS:
            chosen=[r for r in per_seed if r['variant']==v and r['transform']==t]
            row=dict(variant=v,transform=t,seeds=4)
            for n in names:
                row[n+'_mean']=statistics.mean(a[n] for a in chosen)
                row[n+'_sample_sd']=statistics.stdev(a[n] for a in chosen)
            summary.append(row)
    table(evidence/'source_all5040_cells.csv',flat);table(evidence/'source_all56_rows.csv',per_seed);table(evidence/'source_summary.csv',summary)
    write(evidence/'source_summary.json',summary)
    validation=dict(status='PASS',rows=56,all5040_cells_recomputed=True,full_source_V_per_row=27000,
        same_source_E200_accuracy_reproduced=True,all_state_unchanged=True,zero_optimizer_steps=True,
        target_access=False,candidate_selection=False,goal_achieved=False,
        scope='Received interventions on complete sourceV; no unique TX hardware recovery or target performance result',
        max_equivariant_real_V_phase_logit_error=max(r['whole_logit_max_abs_error'] for r in per_seed if r['variant']=='equivariant_memory' and r['transform']=='constant_phase'))
    write(evidence/'analysis_validation.json',validation)
    text=(report/'report.md').read_text(encoding='utf-8')+'\n## 全矩阵完成与独立复算\n\nVERIFIED：worker 自然退出；56 行各 27000 包、90 单元，全部 5040 单元、按包加权均值和最大值已独立复算。8 个原始源 V 准确率逐个完全复现，零优化、模型 state 完全不变、数值策略与原源训练一致，backend flags 恢复。没有 target 访问或候选重排。\n\n数值为四 seed 均值 ± 样本标准差；差值始终对同权重、同包原始输入配对。\n\n| 网络 | Received 变换 | 源 V 准确率（%） | 配对变化（百分点） | 预测一致率（%） | 单位嵌入距离 |\n|---|---|---:|---:|---:|---:|\n'
    for r in summary:text+=f"| {r['variant']} | {r['transform']} | {r['accuracy_pct_mean']:.4f} ± {r['accuracy_pct_sample_sd']:.4f} | {r['paired_accuracy_delta_pp_mean']:+.4f} | {r['agreement_pct_mean']:.4f} | {r['unit_embedding_distance_mean']:.8g} |\n"
    text+='\n## 逐路径响应\n\n距离为节点输出归一化后对原始输入的同包平均欧氏距离；近零向量使用固定 eps=1e-4，不能把局部距离当作因果贡献或恢复的硬件参数。\n\n| 网络 | 变换 | 时间投影 | 频率投影 | 行为投影 | 完整 base | 完整 physical |\n|---|---|---:|---:|---:|---:|---:|\n'
    for r in summary:
        text+='| '+r['variant']+' | '+r['transform']+' | '+' | '.join(f"{r[s+'_unit_distance_mean']:.7g}" for s in STAGES)+' |\n'
    text+=f"\n整体相位约束在真实源 V 上也成立：equivariant_memory 的公共相位预测完全一致，全部真实源包最大 logit 误差 {validation['max_equivariant_real_V_phase_logit_error']:.9g}。但固定正负 CFO 同时改变全部身份路径与预测，说明常相位不变不能替代频偏处理。相对频偏可能包含身份信息，因此不能仅按该扰动把 CFO 全删除；下一步需在原源数据上验证逐包同步的可用性、相位斜率估计歧义以及保留身份信息的条件。\n"
    text+='\nLTI/RX 镜像/RX 三阶也只是本轮已处理 IQ 上的规定变换，不能由这些数值分解真实 TX/RX 失真。原始 source 源选择、既有 clean 预测与评分全部固定，未重新拟合、修改或重跑。下一轮结构/参数/候选只能由源证据及明确物理假设确定，不能读目标分层决定。没有新候选，因此 default clean 为 N/A，旧完成的测试不重复。整体目标仍未完成。\n\n[全 56 行](evidence/source_all56_rows.csv) · [全 5040 单元](evidence/source_all5040_cells.csv) · [独立复算](evidence/analysis_validation.json) · [远端最终读回](evidence/final_readback.json)。\n'
    (report/'report.md').write_text(text,encoding='utf-8')
    write(evidence/'next_source_handoff.json',dict(status='SOURCE_ONLY_HANDOFF',run_id=RUN,commit=d['submit']['commit'],target_scores_included=False,
        source_summary=summary,validation=validation,old_source_clean_frozen=True,
        next_action='Evaluate source-only per-packet synchronization estimability/branch ambiguity before proposing a candidate. Preserve relative-CFO identity tradeoff and upstream chain assumptions; no target feedback.'))
    print(json.dumps(validation))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);analyze(p.parse_args().report)
