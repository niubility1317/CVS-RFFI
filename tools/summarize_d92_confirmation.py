"""Report all paired confirmation cells; never choose a new candidate from scores."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

METHODS=('D92','D92-SCV-v1')
METRICS=('accuracy','old_accuracy','new_accuracy','harmonic_mean','macro_f1','old_macro_f1','new_macro_f1','old_floor','new_floor','forgetting')
KEY=('model_seed','receiver','scenario','k','new_count','support_seed')


def mean(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def summarize(data):
    if data['status']!='SCORED' or data['selection_feedback_forbidden'] is not True:raise ValueError('Not a frozen complete evaluation')
    rows=data['results'];methods={m:[r for r in rows if r['method']==m] for m in METHODS}
    expected={(s,rx,scene,k,n,t) for s in range(2026092701,2026092705) for rx in ['19-1','8-14','8-7']
        for scene in ['practical_high','practical_mid','practical_low_urban'] for k in [1,5,10,20] for n in [0,2,5,10,20] for t in range(2026092711,2026092716)}
    for method,group in methods.items():
        if len(group)!=3600 or {tuple(r[k] for k in KEY) for r in group}!=expected:raise ValueError('Incomplete paired matrix: '+method)
    if len(rows)!=7236 or sum(r['method']=='frozen_dg' for r in rows)!=36:raise ValueError('Wrong final coverage')
    tables={}
    for name,dimensions,joint in [('per_k',['k'],True),('per_k_new',['k','new_count'],False),
        ('per_seed_k',['model_seed','k'],True),('per_rx_scene_k',['receiver','scenario','k'],True)]:
        groups=defaultdict(list)
        for r in rows:
            if r['method'] not in METHODS or (joint and not r['new_count']):continue
            groups[tuple(r[k] for k in dimensions)+(r['method'],)].append(r)
        table=[]
        for key,group in sorted(groups.items()):
            values={m:mean(group,m) for m in METRICS}
            table.append(dict(zip(dimensions+['method'],key),cells=len(group),**values))
        tables[name]=table
    comparisons=[]
    for k in [1,5,10,20]:
        baseline=next(r for r in tables['per_k'] if r['k']==k and r['method']==METHODS[0])
        candidate=next(r for r in tables['per_k'] if r['k']==k and r['method']==METHODS[1])
        delta={m:candidate[m]-baseline[m] for m in METRICS}
        seed_delta=[]
        for seed in range(2026092701,2026092705):
            pair={r['method']:r for r in tables['per_seed_k'] if r['k']==k and r['model_seed']==seed}
            seed_delta.append({m:pair[METHODS[1]][m]-pair[METHODS[0]][m] for m in METRICS})
        comparisons.append(dict(k=k,delta=delta,paired_seed_delta_range={m:[min(r[m] for r in seed_delta),max(r[m] for r in seed_delta)] for m in METRICS},
            h_improved=delta['harmonic_mean']>0,old_guard=delta['old_accuracy']>=-0.01,new_guard=delta['new_accuracy']>=-0.01,
            new_improved=delta['new_accuracy']>0))
    result=dict(status='ANALYZED',rows=len(rows),aggregation='Equal mean over fixed RX/scene/new_count/support seed cells; H is mean of per-cell H, not H of marginal means.4 model seeds are the independent model replicates; support draws share query and are not independent samples.',
        comparisons=comparisons,tables=tables,preregistered_guard_pass=all(r['h_improved'] and r['old_guard'] and r['new_guard'] for r in comparisons),
        new_improved_every_k=all(r['new_improved'] for r in comparisons),selection_feedback_forbidden=True,
        claim_scope=data['claim_scope'])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();result=summarize(json.loads(a.scores.read_text(encoding='utf-8')))
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name,rows in result['tables'].items():
        with (a.output/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# D92-SCV新旧类配对确认结果','',f"记录：{result['rows']}。4个固定Phase1模型、3RX、3场景、4个K、5种新增类规模、5组support抽样。",
        '','单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。',
        '','|K|方法|旧类准确率|新类准确率|H|宏F1|','|---|---|---:|---:|---:|---:|']
    for r in result['tables']['per_k']:
        lines.append(f"|{r['k']}|{r['method']}|{100*r['old_accuracy']:.2f}%|{100*r['new_accuracy']:.2f}%|{100*r['harmonic_mean']:.2f}%|{100*r['macro_f1']:.2f}%|")
    lines+=['','|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|','|---|---:|---:|---:|']
    for r in result['comparisons']:
        d=r['delta'];lines.append(f"|{r['k']}|{100*d['old_accuracy']:+.2f}|{100*d['new_accuracy']:+.2f}|{100*d['harmonic_mean']:+.2f}|")
    lines+=['',f"预登记H及双侧退化约束通过：{result['preregistered_guard_pass']}。每个K的新类均提升：{result['new_improved_every_k']}。",
        '','完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。',
        '','适用范围：相对本轮源域和最近目标接收机独立的3RX；不宣称从未在所有历史实验使用。评分不回流本候选调参或选择性重跑。']
    (a.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='tables'},indent=2))


if __name__=='__main__':main()
