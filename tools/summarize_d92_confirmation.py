"""Report all paired confirmation cells; never choose a new candidate from scores."""
import argparse
from collections import defaultdict
import csv
from itertools import product
import json
from pathlib import Path

import numpy as np
from run_d92_confirmation import candidate_definition

METHODS=('D92','D92-SCV-v1')
METRICS=('accuracy','old_accuracy','new_accuracy','harmonic_mean','macro_f1','old_macro_f1','new_macro_f1','old_floor','new_floor','forgetting')
KEY=('model_seed','receiver','scenario','k','new_count','support_seed')


def mean(rows,key):
    values=[r[key] for r in rows if r[key] is not None]
    return float(np.mean(values)) if values else None


def matrix_definition(spec=None):
    matrix=dict(model_seed=list(range(2026092701,2026092705)),receiver=['19-1','8-14','8-7'],
        scenario=['practical_high','practical_mid','practical_low_urban'],k=[1,5,10,20],
        new_count=[0,2,5,10,20],support_seed=list(range(2026092711,2026092716)))
    methods=METHODS
    acceptance=dict(old_max_drop=0.01,new_max_drop=0.01,require_new_improvement=False,require_h_improvement=True)
    if spec is not None:
        candidate=candidate_definition(spec['confirmation'])
        methods=('D92',candidate['candidate_method'])
        source=spec['data']
        matrix=dict(model_seed=[r['seeds']['model'] for r in spec['rows']],receiver=source['target_receivers'],
            scenario=source['scenarios'],k=source['k'],new_count=source['new_class_counts'],support_seed=source['support_seeds'])
        strict_new=methods[1] in ('D92-SFHead-v1','D92-SGJoint-v1')
        acceptance['require_new_improvement']=strict_new
        declared=spec.get('metrics_plan',{}).get('acceptance',{})
        if not isinstance(declared,dict) or set(declared)-{'old_max_drop','require_new_improvement'}:
            raise ValueError('Unsupported acceptance rule')
        if 'old_max_drop' in declared and declared['old_max_drop']!=0.01:
            raise ValueError('The registered old-class guard must be 0.01')
        if 'require_new_improvement' in declared:
            if type(declared['require_new_improvement']) is not bool or (strict_new and not declared['require_new_improvement']):
                raise ValueError(methods[1]+' requires strict new-class improvement')
            acceptance['require_new_improvement']=declared['require_new_improvement']
    for name,values in matrix.items():
        if not isinstance(values,list) or not values or len(set(values))!=len(values):
            raise ValueError('Matrix axis must be a nonempty unique list: '+name)
        if name in ('receiver','scenario'):
            valid=all(isinstance(v,str) and v for v in values)
        else:
            valid=all(type(v) is int and v>=(0 if name=='new_count' else 1) for v in values)
        if not valid:raise ValueError('Invalid matrix axis: '+name)
    if 0 not in matrix['new_count'] or not any(v>0 for v in matrix['new_count']):
        raise ValueError('Both old-only and joint cells are required')
    expected=set(product(*(matrix[name] for name in KEY)))
    expected_dg=set(product(matrix['model_seed'],matrix['receiver'],matrix['scenario']))
    if spec is not None:
        confirmation=spec['confirmation']
        per_model=len(expected)//len(matrix['model_seed'])
        for name,value in [('expected_split_count',per_model),('splits_per_model',per_model),
                           ('model_rows',len(matrix['model_seed'])),('predictions_total',2*len(expected)+len(expected_dg))]:
            if name in confirmation and confirmation[name]!=value:
                raise ValueError('Spec matrix/count mismatch: '+name)
    return methods,matrix,acceptance,expected,expected_dg


def summarize(data,spec=None):
    if data['status']!='SCORED' or data['selection_feedback_forbidden'] is not True:raise ValueError('Not a frozen complete evaluation')
    methods,matrix,acceptance,expected,expected_dg=matrix_definition(spec)
    rows=data['results'];groups_by_method={m:[r for r in rows if r['method']==m] for m in methods}
    for method,group in groups_by_method.items():
        if len(group)!=len(expected) or {tuple(r[k] for k in KEY) for r in group}!=expected:raise ValueError('Incomplete paired matrix: '+method)
    dg=[r for r in rows if r['method']=='frozen_dg']
    if (len(rows)!=2*len(expected)+len(expected_dg) or len(dg)!=len(expected_dg)
            or {(r['model_seed'],r['receiver'],r['scenario']) for r in dg}!=expected_dg
            or any(r['new_count']!=0 or tuple(r[k] for k in KEY) not in expected for r in dg)):
        raise ValueError('Wrong final coverage')
    tables={}
    for name,dimensions,joint in [('per_k',['k'],True),('per_k_new',['k','new_count'],False),
        ('per_seed_k',['model_seed','k'],True),('per_rx_scene_k',['receiver','scenario','k'],True)]:
        groups=defaultdict(list)
        for r in rows:
            if r['method'] not in methods or (joint and not r['new_count']):continue
            groups[tuple(r[k] for k in dimensions)+(r['method'],)].append(r)
        table=[]
        for key,group in sorted(groups.items()):
            values={m:mean(group,m) for m in METRICS}
            table.append(dict(zip(dimensions+['method'],key),cells=len(group),**values))
        tables[name]=table
    comparisons=[]
    for k in matrix['k']:
        baseline=next(r for r in tables['per_k'] if r['k']==k and r['method']==methods[0])
        candidate=next(r for r in tables['per_k'] if r['k']==k and r['method']==methods[1])
        delta={m:candidate[m]-baseline[m] for m in METRICS}
        seed_delta=[]
        for seed in matrix['model_seed']:
            pair={r['method']:r for r in tables['per_seed_k'] if r['k']==k and r['model_seed']==seed}
            seed_delta.append({m:pair[methods[1]][m]-pair[methods[0]][m] for m in METRICS})
        comparisons.append(dict(k=k,delta=delta,paired_seed_delta_range={m:[min(r[m] for r in seed_delta),max(r[m] for r in seed_delta)] for m in METRICS},
            h_improved=delta['harmonic_mean']>0,old_guard=delta['old_accuracy']>=-acceptance['old_max_drop'],new_guard=delta['new_accuracy']>=-acceptance['new_max_drop'],
            new_improved=delta['new_accuracy']>0))
    result=dict(status='ANALYZED',rows=len(rows),methods=list(methods),matrix=matrix,acceptance=acceptance,
        aggregation=f"Equal mean over fixed RX/scene/new_count/support seed cells; H is mean of per-cell H, not H of marginal means. {len(matrix['model_seed'])} model seeds are the independent model replicates; support draws share query and are not independent samples.",
        comparisons=comparisons,tables=tables,preregistered_guard_pass=all(r['h_improved'] and r['old_guard'] and r['new_guard'] and (not acceptance['require_new_improvement'] or r['new_improved']) for r in comparisons),
        new_improved_every_k=all(r['new_improved'] for r in comparisons),selection_feedback_forbidden=True,
        claim_scope=spec.get('permissions',{}).get('claim_scope',data['claim_scope']) if spec is not None else data['claim_scope'])
    return result


def write_summary(result, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name,rows in result['tables'].items():
        with (output/(name+'.csv')).open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    matrix=result['matrix']
    scope='重复基准比较' if 'REPEATED_BENCHMARK' in result['claim_scope'] else '新旧类配对确认'
    lines=[f"# {result['methods'][1]}{scope}结果",'',f"记录：{result['rows']}。{len(matrix['model_seed'])}个固定Phase1模型、{len(matrix['receiver'])}RX、{len(matrix['scenario'])}场景、{len(matrix['k'])}个K、{len(matrix['new_count'])}种新增类规模、{len(matrix['support_seed'])}组support抽样。",
        '','单元等权平均；H先在单元内计算再平均。不同support抽样复用query，不作为独立模型重复。',
        '','|K|方法|旧类准确率|新类准确率|H|宏F1|','|---|---|---:|---:|---:|---:|']
    for r in result['tables']['per_k']:
        lines.append(f"|{r['k']}|{r['method']}|{100*r['old_accuracy']:.2f}%|{100*r['new_accuracy']:.2f}%|{100*r['harmonic_mean']:.2f}%|{100*r['macro_f1']:.2f}%|")
    lines+=['','|K|Δ旧类（百分点）|Δ新类（百分点）|ΔH（百分点）|','|---|---:|---:|---:|']
    for r in result['comparisons']:
        d=r['delta'];lines.append(f"|{r['k']}|{100*d['old_accuracy']:+.2f}|{100*d['new_accuracy']:+.2f}|{100*d['harmonic_mean']:+.2f}|")
    acceptance_text='每个K的H和新类严格提升、旧类退化不超过1个百分点' if result['acceptance']['require_new_improvement'] else '每个K的H严格提升、新旧类各自退化不超过1个百分点'
    lines+=['',f"验收标准：{acceptance_text}。通过：{result['preregistered_guard_pass']}。每个K的新类均提升：{result['new_improved_every_k']}。",
        '','完整K×新增类规模、每seed和RX×场景结果见同目录CSV。最弱类别准确率、分组宏F1与遗忘均保留，不能用总体H掩盖局部退化。',
        '',f"适用范围：{result['claim_scope']}。评分不回流本候选调参或选择性重跑。"]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--scores',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--spec',type=Path)
    a=p.parse_args();spec=json.loads(a.spec.read_text(encoding='utf-8')) if a.spec else None
    result=summarize(json.loads(a.scores.read_text(encoding='utf-8')),spec=spec)
    write_summary(result,a.output)
    print(json.dumps({k:v for k,v in result.items() if k!='tables'},indent=2))


if __name__=='__main__':main()
