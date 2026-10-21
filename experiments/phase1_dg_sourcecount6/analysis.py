"""Recompute frozen metrics and keep count, method, PL and budget contrasts apart."""
from collections import defaultdict
import csv
from pathlib import Path
import statistics
from experiments.phase1_external_dg.data import read,write
from experiments.phase1_external_dg.driver import atomic
from experiments.phase1_external_dg.analysis import extend_frozen_scores,summary_records
from experiments.phase1_external_dg_aug.analysis import check_source
from .plan import make_plan,VIEWS,SEEDS
from .runtime import bind


def paired_pl(rows):
    lookup={r['config']['row_id']:r for r in rows};groups=defaultdict(list)
    for r in rows:
        c=r['config']
        if not c['U_s_use']:continue
        control=lookup.get(c['supervised_control_row_id'])
        if control is None:continue
        for view in VIEWS:
            def value(item):
                rx=[x['accuracy'] for x in item['results'] if x['dimension']=='receiver' and x['view']==view]
                return statistics.mean(rx)
            groups[(c['source_count'],c['family'],c['variant'],view)].append(
                dict(model_seed=c['model_seed'],delta=value(r)-value(control)))
    return [dict(zip(('source_count','family','variant','view'),key),actual_pairs=len(values),
        mean=statistics.mean(v['delta'] for v in values),
        sample_sd=statistics.stdev(v['delta'] for v in values) if len(values)>1 else None,
        pairs=values) for key,values in sorted(groups.items())]


def analyze(rt):
    _,evaluation=bind();root=Path(rt['run_root']);plan=make_plan();rows=[];missing=[]
    for row in plan['rows']:
        folder=root/row['row_id']
        if not (folder/'complete.json').is_file():missing.append(row['row_id']);continue
        done=read(folder/'complete.json')
        if done['status']!='ANALYZED' or done['row_id']!=row['row_id']:raise ValueError('Invalid row closure')
        scoring=read(folder/'score_config.json');prepared=evaluation.score_preflight(scoring)
        scores=read(folder/'scores/scores.json')
        if not scores['truth_last'] or not scores['target_feedback_forbidden']:raise ValueError('Illegal scoring')
        for test,(ids,predictions) in prepared.items():
            extend_frozen_scores(ids,predictions,read(scoring['tests'][test]['truth']),6,
                {'2021_03_08','2021_03_15','2021_03_23'},test=test,original=scores['results'],class_names=scoring['classes'])
        if row['reuse_parent_row_id']:
            reused=read(folder/'reused_source.json');cost=dict(reused=True,new_training_seconds=0.,original=reused['original_cost'])
        else:cost=check_source(folder/'source',row)
        rows.append(dict(config=row,results=scores['results'],cost=cost,independent_prediction_audit=True))
    summary,dg=summary_records(rows);pl=paired_pl(rows)
    complete=not missing and len(rows)==432;status='ANALYZED' if complete else 'PARTIAL'
    write(root/'summary.json',dict(status=status,rows=len(rows),expected_rows=432,
        new_training=sum(not r['config']['reuse_parent_row_id'] for r in rows),
        reused=sum(bool(r['config']['reuse_parent_row_id']) for r in rows),missing=missing,
        summary=summary,dg_paired=dg,pl_paired=pl,no_target_feedback=True))
    write(root/'audited_rows.json',dict(status=status,rows=rows))
    columns=['group','family','version','test','view','actual_seeds','equal_rx_accuracy_mean',
        'equal_rx_accuracy_sample_sd','macro_f1_mean','worst_rx_mean']
    with (root/'summary.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(summary)
    lines=['# 1 至 6 个源接收机对比','',f'状态：{status}；实际 {len(rows)}/432 行；缺项 {len(missing)}。','',
        '固定目标 6 RX/6 TX/144000 包；全部七视图；没有独立 ResNet18 CE 或 PAMF 行。',
        '自然预算：1260m/11340m/5400m，监督和普通 PL；固定预算：L=1080、V=5400，仅监督。',
        '默认拼接式星地增强；固定 .95 PL；原生训练预算；冻结默认头；独立 truth-last。',
        '一条预定嵌套接收机链只能支持该链的条件性结论，不推广为所有接收机组合的定律。',
        '单源时 RIEI/DRIFT 的真实接收机分类/对抗项退化为单类，保留其他原机制，不虚构域。','',
        '| 设置 | 方法 | 版本 | seed 数 | clean RX 等权 Accuracy（%） | SD |','|---|---|---|---:|---:|---:|']
    for s in summary:
        if s['view']=='clean':lines.append('| %s | %s | %s | %d | %.4f | %.4f |'%(
            s['group'],s['family'],s['version'],s['actual_seeds'],100*s['equal_rx_accuracy_mean'],100*(s['equal_rx_accuracy_sample_sd'] or 0)))
    (root/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(root/'completion.json',dict(status=status,complete=complete,rows=len(rows),expected_rows=432,
        expected_new_training=384,expected_reuse=48,missing=missing))
    state=read(root/'pipeline_state.json');state['status']=status;atomic(root/'pipeline_state.json',state)
