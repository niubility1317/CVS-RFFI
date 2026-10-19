"""Independently audit frozen rows and report DG and PL as separate contrasts."""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

from experiments.phase1_external_dg.analysis import extend_frozen_scores, summary_records
from experiments.phase1_dg_nativebudget import pipeline as native
from .data import read, write
from .driver import atomic
from .evaluation import score_preflight
from .plan import make_plan, SEEDS
from .source import check_training

DATES=('2021_03_01','2021_03_08','2021_03_15','2021_03_23')


def check_source(root,row):
    root=Path(root);frozen=read(root/'freeze.json');c=frozen['config'];check_training(c)
    if (frozen['status']!='SOURCE_TRAINED_FROZEN' or frozen['target_access'] or frozen['target_score_used'] or
        any(c.get(k)!=v for k,v in row.items()) or not frozen['satellite_applied_examples']):
        raise ValueError('Incomplete augmented source freeze')
    payload=native.load(frozen['checkpoint'],c,frozen['source_contract'])
    if payload.get('default_head_joint_training') is not True or payload.get('separate_head_training') is not False:
        raise ValueError('Default head evidence differs')
    selected_telemetry=payload.get('training_telemetry',{})
    if not selected_telemetry.get('satellite_applied_examples') or c['U_s_use'] and not selected_telemetry.get('pl_seen'):
        raise ValueError('Selected checkpoint predates actual required training')
    chosen_epoch,chosen_stage=payload['epoch'],payload['stage'];del payload
    if len(frozen['stages'])!=row['stages']:raise ValueError('Native stage budget differs')
    for stage in frozen['stages']:
        records=[json.loads(s) for s in (root/('stage%d'%stage['stage'])/'epochs.jsonl').read_text().splitlines() if s.strip()]
        if not records or records[-1]['updates']!=stage['updates']:raise ValueError('Missing full source curve')
        for record in records:
            if record['target_access'] or record['pseudo_label_truth_access']:
                raise ValueError('Training permissions violated')
            if not all(math.isfinite(record[k]) for k in ('loss','identity_ce','satellite_ce','pl_loss','source_val_loss')):
                raise ValueError('Nonfinite training evidence')
            if record['satellite_weight']!=.68 or not record['concat_sat_ce_only'] or record['U_s_use']!=row['U_s_use']:
                raise ValueError('Actual concat/PL telemetry differs')
        if row['budget_kind']=='updates':
            if stage['updates']!=row['updates'] or any(r['epoch'] is not None for r in records):
                raise ValueError('Fixed-update budget mislabeled or incomplete')
        else:
            epochs=stage['epochs']
            if [r['epoch'] for r in records]!=list(range(1,epochs+1)) or epochs>row['epochs']:
                raise ValueError('Epoch coverage differs')
            if row['family']!='ibrff' and epochs!=row['epochs']:raise ValueError('Shortened fixed native budget')
            steps=(frozen['source_contract']['counts']['L_s']+row['batch_size']-1)//row['batch_size']
            if stage['updates']!=epochs*steps:raise ValueError('Actual minibatch budget differs')
        if stage['optimizer_steps']!=stage['updates']*(3 if row['family']=='riei' and row['variant']=='full' else 1):
            raise ValueError('Optimizer update evidence differs')
        if stage['satellite_examples']!=stage['labeled_exposure']:
            raise ValueError('Concat batch ratio differs')
        if stage['pl_seen']!=(stage['labeled_exposure'] if row['U_s_use'] else 0):
            raise ValueError('U/L exposure or supervised permission differs')
        if not 0<=stage['pl_accepted']<=stage['pl_seen']:raise ValueError('Invalid PL acceptance')
        if stage['stage']==chosen_stage:
            if row['family'] in ('ibrff','sdg'):
                if frozen['selection']!='source_validation_nll' or chosen_epoch!=min(records,key=lambda r:r['source_val_loss'])['epoch']:
                    raise ValueError('Source NLL checkpoint selection differs')
            elif (frozen['selection']!='fixed_final_budget' or
                  chosen_epoch!=(None if row['budget_kind']=='updates' else row['epochs'])):
                raise ValueError('Fixed final checkpoint differs')
    return dict(stages=frozen['stages'],seconds=frozen['seconds'],
        satellite_examples=frozen['satellite_examples'],satellite_applied_examples=frozen['satellite_applied_examples'],
        pl_seen=frozen['pl_seen'],pl_accepted=frozen['pl_accepted'],selected_epoch=chosen_epoch,
        selected_training_telemetry=selected_telemetry,process_cost=read(root/'training_cost.json'))


def pl_pairs(rows):
    lookup={r['config']['row_id']:r for r in rows};groups={}
    for row in rows:
        c=row['config']
        if not c['U_s_use']:continue
        control=lookup.get(c['supervised_control_row_id'])
        if control is None:continue
        for value in row['results']:
            if value['dimension']!='overall':continue
            test,view=value.get('test','main'),value['view']
            baseline=next(v for v in control['results'] if v['dimension']=='overall' and v.get('test','main')==test and v['view']==view)
            key=(c['family'],c['variant'],test,view)
            rx=lambda r:[v['accuracy'] for v in r['results'] if v['dimension']=='receiver' and v.get('test','main')==test and v['view']==view]
            delta=dict(seed=c['model_seed'],pl_row=c['row_id'],supervised_row=control['config']['row_id'],
                accuracy=value['accuracy']-baseline['accuracy'],macro_f1=value['macro_f1']-baseline['macro_f1'],
                equal_rx_accuracy=statistics.mean(rx(row))-statistics.mean(rx(control)),worst_rx=min(rx(row))-min(rx(control)))
            groups.setdefault(key,[]).append(delta)
    output=[]
    for key,values in sorted(groups.items()):
        if len({x['seed'] for x in values})!=len(values):raise ValueError('Duplicate PL pair seed')
        item=dict(zip(('family','variant','test','view'),key));item.update(actual_pairs=len(values),pairs=values)
        for metric in ('accuracy','macro_f1','equal_rx_accuracy','worst_rx'):
            d=[x[metric] for x in values]
            item[metric+'_mean']=statistics.mean(d);item[metric+'_sample_sd']=statistics.stdev(d) if len(d)>1 else None
            item[metric+'_positive']=sum(x>0 for x in d)
        output.append(item)
    return output


def analyze(rt):
    plan=make_plan();root=Path(rt['run_root']);state=read(root/'pipeline_state.json')
    rows=[];missing=[]
    for row in plan['rows']:
        folder=root/row['row_id']
        if not (folder/'complete.json').exists():missing.append(row['row_id']);continue
        done=read(folder/'complete.json')
        if done['status']!='ANALYZED' or done['row_id']!=row['row_id'] or not done['satellite_train_augmentation']:
            raise ValueError('Row completion differs')
        config=read(folder/'score_config.json');prepared=score_preflight(config)
        scores=read(folder/'scores/scores.json')
        if not scores['truth_last'] or not scores['target_feedback_forbidden']:raise ValueError('Independent scoring differs')
        cost=check_source(folder/'source',row)
        dates={DATES[i] for i in plan['protocols'][row['protocol_id']]['source_days']}
        for test,(ids,predictions) in prepared.items():
            extend_frozen_scores(ids,predictions,read(config['tests'][test]['truth']),len(config['classes']),
                dates,test=test,original=scores['results'],class_names=config['classes'])
        rows.append(dict(config=row,results=scores['results'],cost=cost,independent_prediction_audit=True))
    summary,dg_pairs=summary_records(rows);pl=pl_pairs(rows)
    complete=len(rows)==219 and not missing
    status='ANALYZED' if complete else 'PARTIAL'
    write(root/'summary.json',dict(status=status,new_training_rows=len(rows),
        supervised_rows=sum(not r['config']['U_s_use'] for r in rows),pl_rows=sum(r['config']['U_s_use'] for r in rows),
        missing=missing,summary=summary,dg_paired=dg_pairs,pl_paired=pl,target_feedback_forbidden=True))
    write(root/'audited_rows.json',dict(status=status,rows=rows,exclude_pamf=True))
    columns=['group','family','version','test','view','actual_seeds','accuracy_mean','accuracy_sample_sd',
             'equal_rx_accuracy_mean','equal_rx_accuracy_sample_sd','macro_f1_mean','macro_f1_sample_sd','worst_rx_mean','worst_rx_sample_sd']
    with (root/'summary.csv').open('x',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(summary)
    lines=['# 非PAMF增强监督与普通PL对比结果','',f'性能矩阵状态：{status}，实际{len(rows)}/219行，缺项{len(missing)}。',
        '','全部正式行均训练默认clean＋satellite拼接；PL行同时保留增强。监督/PL差仅对应源U伪标签项。',
        '默认头联合训练；源V选模；七视图预测闭合后独立truth-last评分。历史无增强24行不混入正式排名。',
        '','| 组 | 方法 | 版本 | 测试 | 视图 | seed数 | RX等权Accuracy（%） | SD | Macro-F1（%） | 最差RX（%） |',
        '|---|---|---|---|---|---:|---:|---:|---:|---:|']
    for s in summary:
        if s['view']!='clean':continue
        lines.append('| %s | %s | %s | %s | %s | %d | %.4f | %.4f | %.4f | %.4f |'%(
            s['group'],s['family'],s['version'],s['test'],s['view'],s['actual_seeds'],
            100*s['equal_rx_accuracy_mean'],100*(s['equal_rx_accuracy_sample_sd'] or 0),100*s['macro_f1_mean'],100*s['worst_rx_mean']))
    lines+=['','完整七视图、每RX/日期/RX×日期/TX及DG/PL原始配对差见summary.json和audited_rows.json，所有负差与失败保留。',
        '','源数量固定/自然标签预算、低标签、跨日期和ManyRx的结论分别限定；ManyRx不是独立采集系统。',
        '阈值.95、PL权重1、卫星CE权重.68及首batch启用均在本次目标访问前固定。当前基准已有历史目标暴露，不声明新的独立盲测。',
        '三个model seed的样本SD不代替接收机总体推断；同包七视图不增加独立样本。',
        '','本文件仅证明性能矩阵；整体交付还需96个冻结checkpoint的独立成本测量，不因性能ANALYZED就标记整个研究目标完成。']
    (root/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(root/'completion.json',dict(status=status,complete=complete,new_training=len(rows),
        expected_training=219,supervised_training=192,pl_training=27,frozen_eval=0,
        cost_rows_expected=96,additional_target_feedback=False,missing=missing))
    state['status']=status;atomic(root/'pipeline_state.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    analyze(read(parser.parse_args().config))
