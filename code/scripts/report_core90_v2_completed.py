"""Build detailed source-test tables from the fully parsed frozen audit."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics as st
from collect_core90_v2_completed import SCENES

def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def table(headers,rows):
    return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])+'\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--audit',required=True);args=p.parse_args()
    path=Path(args.audit);out=path.parent;data=json.loads(path.read_text(encoding='utf-8'))
    completed=[r for r in data['rows'] if r['queue_state']=='completed'];byid={r['run_id']:r for r in completed}
    scene_rows=[];tx_rows=[];rx_rows=[];day_rows=[];cm_rows=[];epoch_rows=[];run_rows=[];activation=[];progress=[]
    for r in data['rows']:
        a=r.get('actions',{});epochs=r.get('epochs',[])
        progress.append(dict(run_id=r['run_id'],state=r['queue_state'],last_complete_epoch=epochs[-1]['epoch'] if epochs else None,
            latest_action_epoch=a.get('last_epoch'),steps=a.get('records'),process_alive=r['process_alive']))
        for e in epochs:
            epoch_rows.append(dict(run_id=r['run_id'],**{k:v for k,v in e.items() if k!='terms'},**{'loss_'+k:v for k,v in e['terms'].items()}))
        if r not in completed:continue
        c=r['config'];s=r['independent_scores']['scenes']
        assert a['accepted']==9800 and a['records']==9800 and not a['rejected'] and not a['nonfinite_losses']
        assert c['from_scratch'] and not c['baseline_ckpt'] and not c['game_resume']
        assert r['completion.json']['target_evaluated'] is False
        run_rows.append(dict(run_id=r['run_id'],row=r['row'],seed=r['seed'],solver=c['game_solver'],b8_impl=c['game_b8_impl'],lambda_adv=c['lambda_adv'],
            epochs=len(epochs),accepted_steps=a['accepted'],field_evaluations=a['field_evaluations'],telemetry_backward_evaluations=a['telemetry_backward_evaluations'],
            train_seconds=r['resource_summary.json']['total_seconds'],peak_allocated_mib=max(e['peak_memory_bytes'] for e in epochs)/1024**2,
            initial_loss=epochs[0]['mean_loss'],final_loss=epochs[-1]['mean_loss'],minimum_loss=min(e['mean_loss'] for e in epochs),
            mean_leo_accuracy=st.mean(s[x]['accuracy'] for x in SCENES[1:]),mean_four_accuracy=st.mean(s[x]['accuracy'] for x in SCENES),
            mean_leo_macro_f1=st.mean(s[x]['macro_f1'] for x in SCENES[1:]),mean_four_macro_f1=st.mean(s[x]['macro_f1'] for x in SCENES),
            pseudo_selected=a['pseudo_selected'],unlabeled_exposures=a['unlabeled_samples'],satellite_exposures=a['satellite_count']))
        for term in epochs[0]['terms']:
            nonzero=[e for e in epochs if abs(e['terms'].get(term,0))>1e-12]
            activation.append(dict(run_id=r['run_id'],term=term,first_nonzero_epoch=nonzero[0]['epoch'] if nonzero else None,
                nonzero_epoch_count=len(nonzero),final_value=epochs[-1]['terms'].get(term,0),
                **{name:st.mean(e['terms'].get(term,0) for e in epochs if lo<=e['epoch']<=hi) for name,lo,hi in [('E1_79_mean',1,79),('E80_130_mean',80,130),('E131_200_mean',131,200)]}))
        for scene in SCENES:
            m=s[scene];rx=m['per_rx']
            scene_rows.append(dict(run_id=r['run_id'],scene=scene,**{k:m[k] for k in ['count','correct','accuracy','macro_precision','macro_recall','macro_f1','worst_tx_accuracy']},
                worst_tx=min(m['per_tx'],key=lambda k:m['per_tx'][k]['recall']),worst_rx=min(rx,key=lambda k:rx[k]['accuracy']),worst_rx_accuracy=min(v['accuracy'] for v in rx.values())))
            for tx,v in m['per_tx'].items():tx_rows.append(dict(run_id=r['run_id'],scene=scene,tx_label=tx,**v))
            for kind,target in [('per_rx',rx_rows),('per_day',day_rows)]:
                for group,v in m[kind].items():target.append(dict(run_id=r['run_id'],scene=scene,group_id=group,**{k:v[k] for k in ['count','correct','accuracy','macro_precision','macro_recall','macro_f1','worst_tx_accuracy']}))
            for y,line in enumerate(m['confusion_matrix']):
                for pred,count in enumerate(line):cm_rows.append(dict(run_id=r['run_id'],scene=scene,truth=y,prediction=pred,count=count))
    aggregates=[];paired=[]
    for method in sorted({r['row'] for r in completed}):
        group=[r for r in completed if r['row']==method]
        for scene in [*SCENES,'LEO_mean','four_scene_mean']:
            rr=dict(row=method,scene=scene,n_seeds=len(group))
            for metric in ('accuracy','macro_f1'):
                vals=[(r['independent_scores']['scenes'][scene][metric] if scene in SCENES else st.mean(r['independent_scores']['scenes'][s][metric] for s in (SCENES[1:] if scene=='LEO_mean' else SCENES))) for r in group]
                rr[metric+'_mean']=st.mean(vals);rr[metric+'_sample_std']=st.stdev(vals) if len(vals)>1 else None
            aggregates.append(rr)
    for seed in (392005,392006,392007):
        a=byid.get(f'V2_A_seed{seed}');b=byid.get(f'V2_B_seed{seed}')
        if not a or not b:continue
        for scene in [*SCENES,'LEO_mean']:
            scenes=[scene] if scene in SCENES else SCENES[1:]
            paired.append(dict(seed=seed,comparison='B_minus_A',scene=scene,**{metric+'_delta_pp':100*st.mean(b['independent_scores']['scenes'][s][metric]-a['independent_scores']['scenes'][s][metric] for s in scenes) for metric in ('accuracy','macro_f1')}))
    files={'run_summary.csv':run_rows,'scene_metrics.csv':scene_rows,'per_tx.csv':tx_rows,'per_rx.csv':rx_rows,'per_day.csv':day_rows,
        'confusion_matrices.csv':cm_rows,'epoch_metrics_all_rows.csv':epoch_rows,'mechanism_terms.csv':activation,'seed_aggregates.csv':aggregates,'paired_deltas.csv':paired,'matrix_progress.csv':progress}
    for name,rows in files.items():write_csv(out/name,rows)
    summary=dict(snapshot=data['snapshot_finished'],completed=len(completed),active=sum(r['queue_state']=='active' for r in data['rows']),
        epoch_records=len(epoch_rows),completed_epoch_records=sum(len(r['epochs']) for r in completed),
        action_records=sum(r['actions']['records'] for r in data['rows']),completed_action_records=sum(r['actions']['records'] for r in completed),
        prediction_records=sum(r['independent_scores']['prediction_count'] for r in completed),
        max_score_error=max(r['independent_scores']['saved_metric_max_absolute_error'] for r in completed),
        all_stdout_marker_count=sum(len(r['stdout_markers']) for r in data['rows']),run_summary=run_rows,seed_aggregates=aggregates,paired_deltas=paired)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    pct=lambda x:f'{100*x:.3f}'
    lines=['# CORE90 GAME V2已完成实验详细测试数据','',
        f"数据截止：{data['snapshot_finished']}。**{len(completed)}/21行完成E200和四场景预测评分，其余{summary['active']}行仍运行或评分中。**本报告只汇总已完成行的最终测试指标。",'',
        '## 1. 数据口径与完整性','',
        '这些数据是source验证集V的冻结final E200评估，不是目标接收机测试。source RX=1/3/4/6/8，day=1/2/3，6个已知TX标签0—5；15个RX×day域。L_s/U_s/V=6300/56700/27000，标签训练占训练池10%。每场景27000个样本，每TX4500、每RX5400、每day9000。四场景为同一批物理样本的不同视图，不能当作108000个独立物理样本。','',
        f"全量解析21行共{summary['epoch_records']}条epoch记录、{summary['action_records']}条动作记录，并扫描全部stdout。已完成行覆盖{summary['completed_epoch_records']}个epoch、{summary['completed_action_records']}次更新；独立重算{summary['prediction_records']}条预测，先验证prediction完整性再连接truth，重复ID/缺样本/越界类别/非法置信度均未发现。与原scorer主指标及RX/day/TX指标最大绝对误差={summary['max_score_error']:.3g}。",'',
        '## 2. 已完成行：四场景Accuracy','',
        '以下所有性能数值单位为%；LEO均值只平均三个LEO场景，四场景均值包含clean。A=ordinary、adv=0；B=ordinary、adv=0.35；C=B8-D1 head_grad_only、adv=0。','',
        table(['实验','seed','clean','clear','low_elev','rain','LEO均值','四场景均值'],[[r['row'],r['seed'],*[pct(byid[r['run_id']]['independent_scores']['scenes'][s]['accuracy']) for s in SCENES],pct(r['mean_leo_accuracy']),pct(r['mean_four_accuracy'])] for r in run_rows]),
        '## 3. 四场景Macro-F1','',
        table(['实验','seed','clean','clear','low_elev','rain','LEO均值'],[[r['row'],r['seed'],*[pct(byid[r['run_id']]['independent_scores']['scenes'][s]['macro_f1']) for s in SCENES],pct(r['mean_leo_macro_f1'])] for r in run_rows]),
        'Macro-F1为6类F1的非加权平均；本次每类样本量相等，Accuracy与Macro-recall相等。所有Precision/Recall、正确数和错误数可由scene_metrics.csv与per_tx.csv直接核验。','',
        '## 4. 跨seed汇总与配对差值','',
        '均值±样本标准差，单位%；标准差描述3个训练seed的离散性，不是置信区间。单seed不填标准差。','',
        table(['方法','场景','seed数','Accuracy均值±SD','Macro-F1均值±SD'],[[r['row'],r['scene'],r['n_seeds'],pct(r['accuracy_mean'])+(' ± '+pct(r['accuracy_sample_std']) if r['accuracy_sample_std'] is not None else '（单seed）'),pct(r['macro_f1_mean'])+(' ± '+pct(r['macro_f1_sample_std']) if r['macro_f1_sample_std'] is not None else '（单seed）')] for r in aggregates]),
        'ordinary对抗项配对效应B−A，单位为百分点：','',
        table(['seed','clean ΔAcc','clear ΔAcc','low_elev ΔAcc','rain ΔAcc','LEO均值 ΔAcc'],[[seed,*[f"{next(r['accuracy_delta_pp'] for r in paired if r['seed']==seed and r['scene']==s):+.3f}" for s in [*SCENES,'LEO_mean']]] for seed in sorted({r['seed'] for r in paired})]),
        '## 5. TX薄弱项','',
        '每行列出6个TX标签的Recall（%）；标签是数据索引，不推断硬件序列号。','',
        table(['实验/seed','场景',*[f'TX{i}' for i in range(6)]],[[r['run_id'],s,*[pct(r['independent_scores']['scenes'][s]['per_tx'][str(i)]['recall']) for i in range(6)]] for r in completed for s in SCENES]),
        '## 6. RX分解','',
        table(['实验/seed','场景',*[f'RX{i}' for i in (1,3,4,6,8)]],[[r['run_id'],s,*[pct(r['independent_scores']['scenes'][s]['per_rx'][str(i)]['accuracy']) for i in (1,3,4,6,8)]] for r in completed for s in SCENES]),
        '逐day指标见per_day.csv；全部6×6混淆矩阵见confusion_matrices.csv，行是真实标签、列是预测标签，值为样本数。','',
        '## 7. 训练轨迹、计算量与实际机制','',
        table(['实验/seed','训练记录耗时h','E1 loss','E200 loss','接受更新','field评估','遥测反向','伪标签通过/曝光'],[[r['run_id'],f"{r['train_seconds']/3600:.3f}",f"{r['initial_loss']:.5f}",f"{r['final_loss']:.5f}",r['accepted_steps'],r['field_evaluations'],r['telemetry_backward_evaluations'],f"{r['pseudo_selected']}/{r['unlabeled_exposures']}"] for r in run_rows]),
        '每行200×49=9800个训练更新全部接受，无非有限loss、无拒绝更新。耗时来自resource_summary的训练计时，包含共享服务器负载影响，不作为隔离吞吐基准。不同目标项/阶段的总loss定义不同，不能用B比A的总loss高判定性能退化。','',
        '共用配置：scratch、AdamW、lr=2e-4、weight_decay=1e-4、batch=128、eval_batch=256、FP32、确定性开启；固定课程、正常增强和EMA开启。game_no_audit=true、control=off是F1设计要求，因此不声称自适应CORRECT/CATCHUP/capability已经激活。','',
        '以下是完整日志观测到的首次非零epoch，空值表示整个训练期未出现非零。非零项用于确认执行，不单独证明有益。','',
        table(['实验/seed',*['adv','dom','fishr','proto','sat_cls','cons'],'U/伪标签起点'],[[r['run_id'],*[next(x['first_nonzero_epoch'] for x in activation if x['run_id']==r['run_id'] and x['term']==term) for term in ['adv','dom','fishr','proto','sat_cls','cons']],f"E{r['actions']['first_u_epoch']}/E{r['actions']['first_pseudo_epoch']}"] for r in completed]),
        'E1—79、E80—130、E131—200各loss均值及全部epoch曲线见mechanism_terms.csv和epoch_metrics_all_rows.csv。早期loss最低点不作为checkpoint选择，所有最终分数来自固定E200。','',
        '## 8. 逐样本负对照与边界','']
    for pair in data['paired_predictions']:
        if pair['variant'].startswith('V2_C'):
            differences=sum(v['prediction_disagreements'] for v in pair['scenes'].values());conf=max(v['max_confidence_delta'] for v in pair['scenes'].values())
            lines.append(f"- {pair['variant']}相对{pair['baseline']}：108000条预测中{differences}条不同，最大置信度差={conf}。该adv=0负对照不能证明B8在adv开启时有性能收益。")
    deltas=[r['accuracy_delta_pp'] for r in paired if r['scene']=='LEO_mean']
    lines+=['',f"ordinary加入adv后，3个配对seed的LEO均值Accuracy平均变化为{st.mean(deltas):+.3f}pp；逐seed变化为"+'、'.join(f'{x:+.3f}pp' for x in deltas)+'。这是当前source验证集上的描述性结果，不能推断目标域泛化或统计显著收益。',
        '尚未完整获得D/E/F与strong-source候选结果，不能给出完整solver×adv结论，也不能宣布已确认强基线或科学晋级。无target结果回流调参；本次只读分析未重训或干预任何健康任务。','',
        '## 9. 未完成行','',
        table(['实验','最近完整epoch','当前动作epoch','累计步数','进程存活'],[[r['run_id'],r['last_complete_epoch'],r['latest_action_epoch'],r['steps'],r['process_alive']] for r in progress if r['state']!='completed']),
        'E200日志存在但最终prediction/scorer/completion未闭合的行仍列为未完成。上述运行数据来自采集时间窗口，各文件为顺序读取，并非原子快照。','',
        '## 10. 可下载数据','',*['- ['+name+']('+name+')' for name in files],f'- [完整审计JSON]({path.name})','- [统计摘要](summary.json)','',
        '状态：PARTIAL_MATRIX_ANALYSIS / COMPLETED_ROWS_VERIFIED。代码发布版本5bd1665631b15b1ed97fae0f6b0ed57f25b68ec9；本报告不修改远端release。']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('run_summary','seed_aggregates','paired_deltas')},indent=2))
    print(json.dumps(dict(leo_adv_delta_pp=st.mean(deltas),per_seed_deltas=deltas),indent=2))

if __name__=='__main__':main()
