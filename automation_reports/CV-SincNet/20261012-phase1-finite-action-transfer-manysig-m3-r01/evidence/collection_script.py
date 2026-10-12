from pathlib import Path
import sys,json,time,math,statistics
R=Path('E:/type10-7/code/worktrees/report_physical_ssl_delivery_20261011');sys.path.insert(0,str(R))
from experiments.cvs_finite_action import design as d
from experiments.cvs_phase1_stack.publish import ssh
folder=R/'automation_reports/CV-SincNet'/d.RUN
payload=r'''
from pathlib import Path
import json
p=Path(BASE)
completion=json.loads((p/'completion.json').read_text())
assert completion['status']=='SOURCE_DIAGNOSTIC_COMPLETE' and len(completion['rows'])==3 and not completion['identity_updates'] and not completion['target_access']
names=['completion.json','analysis.json','final_analysis.md','experiment.json']
print(json.dumps({name:(p/name).read_text(encoding='utf-8') for name in names}))
'''
values=json.loads(ssh(payload.replace('BASE',repr(d.BASE.as_posix()))));v=json.loads(values['analysis.json'])
assert v['status']=='COMPLETE' and v['source_only'] and not v['identity_updates'] and not v['target_access'] and not v['U_truth']
assert v['auxiliary_fits']==351 and v['auxiliary_optimizer_steps']==351540 and v['source_model_packet_occurrences']==1474200
assert [r['row_id'] for r in v['rows']]==[r['row_id'] for r in d.rows()]
for row,expected in zip(v['rows'],d.rows()):
 c=d.config(expected);res=row['result'];done=row['completion']
 assert res['config']==done['config']==c and not res['U_fit'] and not res['U_truth'] and not res['V_forwards'] and not res['target_access']
 assert len(res['fits'])==117 and len(res['records'])==630 and len(res['coverage'])==45 and len(row['epoch_groups'])==117
 assert sum(len(e) for e in row['epoch_groups'].values())==2340 and sum(f['optimizer_steps'] for f in res['fits'])==117180
 assert done['budget']['reference_parameters_buffers_modes_gradients_unchanged'] and not done['identity_updates'] and not done['head_updates']
 for fit in res['fits']:
  epochs=row['epoch_groups'][':'.join(fit[k] for k in ('split','kind','level'))]
  assert [e['epoch'] for e in epochs]==list(range(1,21)) and epochs[-1]['optimizer_steps']==fit['optimizer_steps']
  assert fit['fit_eval_PID_disjoint'] and not fit['U_fit'] and fit['fit_normalization_only']
for name in ['completion.json','final_analysis.md']:(folder/name).write_text(values[name],encoding='utf-8')
d.write(folder/'evidence/source_full_analysis.json',v)
summary=[];coverage=[];operations=[]
families={'PID':['PID80_20'],'heldTX':[f'TX{i}' for i in range(6)],'heldDay':[f'day{i}' for i in (1,2,3)],'heldQuality':[f'quality{i}' for i in range(3)],'U':['U_coverage']}
for kind in d.KINDS:
 for level in d.LEVELS:
  for family,splits in families.items():
   paired=[]
   for row in v['rows']:
    rec={(r['split'],r['method']):r for r in row['result']['records'] if r['kind']==kind and r['level']==level}
    for split in splits:
     a=rec[split,'MLP'];b=rec[split,'bilinear_ridge'];l=rec[split,'lowrank8'];zero=rec[split,'zero']
     paired.append(dict(seed=row['model_seed'],split=split,MLP=a['normalized_repeat_MSE'],bilinear=b['normalized_repeat_MSE'],lowrank=l['normalized_repeat_MSE'],zero=zero['normalized_repeat_MSE'],MLP_mean_MSE=a['mean_prediction_MSE'],bilinear_mean_MSE=b['mean_prediction_MSE'],irreducible_fraction=a['empirical_within_repeat_variance']/a['total_empirical_delta_energy'] if a['total_empirical_delta_energy']>1e-12 else None))
   def avg(key):return None if any(p[key] is None for p in paired) else statistics.mean(p[key] for p in paired)
   gain=[p['bilinear']-p['MLP'] for p in paired if p['bilinear'] is not None and p['MLP'] is not None]
   summary.append(dict(kind=kind,level=level,family=family,pairs=paired,MLP=avg('MLP'),bilinear=avg('bilinear'),lowrank=avg('lowrank'),zero=avg('zero'),bilinear_minus_MLP=statistics.mean(gain) if len(gain)==len(paired) else None,positive_pairs=sum(g>0 for g in gain),n=len(paired),irreducible_fraction=avg('irreducible_fraction')))
  for method in c['methods']:
   a=[next(x for x in row['result']['coverage'] if x['kind']==kind and x['level']==level and x['method']==method) for row in v['rows']]
   coverage.append(dict(kind=kind,level=level,method=method,per_seed=[dict(seed=row['model_seed'],**x) for row,x in zip(v['rows'],a)],coverage_mean=statistics.mean(x['coverage_fraction'] for x in a) if all(x['coverage_fraction'] is not None for x in a) else None))
for row in v['rows']:
 res=row['result']
 for op in res['L_operations']:operations.append(dict(seed=row['model_seed'],clean_head_accuracy=res['L_clean_head_accuracy'],clean_minus_operated_pp=100*(res['L_clean_head_accuracy']-op['head_accuracy']),**op))
evidence=dict(status='VERIFIED',run=d.RUN,source_only=True,selection=False,target_access=False,identity_updates=0,summary=summary,coverage=coverage,operations=operations)
d.write(folder/'evidence/source_interpretation.json',evidence)
fmt=lambda n:'N/A' if n is None else f'{n:.6f}'
lines=['# 有限作用：迁移、简单基线与身份保留证据','', '完整三种子、351个拟合、351540次辅助更新、1474200次源模型样本读出已经闭合。完整7020个epoch由远端自动解析JSONL/CSV并核对，本地读回全量分组再核对。身份更新0，U拟合0，target访问0。','', '所有作用、层和留出条件均完整保留。以下为固定比较MLP与双线性ridge，按seed×留出条件等权均值；没有选出某个最佳seed/作用，不能把辅助拟合称为身份泛化提升。','', '|作用|层|留出组|零均值|双线性|低秩8|MLP|双线性−MLP|正差/总数|经验方差占比|','|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in summary:lines.append('|'+x['kind']+'|'+x['level']+'|'+x['family']+'|'+'|'.join(fmt(x[k]) for k in ['zero','bilinear','lowrank','MLP','bilinear_minus_MLP'])+f"|{x['positive_pairs']}/{x['n']}|{fmt(x['irreducible_fraction'])}|")
lines+=['','误差为（预测均值误差＋四次噪声实现样本内方差）/有限差分总能量；越小越好。正差只说明该固定源验证条件下MLP优于双线性，不是独立身份收益或显著性结论。noise方差占比为有限四次实现的经验下限，不是已知真实条件方差；均值MSE/NLL/方差误差/逐seed数据仍见原完整analysis。','', '|作用|层|方法|独立真实源RX残差覆盖均值|','|---|---|---|---:|']
for x in coverage:lines.append(f"|{x['kind']}|{x['level']}|{x['method']}|{fmt(x['coverage_mean'])}|")
lines+=['','每个子空间只由fit有限差分估计，rank最多6。独立L的同TX/day五RX条件均值用于投影；90个条件单元、不是配对包，也不是未知RX。覆盖高不证明身份保留，覆盖低不等于所有非线性物理模型无效；浅中表示仅通道均值/std，不能代表完整时序图。','', '|seed|作用|参数点|clean源头准确率%|操作后源头准确率%|下降百分点|IQ变化能量|','|---|---|---|---:|---:|---:|---:|']
for x in operations:lines.append(f"|{x['seed']}|{x['kind']}|{x['a']}|{100*x['clean_head_accuracy']:.4f}|{100*x['head_accuracy']:.4f}|{x['clean_minus_operated_pp']:.4f}|{x['relative_IQ_change_energy']:.6f}|")
lines+=['','固定身份头在操作后的源准确率仅为兼容性诊断；下降可能是头脆弱或表示变化，不能单独证明IQ身份被破坏。若以后用作用约束身份，仍须同时检验类别margin、净修复及参数更新方向；本批没有这些身份训练证据，84%目标未达成。','', '低秩基线严格为fit输出PCA rank8＋双线性ridge，并非最优RRR；数字FIR/插值/加性噪声仅受控代理，不称真实RX反事实。前导N/A：当前256窗协议确认重复区间未获得，lag16仅质量代理。所有数字与表由完整源记录产生；无target训练/选择/排名/重跑。']
(folder/'source_interpretation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
s=d.read(folder/'experiment.json');s.update(status='ANALYZED',delivery_verdict='VERIFIED',progress=dict(completed_rows=3,auxiliary_fits=351,auxiliary_optimizer_steps=351540,source_packets=1474200,source_epochs=7020,records=1890,coverage_records=135,identity_updates=0,target_access=False,goal_status='NOT_MET'))
for row in s['rows']:row.update(status='SOURCE_DIAGNOSTIC_COMPLETE',auxiliary_fits=117,auxiliary_optimizer_steps=117180)
d.write(folder/'experiment.json',s)
with (folder/'events.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(event='COMPLETE_ANALYZED_VERIFIED',time=time.time(),**s['progress']))+'\n')
with (folder/'report.md').open('a',encoding='utf-8') as f:f.write('\nVERIFIED / ANALYZED：3/3源作用诊断完成，351fit/351540aux更新/1474200源读出/7020完整epoch/1890评估/135覆盖，身份更新0。完整比较见[final_analysis.md](final_analysis.md)，迁移/基线/身份保留解释见[source_interpretation.md](source_interpretation.md)。没有新的target成绩，84%仍未达成。\n')
d.write(folder/'handoff.json',dict(status='ANALYZED',run=d.RUN,execution_commit=s['execution_commit'],results='source_interpretation.md',next='No relaunch. All source evidence retained. Inspect source transfer/control/realRX coverage/identity-preservation before independently preregistered identity mechanism; no target-driven tuning.'))
print(json.dumps(dict(status='VERIFIED',auxiliary_fits=351,source_packets=1474200,identity_updates=0,summary=[{k:value for k,value in item.items() if k!='pairs'} for item in summary]),ensure_ascii=False))
