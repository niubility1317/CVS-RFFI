"""Post-test only: full logs and same-row test reporting, never source selection."""
from pathlib import Path
import json,csv,statistics,math,sys,shutil
R=Path('E:/type10-7');W=R/'code/snapshots/daot_practical_three_20260918_wt';sys.path[:0]=[str(W),str(R/'tools')]
import experiment_registry as registry
from experiments.cvs_reference_residual_clean.contracts import RUN,SOURCE_RUN,RELEASE,SEEDS,VARIANTS,expected_rows,SOURCE_COMMIT
from experiments.cvs_reference_residual_identity.contracts import PRECISION
P=W/'automation_reports/CV-SincNet'/RUN;S=W/'automation_reports/CV-SincNet'/SOURCE_RUN
A=W/'local_artifacts'/RELEASE/'completed';T=A/'test';E=P/'evidence'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def write(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2,allow_nan=False)
def update(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def csvwrite(p,rows):
 with p.open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def pct(xs):return f'{100*statistics.mean(xs):.4f}%±{100*statistics.stdev(xs):.4f}%'
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
done=read(T/'scoring_clean_complete.json');proof=read(T/'independent_recount.json');data=read(T/'clean_scored_results.json');summary=read(T/'clean_summary.json')
assert done['status']=='SCORED_COMPLETE' and proof['status']=='VERIFIED' and done['rows']==8 and done['decisions']==1344000
assert set(expected_rows())=={r['row_id'] for r in data['results']} and len(data['results'])==64 and len(data['class_results'])==48
assert not (E/'full_training_audit.json').exists(),'Preserve completed analysis'
audits=[];resources=[];curves=[]
for rid,(variant,seed) in sorted(expected_rows().items()):
 folder=A/'source'/rid;epochs=lines(folder/'epoch_metrics.jsonl');compact=lines(folder/'epoch_compact.jsonl');steps=lines(folder/'step_metrics.jsonl')
 resolved=read(folder/'resolved_config.json');completion=read(folder/'completion.json');profile=read(folder/'resource_profile.json');initial=read(folder/'initialization.json')
 with (folder/'epoch_metrics.csv').open(encoding='utf-8',newline='') as f:table=list(csv.DictReader(f))
 assert len(epochs)==len(compact)==len(table)==200 and len(steps)==10000
 assert [r['epoch'] for r in epochs]==list(range(1,201)) and [r['step'] for r in steps]==list(range(1,10001))
 assert completion['epoch']==200 and completion['steps']==10000 and resolved['commit']==SOURCE_COMMIT and resolved['precision']==PRECISION
 assert initial['scratch_only'] is True and initial['ancestors']==[] and initial['checkpoint_sources']==[]
 assert all(r['clean_ce']==r['total_loss'] and r['ce_weight']==1 and not r['augmentation_active'] and not r['extra_losses_active'] and not r['pseudo_labels_active'] and r['response_active'] for r in steps)
 assert all(r['gradient_used_parameters']==189562 and all(math.isfinite(r[k]) for k in ('clean_ce','gradient_norm','ridge_gradient_norm','branch_gradient_norm')) for r in steps)
 assert sum(r['source_samples'] for r in steps)==1260000 and all(r['source_sample_exposure']==6300 and r['optimizer_steps']==50 for r in epochs)
 for i,(full,small,tab) in enumerate(zip(epochs,compact,table)):
  assert full['epoch']==small['epoch']==int(tab['epoch']) and full['clean_ce']==small['clean_ce']==float(tab['clean_ce'])
  part=steps[i*50:(i+1)*50];assert all(r['epoch']==i+1 and r['learning_rate']==full['learning_rate'] for r in part)
  assert math.isclose(statistics.mean(r['clean_ce'] for r in part),full['clean_ce'],rel_tol=1e-12)
  diag=full['response_diagnostics'];assert len(diag['ridge'])==9 and all(.0001<=v<=.1001 for v in diag['ridge'])
  assert all(math.isfinite(diag[k]) for k in ('fit_error_mean','inverse_abs_max','reference_quality_mean','residual_feature_rms','projection_weight_norm'))
  curves.append(dict(row_id=rid,variant=variant,model_seed=seed,epoch=full['epoch'],CE=full['clean_ce'],learning_rate=full['learning_rate'],
   gradient_norm=full['gradient_norm'],ridge_gradient_norm=full['ridge_gradient_norm'],branch_gradient_norm=full['branch_gradient_norm'],
   source_V_accuracy=full['source_val_accuracy'],source_V_worst_RX=full['source_val_worst_rx'],fit_error=diag['fit_error_mean'],
   inverse_abs_max=diag['inverse_abs_max'],residual_feature_rms=diag['residual_feature_rms'],projection_weight_norm=diag['projection_weight_norm']))
 stdout=(folder/'stdout.log').read_text(encoding='utf-8');markers=[s for s in ('Traceback (most recent call last)','CUDA out of memory','Nonfinite gradient','Nonfinite CE','Killed') if s in stdout]
 assert not markers,markers
 assert sum(s.startswith('EPOCH ') for s in stdout.splitlines())==200
 final=read(folder/'source_final_diagnostics.json');assert final['count']==27000 and len(final['groups'])==90 and sum(g['count'] for g in final['groups'])==27000
 assert all(len(g['unit_embedding_mean'])==160 for g in final['groups'])
 diag=epochs[-1]['response_diagnostics']
 audits.append(dict(row_id=rid,variant=variant,model_seed=seed,epochs=200,steps=10000,source_exposures=1260000,source_V_count=27000,
  first_CE=epochs[0]['clean_ce'],last_CE=epochs[-1]['clean_ce'],min_CE=min(r['clean_ce'] for r in epochs),
  first_LR=epochs[0]['learning_rate'],last_LR=epochs[-1]['learning_rate'],source_final_accuracy=completion['final_source_metrics']['source_val_accuracy'],
  source_final_worst_RX=completion['final_source_metrics']['source_val_worst_rx'],all_gradients_finite=True,
  ridge_gradient_zero_steps=sum(r['ridge_gradient_norm']==0 for r in steps),branch_gradient_zero_steps=sum(r['branch_gradient_norm']==0 for r in steps),
  final_ridge_min=min(diag['ridge']),final_ridge_max=max(diag['ridge']),final_projection_norm=diag['projection_weight_norm'],
  final_fit_error=diag['fit_error_mean'],final_residual_feature_rms=diag['residual_feature_rms'],final_inverse_abs_max=diag['inverse_abs_max'],
  train_seconds=completion['elapsed_seconds'],source_stdout_error_markers=markers,stdout_warning_lines=sorted({s.strip() for s in stdout.splitlines() if 'warning' in s.lower()}),logs_scope='Full200epoch/10000step/stdout/CSV/compact; no target feedback'))
 resources.append(dict(row_id=rid,variant=variant,model_seed=seed,hardware=profile['hardware'],precision=profile['precision'],
  total_parameters=profile['total_parameters'],trainable_parameters=profile['trainable_parameters'],resident_state_bytes=profile['resident_state_bytes'],
  nonpersistent_buffer_bytes=profile['nonpersistent_buffer_bytes'],partial_mixed_macs=profile['conv_linear_macs_per_sample'],
  inference_batch1_ms=profile['inference_batch1_ms'],inference_batch128_ms=profile['inference_batch128_ms'],training_batch128_ms=profile['training_batch128_ms'],
  benchmark_peak_cuda_bytes=profile['benchmark_peak_cuda_allocated_bytes'],source_peak_cuda_bytes=max(r['peak_cuda_allocated_bytes'] for r in epochs),
  clean_prediction_seconds=read(T/rid/'clean_complete.json')['prediction_seconds'],clean_prediction_peak_cuda_bytes=read(T/rid/'clean_complete.json')['peak_cuda_allocated_bytes']))
for name in ('clean_scored_results.json','clean_summary.json','clean_scored_results.csv','clean_summary.csv','clean_class_results.csv','clean_paired.csv','scoring_clean_complete.json','independent_recount.json'):
 assert not (E/name).exists();shutil.copyfile(T/name,E/name)
write(E/'full_training_audit.json',dict(status='VERIFIED',rows=audits,total_epochs=1600,total_steps=80000,source_exposures=10080000,used_for_selection=False))
csvwrite(P/'full_training_audit.csv',audits);csvwrite(P/'full_training_curves.csv',curves);csvwrite(P/'resource_measurements.csv',resources)
lookup={(r['variant'],r['group'],r['model_seed']):r for r in data['results']}
old=read(W/'automation_reports/CV-SincNet/20261003-phase1-cvs-all-frozen-clean-backfill-manysig-m304-r01/evidence/test_results.json')['results']
comparators=['native','residual_fusion'];oldlookup={(r['variant'],r['group'],r['model_seed']):r for r in old if r['variant'] in comparators}
assert len(oldlookup)==2*8*4
paired=[]
for variant in VARIANTS:
 for baseline in comparators:
  delta=[100*(lookup[(variant,'ALL',s)]['accuracy']-oldlookup[(baseline,'ALL',s)]['accuracy']) for s in SEEDS]
  paired.append(dict(variant=variant,baseline=baseline,mean_pp=statistics.mean(delta),seed_sd_pp=statistics.stdev(delta),positive_seeds=sum(v>0 for v in delta),per_seed_pp=delta))
csvwrite(P/'paired_historical_comparisons.csv',paired)
text='''# 参考约束残差网络：完整测试结果

**状态：VERIFIED。全部8份E200权重完成同168000物理query的clean测试、独立评分和NumPy重算。** 共1344000个分类决定，64条ALL/RX和48条TX记录。完整日志核查覆盖1600轮、80000步、10080000次源样本曝光。

本报告以真实测试判断识别效果。源域指标只描述训练过程。两臂各四seed全部保留，无额外损失、增强、预训练、适应或query拟合。

## 测试准确率

四seed均值±样本标准差；每seed先取最差RX再跨seed汇总。历史对照使用已完成同物理ID预测，仅用于报告比较，没有用于本轮设计或选择。

| 结构 | 准确率 | Macro-F1 | 最差RX准确率 |
|---|---:|---:|---:|
'''
for variant in [*comparators,*VARIANTS]:
 table=lookup if variant in VARIANTS else oldlookup
 worst=[min(r['accuracy'] for (v,g,s),r in table.items() if v==variant and s==seed and g!='ALL') for seed in SEEDS]
 text+=f"| {variant} | {pct([table[(variant,'ALL',s)]['accuracy'] for s in SEEDS])} | {pct([table[(variant,'ALL',s)]['macro_f1'] for s in SEEDS])} | {pct(worst)} |\n"
text+='\n## 配对差值\n\n单位为百分点，四个相同model seed逐一相减。完整差值见[配对CSV](paired_historical_comparisons.csv)。\n\n| 新结构 | 对照 | 平均变化±标准差 | 正向seed |\n|---|---|---:|---:|\n'
for q in paired:text+=f"| {q['variant']} | {q['baseline']} | {q['mean_pp']:+.4f}±{q['seed_sd_pp']:.4f} | {q['positive_seeds']}/4 |\n"
direct=next(r for r in summary['paired'] if r['group']=='ALL')
text+=f"\n逐频减标量：{direct['mean_pp']:+.4f}±{direct['sd_pp']:.4f}个百分点，{direct['positive_seeds']}/4个seed为正。\n"
text+='\n## 逐接收机\n\n准确率为四seed均值。完整标准差和各seed值见CSV。\n\n| RX | 标量补偿 | 逐频补偿 | 逐频减标量（百分点） |\n|---|---:|---:|---:|\n'
for group in sorted({g for _,g,_ in lookup if g!='ALL'}):
 a=statistics.mean(lookup[(VARIANTS[0],group,s)]['accuracy'] for s in SEEDS);b=statistics.mean(lookup[(VARIANTS[1],group,s)]['accuracy'] for s in SEEDS)
 text+=f'| {group} | {100*a:.4f}% | {100*b:.4f}% | {100*(b-a):+.4f} |\n'
text+='\n## 结构执行与资源\n\n完整步骤核查只执行原CE，所有参数梯度有限。九个正则值保持登记范围；正则/分支梯度的零步数和最终出口范数见[完整训练审计](full_training_audit.csv)。可微分支实际执行不代表学到了真实信道逆。\n\n| 结构 | 参数 | 单样本推理ms | batch128训练ms | 持久状态字节 | 非持久buffer字节 |\n|---|---:|---:|---:|---:|---:|\n'
for variant in VARIANTS:
 rr=[r for r in resources if r['variant']==variant];mean=lambda k:statistics.mean(r[k] for r in rr)
 text+=f"| {variant} | {rr[0]['total_parameters']} | {mean('inference_batch1_ms'):.4f} | {mean('training_batch128_ms'):.4f} | {mean('resident_state_bytes'):.0f} | {mean('nonpersistent_buffer_bytes'):.0f} |\n"
text+='''
资源实测使用N607 RTX3090，显式FP64/complex128物理层，FP32主干与参数，无TF32。独立clone做5次warmup、20次推理、10次训练计时；并行负载影响时延。profile峰值含冻结原模型与clone，原训练/预测峰值分别报告。MAC为混合实复数的局部计数，未覆盖solve/bmm/FFT，不能称为完整FLOPs。[资源CSV](resource_measurements.csv)保留所有测量。

## 科学边界与完整输出

单个周期已知激励不能普遍分离TX、H、RX。公开未训练探针中，逐频补偿降低部分IQ漂移，却放大PA/记忆漂移；负结果完整保留。真实识别测试检验组合架构，不能单独把差异归因于信道消除、某个正则参数或完全TX/RX解耦。原始IQ/PA路径保留的作用仍须额外固定消融证据。论文新颖性不由一次分数或参考均衡本身成立。

历史基准已暴露，本次不宣称全新盲测；禁止评分回流调参、结构选择或选择性重跑。无Phase2、support、新类或卫星视图，K×新增类数和适应前后指标N/A；星载传输/部署未测量。

此前登记范围304份历史权重补测、8份validdual测试已完成，本轮再完成8份，总计320份具备clean测试结果；不是所有项目/场景的无限范围覆盖。

完整输出：[逐seed/RX](evidence/clean_scored_results.csv)、[逐TX](evidence/clean_class_results.csv)、[四seed汇总](evidence/clean_summary.csv)、[原始预测独立复算](evidence/independent_recount.json)、[1600轮完整曲线](full_training_curves.csv)。数据、权重、预测、日志均保留原路径。
'''
text+='\n源执行提交 `'+SOURCE_COMMIT+'`。完整收集目录：`'+str(A)+'`。\n'
(P/'report.md').write_text(text,encoding='utf-8')
for folder in (P,S):
 spec=read(folder/'experiment.json');spec['status']='ANALYZED';spec['outcome']=dict(status='VERIFIED',test_run=RUN,test_rows=8,test_decisions=1344000,full_log_epochs=1600,full_log_steps=80000,target_feedback=False,paper_claim_established=False)
 for row in spec['rows']:
  row['status']='ANALYZED'
  registry.event(W,spec['run_id'],'ANALYZED',['../'+RUN+'/evidence/independent_recount.json'],'Fixed row E200 and clean prediction/scoring complete',row['row_id'])
 if folder==S:spec['test_completion_plan'].update(status='VERIFIED',completed_test_run=RUN)
 update(folder/'experiment.json',spec)
 h=read(folder/'handoff.json');h.update(status='ANALYZED',test_run=RUN,all_registered_test_rows_complete=True,completed_clean_checkpoint_scope=320,performance_goal_achieved=False,paper_claim_established=False,next=['All8 source/test/recount complete; preserve outcomes','Do not rerun or use target scores for architecture/hyperparameter tuning']);update(folder/'handoff.json',h)
p=S/'report.md';s=p.read_text(encoding='utf-8').replace('状态：RUNNING，远端发布及独立启动读回VERIFIED。','状态：ANALYZED，全部训练及预登记测试完成，独立复算VERIFIED。');s+='\n全部8行E200/80000步与完整日志已核查；最终性能以[测试报告](../'+RUN+'/report.md)为准。\n';p.write_text(s,encoding='utf-8')
registry.event(W,RUN,'ANALYZED',['evidence/independent_recount.json','evidence/full_training_audit.json','report.md'],'All8 clean test/recount and full80000step logs VERIFIED; complete outcomes retained')
registry.event(W,SOURCE_RUN,'ANALYZED',['../'+RUN+'/report.md'],'All8 fixedE200 source models received complete registered final clean test')
print(json.dumps(dict(status='VERIFIED',paired=paired,inverse_minus_scalar=direct),ensure_ascii=False))
