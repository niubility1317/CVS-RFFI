from pathlib import Path
import json,csv,statistics,shutil,sys
R=Path('E:/type10-7');W=R/'code/snapshots/daot_practical_three_20260918_wt';sys.path.insert(0,str(W))
from experiments.cvs_reference_residual_clean.contracts import RUN,SOURCE_RUN,RELEASE,SOURCE_COMMIT,VARIANTS,SEEDS
P=W/'automation_reports/CV-SincNet'/RUN;S=W/'automation_reports/CV-SincNet'/SOURCE_RUN;A=W/'local_artifacts'/RELEASE
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def write(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert read(P/'experiment.json')['status']=='ANALYZED'
final=read(P/'evidence/final_readback.json');assert final['pipeline']['status']=='SCORED_COMPLETE'
shutil.copyfile(A/'completed/source/frozen_source_matrix.json',S/'evidence/frozen_source_matrix.json')
base=W/'automation_reports/CV-SincNet/20261003-phase1-cvs-all-frozen-clean-backfill-manysig-m304-r01'
with (base/'test_all_models.csv').open(encoding='utf-8',newline='') as f:
 reader=csv.DictReader(f);columns=reader.fieldnames;allmodels=list(reader)
assert len(allmodels)==76 and len({r['model_id'] for r in allmodels})==76
old=read(base/'evidence/test_results.json')['results'];baseline={(r['variant'],r['group'],r['model_seed']):r for r in old if r['variant'] in ['native','residual_fusion']}
for test_run,source_run,variants,prefix in [
 ('20261003-phase1-cvs-validdual-clean-manysig-m8-r01','20261003-phase1-cvs-validdual-identity-manysig-m8-r01',('validdual_static','validdual_dynamic'),'validdual'),
 (RUN,SOURCE_RUN,VARIANTS,'reference_residual')]:
 records=read(W/'automation_reports/CV-SincNet'/test_run/'evidence/clean_scored_results.json')['results']
 lookup={(r['variant'],r['group'],r['model_seed']):r for r in records}
 for v in variants:
  row=dict(group='ALL',model_id=prefix+'/'+v,source_run=source_run,variant=v,query_count_per_seed=168000)
  for key in ('accuracy','macro_accuracy','macro_f1'):
   values=[lookup[(v,'ALL',s)][key] for s in SEEDS];row[key+'_mean']=statistics.mean(values);row[key+'_seed_sd']=statistics.stdev(values)
  for b,k in [('native','native'),('residual_fusion','residual')]:
   values=[100*(lookup[(v,'ALL',s)]['accuracy']-baseline[(b,'ALL',s)]['accuracy']) for s in SEEDS]
   row[k+'_delta_pp_mean']=statistics.mean(values);row[k+'_delta_pp_sd']=statistics.stdev(values)
  worst=[min(r['accuracy'] for (variant,g,s),r in lookup.items() if variant==v and s==seed and g!='ALL') for seed in SEEDS]
  row['worst_rx_accuracy_mean']=statistics.mean(worst);row['worst_rx_accuracy_seed_sd']=statistics.stdev(worst)
  assert set(row)==set(columns);allmodels.append(row)
assert len(allmodels)==80 and len({r['model_id'] for r in allmodels})==80
with (P/'all_registered_clean_80_groups.csv').open('x',encoding='utf-8',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(allmodels)
write(P/'evidence/test_coverage.json',dict(status='VERIFIED',scope='Registered Phase1 clean architecture/precision groups; not satellite or Phase2',
 historical_backfill_checkpoints=304,historical_new_predictions=216,historical_reused_predictions=88,completed_validdual_checkpoints=8,
 new_reference_residual_checkpoints=8,total_checkpoints=320,total_model_groups=80,seeds_per_group=4,query_count_per_checkpoint=168000,
 total_classification_decisions=53760000,new_test_decisions_this_round=1344000,pending_frozen_checkpoints_in_this_registered_scope=0,
 sources=[str(base/'report.md'),'../20261003-phase1-cvs-validdual-clean-manysig-m8-r01/report.md','report.md']))
summary=read(P/'evidence/clean_summary.json');results=read(P/'evidence/clean_scored_results.json')['results'];pairs=list(csv.DictReader((P/'paired_historical_comparisons.csv').open(encoding='utf-8')))
conclusion='''## 性能结论

新增残差补偿分支没有证明超越其原始残差融合基准。标量补偿测试78.3841%±1.0039%，逐频补偿78.2048%±0.8133%；相对残差融合的平均变化分别为−0.0702和−0.2496个百分点。两种结构虽高于原生CVS，但原始残差融合本身已有该优势，不能把与原生CVS的差值全部归功于新增模块。

逐频减标量为−0.1793±0.4374个百分点，四seed中两正两负；当前结果不支持逐频逆响应优于同参数量的标量控制。此处没有做统计显著性检验，不能把均值小幅下降写成确定的总体劣势，也不能宣称性能优化目标已达成。论文贡献仍未由本轮结果建立。

'''
p=P/'report.md';text=p.read_text(encoding='utf-8').replace('## 测试准确率',conclusion+'## 测试准确率',1)
text+='\n## 全量补测覆盖与交付\n\n[80组、320份权重完整测试汇总](all_registered_clean_80_groups.csv)保留76组历史补测、2组validdual及本轮2组，按登记顺序列出，不按测试分数选择或重排。此登记范围内待补测的冻结权重为0。共53760000个分类决定均有已完成评分依据；本轮新增1344000个。\n\n测试执行提交 `'+final['pipeline']['commit']+'`，dispatcher PID `'+str(final['pipeline']['pid'])+'`。初始启动读回遇到一份已完成worker退出时的/proc竞态；其完成产物和后续终态独立读回均已确认，未重启、未重复预测。完整8份源E200、8份测试预测、独立评分和复算VERIFIED。\n'
p.write_text(text,encoding='utf-8')
for folder in (P,S):
 spec=read(folder/'experiment.json');spec['outcome'].update(all_registered_test_rows_complete=True,completed_clean_checkpoint_scope=320,performance_goal_achieved=False,paper_claim_established=False);write(folder/'experiment.json',spec)
 h=read(folder/'handoff.json');h.update(status='ANALYZED',real_test_complete=True,goal_complete=False,performance_goal_achieved=False,paper_claim_established=False,
  completed_clean_checkpoint_scope=320,test_summary={'reference_residual_scalar_accuracy':.7838407738095238,'reference_residual_inverse_accuracy':.7820476190476191},
  next=['All320 registered Phase1 clean checkpoints tested; no pending frozen weights in this scope','Current added branch did not demonstrate gain over its raw residual-fusion base; preserve all outcomes','No target-feedback redesign/tuning/selective rerun; further research needs source/public evidence']);write(folder/'handoff.json',h)
for name in ('analyze_reference_results.py','finalize_reference_results.py'):
 shutil.copyfile(R/'.codex_tmp'/name,P/'evidence'/name)
print(json.dumps(dict(status='VERIFIED',model_groups=80,checkpoints=320,decisions=53760000,pending_in_registered_scope=0)))
