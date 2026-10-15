from pathlib import Path
import sys,json,csv,io,importlib.util,statistics
W=Path(__file__).resolve().parents[2];sys.path[:0]=[str(W),str(W/'code')]
from experiments.cvs_rdrop_budget import design as d
from experiments.cvs_rdrop_budget.publish import PYTHON
A=W/'automation_reports/CV-SincNet'/d.RUN;P=W/'local_artifacts'/d.RELEASE
if (A/'results_analysis.json').is_file():print('ALREADY_RECORDED');sys.exit(0)
v=d.read(P/'independent_readback.json');x=d.read(P/'independent_analysis.json')
assert x['status']=='VERIFIED' and v['completion']['status']=='ARTIFACTS_COMPLETE' and not v['dispatcher'] and not any(r['process'] for r in v['rows']) and not v['failure']
assert len(x['rows'])==12 and len(x['records'])==216 and sum(r['optimizer_steps'] for r in x['rows'].values())==93240
def public(value):
 if isinstance(value,str):return value.replace(d.PROJECT,'PROJECT').replace(PYTHON.rsplit('/bin/python',1)[0],'PYTHON_ENV')
 if isinstance(value,list):return [public(a) for a in value]
 if isinstance(value,dict):return {k:public(a) for k,a in value.items()}
 return value
x=public(x)
def table(name,rows):
 buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows);(A/name).write_text(buf.getvalue(),encoding='utf-8',newline='\n')
overview=[];curves=[];cost=[]
for rid,r in x['rows'].items():
 overall=next(a for a in x['records'] if a['row_id']==rid and a['dimension']=='overall');source=r['source_V'];last=r['complete_epoch_curves'][-1]
 overview.append(dict(row_id=rid,arm=r['arm'],seed=r['model_seed'],epochs=r['epochs'],accuracy=overall['accuracy'],macro_f1=overall['macro_f1'],source_V_accuracy=source['source_val_accuracy'],source_V_worst_RX=source['source_val_worst_rx'],final_train_CE1=r['final_training_L_CE'],final_pairedCE=last['rdrop/paired_CE'],final_KL=last['rdrop/symmetric_KL'],selected_U=r['pseudo_selected_observations'],U_observations=r['pseudo_observations'],training_seconds=r['elapsed_seconds'],prediction_seconds=r['prediction_seconds'],feature_L2_delta=r['feature_parameter_L2_delta'],default_W_L2_delta=r['default_W_L2_delta'],source_health=r['source_health']))
 for epoch in r['complete_epoch_curves']:curves.append(dict(row_id=rid,arm=r['arm'],seed=r['model_seed'],**epoch))
 for stage,name in ((True,'L_only'),(False,'L_plus_U')):
  eps=[a for a in r['complete_epoch_curves'] if a['label_stage']==stage]
  cost.append(dict(row_id=rid,arm=r['arm'],seed=r['model_seed'],stage=name,epochs=len(eps),mean_epoch_seconds=statistics.mean(a['epoch_time_s'] for a in eps),total_epoch_seconds=sum(a['epoch_time_s'] for a in eps)))
table('summary_by_seed.csv',overview);table('training_complete_curves.csv',curves);table('paired_errors.csv',x['paired_errors']);table('paired_strata.csv',x['stratum_pairs']);table('training_stage_cost.csv',cost)
metric=[{k:v for k,v in r.items() if k!='confusion'} for r in x['records']];table('metrics_strata.csv',metric);d.write(A/'results_analysis.json',x)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
colors={'E20_TWO_CE':'#24549b','E20_RDROP':'#c34e36','E50_TWO_CE':'#459eac','E50_RDROP':'#8d4e9d'}
keys=('rdrop/paired_CE','rdrop/symmetric_KL','train/loss','source_V_accuracy_percent','observed/U_selected','epoch_time_s')
titles=('Mean labeled CE','Symmetric KL','Actual training objective','Source V accuracy (%)','Accepted U / update','Epoch wall time (s)')
for arm in d.ARMS:
 rs=[r for r in x['rows'].values() if r['arm']==arm];count=rs[0]['epochs'];grid=range(1,count+1)
 for ax,key in zip(axes.flat,keys):
  values=[[r['complete_epoch_curves'][i][key] for r in rs] for i in range(count)];means=[statistics.mean(a) for a in values];sds=[statistics.stdev(a) for a in values]
  ax.plot(grid,means,label=arm,color=colors[arm],linewidth=1.5);ax.fill_between(grid,[a-b for a,b in zip(means,sds)],[a+b for a,b in zip(means,sds)],color=colors[arm],alpha=.09)
for ax,title in zip(axes.flat,titles):ax.set_title(title,fontsize=10);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
axes[0,0].legend(fontsize=8);fig.suptitle('Full five-source-RX scratch training | 3 fixed seeds | mean +/- sample SD',fontsize=12)
fig.savefig(A/'training_curves.png',dpi=170);plt.close(fig)
title=(A/'report.md').read_text(encoding='utf-8').split('\n',1)[0]
lines=[title,'','状态 **ANALYZED / 全12行训练、clean预测、独立truth-last评分与完整产物分析 VERIFIED**。E20/E50各TWO_CE和RDROP三种子，均使用从零训练后固定final默认头。目标仍active；本轮结果按以下预登记4组和4个配对完整报告，不按最高target分数换主条件。','',
'| 条件 | seed01 | seed02 | seed03 | 均值±样本SD | 最差RX均值 | MacroF1均值 |','|---|---:|---:|---:|---:|---:|---:|']
for arm in d.ARMS:
 s=x['summary'][arm];ps=sorted(s['per_seed'],key=lambda a:a['model_seed'])
 lines.append(f"| {arm} | {ps[0]['accuracy']*100:.6f}% | {ps[1]['accuracy']*100:.6f}% | {ps[2]['accuracy']*100:.6f}% | {s['accuracy_mean']*100:.6f}% ± {s['accuracy_sample_SD']*100:.6f}pp | {s['worst_RX_accuracy_mean']*100:.6f}% | {s['macro_f1_mean']:.6f} |")
lines+=['','| 预登记配对 | 平均差 | 配对样本SD | 正收益种子 |','|---|---:|---:|---:|']
for name,s in x['paired_summary'].items():lines.append(f"| {name} | {s['mean_pp']:+.6f}pp | {s['sample_SD_pp']:.6f}pp | {s['positive_seeds']}/3 |")
lines+=['',f"主条件E50_RDROP clean三seed均值为{x['summary']['E50_RDROP']['accuracy_mean']*100:.6f}%，84%检查={x['E50_RDROP_goal84_check']}；E20_RDROP预算条件为{x['summary']['E20_RDROP']['accuracy_mean']*100:.6f}%，84%检查={x['E20_RDROP_goal84_check']}。这两项属于已暴露benchmark上的默认头验证，不证明未见TX、卫星或Phase2能力；完整研究目标未据此自动完成。",'',
f"独立CPU分析完整解析12份step压缩日志的全部93240条更新、420条epoch、全部12份stdout、24份初末checkpoint、2016000个既有默认头决定和216个分层混淆矩阵。每行实际更新等于epochs×222，13L+7U/33L+17U，原始物理角色6300/56700/27000、scratch空祖先、ownEMA、固定final、headfit0均核实。全部四条件的每seed初始200state一致：{x['initial_state_equal_count_by_seed']}。源V与L健康仅诊断，未回流择轮/配方或选择性重跑。",'',
f"真实目标逐步核验：native已含CE1，实际额外项为.5(CE2−CE1)+alphaKL；最大绝对重构误差{max(r['objective_max_abs_error'] for r in x['rows'].values()):.3g}，全部相关损失有限。原生熵没有独立日志标量，表中`derived/original_weighted_U_entropy`是closedloss扣除CE1与weightedU后的派生量，不能当作独立观测。U真值读取0，接受数只反映覆盖量，不能推断真实U正确率。",'',
f"全部模型实际特征参数和默认W均更新；96个合成packet前向进一步确认初始化至末态joint160改变、feature-only等于部署joint160逐位一致，临时扰动W不改变joint160。该分析新增真实IQ读取0、优化/头拟合0、checkpoint改写0、新target预测0；truth只在全row完整预测验证后独立读取复算。特征路径检查不等于迁移泛化证明。",'',
f"总训练成本420modelepoch/93240更新/11934720额外L前向，主L前向同量、8164800次U重复观察。矩阵从首次worker准入至12行评分汇总完成墙钟{x['matrix_elapsed_seconds']:.2f}s，实测吞吐{x['aggregate_completed_rows_per_hour']:.3f}模型/小时；受共享GPU上其他owner负载影响，不冒称单模型独占成本。各stage/row训练与预测耗时见training_stage_cost/summary。初期12worker分散8卡、各3348MiB；所有进程自然结束，原路径完整保留，没有干预其他任务。",'',
'E20与E50差异同时包括cosine学习率终点和13+7/33+17阶段分配，因此预算效果不能解释为同一20轮轨迹继续训练。两次L前向在TWO_CE和RDROP中匹配；KL的隔离效应只读同预算配对，跨预算主条件直接比较不能替代该消融。固定alpha1/0，LS0、原生dropout、SIG_TIME/CENTER25K/默认头/旧U配方不变，没有额外候选、换头或测试反馈调参。', '',
'此前源RX1留出单seed的+5.873016pp是开展依据，不等同本轮目标泛化结论。NONE默认头E20三seed小幅正收益、CENTER25K等历史正证据和全部负结果继续独立保留；本轮没有把它们或交叉拟合教师/权重平均叠加。RDrop为已有方法，不宣称全新训练理论；创新机制仍需实际源因果/互补证据和后续完整验证。','',
'完整训练曲线，阴影为三seed样本SD：','','![完整训练曲线](training_curves.png)','',
'配置、数据权限、停止条件、逐行命令和预登记主条件见experiment.json；发布前检查/独立P0P1及启动证据保持原文件。完整统计见results_analysis.json和CSV。']
(A/'report.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8',newline='\n')
spec=d.read(A/'experiment.json');spec['result']=dict(status='ANALYZED',evidence_ref='results_analysis.json',actual_rows=12,actual_optimizer_steps=93240,actual_model_epochs=420,clean_decisions=2016000,metric_records=216,primary='E50_RDROP',E50_RDROP_goal84_check=x['E50_RDROP_goal84_check'],E20_RDROP_goal84_check=x['E20_RDROP_goal84_check'],goal_completed=False);d.write(A/'experiment.json',spec)
mod=importlib.util.spec_from_file_location('registry',W.parents[2]/'tools/experiment_registry.py');reg=importlib.util.module_from_spec(mod);mod.loader.exec_module(reg);reg.event(W,d.RUN,'ANALYZED',['results_analysis.json','summary_by_seed.csv','paired_errors.csv','paired_strata.csv','training_complete_curves.csv'],'All12 E20/E50 fixedbudget fullsource/defaulthead models complete;93240steps/420epochs/2016000decisions/216CM independently recounted;no new predictions or fitting');reg.build(W,managed_only=True)
goal=W/'docs/CVS_PHASE1_FEATURE_EXTRACTION_GOAL_20261013.md';marker='<!-- RDROP_E20_E50_COMPLETE_20261015 -->';text=goal.read_text(encoding='utf-8')
if marker not in text:
 means='/'.join(f"{x['summary'][a]['accuracy_mean']*100:.6f}%" for a in d.ARMS)
 contrasts='; '.join(f"{k}: {s['mean_pp']:+.6f}pp/{s['positive_seeds']}正" for k,s in x['paired_summary'].items())
 text+=f'\n\n{marker}\n2026-10-15 `{d.RUN}` 全12行完整ANALYZED，93240更新/420epoch/2016000决定/216CM及完整source日志独立核验，默认头joint/headfit0。E20_TWO_CE/E20_RDROP/E50_TWO_CE/E50_RDROP均值依次{means}；配对{contrasts}。预登记主条件E50_RDROP的84%检查={x["E50_RDROP_goal84_check"]}；E20比较检查={x["E20_RDROP_goal84_check"]}。预算含日程与U轮次变化，既有RDrop不是全球新方法；不以目标结果挑主条件、重跑或自动完成全研究目标。[完整报告](../automation_reports/CV-SincNet/{d.RUN}/report.md)。goal仍active，下一机制按合法源证据推进。\n'
 goal.write_text(text,encoding='utf-8',newline='\n')
paths=[p.relative_to(W).as_posix() for p in A.iterdir() if p.is_file() and p.name in ('experiment.json','events.jsonl','report.md','results_analysis.json','summary_by_seed.csv','training_complete_curves.csv','paired_errors.csv','paired_strata.csv','training_stage_cost.csv','metrics_strata.csv','training_curves.png')]+[goal.relative_to(W).as_posix()]
d.write(W/'.codex_tmp/rdrop_budget_analysis_plan.json',dict(paths=paths,run_id=d.RUN,status='ANALYZED',message='Analyze matched E20 E50 three-seed RDrop training comparison',evidence=str(P/'analysis_delivery.json')))
print(json.dumps(dict(status='ANALYZED',summary=x['summary'],paired=x['paired_summary'],goal_completed=False),ensure_ascii=False))
