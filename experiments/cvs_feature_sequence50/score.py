"""Fresh CPU scoring process, reached only after default-head predictions are fixed."""
import json
from . import design as d

def score(row):
 import numpy as np
 from .predict import preflight
 from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
 c,ids=preflight(row);out=d.BASE/row/'scores'
 if out.exists():raise FileExistsError('Existing scoring output; preserve it')
 out.mkdir();y,rx,days=truth_arrays(d.read(d.TRUTH),ids);records=[]
 with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as p:
  for readout in d.READOUTS:
   rr,dd=metric_rows(c,p[readout],ids,y,rx,days)
   for r in rr+dd:r['readout']=readout
   records.extend(rr+dd)
 d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
 d.write(d.BASE/row/'completion.json',dict(status='TRAINED_DEFAULT_PREDICTED_SCORED',config=c,views=['clean'],readouts=list(d.READOUTS),count_per_readout=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED'))
 print('SCORED '+row,flush=True)

def finish():
 import statistics as st
 records=[];source=[]
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json');pv=d.read(d.BASE/row['row_id']/'diagnostics/summary.json')
  if done!=dict(status='TRAINED_DEFAULT_PREDICTED_SCORED',config=c,views=['clean'],readouts=list(d.READOUTS),count_per_readout=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED') or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=18:raise ValueError('Incomplete row closure')
  for r in v['records']:
   cm=r['confusion'];n=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/n
   f1=sum(2*cm[k][k]/(sum(cm[k])+sum(row[k] for row in cm)) if (sum(cm[k])+sum(row[k] for row in cm)) else 0. for k in range(6))/6
   if n!=r['query_count'] or (r['dimension']=='overall' and n!=168000) or abs(acc-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Confusion matrix independent recount differs')
  records.extend(v['records']);source.append(pv)
 summary=[];paired=[]
 for readout in d.READOUTS:
  for arm in d.ARMS:
   rr=[r for r in records if r['arm']==arm and r['readout']==readout and r['dimension']=='overall'];a=[r['accuracy'] for r in rr]
   if len(rr)!=3 or {r['model_seed'] for r in rr}!=set(d.SEEDS):raise ValueError('Incomplete three-seed matrix')
   summary.append(dict(arm=arm,readout=readout,accuracy_mean=st.mean(a),accuracy_sd=st.stdev(a),macro_f1_mean=st.mean(r['macro_f1'] for r in rr),seed_accuracy={str(r['model_seed']):r['accuracy'] for r in rr}))
  index={(r['arm'],r['model_seed']):r['accuracy'] for r in records if r['readout']==readout and r['dimension']=='overall'}
  for treatment,control in [(a,'BASE_REF') for a in d.ARMS[1:]]+[(a,'RFF') for a in ('MEAN','MEAN_VAR','NONLINEAR_MEAN')]+[('SIG','COV'),('SIG_TIME','COV'),('SIG_TIME','SIG')]:
   delta=[index[treatment,seed]-index[control,seed] for seed in d.SEEDS];paired.append(dict(readout=readout,treatment=treatment,control=control,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 verdict={a:next(r for r in summary if r['arm']==a)['accuracy_mean']>=.84 for a in d.PRIMARY}
 result=dict(status='VERIFIED',run_id=d.RUN,rows=24,new_backbones=21,existing_backbone_references=3,readouts=24,views=['clean'],prediction_count=4032000,reused_native_decisions=504000,new_prediction_decisions=3528000,metric_records=432,head_fit_updates=0,default_classifier_only=True,backbone_updates_new=233100,historical_baseline_updates=33300,summary=summary,paired=paired,source_diagnostics=source,records=records,co_primary=list(d.PRIMARY),goal_by_primary=verdict,goal_84=any(verdict.values()),truth_last=True,exposed_benchmark=True)
 d.write(d.BASE/'analysis.json',result)
 lines=['# 局部分布与路径顺序：默认分类头完整结果','','|结构|准确率均值|SD|Macro-F1|','|---|---:|---:|---:|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['macro_f1_mean']:.6f}|")
 lines+=['','固定默认 ResidualPhysicalHead 随骨干联合训练；无另训或替换分类头。每新增骨干独立 scratch50、33标签＋17旧伪标签门控、EMA＋cos，不使用星地训练。仅 clean 自动预测后独立 truth-last 评分。所有预登记控制和负结果保留；source V、特征几何和局部顺序诊断不拟合任何分类头。结果不回流训练、参数、候选排序或选择性重跑。本轮仅 Phase1，已暴露基准，不称新的独立盲测。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
