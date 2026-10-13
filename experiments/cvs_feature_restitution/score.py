"""Fresh CPU scoring process, reached only after both predictions are fixed."""
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
 d.write(d.BASE/row/'completion.json',dict(status='TRAINED_PROBED_PREDICTED_SCORED',config=c,views=['clean'],readouts=list(d.READOUTS),count_per_readout=168000,metric_records=36,truth_last=True,independent_recount='VERIFIED'))
 print('SCORED '+row,flush=True)

def finish():
 import statistics as st
 records=[];source=[]
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json');pv=d.read(d.BASE/row['row_id']/'probe/summary.json')
  if done!=dict(status='TRAINED_PROBED_PREDICTED_SCORED',config=c,views=['clean'],readouts=list(d.READOUTS),count_per_readout=168000,metric_records=36,truth_last=True,independent_recount='VERIFIED') or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=36:raise ValueError('Incomplete row closure')
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
  for treatment,control in [('ORDINARY','BASE_REF'),('RESTORE','BASE_REF'),('COMP','BASE_REF'),('RESTORE_COMP','BASE_REF'),('RESTORE','ORDINARY'),('RESTORE_COMP','RESTORE'),('RESTORE_COMP','COMP')]:
   delta=[index[treatment,seed]-index[control,seed] for seed in d.SEEDS];paired.append(dict(readout=readout,treatment=treatment,control=control,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 native=next(r for r in summary if r['arm']=='RESTORE_COMP' and r['readout']=='NATIVE_HEAD')
 feature=next(r for r in paired if r['treatment']=='RESTORE_COMP' and r['control']=='BASE_REF' and r['readout']=='REFIT_COSINE')
 result=dict(status='VERIFIED',run_id=d.RUN,new_backbones=12,existing_backbone_references=3,rows=15,readouts=30,views=['clean'],prediction_count=5040000,reused_native_decisions=504000,new_prediction_decisions=4536000,metric_records=540,backbone_updates_new=133200,historical_baseline_updates=33300,head_epochs_per_row=70,summary=summary,paired=paired,source_diagnostics=source,records=records,primary='RESTORE_COMP',native_goal_84=native['accuracy_mean']>=.84,feature_evidence_positive=feature['mean_difference']>0,goal_84=native['accuracy_mean']>=.84 and feature['mean_difference']>0,Phase2_evidence_complete=False,truth_last=True,exposed_benchmark=True)
 d.write(d.BASE/'analysis.json',result)
 lines=['# 归一化信息恢复与分支竞争：完整结果','','|结构|读出|准确率均值|SD|Macro-F1|','|---|---|---:|---:|---:|']
 for r in summary:lines.append(f"|{r['arm']}|{r['readout']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['macro_f1_mean']:.6f}|")
 lines+=['','旧伪标签门控＋EMA＋cos，新增骨干从零固定50轮，分类头结构保持。普通cosine重训只拟合冻结的160维特征，L_s训练10轮；五个留接收机头与一个接收机信息头另计，共70个head epoch/row。基线来自此前同种子、同物理源数据的固定第50轮，只复用原生头预测。所有负结果保留，目标结果不回流训练、参数或候选顺序。尚无Phase2迁移证据；不以Phase1换头结果声称旧新身份联合识别已改善。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return result
