"""Independent process; training and prediction never import this scorer."""
import json,time
from . import design as d
def score(row):
 import numpy as np
 from .predict import preflight
 from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
 c,ids=preflight(row) # Validate every registered view before opening truth.
 out=d.BASE/row/'scores'
 if out.exists():raise FileExistsError('Existing score artifact; reconcile, never replace')
 out.mkdir();truth=d.read(d.TRUTH);y,rx,days=truth_arrays(truth,ids);records=[]
 with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as p:
  for view in d.VIEWS:
   rr,dd=metric_rows(c,p[view],ids,y,rx,days)
   for r in rr+dd:r['view']=view
   records.extend(rr+dd)
 d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'))
 d.write(d.BASE/row/'completion.json',dict(status='TRAINED_PREDICTED_SCORED',config=c,views=list(d.VIEWS),count_per_view=168000,metric_records=len(records),truth_last=True,independent_recount='VERIFIED'))
 print('SCORED '+json.dumps(dict(row_id=row,views=len(d.VIEWS),metric_records=len(records))),flush=True)
def finish():
 import statistics as st
 records=[]
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json')
  expected=dict(status='TRAINED_PREDICTED_SCORED',config=c,views=['clean'],count_per_view=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED')
  if done!=expected or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=18:raise ValueError('Incomplete sampling row scoring')
  for r in v['records']:
   cm=r['confusion'];n=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/n
   if n!=r['query_count'] or (r['dimension']=='overall' and n!=168000) or abs(acc-r['accuracy'])>1e-12:raise ValueError('Sampling confusion recount')
  records.extend(v['records'])
 summary=[];paired=[]
 for arm in d.ARMS:
  rr=[r for r in records if r['arm']==arm and r['view']=='clean' and r['dimension']=='overall'];a=[r['accuracy'] for r in rr]
  if len(rr)!=3 or {r['model_seed'] for r in rr}!=set(d.SEEDS):raise ValueError('Three sampling primary seeds incomplete')
  summary.append(dict(arm=arm,view='clean',accuracy_mean=st.mean(a),accuracy_sd=st.stdev(a),seed_accuracy={str(r['model_seed']):r['accuracy'] for r in rr},macro_f1_mean=st.mean(r['macro_f1'] for r in rr),goal_84=st.mean(a)>=.84))
 index={(r['arm'],r['model_seed']):r['accuracy'] for r in records if r['dimension']=='overall'}
 for treatment,control in [('ORDINARY_POINT_GN','ORDINARY_GLOBAL_GN'),('LOWPASS_POINT_GN','ORDINARY_POINT_GN'),('CARRY_POINT_GN','ORDINARY_POINT_GN'),('CARRY_POINT_GN','LOWPASS_POINT_GN'),('CARRY_POINT_GN','ORDINARY_GLOBAL_GN')]:
  delta=[index[treatment,seed]-index[control,seed] for seed in d.SEEDS];paired.append(dict(treatment=treatment,control=control,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 primary=next(r for r in summary if r['arm']=='CARRY_POINT_GN');goal=primary['accuracy_mean']>=.84
 value=dict(status='VERIFIED',run_id=d.RUN,models=12,readouts=12,views=['clean'],prediction_count=2016000,gradient_updates=79920,summary=summary,paired=paired,goal_84=goal,primary='CARRY_POINT_GN',truth_last=True,records=records)
 d.write(d.BASE/'analysis.json',value)
 lines=['# 采样结构四组：完整三种子 clean 结果','','|结构|准确率均值|SD|Macro-F1|≥84%|','|---|---:|---:|---:|---|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['macro_f1_mean']:.6f}|{r['goal_84']}|")
 lines+=['','普通 cos160，每模型从头30轮/6660更新，纯CE，sourceU/teacher/EMA/satellite/Vforward实际关闭，固定末轮逐row clean prediction后独立truth-last。12条轨迹不继承按目标成绩选择的变体，所有负结果保留。carry保留两相位及fine-grid t3，时序长度翻倍；相同epoch/update/L预算不表示相同计算成本。源机制适用性不是身份性能或完整整网平移不变保证。既有测试基准已暴露，不声称新的独立盲测。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return value
