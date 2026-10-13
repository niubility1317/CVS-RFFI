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
  if done!=dict(status='TRAINED_PREDICTED_SCORED',config=c,views=list(d.VIEWS),count_per_view=168000,metric_records=18,truth_last=True,independent_recount='VERIFIED') or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=18:raise ValueError('Incomplete row scoring')
  for r in v['records']:
   cm=r['confusion'];n=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/n
   if n!=r['query_count'] or abs(acc-r['accuracy'])>1e-12:raise ValueError('Final score recount')
  records.extend(v['records'])
 summary=[];paired=[]
 for arm in d.ARMS:
  for view in d.VIEWS:
   rr=[r for r in records if r['arm']==arm and r['view']==view and r['dimension']=='overall'];a=[r['accuracy'] for r in rr]
   if len(rr)!=3 or {r['model_seed'] for r in rr}!=set(d.SEEDS):raise ValueError('Incomplete three-seed comparison')
   summary.append(dict(arm=arm,view=view,accuracy_mean=st.mean(a),accuracy_sd=st.stdev(a),seed_accuracy={str(r['model_seed']):r['accuracy'] for r in rr},macro_f1_mean=st.mean(r['macro_f1'] for r in rr),worst_RX_mean=st.mean(min(r['accuracy'] for r in records if r['row_id']==x['row_id'] and r['view']==view and r['dimension']=='receiver') for x in rr)))
 for view in d.VIEWS:
  index={(r['arm'],r['model_seed']):r['accuracy'] for r in records if r['dimension']=='overall' and r['view']==view}
  for arm,control in [('RANDOM_INPUT','CE50'),('VAT_INPUT','CE50'),('VAT_FEATURE','CE50'),('VAT_INPUT','RANDOM_INPUT'),('VAT_FEATURE','VAT_INPUT')]:
   delta=[index[arm,s]-index[control,s] for s in d.SEEDS];paired.append(dict(arm=arm,control=control,view=view,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 primary=next(r for r in summary if r['arm']=='VAT_FEATURE' and r['view']=='clean');goal=primary['accuracy_mean']>=.84
 value=dict(status='VERIFIED',run_id=d.RUN,models=12,prediction_count=2016000,summary=summary,paired=paired,goal_84=goal,truth_last=True,records=records);d.write(d.BASE/'analysis.json',value)
 lines=['# 虚拟对抗一致性：固定三种子 clean 结果','','VERIFIED：12个从零训练50轮模型，每row168000个clean预测固定后独立truth-last评分。','','|方案|clean均值|SD|最差RX均值|','|---|---:|---:|---:|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['worst_RX_mean']:.4%}|")
 lines+=['',f'预登记VAT_FEATURE三seedclean均值达到84%：{goal}。','','所有模型保持普通cos160、固定final50。CE不打开U，其他3组合法U在第6轮开始，每步当前student无梯度弱分布、一轮扰动方向计算、可回传参数的确定性一致性前向；前5轮关闭，5轮ramp到0.25。U标签始终隐藏，无EMA/教师权重或目标训练。','','扰动为每包相对tensorL2半径2%，搜索半径0.1%。随机IQ对照计算同一power梯度后丢弃；VAT_INPUT和VAT_FEATURE分别约束IQ与浅中层特征，FEATURE每step只选一层。KL用float64，网络前向fullFP32。源V仅健康，目标不回流选择/调参/重跑；所有负结果保留，不声称新盲测或卫星部署。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return value
