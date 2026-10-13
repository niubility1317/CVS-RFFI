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
  for arm,control in [('RANDOM_MASK','CE30'),('GRAD_MASK','CE30'),('GRAD_MASK','RANDOM_MASK')]:
   delta=[index[arm,s]-index[control,s] for s in d.SEEDS];paired.append(dict(arm=arm,control=control,view=view,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 primary=next(r for r in summary if r['arm']=='GRAD_MASK' and r['view']=='clean');goal=primary['accuracy_mean']>=.84
 value=dict(status='VERIFIED',run_id=d.RUN,models=9,prediction_count=1512000,summary=summary,paired=paired,goal_84=goal,truth_last=True,records=records);d.write(d.BASE/'analysis.json',value)
 lines=['# 浅中层特征自挑战：固定三种子 clean 对照','','VERIFIED：九个从零训练30轮模型；每row固定末轮后完成168000个clean预测，独立truth-last评分。','','|方案|clean均值|SD|最差RX均值|','|---|---:|---:|---:|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['worst_RX_mean']:.4%}|")
 lines+=['',f'固定主方案GRAD_MASK clean三seed均值达到84%：{goal}。','','遮挡只用L_s真值，在time_down和t2.pool交替遮25%通道；前5轮关闭，10轮ramp到0.5，保留完整clean CE。随机组具有相同梯度排名和第二前向成本。普通cos160头不改，无U/teacher/EMA/卫星增强。V仅健康，固定finalE30不选择。','','遮挡不证明去除了RX信息或物理不变量；训练和测试成本分别实报。历史测试已暴露，不声明新的独立盲测；结果不用于本矩阵调参、重排或重跑，全部负结果保留。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return value
