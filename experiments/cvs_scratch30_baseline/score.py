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
 print('SCORED '+json.dumps(dict(row_id=row,views=7,metric_records=len(records))),flush=True)
def finish():
 import statistics as st
 records=[]
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json')
  if done!=dict(status='TRAINED_PREDICTED_SCORED',config=c,views=list(d.VIEWS),count_per_view=168000,metric_records=126,truth_last=True,independent_recount='VERIFIED') or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=126:raise ValueError('Incomplete row scoring')
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
  index={(r['arm'],r['model_seed']):r['accuracy'] for r in records if r['dimension']=='overall' and r['view']==view};delta=[index['SELF_SSL30',s]-index['CE30',s] for s in d.SEEDS];paired.append(dict(view=view,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 primary=next(r for r in summary if r['arm']=='SELF_SSL30' and r['view']=='clean');goal=primary['accuracy_mean']>=.84
 value=dict(status='VERIFIED',run_id=d.RUN,models=6,prediction_count=7056000,summary=summary,paired=paired,goal_84=goal,truth_last=True,records=records);d.write(d.BASE/'analysis.json',value)
 lines=['# 从头30轮 CE／当前学生半监督：完整最终结果','','VERIFIED：六个固定finalE30模型，三个seed；每模型clean和六practical各168000，全部view先prediction后独立truth-last评分。每row自动收尾，不等待全批训练结束。','','|方案|视图|三seed准确率均值|SD|最差RX均值|','|---|---|---:|---:|---:|']
 for r in summary:lines.append(f"|{r['arm']}|{r['view']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['worst_RX_mean']:.4%}|")
 lines+=['',f'预登记主方案SELF_SSL30 clean三seed均值≥84%：{goal}。','', '全部source从零30epoch/6660更新，无旧模型/固定老师、EMA、卫星训练或辅助网络。CE30实际不打开U；SELF_SSL30前5epoch纯CE，之后当前student clean eval软目标、95%置信度及相邻完整遍历两次稳定门控、.015固定噪声强视图、5epoch线性ramp至.25。U真值始终隐藏，V仅epoch10/20/30健康评估，无选模。','', '逐seed、RX/day/TX、混淆和配对差见analysis.json，完整6660步source日志保留。目标结果未回流训练或候选排序，所有负结果保留。数据为地面代理及固定模拟practical，不能称真实卫星部署或新的独立盲测；K与Phase2注册均N/A。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return value
