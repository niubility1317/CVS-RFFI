"""Independent addressspace opens truth only after every same-row readout."""
import json
from . import design as d
def score(row):
 import numpy as np
 from .predict import preflight
 from experiments.cvs_report_ssl.evaluate import truth_arrays,metric_rows
 c,ids=preflight(row);out=d.BASE/row/'scores';out.mkdir(exist_ok=False);y,rx,day=truth_arrays(d.read(d.TRUTH),ids);records=[]
 with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as p:
  for name in d.readouts(c):
   rr,dd=metric_rows(c,p[name],ids,y,rx,day)
   for r in rr+dd:r.update(view='clean',readout=name)
   records+=rr+dd
 d.write(out/'metrics.json',dict(records=records,truth_last=True,independent_recount='VERIFIED'));d.write(d.BASE/row/'completion.json',dict(status='TRAINED_PREDICTED_SCORED',config=c,views=['clean'],readouts=list(d.readouts(c)),count_per_view=168000,metric_records=len(records),truth_last=True,independent_recount='VERIFIED'));print('SCORED '+row,flush=True)
def finish():
 import statistics as st
 records=[]
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json');n=18*len(d.readouts(c))
  if done!=dict(status='TRAINED_PREDICTED_SCORED',config=c,views=['clean'],readouts=list(d.readouts(c)),count_per_view=168000,metric_records=n,truth_last=True,independent_recount='VERIFIED') or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=n:raise ValueError('Incomplete clean scoring')
  for r in v['records']:
   cm=r['confusion'];count=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/count
   if count!=r['query_count'] or abs(acc-r['accuracy'])>1e-12:raise ValueError('Confusion recount differs')
  records+=v['records']
 summary=[]
 for arm in d.ARMS:
  c=d.config(next(r for r in d.rows() if r['arm']==arm))
  for name in d.readouts(c):
   rr=[r for r in records if r['arm']==arm and r['readout']==name and r['dimension']=='overall'];a=[r['accuracy'] for r in rr]
   if len(rr)!=3 or {r['model_seed'] for r in rr}!=set(d.SEEDS):raise ValueError('Threeprimaryseeds incomplete')
   summary.append(dict(arm=arm,readout=name,accuracy_mean=st.mean(a),accuracy_sd=st.stdev(a),seed_accuracy={str(r['model_seed']):r['accuracy'] for r in rr},macro_f1_mean=st.mean(r['macro_f1'] for r in rr),goal_84=st.mean(a)>=.84))
 a=dict(status='VERIFIED',run_id=d.RUN,models=36,readouts=39,views=['clean'],prediction_count=6552000,gradient_updates=239760,summary=summary,goal_84=any(s['goal_84'] for s in summary),truth_last=True,records=records);d.write(d.BASE/'analysis.json',a);lines=['# 报告独立机制：三种子 clean 结果','','|机制|读出|三seed均值|SD|≥84%|','|---|---|---:|---:|---|']
 for s in summary:lines.append(f"|{s['arm']}|{s['readout']}|{s['accuracy_mean']:.4%}|{s['accuracy_sd']:.4%}|{s['goal_84']}|")
 lines+=['','36条从零30轮轨迹，共239760更新；SWA另加三组同轨迹零额外轮次readout，总39组各168000clean。所有方法统一普通cos，原实际物理角色7/63/30、标签占训练池10%，U只两SSL变体且真值隐藏；V无性能筛选，无目标训练/拟合。每训练row自动完成clean预测及独立truth-last评分，全矩阵负结果保留。','报告理论不等于实际新RX性能保证：多项式并非累积量；仿射需满秩且良态；子带需局部近似平坦信道/避免深衰落；log谱固定低频基不能消除任意信道；闭环有限包内8片段共同链假设尚未实证；C4仅有限群且通过非线性特征平均实现；切向项仅约束两个明示数字代理动作。附加支路不使整个含native表示自动严格不变。源L内容/退化检查不是性能门槛。','三seed原样报告，无按目标分数调参、选seed、停止其他健康训练或选择性重跑。本次为已暴露benchmark研究，不声称新盲测或Phase2部署合规。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
