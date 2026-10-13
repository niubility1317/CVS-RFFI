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
 control_records=[];paired=[]
 from experiments.cvs_report_mechanisms.predict import preflight as control_preflight
 from experiments.cvs_report_mechanisms import design as md
 for seed in d.SEEDS:
  cc=d.control(seed);control_preflight(cc['row_id']);v=d.read(md.BASE/cc['row_id']/'scores/metrics.json');control_records+=v['records']
 index={(r['arm'],r['model_seed']):r['accuracy'] for r in records+control_records if r['dimension']=='overall'}
 for arm in d.ARMS:
  delta=[index[arm,seed]-index['UNIFORM_SSL',seed] for seed in d.SEEDS];paired.append(dict(treatment=arm,control='UNIFORM_SSL',seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 delta=[index['TX_CONDITIONAL',seed]-index['UNCONDITIONAL',seed] for seed in d.SEEDS];paired.append(dict(treatment='TX_CONDITIONAL',control='UNCONDITIONAL',seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 a=dict(status='VERIFIED',run_id=d.RUN,models=6,readouts=6,views=['clean'],prediction_count=1008000,gradient_updates=39960,summary=summary,paired=paired,goal_84=any(s['goal_84'] for s in summary),truth_last=True,records=records,control_records=control_records);d.write(d.BASE/'analysis.json',a);lines=['# 浅中层TX条件矩：完整三seedclean结果','','|机制|三seed均值|SD|≥84%|','|---|---:|---:|---|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['goal_84']}|")
 lines+=['','|比较|三seed配对差均值|正向seed|','|---|---:|---:|']
 for r in paired:lines.append(f"|{r['treatment']}-{r['control']}|{r['mean_difference']:.4%}|{r['positive_seeds']}|")
 lines+=['','6新模型从零各30轮/6660步、共39960optimizerupdates，全部sourceL6300/U56700隐藏标签；V无前向或性能资格门槛，每rowclean168000预测固定后独立truthlast，完整108条same-row指标。所有模型同普通cos，Phase2可重新拟合W；矩统计仅当前训练Lbatch计算，不保存为query运行状态。','固定time_down[B,48,128]与t2.pool[B,96,32]浅/中层时间均值及populationstd，合法L当前batch detached均值/标准差归一floor1e-4，源TX条件下RX质心平方偏差；UNCONDITIONAL省去TX分组，其余同。固定warm5/ramp10至.1，CE保护源身份类差异，不声称完整未来新TX信息或真实硬件RX因果分解。','sourceU沿已有UNIFORM_SSL同数据/目标/采样/随机性：5RX L26/26/26/25/25 TX/day平衡，U4其他RX32，warm5/ramp5/KL.25/conf95/twopass/noise.015。复用合法已完成UNIFORM_SSL三seed固定prediction作loss匹配控制，不加载它的checkpoint训练，不重跑target；冗余meta梯度诊断不在新方法执行，不冒充相同计算成本。','系数/层位沿此前未启动矩对齐核心固定；本轮不按target错例/分数调参、挑seed或选择性重跑，全部负结果保留。数学动作检验不是跨新RX性能保证，已暴露benchmark不称新盲测。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
