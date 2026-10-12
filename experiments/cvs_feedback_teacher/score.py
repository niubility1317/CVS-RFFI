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
   actual_seed=c['model_seed'] if name=='STUDENT' else c['feedback_teacher']['teacher_init_seed']
   for r in rr+dd:r.update(view='clean',readout=name,model_seed=actual_seed,paired_model_seed=c['model_seed'])
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
   expected=set(d.SEEDS) if name=='STUDENT' else {s+10000 for s in d.SEEDS}
   if len(rr)!=3 or {r['model_seed'] for r in rr}!=expected:raise ValueError('Threeactual modelseeds incomplete')
   summary.append(dict(arm=arm,readout=name,accuracy_mean=st.mean(a),accuracy_sd=st.stdev(a),seed_accuracy={str(r['model_seed']):r['accuracy'] for r in rr},macro_f1_mean=st.mean(r['macro_f1'] for r in rr),primary_seed_role=name=='STUDENT',goal_84=name=='STUDENT' and st.mean(a)>=.84,diagnostic_teacher_mean_above84=name=='TEACHER' and st.mean(a)>=.84))
 a=dict(status='VERIFIED',run_id=d.RUN,models=12,joint_rows=6,readouts=12,views=['clean'],prediction_count=2016000,gradient_updates=79920,summary=summary,goal_84=any(r['goal_84'] for r in summary),truth_last=True,records=records);d.write(d.BASE/'analysis.json',a);lines=['# 教师目标反馈：完整三种子clean结果','','|方案|模型角色|三seed均值|SD|≥84%|','|---|---|---:|---:|---|']
 for r in summary:lines.append(f"|{r['arm']}|{r['readout']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['goal_84']}|")
 lines+=['','6对联合从零轨迹，12模型各30轮/6660更新，共79920optimizerupdates；STUDENT/TEACHER均为普通cos，各168000clean，全部预测固定后独立truthlast。sourceV无前向/性能资格门槛，原L/U/V7/63/30、训练池标签10%，U TX真值隐藏；无目标拟合、旧checkpoint、EMA或星地训练。','学生实际初始seed为2026092701/02/03，教师实际初始seed为对应+10000；教师角色仅诊断，偏移seed均值即使超过84%也不替代学生primary三seed目标。完整报告实际seed及paired_model_seed，不把辅助初始化冒充primary。','该机制是MPL启发的冻结AdamW二阶度量一阶有限差分教师奖励，并非原MPL或精确虚拟AdamW优化。奖励从训练L另RX的梯度产生，中心化/统一RMSfloor/clip2；只调整teacher backbone的辅助梯度，不调整cos权重的辅助梯度。teacher普通CE与student普通CE照常训练共同cos，Phase2可丢弃W重新拟合。','对照同2模型/虚拟前向/梯度开销，仅teacherauxweight0；固定.1ramp/学生KL.25/warm5/ramp5/conf95/twopass/noise.015，无源V或target条件调参。全部负结果保留，不按测试分数重跑/挑seed；基准已暴露，不称新盲测。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
