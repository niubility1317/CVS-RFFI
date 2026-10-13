from pathlib import Path
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
 from experiments.cvs_crossrx_contrastive_clean.predict import preflight as parent_preflight
 for row in d.rows():
  c=d.config(row);pc=d.parent_config(c);_,ids=preflight(c['row_id']);_,pids=parent_preflight(pc['row_id'])
  if list(ids)!=list(pids):raise ValueError('Paired single member IDs differ')
  control_records+=d.read(Path(pc['output']).parent/'scores/metrics.json')['records']
 index={(r['arm'],r['model_seed'],r['readout']):r['accuracy'] for r in records+control_records if r['dimension']=='overall'}
 for arm in d.ARMS:
  delta=[index[arm,seed,'ENSEMBLE']-index[arm,seed,'FINAL'] for seed in d.SEEDS];paired.append(dict(treatment=arm+' ensemble',control=arm+' member0',seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 # Existing CE ensemble is a disclosed historical same-contract comparison.
 from experiments.cvs_ensemble_recovery import design as ce
 from experiments.cvs_ensemble_recovery.predict import preflight as ce_preflight
 ce_records=[]
 for seed in d.SEEDS:
  rr=next(r for r in ce.rows() if r['model_seed']==seed);cc=ce.config(rr);_,cids=ce_preflight(rr['row_id'])
  _,ids=preflight(next(r['row_id'] for r in d.rows() if r['model_seed']==seed))
  if list(ids)!=list(cids):raise ValueError('Historical CE ensemble IDs differ')
  ce_records += [r for r in d.read(ce.BASE/rr['row_id']/'scores/metrics.json')['records'] if r['view']=='clean']
 ce_index={r['model_seed']:r['accuracy'] for r in ce_records if r['dimension']=='overall'}
 for arm in d.ARMS:
  delta=[index[arm,seed,'ENSEMBLE']-ce_index[seed] for seed in d.SEEDS];paired.append(dict(treatment=arm+' ensemble',control='historical CE ensemble (sampling differs)',seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 a=dict(status='VERIFIED',run_id=d.RUN,models=18,new_models=12,readouts=6,views=['clean'],prediction_count=1008000,gradient_updates=79920,cumulative_member_updates=119880,summary=summary,paired=paired,goal_84=any(s['goal_84'] for s in summary),truth_last=True,records=records,control_records=control_records);d.write(d.BASE/'analysis.json',a);lines=['# 固定三成员监督对比集成：完整三seedclean结果','','|机制|三seed均值|SD|≥84%|','|---|---:|---:|---|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['goal_84']}|")
 lines+=['','|比较|三seed配对差均值|正向seed|','|---|---:|---:|']
 for r in paired:lines.append(f"|{r['treatment']}-{r['control']}|{r['mean_difference']:.4%}|{r['positive_seeds']}|")
 lines+=['','六个三成员集成组，复用六个固定成员0，另十二成员从零各30轮/6660更新，累计540模型轮/119880更新，当前新增79920更新；每组531075参数/3倍单模型成本，各成员均普通cos。sourceU全程TX标签隐藏，V不前向，无性能测试门槛。固定浅中层归一时间mean/std，温度.2，warm5/ramp10至.1。每行固定clean预测后独立truth-last，复用既有UNIFORM_SSL匹配控制原预测。属于条件样本级监督对比扩展，不声称严格接收不变、未来TX完整信息或独立盲测。']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
