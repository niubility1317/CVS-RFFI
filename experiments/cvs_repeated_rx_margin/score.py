"""Independent addressspace opens truth only after every same-row readout."""
import json
from . import design as d
def reclassify(old,new,y,rx,day,arm,seed):
 import numpy as np
 result=[]
 for dimension,values in [('overall',np.zeros(len(y),dtype=int)),('RX',rx),('day',day),('TX',y)]:
  for value in np.unique(values):
   mask=values==value;rescued=int(((~old)&new&mask).sum());regressed=int((old&(~new)&mask).sum());count=int(mask.sum());result.append(dict(arm=arm,model_seed=seed,dimension=dimension,value=value.item() if isinstance(value,np.generic) else value,count=count,rescued=rescued,regressed=regressed,net=(rescued-regressed)/count))
 return result
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
 control_records=[];paired=[];net_reclassification=[]
 import numpy as np
 from .predict import preflight
 from experiments.cvs_report_ssl.evaluate import truth_arrays
 index={(r['arm'],r['model_seed']):r['accuracy'] for r in records if r['dimension']=='overall'}
 for treatment,control in [('RANDOM_ROBUST','MATCHED_BASE'),('MEASURED_MEAN','MATCHED_BASE'),('MEASURED_ROBUST','MATCHED_BASE'),('MEASURED_MEAN','RANDOM_ROBUST'),('MEASURED_ROBUST','RANDOM_ROBUST'),('MEASURED_ROBUST','MEASURED_MEAN')]:
  delta=[index[treatment,seed]-index[control,seed] for seed in d.SEEDS];paired.append(dict(treatment=treatment,control=control,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 for seed in d.SEEDS:
  base=f'MATCHED_BASE-s{seed}';_,ids=preflight(base);y,rx,day=truth_arrays(d.read(d.TRUTH),ids)
  with np.load(d.BASE/base/'prediction/predictions.npz',allow_pickle=False) as p:old=p['FINAL']==y
  for arm in d.ARMS[1:]:
   row=f'{arm}-s{seed}';_,newids=preflight(row)
   if not np.array_equal(ids,newids):raise ValueError('Matched net reclassification IDs differ')
   with np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as p:new=p['FINAL']==y
   net_reclassification+=reclassify(old,new,y,rx,day,arm,seed)
 a=dict(status='VERIFIED',run_id=d.RUN,models=12,readouts=12,views=['clean'],prediction_count=2016000,gradient_updates=79920,summary=summary,paired=paired,goal_84=any(s['goal_84'] for s in summary),truth_last=True,records=records,control_records=control_records,net_reclassification=net_reclassification);d.write(d.BASE/'analysis.json',a);lines=['# 可重复跨RX漂移鲁棒间隔：完整三seedclean结果','','|机制|三seed均值|SD|≥84%|','|---|---:|---:|---|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['goal_84']}|")
 lines+=['','|比较|三seed配对差均值|正向seed|','|---|---:|---:|']
 for r in paired:lines.append(f"|{r['treatment']}-{r['control']}|{r['mean_difference']:.4%}|{r['positive_seeds']}|")
 lines+=['','12 models, scratch30 each, 79920 updates; fixed FINAL ordinary cos/160/scale30. All four arms use identical UNIFORM_SSL objective/sampling, no crossRX contrastive. L-only threefold TX/RX/day current-model mean refresh every epoch, q-weighted bounded .2 feature drift, lambda .25, warm5/ramp10. Random direction preserves exact per-direction norms/q; MEAN averages CE; ROBUST optimizes per-competitor minimum margin. All fixed clean predictions precede independent truth-last; negative results retained. Observational drift is not pure RX causality or unknown RX guarantee. Statistic refresh training cost is included and all statistic banks discarded for query.','Net reclassification counts are preserved by seed/TX/RX/day in analysis.json.']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
