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
 index={(r['arm'],r['model_seed']):r['accuracy'] for r in records if r['dimension']=='overall'}
 for treatment,control in [('MARGINAL_GRL','DETACHED_AUX'),('TX_COND_GRL','DETACHED_AUX'),('TX_COND_GRL','MARGINAL_GRL')]:
  delta=[index[treatment,seed]-index[control,seed] for seed in d.SEEDS];paired.append(dict(treatment=treatment,control=control,seed_differences=delta,mean_difference=st.mean(delta),positive_seeds=sum(v>0 for v in delta)))
 a=dict(status='VERIFIED',run_id=d.RUN,models=9,readouts=9,views=['clean'],prediction_count=1512000,gradient_updates=59940,summary=summary,paired=paired,goal_84=any(s['goal_84'] for s in summary),truth_last=True,records=records,control_records=control_records);d.write(d.BASE/'analysis.json',a);lines=['# 浅中层TX条件RX对抗：完整三seedclean结果','','|机制|三seed均值|SD|≥84%|','|---|---:|---:|---|']
 for r in summary:lines.append(f"|{r['arm']}|{r['accuracy_mean']:.4%}|{r['accuracy_sd']:.4%}|{r['goal_84']}|")
 lines+=['','|比较|三seed配对差均值|正向seed|','|---|---:|---:|']
 for r in paired:lines.append(f"|{r['treatment']}-{r['control']}|{r['mean_difference']:.4%}|{r['positive_seeds']}|")
 lines+=['','9 models scratch30/6660 main updates each, 59940 main updates and59940 auxiliary updates. Source L6300/U56700 hiddenTX/V27000 noforward; ordinary cosW6x160 scale30 unchanged. Two shallow/middle sourceL mean/std RX5 adversaries with equal6head capacity, BASE detachedaux, MARGINAL averages6CE, TXCOND selects trueL TX. Main/aux separate optimizers andclip10/5, RNG-isolated heads/noDropout, GRL coefficient only at backward warm5/ramp10 to.1. Allaux22460 parameters discarded fromquery, noextra identityforward; matchedUNIFORM_SSL sampling/objective retained. Sourcehealthmotivated, no targetselection/tuning; negative outcomes retained, not strictwholeidentity RX invariance or unknownRX guarantee.']
 (d.BASE/'final_analysis.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');return a
