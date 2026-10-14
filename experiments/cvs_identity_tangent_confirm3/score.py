import statistics
from pathlib import Path
from . import design as d
from .predict import preflight
from experiments.cvs_feature_sequence50 import score as shared
FIRST='20261014-phase1-identity-tangent-oldgate-e50-clean-manysig-s01-m4-r01'
def score(row):return shared.score(row,design=d,prediction_preflight=preflight)
def finish():
 from .provenance import source_proof
 source={};qualified=True
 for seed in d.SEEDS:
  source[str(seed)]={}
  for row in [r for r in d.rows() if r['model_seed']==seed]:
   v=d.read(source_proof(d.config(row))/'completion.json');source[str(seed)][row['arm']]=dict(CE=v['final_training_L_CE'],metrics=v['final_source_metrics'])
  base=source[str(seed)]['BASE_REF']['metrics']
  for arm in d.ARMS[1:]:
   v=source[str(seed)][arm];qualified=qualified and v['CE']<=.5 and v['metrics']['source_val_accuracy']>=base['source_val_accuracy']-.01 and v['metrics']['source_val_worst_rx']>=base['source_val_worst_rx']-.02
 d.write(d.BASE/'source_extension_decision.json',dict(source_only=True,source_health_pass=qualified,target_metrics_read=False,source=source,method_parameters_unchanged=True))
 prior=d.read(Path(d.PROJECT)/'runs'/FIRST/'analysis.json')
 if prior['status']!='VERIFIED' or prior['rows']!=4 or not prior['truth_last'] or prior['head_fit_updates'] or not prior['source_health_pass']:raise ValueError('Prior firstseed completion differs')
 records=list(prior['records'])
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json')
  if done['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or done['config']!=c or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=18:raise ValueError('Remainingseed matrix incomplete')
  records.extend(v['records'])
 for r in records:
  cm=r['confusion'];n=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/n;f1=sum(2*cm[k][k]/(sum(cm[k])+sum(rr[k] for rr in cm)) if sum(cm[k])+sum(rr[k] for rr in cm) else 0. for k in range(6))/6
  if n!=r['query_count'] or abs(acc-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Complete3seed CM recount differs')
 seeds=(2026092701,*d.SEEDS);index={(r['arm'],r['model_seed']):r for r in records if r['dimension']=='overall'}
 if set(index)!={(a,s) for a in d.ARMS for s in seeds} or len(records)!=216:raise ValueError('Missing complete3seed records')
 summary=[]
 for arm in d.ARMS:
  v=[index[arm,s]['accuracy'] for s in seeds];summary.append(dict(arm=arm,seeds=list(seeds),seed_count=3,accuracy=statistics.mean(v),sample_sd=statistics.stdev(v),macro_f1=statistics.mean(index[arm,s]['macro_f1'] for s in seeds)))
 paired=[]
 for treatment,control in d.PAIRS:
  delta=[dict(model_seed=s,difference=index[treatment,s]['accuracy']-index[control,s]['accuracy']) for s in seeds];v=[r['difference'] for r in delta];paired.append(dict(treatment=treatment,control=control,seed_count=3,mean_difference=statistics.mean(v),sample_sd=statistics.stdev(v),seeds=delta))
 result=dict(status='VERIFIED',run_id=d.RUN,**d.result_metadata(),combined_seeds=list(seeds),combined_metric_records=216,combined_prediction_count=2016000,combined_new_backbones=9,combined_historical_references=3,prior_completed_run=FIRST,summary=summary,paired=paired,records=records,source=source,source_health_pass=qualified,goal_84=next(r['accuracy'] for r in summary if r['arm']=='IDSAFE')>=.84,head_fit_updates=0,truth_last=True,exposed_benchmark=True,target_feedback=False)
 d.write(d.BASE/'analysis.json',result);return result
