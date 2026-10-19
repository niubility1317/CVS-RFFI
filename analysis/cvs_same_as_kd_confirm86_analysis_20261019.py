"""Matched3 studentseed confirmation from complete frozen artifacts only."""
import json,statistics
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def collect(project):
 import numpy as np
 from experiments.cvs_same_as_kd_confirm86 import design as d
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 if not (d.BASE/'source_matrix_decision.json').is_file():raise ValueError('Source confirmation must finish before final readout')
 predictions={};metrics={};strata={};cost={}
 for row in d.rows():
  key=row['row_id'];b=d.row_root(key);done=read(b/'prediction/complete.json');scored=read(b/'scoring/complete.json')
  if done['status']!='PREDICTIONS_COMPLETE' or done['views']!=list(VIEWS) or done['truth_read'] or scored['status']!='SCORED_COMPLETE':raise ValueError('Incomplete fixed all7: '+key)
  with np.load(b/'prediction/predictions.npz',allow_pickle=False) as f:predictions[key]={k:f[k].copy() for k in f.files}
  rr=read(b/'scoring/scores.json')['results'];strata[key]=rr;metrics[key]={}
  for view in VIEWS:
   x=next(v for v in rr if v['view']==view and v['dimension']=='overall');rx=[v for v in rr if v['view']==view and v['dimension']=='receiver'];metrics[key][view]=dict(accuracy=x['accuracy'],macro_F1=x['macro_f1'],worst_RX=min(v['accuracy'] for v in rx),count=len(predictions[key][view]))
  metrics[key]['urban_mean_accuracy']=sum(metrics[key][v]['accuracy'] for v in VIEWS[-3:])/3
  cost[key]={}
  for seg in ('stage20','source'):
   q=read(b/seg/'result.json');cost[key][seg]={k:v for k,v in q.items() if v is None or isinstance(v,(str,int,float,bool))};cost[key][seg]['sameAsKD']=q.get('sameAsKD');cost[key][seg]['actual_U_student_NN']=q['student_train_NN']-q['L_NN']
 ids=predictions[d.rows()[0]['row_id']]['ids']
 if any(not np.array_equal(x['ids'],ids) for x in predictions.values()):raise ValueError('Fixed physical ordering differs')
 y,rx,day=truth_arrays(read(d.TRUTH),ids);paired={}
 for seed in d.SEEDS:
  a='KD-s'+str(seed);b='T1-s'+str(seed);paired[str(seed)]={}
  for view in VIEWS:
   ac=predictions[a][view]==y;bc=predictions[b][view]==y;paired[str(seed)][view]=dict(repaired=int((ac&~bc).sum()),new_errors=int((~ac&bc).sum()),net=int(ac.sum()-bc.sum()),accuracy_pp=100*float(ac.mean()-bc.mean()),macro_F1_pp=100*(metrics[a][view]['macro_F1']-metrics[b][view]['macro_F1']),worst_RX_pp=100*(metrics[a][view]['worst_RX']-metrics[b][view]['worst_RX']))
  paired[str(seed)]['urban_mean_accuracy_pp']=100*(metrics[a]['urban_mean_accuracy']-metrics[b]['urban_mean_accuracy'])
 summary={};delta={}
 for arm in ('T1','KD'):
  summary[arm]={}
  for view in VIEWS:
   summary[arm][view]={}
   for metric in ('accuracy','macro_F1','worst_RX'):
    values=[100*metrics[arm+'-s'+str(s)][view][metric] for s in d.SEEDS];summary[arm][view][metric]=dict(mean_percent=statistics.mean(values),sample_std_pp=statistics.stdev(values),per_seed_percent=values)
  vals=[100*metrics[arm+'-s'+str(s)]['urban_mean_accuracy'] for s in d.SEEDS];summary[arm]['urban_mean_accuracy']=dict(mean_percent=statistics.mean(vals),sample_std_pp=statistics.stdev(vals),per_seed_percent=vals)
 for view in VIEWS:
  values=[paired[str(s)][view]['accuracy_pp'] for s in d.SEEDS];delta[view]=dict(mean_pp=statistics.mean(values),sample_std_pp=statistics.stdev(values),per_seed_pp=values,positive_seed_count=sum(v>0 for v in values))
 vals=[paired[str(s)]['urban_mean_accuracy_pp'] for s in d.SEEDS];delta['urban_mean_accuracy']=dict(mean_pp=statistics.mean(vals),sample_std_pp=statistics.stdev(vals),per_seed_pp=vals)
 return dict(status='ALL6_ROWS_ALL7_SCORED',run_id=d.RUN,model_seed_count=3,teacher_seed_count=1,metrics=metrics,summary=summary,paired=paired,paired_summary=delta,strata=strata,cost=cost,source_decision=read(d.BASE/'source_matrix_decision.json'),head_fit_updates=0,new_NN=0,raw_IQ_reads=0,goal_complete=False,uncertainty='Sample std across3 studentseeds;fixed test observations,one sharedteacher;not independent blind test or inorbit deployment')
