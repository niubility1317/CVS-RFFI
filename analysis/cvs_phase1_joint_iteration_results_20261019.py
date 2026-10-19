"""Aggregate only completed, frozen seven-view predictions; zero neural work."""
import json
from pathlib import Path
VIEWS=('clean','practical_high','practical_mid','practical_low_suburban','practical_high_urban','practical_mid_urban','practical_low_urban')
R1='20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r01';R2='20261019-phase1-FcrossRX-AsTemperature-e30-s2026092701-m2-r02';RK='20261019-phase1-FcrossRX-sameAsKD-e30-s2026092701-m2-r01';REF='20261018-phase1-FcrossRX-impulse-factorial-e30-s2026092701-m6-r01'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def collect(project):
 import numpy as np
 p=Path(project)/'runs';roots={'T1':p/R2/'T1-s2026092701','T6':p/R1/'T6-s2026092701','KD':p/RK/'KD-s2026092701','AS30':p/REF/'B01-s2026092701','NOAS30':p/REF/'B00-s2026092701'}
 if not all((p/r/'source_matrix_decision.json').is_file() for r in (R2,RK)):raise ValueError('Observe existing preregistered source decisions before final readout')
 decisions={r:read(p/r/'source_matrix_decision.json') for r in (R2,RK)};metrics={};data={};cost={};strata={}
 for arm,b in roots.items():
  done=read(b/'prediction/complete.json');score=read(b/'scoring/complete.json')
  if done['status']!='PREDICTIONS_COMPLETE' or done['views']!=list(VIEWS) or done['truth_read'] or score['status']!='SCORED_COMPLETE':raise ValueError('Full frozen seven-view prediction/score required: '+arm)
  with np.load(b/'prediction/predictions.npz',allow_pickle=False) as f:data[arm]={k:f[k].copy() for k in f.files}
  rr=read(b/'scoring/scores.json')['results'];strata[arm]=rr;metrics[arm]={}
  for view in VIEWS:
   x=next(v for v in rr if v['view']==view and v['dimension']=='overall');rx=[v for v in rr if v['view']==view and v['dimension']=='receiver'];metrics[arm][view]=dict(accuracy=x['accuracy'],macro_F1=x['macro_f1'],worst_RX=min(v['accuracy'] for v in rx),count=len(data[arm][view]))
  metrics[arm]['urban_mean_accuracy']=sum(metrics[arm][v]['accuracy'] for v in VIEWS[-3:])/3
  cost[arm]={}
  for seg in ('stage20','source'):
   q=read(b/seg/'result.json');cost[arm][seg]={k:v for k,v in q.items() if v is None or isinstance(v,(str,int,float,bool))};cost[arm][seg]['sameAsKD']=q.get('sameAsKD');cost[arm][seg]['actual_U_student_NN']=q['student_train_NN']-q['L_NN']
 ids=data['T1']['ids']
 if any(not np.array_equal(x['ids'],ids) for x in data.values()):raise ValueError('Same-row fixed physical ID order differs')
 from experiments.cvs_noise_shape_factorial43 import design as d
 from experiments.cvs_early_geometry.evaluate import truth_arrays
 y,rx,day=truth_arrays(read(d.TRUTH),ids);paired={}
 for a,b in [('T6','T1'),('KD','T1'),('T1','AS30'),('T6','AS30'),('KD','AS30'),('T1','NOAS30'),('T6','NOAS30'),('KD','NOAS30')]:
  out={}
  for view in VIEWS:
   ac=data[a][view]==y;bc=data[b][view]==y;out[view]=dict(repaired=int((ac&~bc).sum()),new_errors=int((~ac&bc).sum()),net=int(ac.sum()-bc.sum()),accuracy_pp=100*float(ac.mean()-bc.mean()),macro_F1_pp=100*(metrics[a][view]['macro_F1']-metrics[b][view]['macro_F1']),worst_RX_pp=100*(metrics[a][view]['worst_RX']-metrics[b][view]['worst_RX']))
  out['urban_mean_accuracy_pp']=100*(metrics[a]['urban_mean_accuracy']-metrics[b]['urban_mean_accuracy']);paired[a+'_minus_'+b]=out
 return dict(status='ALL_THREE_FULL_E30_ALL7_SCORED',metrics=metrics,paired=paired,strata=strata,cost=cost,source_decisions=decisions,model_seed_count=1,channel_draws_are_not_model_seeds=True,crossseed_confirmed=False,head_fit_updates=0,NN=0,raw_IQ_reads=0,goal_complete=False)
