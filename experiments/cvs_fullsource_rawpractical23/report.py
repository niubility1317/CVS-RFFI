"""Report the fixed row; read comparison scores only after both7-view predictions."""
import csv
from . import design as d
def collect():
 row=d.rows()[0]['row_id'];own=d.read(d.BASE/row/'scoring/scores.json');values={q['view']:q for q in own['results'] if q['dimension']=='overall'}
 result=dict(run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,views=values,source_cost=d.read(d.BASE/row/'source/result.json'),Ag_proved=False,crossseed_confirmed=False,target_feedback=False,goal_complete=False)
 path=d.REFERENCE_ROOT/'scoring/scores.json'
 if path.is_file() and (d.REFERENCE_ROOT/'prediction/complete.json').is_file() and (d.REFERENCE_ROOT/'scoring/complete.json').is_file():
  completed=d.read(d.REFERENCE_ROOT/'scoring/complete.json')
  if completed['status']!='SCORED_COMPLETE' or completed['views']!=list(d.VIEWS):raise ValueError('Comparison truth-last score incomplete')
  import numpy as np
  with np.load(d.REFERENCE_ROOT/'prediction/predictions.npz',allow_pickle=False) as a,np.load(d.BASE/row/'prediction/predictions.npz',allow_pickle=False) as b:
   if not np.array_equal(a['ids'],b['ids']):raise ValueError('Fixed comparison physical IDs differ')
  ref={q['view']:q for q in d.read(path)['results'] if q['dimension']=='overall'}
  if set(ref)!=set(d.VIEWS) or set(values)!=set(d.VIEWS):raise ValueError('All fixed7 views required')
  delta={v:100*(values[v]['accuracy']-ref[v]['accuracy']) for v in d.VIEWS}
  result.update(comparison_status='COMPLETE',reference=d.REFERENCE_RUN,accuracy_gain_pp=delta,urban_gain_pp=sum(delta[v] for v in d.SCENES[3:])/3)
 else:result['comparison_status']='REFERENCE_PENDING'
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  fields=['row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(own['results'])
 d.write(d.BASE/'results.json',result);return result
