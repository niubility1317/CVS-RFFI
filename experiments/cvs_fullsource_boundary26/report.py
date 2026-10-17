"""Terminal comparison only; reference scores never enter training."""
import csv
from . import design as d
def collect():
 values={};full={};source={}
 for arm,root in [('BCTX',d.BASE/('BCTX-s'+str(d.SEED))),('B01',d.REFERENCE_RUN/('B01-s'+str(d.SEED))),('B00',d.REFERENCE_RUN/('B00-s'+str(d.SEED)))]:
  done=d.read(root/'scoring/complete.json');pred=d.read(root/'prediction/complete.json');train=d.read(root/'source/completion.json')
  if done['status']!='SCORED_COMPLETE' or not done['truth_last'] or pred['views']!=list(d.VIEWS) or pred['truth_read'] or not pred['all_parameters_frozen'] or train['epoch']!=30 or train['optimizer_steps']!=6660:raise ValueError('All terminal defaulthead7view references required')
  rows=d.read(root/'scoring/scores.json')['results'];overall={q['view']:q for q in rows if q['dimension']=='overall'}
  if set(overall)!=set(d.VIEWS):raise ValueError('Seven fixed views required')
  full[arm]=rows;values[arm]=overall;source[arm]=d.read(root/'source/result.json')
 delta={v:100*(values['BCTX'][v]['accuracy']-values['B01'][v]['accuracy']) for v in d.VIEWS}
 result=dict(run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,views=values,BCTX_minus_B01_accuracy_pp=delta,urban_mean_gain_pp=sum(delta[v] for v in d.SCENES[3:])/3,suburban_gaps_pp={a:{v:100*(q['clean']['accuracy']-q[v]['accuracy']) for v in d.SCENES[:3]} for a,q in values.items()},source_costs=source,reference_models_reused=2,new_models=1,target_feedback=False,crossseed_confirmed=False,Ag_proved=False,goal_complete=False)
 d.write(d.BASE/'results.json',result)
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  fields=['row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for rows in full.values():w.writerows(rows)
 return result
