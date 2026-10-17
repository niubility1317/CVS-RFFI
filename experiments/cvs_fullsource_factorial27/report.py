"""Fixed four-factor terminal reporting; never chooses a model from target scores."""
import csv
from pathlib import Path
from . import design as d
def collect():
 reference=Path(d.PROJECT)/'runs/20261017-phase1-FcrossRX-fullsource-concat-e30-s2026092701-m2-r01'
 roots={a:(d.BASE if a in ('B10','B11') else reference)/(a+'-s'+str(d.SEED)) for a in ('B00','B10','B01','B11')}
 values={};allrows=[];source={};qualified={}
 for arm,root in roots.items():
  pred=d.read(root/'prediction/complete.json');score=d.read(root/'scoring/complete.json');train=d.read(root/'source/completion.json')
  if pred['status']!='PREDICTIONS_COMPLETE' or pred['views']!=list(d.VIEWS) or pred['truth_read'] or not pred['all_parameters_frozen'] or pred['head_fit_updates'] or score['status']!='SCORED_COMPLETE' or not score['truth_last'] or train['epoch']!=30 or train['optimizer_steps']!=6660 or train['target_access'] or train['head_fit_updates'] or train['config']['source_roles']!=dict(L_s=6300,U_s=56700,V=27000) or train['config']['held_RX'] is not None:raise ValueError('Full-source frozen FINAL30 seven-view references required')
  records=d.read(root/'scoring/scores.json')['results'];q={x['view']:x for x in records if x['dimension']=='overall'}
  if set(q)!=set(d.VIEWS):raise ValueError('All fixed views required')
  values[arm]=q;allrows+=records;source[arm]=d.read(root/'source/result.json')
  if arm in ('B10','B11'):
   qual=root/'source_qualification/complete.json'
   qualified[arm]=d.read(qual) if qual.exists() else d.read(root/'source_qualification_failure.json')
 interaction={v:100*(values['B11'][v]['accuracy']-values['B10'][v]['accuracy']-values['B01'][v]['accuracy']+values['B00'][v]['accuracy']) for v in d.VIEWS}
 comparisons={pair:{v:100*(values[a][v]['accuracy']-values[b][v]['accuracy']) for v in d.VIEWS} for pair,a,b in [('Ag_independent','B10','B00'),('joint_net','B11','B00'),('As_on_Ag','B11','B10'),('Ag_on_As','B11','B01')]}
 source_complete=all(q.get('status')=='SOURCE_QUALIFIED_READOUT' for q in qualified.values())
 result=dict(run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,views=values,comparisons_accuracy_pp=comparisons,interaction_accuracy_pp=interaction,suburban_gaps_pp={a:{v:100*(q['clean']['accuracy']-q[v]['accuracy']) for v in d.SCENES[:3]} for a,q in values.items()},source_costs=source,source_qualification=qualified,source_qualification_complete=source_complete,new_models=2,reused_reference_models=2,new_model_epochs=60,new_updates=13320,training_parent_epochs=0,reference_historical_epochs=60,target_feedback=False,crossseed_confirmed=False,Ag_proved=False,goal_complete=False)
 d.write(d.BASE/'results.json',result)
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'],extrasaction='ignore');writer.writeheader();writer.writerows(allrows)
 return result
