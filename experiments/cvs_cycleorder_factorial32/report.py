"""Source decision first; terminal formal factorial never feeds the recipe."""
import csv
import numpy as np
from . import design as d
def collect():
 source={r['arm']:d.read(d.BASE/r['row_id']/'source/result.json') for r in d.rows()};qualification={};complete=True
 for r in d.rows():
  f=d.BASE/r['row_id']/'source_qualification/complete.json'
  if not f.is_file():complete=False;continue
  q=d.read(f)
  if q['config']!=d.config(r) or q['NN']!=12602 or q['target_access'] or q['formal_scores_read']:raise ValueError('Fixed source-only qualification mismatch')
  qualification[r['arm']]=q
 decision=dict(status='SOURCE_PARTIAL',source_only=True,source_qualification_complete=complete,source_replication_evidence=False,automatic_seed_extension=False,formal_scores_read=False)
 if complete:
  cv={a:d.read(d.BASE/(a+'-s'+str(d.SEED))/'source/completion.json')['final_source_metrics']['source_val_accuracy'] for a in source}
  delta=100*(np.asarray(qualification['B11']['source_accuracies'])[:,1:]-np.asarray(qualification['B01']['source_accuracies'])[:,1:]);checks=dict(Ag_clean_V_gain=100*(cv['B10']-cv['B00'])>=.5-1e-10,joint_clean_V_net=100*(cv['B11']-cv['B00'])>=-1e-10,all_urban_draws_gain=bool((delta>=1.-1e-10).all()))
  decision.update(status='SOURCE_DECIDED',checks=checks,source_replication_evidence=all(checks.values()),clean_V=cv,Ag_clean_V_gain_pp=100*(cv['B10']-cv['B00']),joint_clean_V_net_pp=100*(cv['B11']-cv['B00']),urban_draw_delta_pp=delta.tolist())
 d.write(d.BASE/'source_matrix_decision.json',decision) # Before this process reads any formal scores.
 values={};full={}
 for r in d.rows():
  rows=d.read(d.BASE/r['row_id']/'scoring/scores.json')['results'];overall={q['view']:q for q in rows if q['dimension']=='overall'}
  if set(overall)!=set(d.VIEWS):raise ValueError('Each fixed row requires all7 views')
  values[r['arm']]=overall;full[r['arm']]=rows
 effects={v:{'Ag_clean_or_channel_pp':100*(values['B10'][v]['accuracy']-values['B00'][v]['accuracy']),'As_without_Ag_pp':100*(values['B01'][v]['accuracy']-values['B00'][v]['accuracy']),'joint_net_pp':100*(values['B11'][v]['accuracy']-values['B00'][v]['accuracy']),'Ag_added_to_As_pp':100*(values['B11'][v]['accuracy']-values['B01'][v]['accuracy']),'interaction_pp':100*(values['B11'][v]['accuracy']-values['B10'][v]['accuracy']-values['B01'][v]['accuracy']+values['B00'][v]['accuracy'])} for v in d.VIEWS}
 result=dict(run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,views=values,effects=effects,suburban_gaps_pp={a:{v:100*(q['clean']['accuracy']-q[v]['accuracy']) for v in d.SCENES[:3]} for a,q in values.items()},worst_RX_accuracy={a:{v:min(q['accuracy'] for q in rows if q['dimension']=='receiver' and q['view']==v) for v in d.VIEWS} for a,rows in full.items()},source_costs=source,source_qualification_complete=complete,source_decision=decision,target_feedback=False,crossseed_confirmed=False,Ag_proved=False,goal_complete=False)
 d.write(d.BASE/'results.json',result)
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  fields=['row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for rows in full.values():w.writerows(rows)
 return result
