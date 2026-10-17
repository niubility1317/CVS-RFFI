"""Freeze source criterion before reading terminal formal scores;reused controls stay old."""
import csv
import numpy as np
from . import design as d
from .criteria import decide

def collect():
 source={r['arm']:d.read(d.source_row_root(r['arm'])/'source/result.json') for r in d.all_rows()};qualification={};complete=True
 for r in d.all_rows():
  f=d.source_row_root(r['arm'])/'source_qualification/complete.json'
  if not f.is_file():complete=False;continue
  q=d.read(f);expectedNN=16202 if r['arm'] in ('B10','B11') else 12602
  if q['config']!=d.row_config(r) or q['NN']!=expectedNN or q['target_access'] or q['formal_scores_read'] or q['training_updates'] or q['head_fit_updates']:raise ValueError('Fixed source qualification mismatch')
  qualification[r['arm']]=q
 ref=d.read(d.BASE/'reference_timing/complete.json')
 if ref['config']!=d.control_config('B00') or ref['NN']!=3602 or ref['target_access'] or ref['formal_scores_read'] or ref['training_updates']:raise ValueError('Matched reference timing readout differs')
 decision=dict(status='SOURCE_PARTIAL',source_only=True,source_qualification_complete=complete,source_replication_evidence=False,formal_scores_read=False,automatic_seed_extension=False)
 if complete:
  cv={r['arm']:d.read(d.source_row_root(r['arm'])/'source/completion.json')['final_source_metrics']['source_val_accuracy'] for r in d.all_rows()}
  with np.load(d.BASE/'reference_timing/predictions.npz',allow_pickle=False) as f:ids=f['ids'];offset=f['offset_samples']
  for arm in ('B10','B11'):
   with np.load(d.source_row_root(arm)/'source_qualification/predictions.npz',allow_pickle=False) as f:
    if not np.array_equal(f['ids'],ids) or not np.array_equal(f['offset_samples'],offset):raise ValueError('Same private source timing observations required')
  decision=decide(cv,ref['timing_accuracies'],qualification['B10']['timing_accuracies'],np.asarray(qualification['B01']['source_accuracies'])[:,1:],np.asarray(qualification['B11']['source_accuracies'])[:,1:])
 d.write(d.BASE/'source_matrix_decision.json',decision) # first,before any formal score read in this collector
 values={};full={};origins={}
 for r in d.all_rows():
  arm=r['arm'];origin=d.source_row_root(arm);rows=d.read(origin/'scoring/scores.json')['results'];overall={q['view']:q for q in rows if q['dimension']=='overall'}
  if set(overall)!=set(d.VIEWS):raise ValueError('Every fixed or reused row requires original7 views')
  values[arm]=overall;full[arm]=[dict(q,origin_run_id=d.row_config(r)['run_id'],reused_control=arm in ('B00','B01')) for q in rows];origins[arm]=str(origin)
 effects={v:dict(Ag_pp=100*(values['B10'][v]['accuracy']-values['B00'][v]['accuracy']),As_without_Ag_pp=100*(values['B01'][v]['accuracy']-values['B00'][v]['accuracy']),joint_net_pp=100*(values['B11'][v]['accuracy']-values['B00'][v]['accuracy']),Ag_added_to_As_pp=100*(values['B11'][v]['accuracy']-values['B01'][v]['accuracy']),interaction_pp=100*(values['B11'][v]['accuracy']-values['B10'][v]['accuracy']-values['B01'][v]['accuracy']+values['B00'][v]['accuracy'])) for v in d.VIEWS}
 result=dict(run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,views=values,effects=effects,origins=origins,reused_controls=['B00','B01'],new_rows=['B10','B11'],source_costs=source,source_qualification_complete=complete,source_decision=decision,reference_timing_cost=ref,source_qualification_costs={a:q['NN'] for a,q in qualification.items()},suburban_gaps_pp={a:{v:100*(q['clean']['accuracy']-q[v]['accuracy']) for v in d.SCENES[:3]} for a,q in values.items()},worst_RX_accuracy={a:{v:min(q['accuracy'] for q in rows if q['dimension']=='receiver' and q['view']==v) for v in d.VIEWS} for a,rows in full.items()},target_feedback=False,crossseed_confirmed=False,Ag_proved=False,goal_complete=False)
 d.write(d.BASE/'results.json',result)
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  fields=['row_id','origin_run_id','reused_control','model_seed','view','dimension','stratum','accuracy','macro_f1'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for rows in full.values():w.writerows(rows)
 return result
