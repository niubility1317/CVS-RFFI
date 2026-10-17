"""Compare actual full-source scratch30 controls;never an invented Ag factorial."""
import csv
import numpy as np
from . import design as d
from .criteria import decide
def source_decision():
 q=d.read(d.BASE/d.rows()[0]['row_id']/'source_qualification/complete.json');r=d.read(d.source_row_root('AS50')/'source_qualification/complete.json');c0=d.read(d.BASE/'reference_source/complete.json')
 for obj,c in ((q,d.config(d.rows()[0])),(r,d.control_config('AS50')),(c0,d.control_config('C0'))):
  if obj['config']!=c or obj['NN']!=12602 or obj['target_access'] or obj['formal_scores_read'] or obj['training_updates'] or obj['head_fit_updates']:raise ValueError('Only frozen matching source readouts')
 paths=[d.BASE/d.rows()[0]['row_id']/'source_qualification/predictions.npz',d.source_row_root('AS50')/'source_qualification/predictions.npz',d.BASE/'reference_source/predictions.npz'];ids=[]
 for p in paths:
  with np.load(p,allow_pickle=False) as f:ids.append(f['ids'].copy())
 if not all(np.array_equal(ids[0],v) for v in ids):raise ValueError('Same physical source panel required')
 clean={a:d.read(d.source_row_root(a)/'source/completion.json')['final_source_metrics']['source_val_accuracy'] for a in ('C0','R25')};decision=decide(clean,[x[1:] for x in q['source_accuracies']],[x[1:] for x in r['source_accuracies']]);decision['source_qualification_complete']=True;d.write(d.BASE/'source_matrix_decision.json',decision);return decision

def collect():
 p=d.BASE/'source_matrix_decision.json';decision=d.read(p) if p.is_file() else dict(status='SOURCE_PARTIAL',source_qualification_complete=False,automatic_seed_extension=False);values={};full={};source={}
 for row in d.all_rows():
  arm=row['arm'];origin=d.source_row_root(arm);records=d.read(origin/'scoring/scores.json')['results'];v={x['view']:x for x in records if x['dimension']=='overall'}
  if set(v)!=set(d.VIEWS):raise ValueError('All7 formal views required')
  values[arm]=v;full[arm]=[dict(x,comparison_arm=arm,origin_run_id=d.row_config(row)['run_id'],reused_control=arm!='R25') for x in records];source[arm]=d.read(origin/'source/result.json')
 effects={v:{b:100*(values['R25'][v]['accuracy']-values[b][v]['accuracy']) for b in ('C0','AS50')} for v in d.VIEWS}
 result=dict(run_id=d.RUN,model_seed=d.SEED,model_seed_count=1,views=values,effects_pp=effects,new_rows=['R25'],reused_controls=['C0','AS50'],source_costs=source,source_qualification_complete=decision.get('source_qualification_complete',False),source_decision=decision,suburban_gaps_pp={a:{v:100*(q['clean']['accuracy']-q[v]['accuracy']) for v in d.SCENES[:3]} for a,q in values.items()},worst_RX_accuracy={a:{v:min(q['accuracy'] for q in rs if q['dimension']=='receiver' and q['view']==v) for v in d.VIEWS} for a,rs in full.items()},reference_source_cost=d.read(d.BASE/'reference_source/complete.json'),new_source_qualification_cost=12602,target_feedback=False,crossseed_confirmed=False,Ag_proved=False,goal_complete=False)
 d.write(d.BASE/'results.json',result)
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  fields=['comparison_arm','row_id','origin_run_id','reused_control','model_seed','view','dimension','stratum','accuracy','macro_f1'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for rs in full.values():w.writerows(rs)
 return result
