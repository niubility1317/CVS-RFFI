import csv
from . import design as d
from .criteria import decide

def source_decision():
 clean={};urban={};rx={}
 sources={row['arm']:d.BASE/row['row_id'] for row in d.rows()}
 sources.update({arm:d.REFERENCE/(arm+'-s'+str(d.SEED)) for arm in ('B00','B01')})
 for arm,p in sources.items():
  s=d.read(p/'source/completion.json');q=d.read(p/'source_qualification/complete.json')
  if q['target_access'] or q['formal_scores_read'] or q['NN']!=12600 or q['training_updates']:raise ValueError('Source-only frozen readout required')
  if arm.startswith('H') and q['config']!=d.config(next(r for r in d.rows() if r['arm']==arm)):raise ValueError('Own source config differs')
  clean[arm]=s['final_source_metrics']['source_val_accuracy'];urban[arm]=q['source_accuracies'];rx[arm]=s['final_source_metrics']['source_val_rx_accuracy']
 result=decide(clean,urban,rx);d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 records=[];values={};cost={}
 for arm,p in [(a,d.REFERENCE/(a+'-s'+str(d.SEED))) for a in ('B00','B01')]+[(r['arm'],d.BASE/r['row_id']) for r in d.rows()]:
  rows=d.read(p/'scoring/scores.json')['results'];records.extend(dict(x,comparison_arm=arm) for x in rows)
  values[arm]={r['view']:r for r in rows if r['dimension']=='overall'}
  if set(values[arm])!=set(d.VIEWS):raise ValueError('All7 defaulthead metrics required')
  cost[arm]={str(k):d.read(p/('stage20' if k==20 else 'source')/'result.json') for k in (20,30)}
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['comparison_arm','row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'],extrasaction='ignore');w.writeheader();w.writerows(records)
 decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').exists() else None
 result=dict(status='ANALYZED' if decision else 'FORMAL_COMPLETE_SOURCE_PARTIAL',views=values,cost=cost,model_seed_count=1,source_decision=decision,source_only_selection='All2 fixedFINAL30 autoall7;no sourcecriterion skips formal rows',formal_feedback=False,Ag_proved=False,loss_only_ablation=True,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
