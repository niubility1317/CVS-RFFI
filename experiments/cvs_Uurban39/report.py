import csv
from . import design as d
from .criteria import decide

def source_decision():
 clean={};urban={};teacher={}
 for row in d.rows():
  arm=row['arm'];b=d.BASE/row['row_id'];s=d.read(b/'source/completion.json');q=d.read(b/'source_qualification/complete.json');r=d.read(b/'source/result.json')
  if q['target_access'] or q['formal_scores_read'] or q['NN']!=12600 or q['training_updates'] or q['config']!=d.config(row):raise ValueError('Only all3 frozen source rows decide')
  clean[arm]=s['final_source_metrics']['source_val_accuracy'];urban[arm]=q['source_accuracies'];teacher[arm]=r['U_source_qualified']
 result=decide(clean,urban,teacher);d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 values={};records=[];cost={}
 for row in d.rows():
  b=d.BASE/row['row_id'];r=d.read(b/'scoring/scores.json')['results'];values[row['arm']]={v['view']:v for v in r if v['dimension']=='overall'}
  if set(values[row['arm']])!=set(d.VIEWS):raise ValueError('All7 actual frozen defaulthead views required')
  records.extend(dict(x,comparison_arm=row['arm']) for x in r);cost[row['arm']]={str(k):d.read(b/('stage20' if k==20 else 'source')/'result.json') for k in (20,30)}
 with (d.BASE/'per_row_seed.csv').open('x',encoding='utf8',newline='') as f:
  fields=['comparison_arm','row_id','model_seed','view','dimension','stratum','accuracy','macro_f1'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(records)
 result=dict(status='ANALYZED' if (d.BASE/'source_matrix_decision.json').is_file() else 'FORMAL_COMPLETE_SOURCE_PARTIAL',views=values,cost=cost,model_seed_count=1,source_decision=d.read(d.BASE/'source_matrix_decision.json') if (d.BASE/'source_matrix_decision.json').is_file() else None,source_only_selection='fixedFINAL30 allrows;sourcecriterion never skips formalrows',source_decision_timing='Once all source rows complete;agent reads no formal metrics before decision;each fixed row auto-predicts independently',formal_feedback=False,Ag_proved=False,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
