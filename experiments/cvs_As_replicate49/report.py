"""Three-seed As replication; new rows never consume target scores to train."""
import csv,statistics
from . import design as d
SEED1=2026092701

def located_rows():
 rows=[dict(row_id=a+'-s'+str(SEED1),arm=a,model_seed=SEED1,base=d.REFERENCE/(a+'-s'+str(SEED1)),reused=True) for a in ('B00','B01')]
 return rows+[dict(**r,base=d.BASE/r['row_id'],reused=False) for r in d.rows()]

def source_decision():
 records=[];checks=[]
 for seed in (SEED1,*d.SEEDS):
  found={}
  for r in located_rows():
   if r['model_seed']!=seed:continue
   b=r['base'];cp=d.read(b/'source/completion.json');q=d.read(b/'source_qualification/complete.json')
   if cp['target_access'] or q['target_access'] or q['formal_scores_read'] or q['NN']!=12600 or cp['config']['model_seed']!=seed or q['config']!=cp['config']:raise ValueError('Exact independent source record required')
   clean=cp['final_source_metrics'];found[r['arm']]=dict(clean=clean['source_val_accuracy'],rx=clean['source_val_rx_accuracy'],urban=q['source_accuracies'])
  if set(found)!={'B00','B01'}:raise ValueError('Missing matched baseline pair')
  a=found['B00'];b=found['B01'];gains=[[100*(b['urban'][draw][scene]-a['urban'][draw][scene]) for scene in (1,2)] for draw in (0,1)]
  item=dict(model_seed=seed,source_clean_delta_pp=100*(b['clean']-a['clean']),source_urban_mid_low_gain_pp=gains,source_clean_RX_delta_pp={r:100*(b['rx'][r]-a['rx'][r]) for r in a['rx']})
  item['source_As_replicated']=all(g>=5 for v in gains for g in v) and item['source_clean_delta_pp']>=-.5 and min(item['source_clean_RX_delta_pp'].values())>=-1
  records.append(item);checks.append(item['source_As_replicated'])
 result=dict(status='SOURCE_ANALYZED',records=records,source_As_replicated_all3=all(checks),new_Ag=False,Ag_proved=False,goal_complete=False,formal_scores_read=False,decision='BASELINE_SOURCE_AS_REPLICATED' if all(checks) else 'BASELINE_SOURCE_AS_NOT_ALL_SEEDS',scientific_failure_does_not_skip_registered_formal_rows=True)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 if not (d.BASE/'source_matrix_decision.json').exists():source_decision()
 values={};records=[];cost={};pairs=[]
 for r in located_rows():
  b=r['base'];done=d.read(b/'complete.json');pred=d.read(b/'prediction/complete.json');sc=d.read(b/'scoring/complete.json')
  if done['status']!='TRAINED_AND_SCORED' or pred['views']!=list(d.VIEWS) or pred['truth_read'] or not sc['truth_last']:raise ValueError('All7 frozen truth-last rows required')
  scores=d.read(b/'scoring/scores.json')['results'];key=r['row_id'];values[key]={v['view']:v for v in scores if v['dimension']=='overall'}
  if set(values[key])!=set(d.VIEWS) or any(v['model_seed']!=r['model_seed'] for v in scores):raise ValueError('View/seed identity differs')
  records.extend(dict(v,reused_seed01=r['reused']) for v in scores);cost[key]={str(k):d.read(b/('stage20' if k==20 else 'source')/'result.json') for k in (20,30)}
 for seed in (SEED1,*d.SEEDS):
  for view in d.VIEWS:
   a=values['B00-s'+str(seed)][view];b=values['B01-s'+str(seed)][view];pairs.append(dict(model_seed=seed,view=view,accuracy_delta_pp=100*(b['accuracy']-a['accuracy']),macro_F1_delta_pp=100*(b['macro_f1']-a['macro_f1'])))
 summary={view:{metric:dict(mean=statistics.mean([p[metric] for p in pairs if p['view']==view]),SD=statistics.stdev([p[metric] for p in pairs if p['view']==view]),n=3) for metric in ['accuracy_delta_pp','macro_F1_delta_pp']} for view in d.VIEWS}
 for filename,rows in [('per_row_seed.csv',records),('paired_seed_effects.csv',pairs)]:
  with (d.BASE/filename).open('x',encoding='utf8',newline='') as f:
   fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 result=dict(status='ANALYZED',new_model_seed_count=2,reused_model_seed_count=1,aggregate_model_seed_count=3,new_models=4,reused_models=2,rows=values,paired_seed_effects=pairs,paired_summary=summary,cost=cost,source_decision=d.read(d.BASE/'source_matrix_decision.json'),formal_feedback=False,Ag_proved=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
