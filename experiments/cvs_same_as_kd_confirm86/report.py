from . import design as d

def source_decision():
 pairs=[]
 for seed in d.SEEDS:
  a={}
  for row in [r for r in d.rows() if r['model_seed']==seed]:
   b=d.row_root(row['row_id']);s=d.read(b/'source/completion.json');q=d.read(b/'source_qualification/complete.json')
   if q['target_access'] or q['formal_scores_read'] or q['config']!=d.config(row):raise ValueError('Source-only exact fixed final required')
   a[row['arm']]=dict(clean=s['final_source_metrics']['source_val_accuracy'],midlow=sum(q['source_accuracies'][j][i] for j in range(2) for i in (1,2))/4)
  pairs.append(dict(model_seed=seed,midlow_gain_pp=100*(a['KD']['midlow']-a['T1']['midlow']),clean_delta_pp=100*(a['KD']['clean']-a['T1']['clean'])))
 gain=sum(x['midlow_gain_pp'] for x in pairs)/3;clean=sum(x['clean_delta_pp'] for x in pairs)/3
 result=dict(status='SOURCE_DECIDED',pairs=pairs,midlow_gain_pp=gain,clean_delta_pp=clean,passed=gain>=.5 and clean>=-.1,model_seed_count=3,target_scores_read=False,automatic_extension=False,all_rows_still_all7=True,confirmation_fixed_from_seed01_source=True)
 d.write(d.BASE/'source_matrix_decision.json',result);return result

def collect():
 rows={};cost={}
 for row in d.rows():
  b=d.row_root(row['row_id'])
  if d.read(b/'complete.json')['status']!='TRAINED_AND_SCORED' or not (b/'prediction/complete.json').is_file() or not (b/'scoring/complete.json').is_file():raise ValueError('All6 rows must finish all7 before aggregate')
  rows[row['row_id']]=d.read(b/'scoring/scores.json')['results'];cost[row['row_id']]={str(k):d.read(b/('stage20' if k==20 else 'source')/'result.json') for k in (20,30)}
 result=dict(status='ANALYZED',rows=rows,cost=cost,source_decision=d.read(d.BASE/'source_matrix_decision.json'),model_seed_count=3,reused_model_count=2,new_model_count=4,goal_complete=False,teacher_seed_count=1)
 d.write(d.BASE/'results.json',result);return result
