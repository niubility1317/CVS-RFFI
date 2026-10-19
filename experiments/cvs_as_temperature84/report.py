from . import design as d
def source_decision():
 a={}
 for row in d.rows():
  s=d.read(d.row_root(row['row_id'])/'source/completion.json');q=d.read(d.row_root(row['row_id'])/'source_qualification/complete.json')
  if q['target_access'] or q['formal_scores_read'] or q['config']!=d.config(row):raise ValueError('Source-only matched frozen final required')
  a[row['arm']]=dict(clean=s['final_source_metrics']['source_val_accuracy'],urban=q['source_accuracies'])
 midlow={k:sum(v['urban'][j][i] for j in range(2) for i in (1,2))/4 for k,v in a.items()};gain=100*(midlow['T6']-midlow['T1']);clean=100*(a['T6']['clean']-a['T1']['clean']);result=dict(status='SOURCE_DECIDED',passed=gain>=.5 and clean>=-.1,midlow_gain_pp=gain,clean_delta_pp=clean,midlow_min_gain_pp=.5,clean_min_delta_pp=-.1,target_scores_read=False,automatic_extension=False,all_rows_still_all7=True)
 d.write(d.BASE/'source_matrix_decision.json',result);return result
def collect():
 rows={};cost={}
 for row in d.rows():
  b=d.row_root(row['row_id']);rs=d.read(b/'scoring/scores.json')['results'];rows[row['arm']]=rs;cost[row['arm']]={str(k):d.read(b/('stage20' if k==20 else 'source')/'result.json') for k in (20,30)}
 result=dict(status='ANALYZED',rows=rows,cost=cost,source_decision=d.read(d.BASE/'source_matrix_decision.json'),model_seed_count=1,crossseed_confirmed=False,goal_complete=False)
 d.write(d.BASE/'results.json',result);return result
