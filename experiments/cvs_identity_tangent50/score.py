from . import design as d
from .predict import preflight
from experiments.cvs_feature_sequence50 import score as shared
def score(row):return shared.score(row,design=d,prediction_preflight=preflight)
def finish():
 # Source qualification is written before this function opens any target metric.
 source={};qualified=True
 for row in d.rows():
  c=d.config(row)
  from .provenance import source_proof
  cp=source_proof(c);v=d.read(cp/'completion.json');source[row['arm']]=dict(CE=v['final_training_L_CE'],metrics=v['final_source_metrics'])
 base=source['BASE_REF']['metrics']
 for arm in d.ARMS[1:]:
  v=source[arm];qualified=qualified and v['CE']<=.5 and v['metrics']['source_val_accuracy']>=base['source_val_accuracy']-.01 and v['metrics']['source_val_worst_rx']>=base['source_val_worst_rx']-.02
 d.write(d.BASE/'source_extension_decision.json',dict(source_only=True,source_health_pass=qualified,parameters_frozen=True,target_metrics_read=False,source=source,scope='Health criterion only; remaining two seeds need a separate immutable registered run, no automatic parameter change'))
 records=[]
 for row in d.rows():
  c=d.config(row);done=d.read(d.BASE/row['row_id']/'completion.json');v=d.read(d.BASE/row['row_id']/'scores/metrics.json')
  if done['status']!='TRAINED_DEFAULT_PREDICTED_SCORED' or done['config']!=c or not v['truth_last'] or v['independent_recount']!='VERIFIED' or len(v['records'])!=18:raise ValueError('Incomplete fixed first-seed matrix')
  for r in v['records']:
   cm=r['confusion'];n=sum(map(sum,cm));acc=sum(cm[k][k] for k in range(6))/n
   f1=sum(2*cm[k][k]/(sum(cm[k])+sum(rr[k] for rr in cm)) if (sum(cm[k])+sum(rr[k] for rr in cm)) else 0. for k in range(6))/6
   if n!=r['query_count'] or abs(acc-r['accuracy'])>1e-12 or abs(f1-r['macro_f1'])>1e-12:raise ValueError('Independent6class CM recount differs')
  records.extend(v['records'])
 index={r['arm']:r for r in records if r['dimension']=='overall'}
 if set(index)!=set(d.ARMS):raise ValueError('Missing first-seed condition')
 summary=[dict(arm=a,model_seed=d.SEEDS[0],accuracy=index[a]['accuracy'],sample_sd=None,macro_f1=index[a]['macro_f1']) for a in d.ARMS]
 paired=[dict(treatment=t,control=c,model_seed=d.SEEDS[0],difference=index[t]['accuracy']-index[c]['accuracy']) for t,c in d.PAIRS]
 result=dict(status='VERIFIED',run_id=d.RUN,**d.result_metadata(),summary=summary,paired=paired,records=records,source=source,source_health_pass=qualified,goal_84=False,goal_reason='One model seed cannot meet three-predefined-seed mean target',head_fit_updates=0,truth_last=True,exposed_benchmark=True)
 d.write(d.BASE/'analysis.json',result);return result
