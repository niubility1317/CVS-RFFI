"""Default-only truth-last scoring; a single seed cannot complete the three-seed goal."""
import statistics as st
from pathlib import Path
from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():
 records=[]
 for r in d.rows():
  c=d.config(r);p=d.BASE/r['row_id'];metrics=d.read(p/'scores/metrics.json');count=validate_records(c,metrics['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,count,design=d) or not metrics['truth_last'] or metrics['independent_recount']!='VERIFIED':raise ValueError('Default truth-last row completion differs')
  records.extend(metrics['records'])
 summary=[]
 for arm in d.ARMS:
  allrows=[v for v in records if v['arm']==arm];over=next(v for v in allrows if v['dimension']=='overall')
  summary.append(dict(arm=arm,seed_count=1,model_seed=d.SEEDS[0],accuracy=over['accuracy'],accuracy_sd=None,macro_f1=over['macro_f1'],worst_RX=min(v['accuracy'] for v in allrows if v['dimension']=='receiver'),worst_day=min(v['accuracy'] for v in allrows if v['dimension']=='day')))
 index={s['arm']:s['accuracy'] for s in summary};paired=[dict(arm=a,control=b,model_seed=d.SEEDS[0],difference=index[a]-index[b],sample_SD=None) for a,b in d.PAIRS]
 from .feature_check import run
 features=run(d.SEEDS[0]);d.write(d.BASE/'source_features.json',features)
 if features['status']!='VERIFIED' or len(features['paired'])!=1:raise ValueError('Actual paired source feature proof incomplete')
 source=[]
 for r in d.training_rows():
  c=d.config(r);p=Path(c['output_root']);done=d.read(p/'completion.json');diag=d.read(p/'source_style_epoch_diagnostic.json');epochs=diag['records']
  if [v['epoch'] for v in epochs]!=list(range(1,51)) or diag['query_read'] or diag['head_fit'] or diag['U_truth_read']:raise ValueError('Source style epoch witness differs')
  counts={}
  for n in d.STYLE_LAYERS:
   counts[n]={k:sum(v['layers'][n][k] for v in epochs) for k in ('calls','enabled_calls','packet_calls','eligible_packets','changed_packets')}
  enabled=c['identity_mixstyle']['enabled']
  if any(v['calls']!=(11100 if enabled else 0) for v in counts.values()) or any(v['teacher_style_calls'] or v['U_style_calls'] or v['eval_style_calls'] for v in epochs):raise ValueError('Actual fixed L-only hook budget differs')
  source.append(dict(row_id=r['row_id'],final_L_CE=done['final_training_L_CE'],layers=counts))
 pair=features['paired'][0]['source_features']['records'][0];style=next(v for v in source if v['row_id'].startswith(d.PRIMARY+'-'))
 qualified=all(v['final_L_CE']<=.5 for v in source) and all(v['packet_calls']>0 and v['changed_packets']/v['packet_calls']>=.25 for v in style['layers'].values()) and pair['joint160_L2_mean']>=1e-6
 value=dict(status='VERIFIED',run_id=d.RUN,models=3,newly_trained_models=2,baseline_reused_models=1,summary=summary,paired=paired,records=records,source=source,source_falsification_pass=qualified,source_falsification_fixed_before_target=True,additional_seed_launch=False,source_feature_pair=pair,goal_84=False,goal_requires_three_predefined_seeds=True,clean_three_seed_mean=None,truth_last=True,target_feedback_to_training=False,head_fit=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',value);return value
