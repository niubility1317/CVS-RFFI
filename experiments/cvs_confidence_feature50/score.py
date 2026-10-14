import gzip,json
from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion
def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def source_evidence():
 from .feature_check import run
 features=run()
 if features['status']!='VERIFIED' or features['models']!=2 or len(features['paired'])!=1:raise ValueError('Actual fixed feature witnesses incomplete')
 d.write(d.BASE/'source_features.json',features);CE={};obs={};discrepancy={};signal={};all_discrepancy={};retention={}
 for row in d.training_rows():
  p=d.BASE/row['row_id']/'source';CE[row['arm']]=d.read(p/'completion.json')['final_training_L_CE'];v=d.read(p/'feature_mean_teacher_training_observation.json');obs[row['arm']]=v
  with gzip.open(p/'feature_mean_teacher_steps.jsonl.gz','rt',encoding='utf8') as stream:records=[json.loads(t) for t in stream]
  if len(records)!=11100 or sum(v['U_packets'] for v in records)!=963900 or v['steps']!=11100 or v['U_feature_steps']!=3774 or v['U_feature_packets']!=963900 or v['extra_model_forwards'] or v['head_fit_updates'] or v['U_truth_read'] or not v['hard_pseudo_loss_returned_unchanged']:raise ValueError('Feature consistency native budget differs')
  last=[r for r in records if r['epoch']>=46];n=sum(r['U_packets'] for r in last)
  if n!=283500:raise ValueError('Final5 sourceU exposure differs')
  kept=sum(r['retained'] for r in last);retention[row['arm']]=kept/n
  discrepancy[row['arm']]=sum(r['U_packets']*r['feature_loss'] for r in last)/kept if kept else None
  all_discrepancy[row['arm']]=sum(r['U_packets']*r['full_feature_loss'] for r in last)/n
  g=d.read(p/'feature_mean_teacher_gradient_records.json')['batch_records'];signal[row['arm']]=all(any(r['epoch']==ep and r['layers'][k]['effective']['auxiliary_norm']>1e-12 for r in g) for ep in (34,50) for k in range(3))
 z=features['paired'][0]['source_features']['records'][0]['joint160_L2_mean'];den=discrepancy['SIG_TIME'];ratio=discrepancy['CONF_FEATURE_MT']/den if den is not None and den>0 and discrepancy['CONF_FEATURE_MT'] is not None else None;all_ratio=all_discrepancy['CONF_FEATURE_MT']/all_discrepancy['SIG_TIME'] if all_discrepancy['SIG_TIME']>0 else None
 passed=all(x<=.5 for x in CE.values()) and z>=1e-6 and signal['CONF_FEATURE_MT'] and ratio is not None and ratio<=.95 and all_ratio is not None and all_ratio<=1. and retention['CONF_FEATURE_MT']>=.5
 q=dict(status='PASS' if passed else 'FAIL',source_only=True,target_metrics_read_for_decision=False,final_L_CE=CE,last5_U_retained_feature_mean=discrepancy,retained_feature_mean_ratio=ratio,last5_allU_feature_mean=all_discrepancy,allU_feature_mean_ratio=all_ratio,last5_confidence_gate_retention=retention,actual_t1_t2_SIG_gradient_signal=signal,actual_candidate_minus_control_L720zL2=z,actual_observations=obs,fixed_rules=d.config(d.training_rows()[0])['source_falsification'],predefined_remaining_seeds=[2026092702,2026092703],configuration_changes_for_expansion=False,head_fit_updates=0,limitation='Source optimization and exposure qualification only, not DG proof or U correctness')
 d.write(d.BASE/'source_qualification.json',q);return q
def finish():
 qualification=source_evidence();records=[]
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];v=d.read(p/'scores/metrics.json');n=validate_records(c,v['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,n,design=d) or not v['truth_last'] or v['independent_recount']!='VERIFIED':raise ValueError('Fixed defaulthead row closure incomplete')
  records.extend(v['records'])
 from experiments.cvs_signature_stats_seeds.score import summarize
 summary,paired=summarize(records,d.SEEDS,design=d);v=dict(status='VERIFIED',run_id=d.RUN,models=3,newly_trained_models=2,baseline_reused_models=1,prediction_count=504000,metric_records=54,summary=summary,paired=paired,records=records,source_qualification=qualification,goal_84=False,first_seed_is_not_goal_completion=True,goal_requires_three_predefined_seeds=True,head_fit_updates=0,truth_last=True,target_feedback_to_training=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',v);return v
