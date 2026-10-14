from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():
 from .feature_check import run
 f=run()
 if f['status']!='VERIFIED' or f['models']!=2 or len(f['paired'])!=1:raise ValueError('FixedSIG actual feature witnesses incomplete')
 d.write(d.BASE/'source_features.json',f);CE={r['row_id']:d.read(d.BASE/r['row_id']/'source/completion.json')['final_training_L_CE'] for r in d.training_rows()};obs={r['row_id']:d.read(d.BASE/r['row_id']/'source/ssl_training_observation.json') for r in d.training_rows()};z=f['paired'][0]['source_features']['records'][0]['joint160_L2_mean'];pair=obs[f'SIG_PAIR-s{d.SEEDS[0]}'];coverage=pair['U_accepted']/pair['U_packets']
 valid=all(v['status']=='VERIFIED' and v['steps']==11100 and v['sampler_packets']==963900 and v['sampler_epochs']==17 and v['gradient_batches']==666 and v['gradient_epochs']==[33,34,50] and not v['head_fit_updates'] and not v['U_truth_read'] and v['loss_returned_unchanged'] for v in obs.values());passed=valid and all(v<=.5 for v in CE.values()) and coverage>=.25 and z>=1e-6
 q=dict(status='PASS' if passed else 'FAIL',source_only=True,target_metrics_read_for_decision=False,final_L_CE=CE,actual_SIG_PAIR_minus_RANDOM_L720zL2=z,paired_U_acceptance=coverage,actual_observations=obs,fixed_rules=d.config(d.training_rows()[0])['source_falsification'],predefined_remaining_seeds=[2026092702,2026092703],configuration_changes_for_expansion=False,head_fit_updates=0)
 d.write(d.BASE/'source_qualification.json',q);records=[]
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];v=d.read(p/'scores/metrics.json');n=validate_records(c,v['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,n,design=d) or not v['truth_last'] or v['independent_recount']!='VERIFIED':raise ValueError('FixedSIG default-head row closure incomplete')
  records.extend(v['records'])
 from experiments.cvs_signature_stats_seeds.score import summarize
 summary,paired=summarize(records,d.SEEDS,design=d);v=dict(status='VERIFIED',run_id=d.RUN,models=3,newly_trained_models=2,baseline_reused_models=1,prediction_count=504000,metric_records=54,summary=summary,paired=paired,records=records,source_qualification=q,goal_84=False,first_seed_is_not_goal_completion=True,goal_requires_three_predefined_seeds=True,head_fit_updates=0,truth_last=True,target_feedback_to_training=False,integration_not_new_core_algorithm=True,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',v);return v
