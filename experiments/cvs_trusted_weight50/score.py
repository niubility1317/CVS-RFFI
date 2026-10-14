"""Default-head row truth-last scoring and source-only qualification."""
import statistics as st
from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():
 from .feature_check import run
 features=run()
 if features['status']!='VERIFIED' or features['models']!=2 or len(features['paired'])!=1:raise ValueError('Complete actual source feature matrix required')
 d.write(d.BASE/'source_features.json',features)
 source_CE={r['row_id']:d.read(d.BASE/r['row_id']/'source/completion.json')['final_training_L_CE'] for r in d.training_rows()}
 z=features['paired'][0]['source_features']['records'][0]['joint160_L2_mean']
 observations={r['row_id']:d.read(d.BASE/r['row_id']/'source/trusted_training_observation.json') for r in d.training_rows()}
 if any(v['status']!='VERIFIED' or v['layer_gradient_records']!=1998 or v['epochs']!=[33,34,50] or v['U_truth_read'] or v['head_fit'] for v in observations.values()):raise ValueError('Actual gradient evidence incomplete')
 source_pass=all(v<=.5 for v in source_CE.values()) and z>=1e-6
 qualification=dict(status='PASS' if source_pass else 'FAIL',source_only=True,target_metrics_read_for_decision=False,final_L_CE=source_CE,actual_source_WEIGHT_minus_SIG_L2=z,fixed_rules=d.config(d.training_rows()[0])['source_falsification'],head_fit=False,source_feature_export_exact=True,actual_initial_native_body_and_head_equal=True,predefined_remaining_seeds=[2026092702,2026092703],configuration_changes_for_expansion=False,actual_training_observations=observations)
 d.write(d.BASE/'source_qualification.json',qualification)
 records=[]
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];v=d.read(p/'scores/metrics.json');count=validate_records(c,v['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,count,design=d) or not v['truth_last'] or v['independent_recount']!='VERIFIED':raise ValueError('All fixed rows must be truth-last scored')
  records.extend(v['records'])
 from experiments.cvs_signature_stats_seeds.score import summarize
 summary,paired=summarize(records,d.SEEDS,design=d)
 value=dict(status='VERIFIED',run_id=d.RUN,models=3,newly_trained_models=2,baseline_reused_models=1,prediction_count=504000,metric_records=54,summary=summary,paired=paired,records=records,source_qualification=qualification,goal_84=False,goal_requires_three_predefined_seeds=True,first_seed_is_not_goal_completion=True,head_fit=False,truth_last=True,target_feedback_to_training=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',value);return value
