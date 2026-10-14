"""Default-head row truth-last scoring and source-only qualification."""
import statistics as st
from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():
 from .feature_check import run
 features=run()
 if features['status']!='VERIFIED' or features['models']!=3 or len(features['paired'])!=1:raise ValueError('Complete actual source feature matrix required')
 d.write(d.BASE/'source_features.json',features)
 source_CE={r['row_id']:d.read(d.BASE/r['row_id']/'source/completion.json')['final_training_L_CE'] for r in d.training_rows()}
 z=features['paired'][0]['source_features']['records'][0]['joint160_L2_mean']
 source_pass=all(v<=.5 for v in source_CE.values()) and z>=1e-6
 qualification=dict(status='PASS' if source_pass else 'FAIL',source_only=True,target_metrics_read_for_decision=False,final_L_CE=source_CE,actual_source_candidate_minus_X3_L2=z,fixed_rules=d.config(d.training_rows()[0])['source_falsification'],head_fit=False,source_feature_export_exact=True,actual_initial_native_body_and_head_equal=True,predefined_remaining_seeds=[2026092702,2026092703],configuration_changes_for_expansion=False)
 d.write(d.BASE/'source_qualification.json',qualification)
 records=[]
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];v=d.read(p/'scores/metrics.json');count=validate_records(c,v['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,count,design=d) or not v['truth_last'] or v['independent_recount']!='VERIFIED':raise ValueError('All fixed rows must be truth-last scored')
  records.extend(v['records'])
 summary=[];index={}
 for arm in d.ARMS:
  rr=[v for v in records if v['arm']==arm and v['dimension']=='overall']
  if len(rr)!=len(d.SEEDS) or {v['model_seed'] for v in rr}!=set(d.SEEDS):raise ValueError('Fixed seed set incomplete')
  values=[v['accuracy'] for v in rr];floor=lambda dim:st.mean(min(v['accuracy'] for v in records if v['arm']==arm and v['model_seed']==seed and v['dimension']==dim) for seed in d.SEEDS)
  summary.append(dict(arm=arm,seed_count=len(d.SEEDS),accuracy_mean=st.mean(values),accuracy_sd=st.stdev(values) if len(values)>1 else None,seed_accuracy={str(v['model_seed']):v['accuracy'] for v in rr},macro_f1_mean=st.mean(v['macro_f1'] for v in rr),worst_RX_mean=floor('receiver'),worst_day_mean=floor('day')))
  index.update({(arm,v['model_seed']):v['accuracy'] for v in rr})
 paired=[]
 for arm,control in d.PAIRS:
  differences=[index[arm,s]-index[control,s] for s in d.SEEDS];paired.append(dict(arm=arm,control=control,seeds=list(d.SEEDS),seed_differences=differences,mean_difference=st.mean(differences),positive_seeds=sum(v>0 for v in differences)))
 value=dict(status='VERIFIED',run_id=d.RUN,models=4,newly_trained_models=3,baseline_reused_models=1,prediction_count=672000,metric_records=72,summary=summary,paired=paired,records=records,source_qualification=qualification,goal_84=False,goal_requires_three_predefined_seeds=True,first_seed_is_not_goal_completion=True,head_fit=False,truth_last=True,target_feedback_to_training=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',value);return value
