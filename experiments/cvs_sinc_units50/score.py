import json
from . import design as d
from .predict import preflight
from .evidence import recount
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def source_evidence():
 from .feature_check import run
 features=run()
 if features['status']!='VERIFIED' or features['models']!=2 or len(features['paired'])!=1:raise ValueError('Actual fixed feature witnesses incomplete')
 d.write(d.BASE/'source_features.json',features);observations=[];CE={}
 for row in d.training_rows():
  p=d.BASE/row['row_id']/'source';obs=recount(row);observations.append(obs)
  records=[json.loads(line) for line in (p/'epoch_metrics.jsonl').read_text().splitlines()]
  if [r['epoch'] for r in records]!=list(range(1,51)):raise ValueError('Complete fixed source curve missing')
  CE[row['arm']]=sum(r['train_loss_tx_labeled'] for r in records[-5:])/5
 candidate=next(r for r in observations if r['row_id'].startswith(d.PRIMARY+'-'));student=candidate['student_Sinc'];teacher=candidate['ownEMA_Sinc'];z=features['paired'][0]['source_features']['records'][0]['joint160_L2_mean']
 changed_low=24-student['parameter_changes']['low_hz_']['exactly_unchanged'];changed_band=24-student['parameter_changes']['band_hz_']['exactly_unchanged']
 passed=all(v<=.5 for v in CE.values()) and changed_low>=12 and changed_band>=12 and student['kernel_relative_L2']>=1e-6 and teacher['kernel_relative_L2']>=1e-6 and z>=1e-6
 q=dict(status='PASS' if passed else 'FAIL',source_only=True,fixed_rules=d.config(d.training_rows()[0])['source_falsification'],last5_mean_L_CE=CE,unit_changed_cutoffs=changed_low,unit_changed_bandwidths=changed_band,unit_student_kernel_relative_L2=student['kernel_relative_L2'],unit_ownEMA_kernel_relative_L2=teacher['kernel_relative_L2'],actual_candidate_minus_control_L720zL2=z,actual_observations=observations,target_metrics_read_for_decision=False,predefined_remaining_seeds=[2026092702,2026092703],head_fit_updates=0,limitation='Actual source numerical adaptation/health/interface only, not DG or84proof')
 d.write(d.BASE/'source_qualification.json',q);return q

def finish():
 q=source_evidence();records=[]
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];v=d.read(p/'scores/metrics.json');n=validate_records(c,v['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,n,design=d) or not v['truth_last'] or v['independent_recount']!='VERIFIED':raise ValueError('Original default-head scored row incomplete')
  records.extend(v['records'])
 from experiments.cvs_signature_stats_seeds.score import summarize
 summary,paired=summarize(records,d.SEEDS,design=d)
 v=dict(status='VERIFIED',run_id=d.RUN,models=3,newly_trained_models=2,baseline_reused_models=1,prediction_count=504000,metric_records=54,summary=summary,paired=paired,records=records,source_qualification=q,goal_84=False,first_seed_is_not_goal_completion=True,head_fit_updates=0,truth_last=True,target_feedback_to_training=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',v);return v
