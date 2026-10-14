from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def required_backbone_signal(grad):
 return all(any(v['epoch']==ep and v['layers'][k]['effective']['auxiliary_norm']>1e-12 for v in grad) for ep in (17,33,50) for k in range(len(d.FISHR_PARAMETERS)))
def finish():
 from .feature_check import run
 features=run()
 if features['status']!='VERIFIED' or features['models']!=3 or len(features['paired'])!=2:raise ValueError('Actual two-candidate feature comparisons incomplete')
 d.write(d.BASE/'source_features.json',features)
 CE={r['row_id']:d.read(d.BASE/r['row_id']/'source/completion.json')['final_training_L_CE'] for r in d.training_rows()};obs={r['row_id']:d.read(d.BASE/r['row_id']/'source/backbone_fishr_training_observation.json') for r in d.training_rows()};candidate=[]
 for arm in ('BONE_RAW_FISHR','BONE_COND_FISHR'):
  z=next(p['source_features']['records'][0]['joint160_L2_mean'] for p in features['paired'] if p['arm']==arm);grad=d.read(d.BASE/f'{arm}-s{d.SEEDS[0]}'/'source/backbone_fishr_gradient_records.json')['batch_records'];nonzero=required_backbone_signal(grad);candidate.append(dict(arm=arm,actual_L720zL2=z,actual_t1_t2_gradient_nonzero_17_33_50=nonzero,passed=z>=1e-6 and nonzero))
 valid=all(v['status']=='VERIFIED' and v['extra_batches']==650 and v['gradient_batches']==52 and not v['head_fit_updates'] and not v['U_truth_read'] for v in obs.values());passed=valid and all(v<=.5 for v in CE.values()) and all(r['passed'] for r in candidate)
 qualification=dict(status='PASS' if passed else 'FAIL',source_only=True,target_metrics_read_for_decision=False,final_L_CE=CE,candidates=candidate,actual_observations=obs,fixed_rules=d.config(d.training_rows()[0])['source_falsification'],predefined_remaining_seeds=[2026092702,2026092703],expand_both_candidates_only_after_both_source_PASS=True,configuration_changes_for_expansion=False,head_fit_updates=0)
 d.write(d.BASE/'source_qualification.json',qualification)
 records=[]
 for row in d.rows():
  c=d.config(row);p=d.BASE/row['row_id'];v=d.read(p/'scores/metrics.json');count=validate_records(c,v['records'],design=d)
  if d.read(p/'completion.json')!=row_completion(c,count,design=d) or not v['truth_last'] or v['independent_recount']!='VERIFIED':raise ValueError('Fixed rows not closed')
  records.extend(v['records'])
 from experiments.cvs_signature_stats_seeds.score import summarize
 summary,paired=summarize(records,d.SEEDS,design=d);v=dict(status='VERIFIED',run_id=d.RUN,models=4,newly_trained_models=3,baseline_reused_models=1,prediction_count=672000,metric_records=72,summary=summary,paired=paired,records=records,source_qualification=qualification,goal_84=False,goal_requires_three_predefined_seeds=True,first_seed_is_not_goal_completion=True,head_fit_updates=0,truth_last=True,target_feedback_to_training=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',v);return v
