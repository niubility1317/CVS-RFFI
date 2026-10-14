from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score
from experiments.cvs_sigtime_pair_seeds.score import checked_records,summarize
def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():
 from .feature_check import run
 f=run()
 if f['status']!='VERIFIED' or f['models']!=4 or len(f['paired'])!=2:raise ValueError('Four remaining fixedSIG feature witnesses incomplete')
 d.write(d.BASE/'source_features.json',f)
 obs={r['row_id']:d.read(d.BASE/r['row_id']/'source/feature_mean_teacher_training_observation.json') for r in d.training_rows()}
 if any(v['status']!='VERIFIED' or v['steps']!=11100 or v['U_feature_packets']!=963900 or v['U_feature_steps']!=3774 or v['gradient_batches']!=666 or v['gradient_epochs']!=[33,34,50] or v['extra_model_forwards'] or v['head_fit_updates'] or v['U_truth_read'] or v['auxiliary_classW_dependency'] or not v['hard_pseudo_loss_returned_unchanged'] for v in obs.values()):raise ValueError('Remaining actual feature exposure differs')
 records=checked_records(d);ss,pp=summarize(records,d.SEEDS,design=d)
 from experiments.cvs_feature_mean_teacher50 import design as first
 seeds=tuple(first.SEEDS)+tuple(d.SEEDS)
 if seeds!=(2026092701,2026092702,2026092703):raise ValueError('Complete predefined seed set required')
 combined_records=checked_records(first)+records;summary,paired=summarize(combined_records,seeds,design=d);primary=next(r for r in summary if r['arm']==d.PRIMARY)
 combined=dict(status='VERIFIED',method='FixedSIG ownEMA same-packet joint160 consistency vs matchedZERO',run_ids=[first.RUN,d.RUN],model_seeds=list(seeds),models=9,newly_trained_models=6,baseline_reused_models=3,default_decision_count=1512000,metric_records=162,summary=summary,paired=paired,records=combined_records,first_seed_retrained=False,target_feedback_to_training=False,head_fit=False,native_fitted_probes_disabled=True,truth_last=True,goal_84=primary['accuracy_mean']>=.84,goal_requires_three_predefined_seeds=True,Phase2_claim=False)
 d.write(d.BASE/'combined_three_seed_analysis.json',combined)
 v=dict(status='VERIFIED',run_id=d.RUN,models=6,newly_trained_models=4,baseline_reused_models=2,prediction_count=1008000,metric_records=108,summary=ss,paired=pp,records=records,goal_84=combined['goal_84'],combined_three_seed_ref=str(d.BASE/'combined_three_seed_analysis.json'),head_fit=False,truth_last=True,target_feedback_to_training=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',v);return v
