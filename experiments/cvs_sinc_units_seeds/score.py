from . import design as d
from .predict import preflight
from .evidence import recount
from experiments.cvs_pa_cross_delay.score import score as shared_score
from experiments.cvs_sigtime_pair_seeds.score import checked_records,summarize

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():
 from .feature_check import run
 q=d.require_source();features=run()
 if features['status']!='VERIFIED' or features['models']!=4 or len(features['paired'])!=2:raise ValueError('Four actual remaining Sinc feature witnesses incomplete')
 d.write(d.BASE/'source_features.json',features);obs=[recount(r) for r in d.training_rows()]
 d.write(d.BASE/'source_observations.json',dict(status='VERIFIED',first_source_qualification=q,observations=obs,new_selection_gate=False))
 records=checked_records(d);summary,paired=summarize(records,d.SEEDS,design=d);seeds=tuple(d.first.SEEDS)+tuple(d.SEEDS)
 if seeds!=(2026092701,2026092702,2026092703):raise ValueError('Only complete predefined seed set allowed')
 all_records=checked_records(d.first)+records;ss,pp=summarize(all_records,seeds,design=d);primary=next(r for r in ss if r['arm']==d.PRIMARY)
 combined=dict(status='VERIFIED',run_ids=[d.first.RUN,d.RUN],model_seeds=list(seeds),models=9,newly_trained_models=6,baseline_reused_models=3,default_decision_count=1512000,metric_records=162,summary=ss,paired=pp,records=all_records,first_seed_retrained=False,target_feedback_to_training=False,head_fit=False,truth_last=True,goal_84=primary['accuracy_mean']>=.84,goal_requires_three_predefined_seeds=True,Phase2_claim=False)
 d.write(d.BASE/'combined_three_seed_analysis.json',combined)
 v=dict(status='VERIFIED',run_id=d.RUN,models=6,newly_trained_models=4,baseline_reused_models=2,prediction_count=1008000,metric_records=108,summary=summary,paired=paired,records=records,goal_84=combined['goal_84'],combined_three_seed_ref=str(d.BASE/'combined_three_seed_analysis.json'),head_fit=False,truth_last=True,target_feedback_to_training=False)
 d.write(d.BASE/'analysis.json',v);return v
