from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score
from experiments.cvs_signature_stats_seeds.score import checked_records,summarize
def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def finish():
 from .feature_check import run
 f=run()
 if f['status']!='VERIFIED' or f['models']!=6 or len(f['paired'])!=4:raise ValueError('Six remaining fixedSIG feature witnesses incomplete')
 d.write(d.BASE/'source_features.json',f)
 obs={r['row_id']:d.read(d.BASE/r['row_id']/'source/backbone_fishr_training_observation.json') for r in d.training_rows()}
 if any(v['status']!='VERIFIED' or v['steps']!=11100 or v['extra_batches']!=650 or v['extra_source_packet_forwards']!=117000 or v['U_packets']!=963900 or v['gradient_batches']!=52 or v['gradient_epochs']!=[1,17,33,50] or v['head_fit_updates'] or v['U_truth_read'] for v in obs.values()):raise ValueError('Actual remaining gradient/exposure budget incomplete')
 records=checked_records(d);ss,pp=summarize(records,d.SEEDS,design=d)
 from experiments.cvs_backbone_fishr50 import design as first
 seeds=tuple(first.SEEDS)+tuple(d.SEEDS)
 if seeds!=(2026092701,2026092702,2026092703):raise ValueError('Only complete predefined model seed set allowed')
 combined_records=checked_records(first)+records;summary,paired=summarize(combined_records,seeds,design=d);primary=next(r for r in summary if r['arm']==d.PRIMARY)
 combined=dict(status='VERIFIED',method='Actual backbone gradient variance on fixedSIG_TIME; RAW/TXdayCENTERED with matchedZERO',run_ids=[first.RUN,d.RUN],model_seeds=list(seeds),models=12,newly_trained_models=9,baseline_reused_models=3,default_decision_count=2016000,metric_records=216,summary=summary,paired=paired,records=combined_records,first_seed_retrained=False,target_feedback_to_training=False,head_fit=False,native_fitted_probes_disabled=True,truth_last=True,goal_84=primary['accuracy_mean']>=.84,goal_requires_three_predefined_seeds=True,Phase2_claim=False)
 d.write(d.BASE/'combined_three_seed_analysis.json',combined)
 v=dict(status='VERIFIED',run_id=d.RUN,models=8,newly_trained_models=6,baseline_reused_models=2,prediction_count=1344000,metric_records=144,summary=ss,paired=pp,records=records,goal_84=combined['goal_84'],goal_scope='Complete first+remaining predefined seeds; no twoseed or highestsecondary substitute',combined_three_seed_ref=str(d.BASE/'combined_three_seed_analysis.json'),head_fit=False,truth_last=True,target_feedback_to_training=False,Phase2_claim=False)
 d.write(d.BASE/'analysis.json',v);return v
