"""Per-row truth-last; combine the immutable first seed with the remaining two."""
import statistics as st
from . import design as d
from .predict import preflight
from experiments.cvs_pa_cross_delay.score import score as shared_score,validate_records,row_completion

def score(row):return shared_score(row,design=d,preflight_fn=preflight)
def checked_records(design):
 records=[]
 for r in design.rows():
  c=design.config(r);p=design.BASE/r['row_id'];v=design.read(p/'scores/metrics.json');count=validate_records(c,v['records'],design=design)
  if design.read(p/'completion.json')!=row_completion(c,count,design=design) or not v['truth_last'] or v['independent_recount']!='VERIFIED':raise ValueError('Immutable default-head row not fully scored')
  records.extend(v['records'])
 return records

def summarize(records,seeds,*,design=None):
 d=design or globals()['d']
 summary=[];index={}
 for arm in d.ARMS:
  overall=[r for r in records if r['arm']==arm and r['dimension']=='overall']
  if len(overall)!=len(seeds) or {r['model_seed'] for r in overall}!=set(seeds):raise ValueError('Complete predefined seed set required')
  values=[r['accuracy'] for r in overall]
  index.update({(arm,r['model_seed']):r['accuracy'] for r in overall})
  floor=lambda dimension:st.mean(min(r['accuracy'] for r in records if r['arm']==arm and r['model_seed']==seed and r['dimension']==dimension) for seed in seeds)
  summary.append(dict(arm=arm,seed_count=len(seeds),accuracy_mean=st.mean(values),accuracy_sd=st.stdev(values) if len(values)>1 else None,seed_accuracy={str(r['model_seed']):r['accuracy'] for r in overall},macro_f1_mean=st.mean(r['macro_f1'] for r in overall),worst_RX_mean=floor('receiver'),worst_day_mean=floor('day')))
 paired=[]
 for arm,control in d.PAIRS:
  differences=[index[arm,seed]-index[control,seed] for seed in seeds]
  paired.append(dict(arm=arm,control=control,seeds=list(seeds),seed_differences=differences,mean_difference=st.mean(differences),sd_difference=st.stdev(differences) if len(seeds)>1 else None,positive_seeds=sum(x>0 for x in differences)))
 return summary,paired

def finish():
 from .feature_check import run
 from experiments.cvs_sigtime_pair50 import design as first
 f=run()
 if f['status']!='VERIFIED' or f['models']!=4 or len(f['paired'])!=2:raise ValueError('Four actual fixedSIG pair source witnesses incomplete')
 d.write(d.BASE/'source_features.json',f);obs={r['row_id']:d.read(d.BASE/r['row_id']/'source/ssl_training_observation.json') for r in d.training_rows()}
 if any(v['status']!='VERIFIED' or v['steps']!=11100 or v['U_packets']!=963900 or v['sampler_packets']!=963900 or v['sampler_epochs']!=17 or v['gradient_batches']!=666 or v['gradient_epochs']!=[33,34,50] or v['head_fit_updates'] or v['U_truth_read'] or not v['loss_returned_unchanged'] for v in obs.values()):raise ValueError('Actual remaining fixedSIG pair training evidence differs')
 records=checked_records(d);summary,paired=summarize(records,d.SEEDS);seeds=tuple(first.SEEDS)+tuple(d.SEEDS)
 if seeds!=(2026092701,2026092702,2026092703):raise ValueError('Only full predefined fixedSIG pair seed set allowed')
 combined_records=checked_records(first)+records;ss,pp=summarize(combined_records,seeds);primary=next(r for r in ss if r['arm']==d.PRIMARY)
 combined=dict(status='VERIFIED',method='FixedSIG_TIME semi-supervised RANDOMvsphysicalPAIR integration reference',run_ids=[first.RUN,d.RUN],model_seeds=list(seeds),models=9,newly_trained_models=6,baseline_reused_models=3,default_decision_count=1512000,metric_records=162,summary=ss,paired=pp,records=combined_records,first_seed_retrained=False,target_feedback_to_training=False,head_fit=False,native_fitted_probes_disabled=True,truth_last=True,goal_84=primary['accuracy_mean']>=.84,goal_requires_three_predefined_seeds=True,integration_not_new_core_algorithm=True,Phase2_claim=False)
 d.write(d.BASE/'combined_three_seed_analysis.json',combined)
 v=dict(status='VERIFIED',run_id=d.RUN,models=6,newly_trained_models=4,baseline_reused_models=2,prediction_count=1008000,metric_records=108,summary=summary,paired=paired,records=records,goal_84=combined['goal_84'],combined_three_seed_ref=str(d.BASE/'combined_three_seed_analysis.json'),head_fit=False,truth_last=True,target_feedback_to_training=False,integration_not_new_core_algorithm=True)
 d.write(d.BASE/'analysis.json',v);return v
