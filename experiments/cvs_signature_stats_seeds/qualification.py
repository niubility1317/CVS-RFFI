"""Read the existing frozen source verdict, without target scores or new fitting."""
from . import design as d
from experiments.cvs_signature_stats50 import design as first
FIRST_EXECUTION='eda57144de34d479288f42d14c037eb7802c1ecd'
def check():
 q=first.read(first.BASE/'source_qualification.json')
 f=first.read(first.BASE/'source_features.json')
 done=first.read(first.BASE/'completion.json')
 spec=first.read(first.BASE/'experiment.json')
 if spec.get('execution_commit')!=FIRST_EXECUTION or [r['config'] for r in spec['rows']]!=[first.config(r) for r in first.rows()]:raise ValueError('Actual fixed first-seed source configuration differs')
 if done.get('status')!='EXPERIMENT_COMPLETE' or done.get('models')!=4 or done.get('new_trained_models')!=3 or done.get('prediction_count')!=672000 or not done.get('truth_last'):raise ValueError('First seed must finish its declared row closure before expansion')
 if q.get('status')!='PASS' or not q.get('source_only') or q.get('target_metrics_read_for_decision') or q.get('head_fit') or q.get('configuration_changes_for_expansion') or q.get('predefined_remaining_seeds')!=list(d.SEEDS):raise ValueError('Frozen source expansion rule did not pass')
 rules=first.config(first.training_rows()[0])['source_falsification']
 expected_rows={r['row_id'] for r in first.training_rows()}
 if q.get('fixed_rules')!=rules or set(q.get('final_L_CE',{}))!=expected_rows or any(v>rules['final_L_CE_max'] for v in q['final_L_CE'].values()) or q.get('actual_source_candidate_minus_X3_L2',0)<rules['source_added_feature_difference_min']:raise ValueError('Exact preregistered source thresholds differ')
 if f.get('status')!='VERIFIED' or f.get('models')!=3 or len(f.get('paired',[]))!=1 or f.get('query_read') or f.get('head_fit') or not f.get('source_only'):raise ValueError('Actual first-seed source witnesses incomplete')
 return dict(status='PASS',source_only=True,first_execution_commit=FIRST_EXECUTION,first_run_id=first.RUN,qualification=q,remaining_seeds=list(d.SEEDS),configuration_changed=False,target_metric_read=False,first_seed_retrained=False)
