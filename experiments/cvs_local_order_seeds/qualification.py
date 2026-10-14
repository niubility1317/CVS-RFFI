from . import design as d
from experiments.cvs_local_order50 import design as first
FIRST_EXECUTION='f34078d1a0775ef6fa66365689481ad5a1932e4c'
def check():
 q=first.read(first.BASE/'source_qualification.json');f=first.read(first.BASE/'source_features.json');done=first.read(first.BASE/'completion.json');spec=first.read(first.BASE/'experiment.json')
 if spec.get('execution_commit')!=FIRST_EXECUTION or [r['config'] for r in spec['rows']]!=[first.config(r) for r in first.rows()]:raise ValueError('Actual fixed first source matrix differs')
 if done.get('status')!='EXPERIMENT_COMPLETE' or done.get('models')!=4 or done.get('new_trained_models')!=3 or done.get('prediction_count')!=672000 or not done.get('truth_last'):raise ValueError('First fixedrow closure incomplete')
 if q.get('status')!='PASS' or not q.get('source_only') or q.get('target_metrics_read_for_decision') or q.get('head_fit_updates') or q.get('configuration_changes_for_expansion') or q.get('predefined_remaining_seeds')!=list(d.SEEDS):raise ValueError('Fixed source expansion verdict did not pass')
 rules=first.config(first.training_rows()[0])['source_falsification'];expected={r['row_id'] for r in first.training_rows()}
 if q.get('fixed_rules')!=rules or set(q.get('final_L_CE',{}))!=expected or any(v>rules['final_L_CE_max'] for v in q['final_L_CE'].values()) or q.get('actual_ORDER_minus_SIG_L720zL2',0)<rules['source_added_feature_difference_min'] or not q.get('actual_ORDER_effective_gradient_t1_t2_nonzero_17_33_50'):raise ValueError('Exact preregistered source thresholds differ')
 if f.get('status')!='VERIFIED' or f.get('models')!=3 or len(f.get('paired',[]))!=1 or not f.get('source_only') or f.get('query_read') or f.get('head_fit'):raise ValueError('Actual first source witnesses incomplete')
 obs=q.get('actual_observations',{})
 if set(obs)!=expected or any(v.get('status')!='VERIFIED' or v.get('extra_batches')!=2750 or v.get('gradient_batches')!=220 or v.get('gradient_epochs')!=[1,17,33,50] or v.get('head_fit_updates') or v.get('U_truth_read') for v in obs.values()):raise ValueError('Actual first source gradient/exposure incomplete')
 return dict(status='PASS',source_only=True,first_run=first.RUN,first_execution=FIRST_EXECUTION,qualification=q,remaining_seeds=list(d.SEEDS),configuration_changed=False,target_metric_read=False,first_seed_retrained=False)
