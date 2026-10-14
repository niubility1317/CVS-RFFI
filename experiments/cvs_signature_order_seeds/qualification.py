from . import design as d
from experiments.cvs_signature_order50 import design as first
FIRST_EXECUTION='2658815a2fe5ef931f24558a6c6c61146433fd9c'
def check():
 q=first.read(first.BASE/'source_qualification.json');f=first.read(first.BASE/'source_features.json');done=first.read(first.BASE/'completion.json');spec=first.read(first.BASE/'experiment.json')
 if spec.get('execution_commit')!=FIRST_EXECUTION or [r['config'] for r in spec['rows']]!=[first.config(r) for r in first.rows()]:raise ValueError('Actual first signature source matrix differs')
 if done.get('status')!='EXPERIMENT_COMPLETE' or done.get('models')!=4 or done.get('new_trained_models')!=3 or done.get('prediction_count')!=672000 or not done.get('truth_last'):raise ValueError('First fixedrow closure incomplete')
 if q.get('status')!='PASS' or not q.get('source_only') or q.get('target_metrics_read_for_decision') or q.get('head_fit_updates') or q.get('configuration_changes_for_expansion') or q.get('predefined_remaining_seeds')!=list(d.SEEDS) or not q.get('expand_both_candidates_only_after_both_source_PASS'):raise ValueError('Both-candidate source expansion verdict did not pass')
 rules=first.config(first.training_rows()[0])['source_falsification'];expected={r['row_id'] for r in first.training_rows()};candidates=q.get('candidates',[])
 if q.get('fixed_rules')!=rules or set(q.get('final_L_CE',{}))!=expected or any(v>rules['final_L_CE_max'] for v in q['final_L_CE'].values()) or len(candidates)!=2 or {r.get('arm') for r in candidates}!={'SIG_RAW_ORDER','SIG_DYNAMIC_ORDER'} or any(not r.get('passed') or r.get('actual_L720zL2',0)<rules['feature_difference_min'] or not r.get('actual_t1_t2_SIGprojection_gradient_nonzero_17_33_50') for r in candidates):raise ValueError('Fixed signature source thresholds differ')
 if f.get('status')!='VERIFIED' or f.get('models')!=3 or len(f.get('paired',[]))!=2 or not f.get('source_only') or f.get('query_read') or f.get('head_fit'):raise ValueError('Actual first signature feature witnesses incomplete')
 obs=q.get('actual_observations',{})
 if set(obs)!=expected or any(v.get('status')!='VERIFIED' or v.get('extra_batches')!=2750 or v.get('gradient_batches')!=220 or v.get('gradient_epochs')!=[1,17,33,50] or v.get('head_fit_updates') or v.get('U_truth_read') for v in obs.values()):raise ValueError('Actual first signature gradient/exposure incomplete')
 return dict(status='PASS',source_only=True,first_run=first.RUN,first_execution=FIRST_EXECUTION,qualification=q,remaining_seeds=list(d.SEEDS),configuration_changed=False,target_metric_read=False,first_seed_retrained=False)
