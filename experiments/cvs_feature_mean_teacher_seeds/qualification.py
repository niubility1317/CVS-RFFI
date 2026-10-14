from . import design as d
from experiments.cvs_feature_mean_teacher50 import design as first
FIRST_EXECUTION='6b23a5b033ee063dea97d6783d5634e220a1d93d'
def check():
 q=first.read(first.BASE/'source_qualification.json');f=first.read(first.BASE/'source_features.json');done=first.read(first.BASE/'completion.json');spec=first.read(first.BASE/'experiment.json')
 if spec.get('execution_commit')!=FIRST_EXECUTION or [r['config'] for r in spec['rows']]!=[first.config(r) for r in first.rows()]:raise ValueError('Actual first source matrix differs')
 if done.get('status')!='EXPERIMENT_COMPLETE' or done.get('models')!=3 or done.get('new_trained_models')!=2 or done.get('prediction_count')!=504000 or not done.get('truth_last'):raise ValueError('First fixed rows incomplete')
 if q.get('status')!='PASS' or not q.get('source_only') or q.get('target_metrics_read_for_decision') or q.get('head_fit_updates') or q.get('configuration_changes_for_expansion') or q.get('predefined_remaining_seeds')!=list(d.SEEDS):raise ValueError('First feature consistency source qualification did not pass')
 rules=first.config(first.training_rows()[0])['source_falsification'];expected={'SIG_TIME','FEATURE_MT'};ratio=q.get('feature_discrepancy_ratio')
 if q.get('fixed_rules')!=rules or set(q.get('final_L_CE',{}))!=expected or any(v>rules['final_L_CE_max'] for v in q['final_L_CE'].values()) or q.get('actual_candidate_minus_control_L720zL2',0)<rules['source_feature_difference_min'] or ratio is None or not 0<=ratio<=rules['last5_U_feature_discrepancy_ratio_max'] or not q.get('actual_t1_t2_SIG_gradient_signal',{}).get('FEATURE_MT'):raise ValueError('Frozen source thresholds differ')
 if f.get('status')!='VERIFIED' or f.get('models')!=2 or len(f.get('paired',[]))!=1 or not f.get('source_only') or f.get('query_read') or f.get('head_fit'):raise ValueError('First actual source feature witnesses incomplete')
 obs=q.get('actual_observations',{})
 if set(obs)!=expected or any(v.get('status')!='VERIFIED' or v.get('steps')!=11100 or v.get('U_feature_packets')!=963900 or v.get('U_feature_steps')!=3774 or v.get('gradient_batches')!=666 or v.get('gradient_epochs')!=[33,34,50] or v.get('extra_model_forwards') or v.get('head_fit_updates') or v.get('U_truth_read') or not v.get('hard_pseudo_loss_returned_unchanged') or v.get('auxiliary_classW_dependency') for v in obs.values()):raise ValueError('First feature/native exposure incomplete')
 return dict(status='PASS',source_only=True,first_run=first.RUN,first_execution=FIRST_EXECUTION,qualification=q,remaining_seeds=list(d.SEEDS),configuration_changed=False,target_metric_read=False,first_seed_retrained=False)
