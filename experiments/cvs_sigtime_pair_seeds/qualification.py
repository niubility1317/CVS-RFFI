from . import design as d
from experiments.cvs_sigtime_pair50 import design as first
FIRST_EXECUTION='86c22cb76e5eff4a7b106cca769d198879ab22b4'
def check():
 q=first.read(first.BASE/'source_qualification.json');f=first.read(first.BASE/'source_features.json');done=first.read(first.BASE/'completion.json');spec=first.read(first.BASE/'experiment.json')
 if spec.get('execution_commit')!=FIRST_EXECUTION or [r['config'] for r in spec['rows']]!=[first.config(r) for r in first.rows()]:raise ValueError('Actual first fixedSIG pair source matrix differs')
 if done.get('status')!='EXPERIMENT_COMPLETE' or done.get('models')!=3 or done.get('new_trained_models')!=2 or done.get('prediction_count')!=504000 or not done.get('truth_last'):raise ValueError('First fixedSIG pair closure incomplete')
 if q.get('status')!='PASS' or not q.get('source_only') or q.get('target_metrics_read_for_decision') or q.get('head_fit_updates') or q.get('configuration_changes_for_expansion') or q.get('predefined_remaining_seeds')!=list(d.SEEDS):raise ValueError('FixedSIG pair source qualification did not pass')
 rules=first.config(first.training_rows()[0])['source_falsification'];expected={r['row_id'] for r in first.training_rows()}
 if q.get('fixed_rules')!=rules or set(q.get('final_L_CE',{}))!=expected or any(v>rules['final_L_CE_max'] for v in q['final_L_CE'].values()) or q.get('actual_SIG_PAIR_minus_RANDOM_L720zL2',0)<rules['source_feature_difference_min'] or q.get('paired_U_acceptance',0)<rules['minimum_paired_U_acceptance']:raise ValueError('FixedSIG pair source thresholds differ')
 if f.get('status')!='VERIFIED' or f.get('models')!=2 or len(f.get('paired',[]))!=1 or not f.get('source_only') or f.get('query_read') or f.get('head_fit'):raise ValueError('Actual fixedSIG pair source feature witnesses incomplete')
 obs=q.get('actual_observations',{})
 if set(obs)!=expected or any(v.get('status')!='VERIFIED' or v.get('steps')!=11100 or v.get('U_packets')!=963900 or v.get('sampler_epochs')!=17 or v.get('sampler_packets')!=963900 or v.get('gradient_batches')!=666 or v.get('gradient_epochs')!=[33,34,50] or v.get('head_fit_updates') or v.get('U_truth_read') or not v.get('loss_returned_unchanged') for v in obs.values()):raise ValueError('Actual fixedSIG pair sampler/gradient budget differs')
 return dict(status='PASS',source_only=True,first_run=first.RUN,first_execution=FIRST_EXECUTION,qualification=q,remaining_seeds=list(d.SEEDS),configuration_changed=False,target_metric_read=False,first_seed_retrained=False)
