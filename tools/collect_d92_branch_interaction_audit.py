"""Audit recorded BranchInteraction fit/cost metadata; never read query outputs."""
import argparse
from collections import Counter
import csv
import inspect
from itertools import product, zip_longest
import json
import math
from pathlib import Path
import statistics
import subprocess

from collect_d92_mvkme_audit import check, finite_tree, read_json, json_lines, nonnegative, close, stats
from collect_d92_bnna_audit import scalars, csv_matches


def validate_fit(row, compact, config):
    finite_tree(row); finite_tree(compact)
    classes, old, k = row['classes'], config['old_classes'], row['k']
    c = len(classes); n = c*k
    check(row['config'] == config['algorithm'] and classes == row['registered_classes'], 'Frozen formula/class order mismatch')
    check(type(k) is int and k in config['ks'] and c-len(old) in config['new_counts']
          and row['new_count'] == c-len(old), 'K/registered class count mismatch')
    check(len(set(classes)) == c and all(isinstance(v, str) and v for v in classes)
          and set(old) <= set(classes) and row['old_classes'] == sorted(old), 'Class membership mismatch')
    check(row['selected'] == 'interaction' and row['selection'] == 'fixed_no_selection'
          and row['factorization_count'] == 1 and row['optimizer_steps'] == 0
          and row['fold_count'] == 0 and row['folds'] == [] and row['oof'] is None, 'Fixed single full fit only')
    check(row['heldout_unavailable_reason'] == ('K1_NO_INDEPENDENT_PHYSICAL_HOLDOUT' if k == 1 else 'NO_CV_FIXED_CONFIG'), 'Held-out diagnostic semantics mismatch')
    check(row['support_count'] == n and row['phase1_frozen'] is True, 'Support count/frozen model mismatch')
    for key in ('query_rows_used', 'source_rows_used', 'new_source_payload_bytes', 'new_ground_statistics_bytes',
                'query_rows_used_for_fit', 'source_rows_used_for_fit'):
        check(type(row[key]) is int and row[key] == 0, 'Forbidden fitting input '+key)
    check(row['source_validation'] is None and bool(row['source_validation_reason']), 'Source validation must be unavailable')
    support = row['support_records']; ids = [v['physical_id'] for v in support]
    check(len(ids) == len(set(ids)) == n and ids == sorted(ids)
          and all(isinstance(v, str) and v for v in ids), 'Unique canonical physical support required')
    check(Counter(v['class_id'] for v in support) == Counter({cls:k for cls in classes}), 'Support physical class/K mismatch')
    fit = row['final_fit']
    check(fit['scope'] == 'final' and fit['fold'] is None and fit['training_physical_ids'] == ids
          and fit['train_k'] == k and fit['train_physical_count'] == n
          and fit['all_states_estimated_from_current_support_only'] is True
          and fit['status'] == 'CLOSED_FORM_SOLVED', 'Full current support fit mismatch')
    check(fit['physical_loss_mass'] == n and fit['sample_weight'] == 1. and fit['ridge_coefficient'] == 1.
          and fit['optimizer_steps'] == 0 and fit['learning_rate'] is fit['epoch'] is None, 'Physical sum/ridge/optimizer mismatch')
    check(fit['arm'] == 'interaction' and fit['input_feature_dim'] == 736 and fit['implicit_feature_dim'] == 123616
          and fit['output_dim'] == fit['classification_output_dim'] == c and fit['factorization_calls'] == 1
          and fit['factorization_dim'] == n and fit['solver'] == 'float64_centered_kernel_Cholesky',
          'Single actual kernel decomposition mismatch')
    close(fit['target_norm_squared'], n*(1-1/c), 'centered onehot target mass')
    for key in ('loss_data', 'loss_ridge', 'loss_total', 'gradient_norm', 'normal_equation_residual', 'intercept_gradient_norm'):
        nonnegative(fit[key], key)
    close(fit['loss_total'], fit['loss_data']+fit['loss_ridge'], 'full objective components')
    check(fit['loss_total'] <= .5*fit['target_norm_squared']+1e-8, 'Convex solution worse than feasible zero head')
    check(fit['gradient_coordinate'] == 'dual_coefficients', 'Gradient coordinate mismatch')
    close(fit['gradient_norm'],fit['dual_objective_gradient_norm'],'Dual gradient measurement mismatch')
    check(fit['gradient_norm'] <= max(0.,fit['condition_bound']-1)*fit['normal_equation_residual']+1e-8,
          'Dual gradient/kernel residual relation inconsistent')
    check(fit['gram_min_eigenvalue_lower_bound'] == 1. and 1-1e-9 <= fit['condition_bound'] <= 1+3*n+1e-8
          and 0 <= fit['kernel_diagonal_mean'] <= 3+1e-12, 'Kernel energy/SPD condition bound mismatch')
    expected_bytes = dict(training_feature_bytes=8*n*736, gram_bytes=8*n*n, coefficient_bytes=8*n*c)
    for key, value in expected_bytes.items():
        check(type(fit[key]) is int and fit[key] == value, 'Actual fit array byte mismatch '+key)
    check(row['head_bytes'] == row['persistent_state_bytes'] == 8*(n*(736+c+2)+c)+16
          and row['current_support_feature_bytes'] == 8*n*736 and bool(row['state_byte_scope']), 'Numeric deployed state byte mismatch')
    if fit['degenerate_constant_features']:
        close(fit['loss_ridge'], 0., 'constant feature zero head')
        close(fit['loss_total'], .5*fit['target_norm_squared'], 'constant feature exact uniform fit')
    for key in ('gram_seconds','solve_seconds','fit_seconds'):
        nonnegative(fit[key], key)
    check(sum(fit[key] for key in ('gram_seconds','solve_seconds')) <= fit['fit_seconds']+1e-8,
          'Nested analytical duration mismatch')
    check(fit['fit_seconds'] <= row['fit_seconds']+1e-8, 'Core total duration mismatch')
    for key in ('split_id','receiver','scenario','k','new_count','support_seed','selected','selection','fold_count','oof',
                'factorization_count','optimizer_steps','fit_seconds','head_bytes','persistent_state_bytes',
                'fit_call_seconds','query_score_seconds','prediction_write_seconds'):
        check(compact[key] == row[key], 'Compact/trace mismatch '+key)
    check(compact['classes'] == c and compact['final_fit'] == scalars(fit), 'Compact class/final fit mismatch')
    check(compact['gradient_norm'] == fit['gradient_norm'] and compact['source_validation'] is None
          and bool(compact['source_validation_reason']) and compact['learning_rate'] is compact['epoch'] is None
          and bool(compact['unavailable_reason']) and compact['query_rows_used_for_fit'] == compact['source_rows_used_for_fit'] == 0,
          'Compact optimizer/access mismatch')
    for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds'):
        nonnegative(compact[key], key)
    check(row['fit_seconds'] <= row['fit_call_seconds']+1e-8
          and sum(compact[key] for key in ('fit_call_seconds','query_score_seconds','prediction_write_seconds')) <= compact['total_seconds']+1e-8,
          'Fit/scoring/writing clock mismatch')


def validate_resources(row, startup, marker, config):
    root = Path(row['reuse_branch_features_root'])
    feature = read_json(root/'features_complete.json')
    extraction = read_json(root/'startup.json'); provenance = read_json(root/'checkpoint_provenance.json')
    sha, seed = row['expected_checkpoint_sha256'], row['seeds']['model']
    for record in (feature, extraction, provenance, startup):
        check(record['checkpoint_sha256'] == sha and record['model_seed'] == seed, 'Model source/cache binding mismatch')
    check(feature['status'] == 'BRANCH_FEATURES_COMPLETE' and feature['schema'] == extraction['schema'] == 'd92_branch_received_features_v1'
          and feature['capsule_id'] == extraction['capsule_id'] == startup['capsule_id'] == config['capsule_id']
          and feature['classes'] == provenance['classes'] == config['old_classes'], 'Feature/capsule/class binding mismatch')
    check(extraction['config'] == {'algorithm':config['producer_algorithm']}
          and startup['config'] == {'algorithm':config['algorithm']}
          and extraction['provenance'] == startup['provenance'] == provenance, 'Exact provenance/config mismatch')
    check(provenance['verdict'] == 'MATCHED_SOURCE_ONLY_SCRATCH' and provenance['source_role_comparison'] == 'EXACT_MATCH'
          and provenance['checkpoint_epoch'] == 200 and provenance['checkpoint_inheritance'] == []
          and provenance['target_access_before_freeze'] is False, 'Checkpoint inheritance mismatch')
    contract = config['feature_contract']
    check(feature['feature_contract'] == extraction['feature_contract'] == startup['feature_contract'] == contract
          and startup['feature_cache_schema'] == 'd92_branch_received_features_v1'
          and Path(startup['branch_features']) == root, 'Frozen feature formula/path mismatch')
    for record in (feature, extraction, startup):
        check(all(record.get(key) is False for key in ('truth_read','source_data_access','adapted_state_inherited','encoder_updated')), 'Feature/fit forbidden access/state')
    check(feature['query_used_for_fitting'] is extraction['query_fit_access'] is startup['query_fit_access'] is False
          and startup['ground_summary_access'] is startup['cross_row_adapted_state_reuse'] is False, 'Query/source adaptation forbidden')
    check(feature['native_eval'] is extraction['native_eval'] is feature['native_parameters_unchanged'] is feature['native_buffers_unchanged'] is True,
          'Native model was not immutable eval')
    check(feature['dtype'] == 'float32' and feature['native_batch_size'] == extraction['native_batch_size'] == 1
          and feature['view_count_per_observation'] == 1, 'Single-observation original view mismatch')
    count = feature['count']; nonnegative(count, 'received count', integer=True, positive=True)
    check(feature['feature_array_bytes'] == 2944*count
          and feature['shapes'] == {key:[count,96 if key=='fft' else 160] for key in ('z_id','t_emb','f_emb','pa_local','fft')}, 'Feature dimensions/bytes mismatch')
    check(feature['native_physical_forward_count'] == feature['native_batch_calls'] == feature['identity_reference_checks'] == count
          and feature['identity_check_additional_encoder_forwards'] == 0, 'Native singleton forward count mismatch')
    smoke = feature['synthetic_smoke']
    check(feature['smoke_forward_count'] == feature['smoke_batch_calls'] == 1
          and feature['native_total_physical_forward_count'] == feature['native_total_batch_calls'] == count+1
          and feature['support_cache_reuse_count'] == 0, 'Extra synthetic forward accounting mismatch')
    check(smoke['status'] == 'PASS' and smoke['input'] == 'synthetic_PCG64_seed0'
          and smoke['physical_forward_count'] == smoke['native_batch_calls'] == 1
          and smoke['feature_shape'] == [1,736] and smoke['query_rows_read'] == 0
          and smoke['frozen_state_unchanged'] is True, 'Synthetic smoke scope mismatch')
    close(smoke['seconds'], feature['timing']['synthetic_smoke_seconds'], 'synthetic smoke timing')
    attrs=feature['active_branches']
    check(attrs==provenance['active_branches'] and attrs['emb_dim']==attrs['t_dim']==attrs['f_dim']==160
          and attrs['active_defects']==['pa'] and attrs['id_feature_key']=='feat_joint'
          and attrs['enable_dac'] is False and attrs['enable_pa'] is attrs['use_time_path'] is attrs['use_freq_path'] is True,
          'Actual active branch mismatch')
    check((root/'received_branch_features.npz').stat().st_size == feature['feature_file_bytes'], 'Feature container byte mismatch')
    payload = marker['payload_audit']
    check(startup['payload_audit'] == payload and payload['checkpoint_loaded'] is False
          and payload['feature_cache_reused'] is True and payload['model_already_deployed'] is None
          and payload['model_incremental_transfer_bytes'] == 0
          and bool(payload['model_deployment_unknown_reason']), 'Deployment scope mismatch')
    for key in ('new_source_payload_bytes','new_ground_statistics_bytes'):
        check(feature[key] == extraction[key] == startup[key] == marker[key] == payload[key] == 0, 'Source payload mismatch')
    check(feature['model_file_bytes'] == provenance['model_file_bytes'] == extraction['model_file_bytes'] == payload['model_file_bytes'], 'Existing model byte mismatch')
    nonnegative(feature['model_file_bytes'], 'model file bytes', integer=True, positive=True)
    for key in ('feature_array_bytes','feature_file_bytes'):
        check(payload[key] == feature[key], 'Payload measurement mismatch '+key)
    for key in ('native_physical_forward_count','native_batch_calls','smoke_forward_count','smoke_batch_calls',
                'native_total_physical_forward_count','native_total_batch_calls','support_cache_reuse_count','feature_extraction_this_run_seconds'):
        check(payload[key] == 0, 'Unexpected new extraction work '+key)
    check(payload['existing_feature_extraction_timing'] == feature['timing']
          and payload['existing_native_physical_forward_count'] == count, 'Existing cache cost binding mismatch')
    nonnegative(payload['cache_load_seconds'],'cache load seconds')
    for key, value in feature['timing'].items(): nonnegative(value, key)
    check(startup['selection'] == 'fixed_no_selection' and startup['prediction_tie_policy'] == 'physical_class_id_ascending'
          and startup['learning_rate'] is startup['epoch'] is startup['gradient_norm'] is startup['source_validation'] is None,
          'Startup algorithm/diagnostic contract mismatch')
    return dict(checkpoint_sha256=sha, existing_model_file_bytes=feature['model_file_bytes'],
        feature_array_bytes=feature['feature_array_bytes'], feature_file_bytes=feature['feature_file_bytes'],
        registry_array_bytes=feature['registry_array_bytes'], count=count,
        native_physical_forward_count=0, native_batch_calls=0, smoke_forward_count=0,smoke_batch_calls=0,
        native_total_physical_forward_count=0,native_total_batch_calls=0,support_cache_reuse_count=0,
        existing_extraction_timing=feature['timing'],existing_received_native_forward_count=count,
        extraction_timing=dict(total_seconds=0.),cache_load_seconds=payload['cache_load_seconds'],
        model_incremental_transfer_bytes=0,checkpoint_loaded=False,feature_cache_reused=True,
        new_source_payload_bytes=0, new_ground_statistics_bytes=0,
        model_byte_scope='Full existing training checkpoint package; not a measured minimum inference transfer')


def audit_run(config):
    root = Path(config['root']); complete = read_json(root/'complete.json')
    check(complete['status'] == 'SCORED' and complete['selection_feedback_forbidden'] is True, 'Collection requires final SCORED metadata')
    check(complete['model_rows'] == len(config['rows']), 'Completion model row count mismatch')
    startup = read_json(root/'startup.json'); workflow = read_json(root/'workflow_state.json')
    check(startup['spec'] == config['spec'] and startup['commit'] == complete['commit'] == workflow['commit']
          and workflow['status'] == 'SCORED' and workflow['updated'] >= startup['timestamp'], 'Run binding/wall clock mismatch')
    state = read_json(root/'state.json')
    check(set(state) == {row['row_id'] for row in config['rows']} and all(v['status']=='PREDICTIONS_COMPLETE' for v in state.values()), 'Incomplete model lanes')
    expected = set(product(config['receivers'],config['scenarios'],config['ks'],config['new_counts'],config['support_seeds']))
    check(len(expected)==config['splits'], 'Matrix definition mismatch')
    models=[]
    for row in config['rows']:
        folder=Path(row['output_root'])/'branch_interaction';marker=read_json(folder/'predictions_complete.json')
        check(marker['status']=='PREDICTIONS_COMPLETE' and marker['split_count']==marker['predictions']==config['splits']
              and marker['capsule_id']==config['capsule_id'] and marker['checkpoint_sha256']==row['expected_checkpoint_sha256']
              and marker['truth_read'] is marker['source_data_access'] is marker['query_used_for_fitting'] is False
              and marker['factorization_count']==config['splits'] and marker['optimizer_steps']==0
              and marker['mode']=='d92_branch_interaction_registration' and marker['algorithm']==config['algorithm'], 'Prediction completion metadata mismatch')
        resource=validate_resources(row,read_json(folder/'startup.json'),marker,config)
        # CSV is checked against its own streamed compact/stage JSONL; full trace
        # is streamed separately and only scalar summaries survive an episode.
        csv_matches(folder/'compact.csv',json_lines(folder/'compact.jsonl'))
        csv_matches(folder/'fit_stages.csv',json_lines(folder/'fit_stages.jsonl'))
        seen=set();cells=set();compact_records=[];fit_records=[]
        streams=(json_lines(folder/'fit_trace.jsonl'),json_lines(folder/'compact.jsonl'),json_lines(folder/'fit_stages.jsonl'))
        for trace,small,stage in zip_longest(*streams):
            check(trace is not None and small is not None and stage is not None, 'Unequal metadata stream lengths')
            sid=trace['split_id'];check(sid not in seen,'Duplicate split');seen.add(sid)
            cell=tuple(trace[key] for key in ('receiver','scenario','k','new_count','support_seed'))
            check(cell in expected and cell not in cells,'Unexpected/duplicate matrix cell');cells.add(cell)
            validate_fit(trace,small,config)
            check(stage==dict(split_id=sid,**scalars(trace['final_fit'])),'Full trace/analytical stage mismatch')
            check(small['completed']==len(seen) and small['total']==config['splits'],'Compact progress mismatch')
            compact_records.append({key:small[key] for key in ('k','fit_seconds','fit_call_seconds','query_score_seconds',
                'prediction_write_seconds','total_seconds','peak_process_rss_bytes','head_bytes','persistent_state_bytes')})
            fit_records.append(scalars(trace['final_fit']))
        check(cells==expected and len(seen)==config['splits'],'Incomplete full matrix')
        measurement_keys=('loss_data','loss_ridge','loss_total','gradient_norm','normal_equation_residual','intercept_gradient_norm',
            'target_norm_squared','condition_bound','kernel_diagonal_mean','gram_seconds','solve_seconds','fit_seconds',
            'factorization_dim','training_feature_bytes','gram_bytes','coefficient_bytes','dual_objective_gradient_norm')
        models.append(dict(row_id=row['row_id'],model_seed=row['seeds']['model'],fits=len(seen),
            k1_fits=sum(v['k']==1 for v in compact_records),full_fit_calls=len(fit_records),factorizations=len(fit_records),
            solver_counts=dict(Counter(v['solver'] for v in fit_records)),status_counts=dict(Counter(v['status'] for v in fit_records)),
            resources=resource,fit_measurements={key:stats([v[key] for v in fit_records]) for key in measurement_keys},
            timing={key:stats([v[key] for v in compact_records]) for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds')},
            state_bytes={key:stats([v[key] for v in compact_records]) for key in ('head_bytes','persistent_state_bytes')},
            peak_process_rss_bytes=marker.get('peak_process_rss_bytes'),peak_process_rss_reason=marker.get('peak_process_rss_reason'),
            compact_peak_rss_bytes=stats([v['peak_process_rss_bytes'] for v in compact_records]),
            trace_file_bytes=(folder/'fit_trace.jsonl').stat().st_size,stage_file_bytes=(folder/'fit_stages.jsonl').stat().st_size))
    return dict(status='VERIFIED',run_id=config['spec']['run_id'],models=models,commit=complete['commit'],
        total_fits=sum(m['fits'] for m in models),k1_fits=sum(m['k1_fits'] for m in models),
        run_wall_seconds=workflow['updated']-startup['timestamp'],run_started=startup['timestamp'],run_finished=workflow['updated'],
        wall_scope='Complete cohort orchestration including cache binding, prediction, baseline verification and independent scorer; scorer values not read')


def audit_runs(configs):
    check(len(configs)==2 and sorted(len(c['receivers']) for c in configs)==[1,3], 'Joint audit requires both complete cohorts')
    check(set(configs[0]['receivers']).isdisjoint(configs[1]['receivers']), 'Receiver cohorts overlap')
    runs=[audit_run(c) for c in configs];models=[m for run in runs for m in run['models']]
    check(len(models)==8 and sum(m['fits'] for m in models)==4800 and sum(m['k1_fits'] for m in models)==1200,
          'Full eight-row/4800-fit/1200-K1 coverage required')
    fit_calls=sum(m['full_fit_calls'] for m in models);factorizations=sum(m['factorizations'] for m in models)
    check(fit_calls==factorizations==4800,'Exactly one recorded full fit and factorization per episode required')
    by_sha={}
    for model in models:
        r=model['resources'];sha=r['checkpoint_sha256']
        check(sha not in by_sha or by_sha[sha]==r['existing_model_file_bytes'],'Same SHA model byte mismatch')
        by_sha[sha]=r['existing_model_file_bytes']
    check(len(by_sha)==4,'Exactly four unique frozen model packages required')
    measured={key:sum(m['timing'][key]['total'] for m in models) for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds')}
    rss=[m['peak_process_rss_bytes'] for m in models if m['peak_process_rss_bytes'] is not None]
    return dict(status='VERIFIED',method='D92-BranchInteraction-v1',runs=runs,total_fits=4800,k1_fits=1200,
        full_fit_calls=fit_calls,factorizations=factorizations,fold_fit_calls=0,optimizer_steps=0,oof=None,
        feature_dim=736,implicit_feature_dim=123616,numeric_head_bytes_formula='8*(N*(736+C+2)+C)+16; N=C*K; current support retained',
        query_rows_used_for_fit=0,source_rows_used_for_fit=0,new_source_payload_bytes=0,new_ground_statistics_bytes=0,
        selection='fixed_no_selection',raw_traces_preserved=True,parameter_feedback_forbidden=True,
        summed_process_work_seconds=measured,
        numeric_state_bytes={key:dict(min=min(m['state_bytes'][key]['min'] for m in models),
            max=max(m['state_bytes'][key]['max'] for m in models),
            mean=sum(m['state_bytes'][key]['total'] for m in models)/fit_calls,
            summed_episode_bytes=sum(m['state_bytes'][key]['total'] for m in models))
            for key in ('head_bytes','persistent_state_bytes')},
        summed_extraction_seconds=sum(m['resources']['extraction_timing']['total_seconds'] for m in models),
        summed_cache_load_seconds=sum(m['resources']['cache_load_seconds'] for m in models),
        summed_extraction_stage_seconds={key:sum(m['resources']['extraction_timing'][key] for m in models)
            for key in models[0]['resources']['extraction_timing']},
        joint_wall_span_seconds=max(r['run_finished'] for r in runs)-min(r['run_started'] for r in runs),
        summed_cohort_wall_seconds=sum(r['run_wall_seconds'] for r in runs),
        max_observed_process_rss_bytes=max(rss) if rss else None,
        existing_unique_model_file_bytes=sum(by_sha.values()),unique_model_count=4,
        model_incremental_transfer_bytes=0,checkpoint_loaded=False,feature_cache_reused=True,
        received_feature_array_bytes=sum(m['resources']['feature_array_bytes'] for m in models),
        received_feature_file_bytes=sum(m['resources']['feature_file_bytes'] for m in models),
        received_registry_array_bytes=sum(m['resources']['registry_array_bytes'] for m in models),
        received_native_forward_count=sum(m['resources']['native_physical_forward_count'] for m in models),
        synthetic_smoke_forward_count=sum(m['resources']['smoke_forward_count'] for m in models),
        actual_total_native_forward_count=sum(m['resources']['native_total_physical_forward_count'] for m in models),
        actual_total_native_batch_calls=sum(m['resources']['native_total_batch_calls'] for m in models),
        interpretation='Recorded metadata consistency, not numerical reexecution. Process work sums differ from wall span; max process RSS is not aggregate simultaneous memory. Summed episode state bytes are cumulative per-fit snapshots, not concurrent memory. Existing model and feature bytes are storage, not new transfer; extraction/forward costs are zero for this cache-reuse run. No predictions, scores, truth, checkpoints or feature values read.')


def audit_config(spec):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
    from cvsrffi.d92_branch_interaction import FROZEN_CONFIG
    from cvsrffi.d92_branch_ridge import FROZEN_CONFIG as PRODUCER_CONFIG
    from export_d92_branch_features import FEATURE_CONTRACT
    conf,data=spec['confirmation'],spec['data']
    check(conf['candidate']==FROZEN_CONFIG and conf['candidate_method']=='D92-BranchInteraction-v1'
          and conf['candidate_folder']=='branch_interaction','Frozen BranchInteraction configuration only')
    check(len(spec['rows'])==4 and sorted(r['seeds']['model'] for r in spec['rows'])==list(range(2026092701,2026092705)), 'Four exact frozen model seeds required')
    check(data['k']==[1,5,10,20] and data['new_class_counts']==[0,2,5,10,20]
          and len(data['scenarios'])==3 and len(data['support_seeds'])==5 and len(data['target_receivers']) in (1,3), 'Full registered cohort matrix required')
    count=math.prod(len(data[key]) for key in ('target_receivers','scenarios','k','new_class_counts','support_seeds'))
    check(count==conf['expected_split_count']==conf['splits_per_model'],'Spec split count mismatch')
    return dict(root=spec['execution']['remote_run_root'],spec=spec,rows=spec['rows'],splits=count,
        receivers=data['target_receivers'],scenarios=data['scenarios'],ks=data['k'],new_counts=data['new_class_counts'],
        support_seeds=data['support_seeds'],old_classes=conf['old_classes'],algorithm=conf['candidate'],
        feature_contract=FEATURE_CONTRACT,producer_algorithm=PRODUCER_CONFIG,capsule_id=conf['reuse_validated_capsule_id'])


def remote_script(configs):
    functions=(check,finite_tree,read_json,json_lines,nonnegative,close,stats,scalars,csv_matches,
               validate_fit,validate_resources,audit_run,audit_runs)
    return ('import json,math,statistics,csv\nfrom pathlib import Path\nfrom collections import Counter\nfrom itertools import product,zip_longest\n'
            +'\n'.join(inspect.getsource(f) for f in functions)
            +'\nprint(json.dumps(audit_runs('+repr(configs)+'),allow_nan=False))\n')


def main():
    from read_d92_run import FLAGS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    script=remote_script([audit_config(read_json(path)) for path in args.spec]);compile(script,'branch_interaction_metadata_audit','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    value=json.loads(result.stdout);finite_tree(value);check(value.get('status')=='VERIFIED','Audit incomplete')
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='runs'},allow_nan=False))


if __name__=='__main__':main()
