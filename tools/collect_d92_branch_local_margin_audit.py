"""Audit every recorded BranchLocalMargin fit/sweep; never read query outputs."""
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
from collect_d92_bnna_audit import csv_matches


def scalars(value):
    return {key: scalars(item) if isinstance(item, dict) else item for key, item in value.items()
            if isinstance(item, dict) or item is None or isinstance(item, (str, int, float, bool))}


def kernel_diagnostics(fit):
    result = {}
    for name in ('nearest_other_class_squared_distance', 'training_offdiagonal_radial',
                 'training_radial_row_sum', 'centered_training_diagonal'):
        value = fit[name]
        for key in ('count', 'min', 'median', 'max', 'mean'):
            result[name + '.' + key] = value[key] if value is not None else None
        for i, quantile in enumerate((0, 25, 50, 75, 100)):
            result[name + '.q' + str(quantile)] = (value['quantiles'][i]
                if value is not None and value['quantiles'] is not None else None)
    return result


def validate_certificate(value, n, c, final=False):
    """Check scalar certificate identities, not an independent numerical refit."""
    finite_tree(value)
    tolerance = 128*math.ulp(1.)*max(n, c)
    eta = math.sqrt(math.ulp(1.))
    check(value['optimization_tolerance'] == eta
          and value['objective_rounding_tolerance'] == tolerance, 'Frozen certificate tolerance mismatch')
    p, d, gap = (value[k] for k in ('primal_objective','dual_objective','duality_gap'))
    scale = max(1., abs(p), abs(d))
    close(value['loss_total'], p, 'Primal/loss mismatch')
    close(p, value['loss_data']+value['loss_ridge'], 'Primal components mismatch')
    close(gap, p-d, 'Primal dual gap mismatch')
    close(value['duality_gap_scale'], scale, 'Gap scale mismatch')
    close(value['relative_duality_gap'], max(0., gap)/scale, 'Relative gap mismatch')
    check(gap >= -tolerance*scale and value['loss_ridge'] >= -tolerance*max(1., abs(value['loss_ridge'])),
          'Certificate exceeds objective rounding allowance')
    check(value['kkt_scale'] >= 1., 'KKT normalization mismatch')
    close(value['relative_kkt_residual'], value['kkt_residual']/value['kkt_scale'], 'Relative projected KKT mismatch')
    close(value['squared_slack_sum'], 2*value['loss_data'], 'Squared slack objective mismatch')
    for key in ('loss_data','gradient_norm','kkt_residual','relative_kkt_residual',
                'relative_duality_gap','slack_min','slack_mean','slack_max','squared_slack_sum'):
        nonnegative(value[key], key)
    check(value['slack_min'] <= value['slack_mean'] <= value['slack_max']
          and 0 <= value['training_accuracy'] <= 1, 'Slack/training scalar range mismatch')
    for key, maximum in (('active_constraint_count', n*(c-1)), ('active_physical_count', n)):
        check(type(value[key]) is int and 0 <= value[key] <= maximum, 'Active dual count mismatch '+key)
    check(value['certified'] is (value['relative_duality_gap'] <= eta and value['relative_kkt_residual'] <= eta),
          'Certificate flag disagrees with projected KKT/gap')
    if final:
        check(value['certified'] is True and p <= (n/2. if c > 1 else 0.)+eta*scale,
              'Final fit lacks certificate or exceeds feasible zero objective')


def validate_optimization(fit, n, c, config):
    validate_certificate(fit, n, c, final=True)
    trace = fit['optimization_trace']
    iterative = c > 1 and fit['interaction_centered_trace'] > 0
    count = fit['sweeps']
    check(type(count) is int and len(trace) == count and 0 <= count <= config['algorithm']['max_sweeps'],
          'Complete optimization trace/sweep count mismatch')
    check(type(fit['optimizer_steps']) is int and type(fit['row_block_solves']) is int
          and fit['optimizer_steps'] == fit['row_block_solves'] == count*n and fit['epoch'] == count,
          'Actual physical row-block step count mismatch')
    check(fit['normal_equation_residual'] is fit['effective_degrees_of_freedom'] is None
          and bool(fit['normal_equation_residual_reason']) and bool(fit['effective_degrees_of_freedom_reason']),
          'Margin certificate must not claim ridge diagnostics')
    previous_dual, previous_seconds = 0., 0.
    certificate_keys = ('loss_data','loss_ridge','loss_total','primal_objective','dual_objective','duality_gap',
        'relative_duality_gap','duality_gap_scale','kkt_residual','relative_kkt_residual','kkt_scale',
        'gradient_norm','gradient_coordinate','active_constraint_count','active_physical_count',
        'slack_min','slack_mean','slack_max','squared_slack_sum','training_accuracy',
        'optimization_tolerance','objective_rounding_tolerance','certified')
    for index, event in enumerate(trace, 1):
        validate_certificate(event, n, c)
        check(event['event'] == 'sweep' and event['arm'] == 'local_margin'
              and event['sweep'] == event['epoch'] == index
              and event['optimizer_steps'] == event['row_block_solves'] == index*n
              and event['optimizer'] == config['algorithm']['solver'] and event['learning_rate'] is None
              and event['gradient_coordinate'] == 'negative_dual_beta', 'Sweep optimizer metadata mismatch')
        for key in ('scope','parent_k','train_k','fold','trial','bandwidth_tau','trace_scale',
                    'interaction_centered_trace','radial_centered_trace','train_physical_count'):
            check(event[key] == fit[key], 'Sweep frozen geometry/context mismatch '+key)
        check(event['class_count'] == c and event['max_sweeps'] == config['algorithm']['max_sweeps'],
              'Sweep dimensions/cap mismatch')
        close(event['negative_dual_change'], previous_dual-event['dual_objective'], 'Sweep dual change mismatch')
        check(event['negative_dual_change'] <= event['objective_rounding_tolerance']*max(1.,abs(previous_dual),abs(event['dual_objective'])),
              'Sweep dual objective increase exceeds rounding')
        nonnegative(event['sweep_seconds'], 'sweep seconds')
        check(event['solve_seconds'] >= previous_seconds
              and event['sweep_seconds'] <= event['solve_seconds']-previous_seconds+1e-8,
              'Sweep elapsed clock mismatch')
        check(event['certified'] is (index == count), 'Optimizer continued after certificate or stopped early')
        previous_dual, previous_seconds = event['dual_objective'], event['solve_seconds']
    if iterative:
        check(count > 0 and fit['status'] == 'CERTIFIED_MARGIN_SOLVED'
              and fit['solver'] == config['algorithm']['solver'] and fit['gradient_coordinate'] == 'negative_dual_beta',
              'Iterative margin solve mismatch')
        check(all(fit[key] == trace[-1][key] for key in certificate_keys), 'Final/sweep certificate mismatch')
        check(previous_seconds <= fit['solve_seconds']+1e-8, 'Final solver clock mismatch')
    else:
        check(count == 0 and fit['status'] == 'EXACT_ZERO_CLASSIFIER'
              and fit['solver'] == 'NO_OPTIMIZATION_ANALYTIC_ZERO'
              and fit['gradient_coordinate'] == 'analytic_dual_certificate' and fit['solve_seconds'] == 0,
              'Analytic zero-kernel path mismatch')
        expected = 'beta_ic=1/(C-1),s_i=1,W=0; stored alpha=0 is equivalent RKHS function' if c > 1 else 'No rival constraints; W=0,P=D=0'
        check(fit['analytic_certificate'] == expected, 'Analytic dual certificate missing')
        loss = n/2. if c > 1 else 0.
        for key in ('loss_data','loss_total','primal_objective','dual_objective'):
            close(fit[key], loss, 'Analytic zero-kernel objective')
        for key in ('loss_ridge','duality_gap','relative_duality_gap','kkt_residual','relative_kkt_residual','gradient_norm'):
            close(fit[key], 0., 'Analytic zero-kernel residual')
        check(fit['active_constraint_count'] == n*(c-1) and fit['active_physical_count'] == (n if c > 1 else 0),
              'Analytic beta certificate active counts mismatch')


def validate_fit(row, compact, config):
    finite_tree(row); finite_tree(compact)
    classes, old, k = row['classes'], config['old_classes'], row['k']
    c = len(classes); n = c*k
    check(row['config'] == config['algorithm'] and classes == row['registered_classes'], 'Frozen formula/class order mismatch')
    check(type(k) is int and k in config['ks'] and c-len(old) in config['new_counts']
          and row['new_count'] == c-len(old), 'K/registered class count mismatch')
    check(len(set(classes)) == c and all(isinstance(v, str) and v for v in classes)
          and set(old) <= set(classes) and row['old_classes'] == sorted(old), 'Class membership mismatch')
    check(row['selected'] == 'local_margin' and row['selection'] == 'fixed_no_selection'
          and row['paired'] is row['oneshot_proxy'] is None
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
          and fit['all_states_estimated_from_current_support_only'] is True,
          'Full current support fit mismatch')
    check(fit['physical_loss_mass'] == n and fit['physical_loss_weight'] == fit['sample_weight'] == fit['ridge_coefficient'] == fit['margin'] == 1.
          and fit['intercept'] is False and fit['learning_rate'] is None, 'Physical sum/margin/optimizer mismatch')
    check(row['optimizer_steps'] == fit['optimizer_steps'] and row['sweep_count'] == fit['sweeps'], 'Top-level optimizer count mismatch')
    s0, sr, gamma, tau = [fit[key] for key in
        ('interaction_centered_trace', 'radial_centered_trace', 'trace_scale', 'bandwidth_tau')]
    nonnegative(s0, 'interaction centered trace')
    iterative = c > 1 and s0 > 0
    tolerance = 128*math.ulp(1.)*max(n, c)
    check(fit['numerical_tolerance'] == tolerance and fit['degenerate_constant_features'] == (s0 == 0),
          'Frozen numeric tolerance/degeneracy mismatch')
    check(all(fit[key] == config['algorithm'][key] for key in
              ('distance_rule', 'bandwidth_rule')), 'Frozen local geometry changed')
    check(0 <= fit['trace_relative_error'] <= tolerance,
          'Frozen numerical residual exceeded')
    if c == 1:
        check(tau is None and fit['bandwidth_zero'] is None
              and fit['degeneracy_reason'] == 'SINGLE_REGISTERED_CLASS', 'Single registered class path mismatch')
    else:
        check(tau is not None and tau >= 0 and fit['bandwidth_zero'] == (tau == 0), 'Bandwidth mismatch')
        close(tau, fit['nearest_other_class_squared_distance']['median'], 'Train-only median bandwidth')
        check(fit['nearest_other_class_squared_distance']['count'] == n, 'Bandwidth physical count mismatch')
        reason = 'IDENTICAL_COMPLETE_FEATURES' if s0 == 0 else ('ZERO_BANDWIDTH_EQUIVALENCE_KERNEL' if tau == 0 else None)
        check(fit['degeneracy_reason'] == reason, 'Exact local degeneracy mismatch')
    if iterative:
        check(sr is not None and sr > 0 and gamma is not None and gamma > 0,
              'Positive radial trace/scale required')
        check(abs(sr*gamma-s0)/s0 <= tolerance, 'Trace matching mismatch')
    else:
        check(sr is gamma is None, 'Exact zero geometry mismatch')
    check(fit['arm'] == 'local_margin' and fit['input_feature_dim'] == 736 and fit['implicit_feature_dim'] == 123616
          and fit['output_dim'] == fit['classification_output_dim'] == c
          and type(row['factorization_count']) is int and type(fit['factorization_calls']) is int
          and row['factorization_count'] == fit['factorization_calls'] == fit['factorization_dim'] == 0,
          'Margin candidate must perform zero matrix factorizations')
    check(fit['dual_hessian_materialized'] is False and fit['maximum_dual_variable_count'] == n*(c-1)
          and fit['all_states_estimated_from_trainfold_only'] is True, 'Margin optimization state scope mismatch')
    validate_optimization(fit, n, c, config)
    expected_bytes = dict(training_feature_bytes=8*n*736, gram_bytes=8*n*n,
                          coefficient_bytes=8*n*c, kernel_bytes=8*n*n)
    for key, value in expected_bytes.items():
        check(type(fit[key]) is int and fit[key] == value, 'Actual fit array byte mismatch '+key)
    arrays = dict(support_background=8*n*256, support_auxiliary=8*n*480, alpha=8*n*c,
                  reference_kernel=8*n, center_mean=8*n)
    scalar_bytes = 16+8*int(tau is not None)+8*int(gamma is not None)
    check(row['state_array_bytes'] == arrays and row['state_scalar_bytes'] == scalar_bytes
          and row['head_bytes'] == row['persistent_state_bytes'] == sum(arrays.values())+scalar_bytes
          and row['current_support_feature_bytes'] == 8*n*736 and bool(row['state_byte_scope']), 'Numeric deployed state byte mismatch')
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
          and bool(compact['source_validation_reason']) and compact['learning_rate'] is None and compact['epoch'] == fit['sweeps']
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


def sweep_text_lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield line.rstrip('\r\n')


def sweep_text(event):
    return (('[CVS LocalMargin] split={split_id} scope={scope} sweep={sweep} '
             'steps={optimizer_steps} margin_loss={loss_data} rkhs_penalty={loss_ridge} '
             'P={primal_objective} D={dual_objective} gap={relative_duality_gap} '
             'KKT={relative_kkt_residual} LR=N/A source_validation=N/A').format_map(event)
            + ' measured=' + json.dumps(event, sort_keys=True, allow_nan=False))


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
        folder=Path(row['output_root'])/'branch_local_margin';marker=read_json(folder/'predictions_complete.json')
        check(not (folder/'technical_failure.json').exists(), 'Failed candidate artifacts cannot certify completion')
        check(marker['status']=='PREDICTIONS_COMPLETE' and marker['split_count']==marker['predictions']==config['splits']
              and marker['capsule_id']==config['capsule_id'] and marker['checkpoint_sha256']==row['expected_checkpoint_sha256']
              and marker['truth_read'] is marker['source_data_access'] is marker['query_used_for_fitting'] is False
              and type(marker['factorization_count']) is int and marker['factorization_count']==0
              and marker['mode']=='d92_branch_local_margin_registration' and marker['algorithm']==config['algorithm'], 'Prediction completion metadata mismatch')
        resource=validate_resources(row,read_json(folder/'startup.json'),marker,config)
        # CSV is checked against its own streamed compact/stage JSONL; full trace
        # is streamed separately and only scalar summaries survive an episode.
        csv_matches(folder/'compact.csv',json_lines(folder/'compact.jsonl'))
        csv_matches(folder/'fit_stages.csv',json_lines(folder/'fit_stages.jsonl'))
        csv_matches(folder/'solver_sweeps.csv',json_lines(folder/'solver_sweeps.jsonl'))
        sweep_stream=iter(json_lines(folder/'solver_sweeps.jsonl'))
        text_stream=iter(sweep_text_lines(folder/'solver_sweeps.log'))
        seen=set();cells=set();compact_records=[];fit_records=[];diagnostic_records=[]
        streams=(json_lines(folder/'fit_trace.jsonl'),json_lines(folder/'compact.jsonl'),json_lines(folder/'fit_stages.jsonl'))
        for trace,small,stage in zip_longest(*streams):
            check(trace is not None and small is not None and stage is not None, 'Unequal metadata stream lengths')
            sid=trace['split_id'];check(sid not in seen,'Duplicate split');seen.add(sid)
            cell=tuple(trace[key] for key in ('receiver','scenario','k','new_count','support_seed'))
            check(cell in expected and cell not in cells,'Unexpected/duplicate matrix cell');cells.add(cell)
            validate_fit(trace,small,config)
            for event in trace['final_fit']['optimization_trace']:
                recorded=next(sweep_stream,None)
                check(recorded is not None, 'Missing sweep JSONL record')
                extras={'split_id','source_validation','source_validation_reason','learning_rate_reason'}
                check({key:value for key,value in recorded.items() if key not in extras} == scalars(event)
                      and recorded['split_id']==sid and recorded['source_validation'] is None
                      and bool(recorded['source_validation_reason']) and bool(recorded['learning_rate_reason']),
                      'Live sweep record differs from complete fit trace')
                check(next(text_stream,None)==sweep_text(recorded), 'Sweep text differs from measured JSONL record')
            check(stage==dict(split_id=sid,**scalars(trace['final_fit'])),'Full trace/analytical stage mismatch')
            check(small['completed']==len(seen) and small['total']==config['splits'],'Compact progress mismatch')
            compact_records.append({key:small[key] for key in ('k','fit_seconds','fit_call_seconds','query_score_seconds',
                'prediction_write_seconds','total_seconds','peak_process_rss_bytes','head_bytes','persistent_state_bytes')})
            fit_records.append(scalars(trace['final_fit']))
            diagnostic_records.append(kernel_diagnostics(trace['final_fit']))
        check(cells==expected and len(seen)==config['splits'],'Incomplete full matrix')
        check(next(sweep_stream,None) is None and next(text_stream,None) is None, 'Unbound extra sweep records')
        factors=sum(v['factorization_calls'] for v in fit_records)
        check(marker['factorization_count']==factors, 'Completion actual factorization total mismatch')
        steps=sum(v['optimizer_steps'] for v in fit_records); sweeps=sum(v['sweeps'] for v in fit_records)
        check(type(marker['optimizer_steps']) is type(marker['sweep_count']) is int
              and marker['optimizer_steps']==steps and marker['sweep_count']==sweeps,
              'Completion actual optimizer/sweep totals mismatch')
        measurement_keys=('loss_data','loss_ridge','loss_total','gradient_norm','primal_objective','dual_objective',
            'duality_gap','relative_duality_gap','kkt_residual','relative_kkt_residual','kkt_scale',
            'active_constraint_count','active_physical_count','slack_min','slack_mean','slack_max','squared_slack_sum',
            'sweeps','optimizer_steps','row_block_solves','maximum_dual_variable_count',
            'bandwidth_tau','interaction_centered_trace','radial_centered_trace','trace_scale','trace_relative_error',
            'zero_nearest_other_class_fraction',
            'gram_seconds','solve_seconds','fit_seconds','factorization_dim','training_feature_bytes','gram_bytes',
            'kernel_bytes','coefficient_bytes','training_accuracy')
        models.append(dict(row_id=row['row_id'],model_seed=row['seeds']['model'],fits=len(seen),
            k1_fits=sum(v['k']==1 for v in compact_records),full_fit_calls=len(fit_records),factorizations=factors,
            optimizer_steps=steps,row_block_solves=steps,sweep_count=sweeps,
            certified_fits=len(fit_records),iterative_fits=sum(v['sweeps']>0 for v in fit_records),
            analytic_zero_fits=sum(v['sweeps']==0 for v in fit_records),
            verified_sweep_records=sweeps,verified_sweep_text_records=sweeps,
            bandwidth_zero_fits=sum(v['bandwidth_zero'] is True for v in fit_records),
            degeneracy_counts=dict(Counter(v['degeneracy_reason'] or 'NONE' for v in fit_records)),
            solver_counts=dict(Counter(v['solver'] for v in fit_records)),status_counts=dict(Counter(v['status'] for v in fit_records)),
            resources=resource,fit_measurements={key:stats([v[key] for v in fit_records]) for key in measurement_keys},
            kernel_diagnostics={key:stats([v[key] for v in diagnostic_records]) for key in diagnostic_records[0]},
            timing={key:stats([v[key] for v in compact_records]) for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds')},
            state_bytes={key:stats([v[key] for v in compact_records]) for key in ('head_bytes','persistent_state_bytes')},
            peak_process_rss_bytes=marker.get('peak_process_rss_bytes'),peak_process_rss_reason=marker.get('peak_process_rss_reason'),
            compact_peak_rss_bytes=stats([v['peak_process_rss_bytes'] for v in compact_records]),
            trace_file_bytes=(folder/'fit_trace.jsonl').stat().st_size,stage_file_bytes=(folder/'fit_stages.jsonl').stat().st_size,
            sweep_file_bytes={name:(folder/name).stat().st_size for name in ('solver_sweeps.jsonl','solver_sweeps.csv','solver_sweeps.log')}))
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
    check(fit_calls==4800 and factorizations==0 and sum(m['certified_fits'] for m in models)==4800
          and sum(m['iterative_fits']+m['analytic_zero_fits'] for m in models)==4800,
          'Exactly one certified full fit with zero decomposition per episode required')
    sweeps=sum(m['sweep_count'] for m in models);steps=sum(m['optimizer_steps'] for m in models)
    check(sum(m['row_block_solves'] for m in models)==steps
          and sum(m['verified_sweep_records'] for m in models)==sweeps
          and sum(m['verified_sweep_text_records'] for m in models)==sweeps, 'Incomplete recorded optimizer work')
    by_sha={}
    for model in models:
        r=model['resources'];sha=r['checkpoint_sha256']
        check(sha not in by_sha or by_sha[sha]==r['existing_model_file_bytes'],'Same SHA model byte mismatch')
        by_sha[sha]=r['existing_model_file_bytes']
    check(len(by_sha)==4,'Exactly four unique frozen model packages required')
    measured={key:sum(m['timing'][key]['total'] for m in models) for key in ('fit_seconds','fit_call_seconds','query_score_seconds','prediction_write_seconds','total_seconds')}
    rss=[m['peak_process_rss_bytes'] for m in models if m['peak_process_rss_bytes'] is not None]
    return dict(status='VERIFIED',method='D92-BranchLocalMargin-v1',runs=runs,total_fits=4800,k1_fits=1200,
        full_fit_calls=fit_calls,factorizations=0,fold_fit_calls=0,optimizer_steps=steps,row_block_solves=steps,
        sweep_count=sweeps,verified_sweep_records=sweeps,verified_sweep_text_records=sweeps,oof=None,
        certified_fits=fit_calls,iterative_fits=sum(m['iterative_fits'] for m in models),
        analytic_zero_fits=sum(m['analytic_zero_fits'] for m in models),
        bandwidth_zero_fits=sum(m['bandwidth_zero_fits'] for m in models),
        feature_dim=736,implicit_feature_dim=123616,numeric_head_bytes_formula='8*N*(736+C+2)+16+8*(tau is not null)+8*(gamma is not null); N=C*K; five arrays and present scalars',
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
        model_incremental_transfer_bytes=0,model_already_deployed=None,
        model_deployment_unknown_reason='Existing cache reused; deployment state not inferred from storage bytes',
        checkpoint_loaded=False,feature_cache_reused=True,
        received_feature_array_bytes=sum(m['resources']['feature_array_bytes'] for m in models),
        received_feature_file_bytes=sum(m['resources']['feature_file_bytes'] for m in models),
        received_registry_array_bytes=sum(m['resources']['registry_array_bytes'] for m in models),
        received_native_forward_count=sum(m['resources']['native_physical_forward_count'] for m in models),
        synthetic_smoke_forward_count=sum(m['resources']['smoke_forward_count'] for m in models),
        actual_total_native_forward_count=sum(m['resources']['native_total_physical_forward_count'] for m in models),
        actual_total_native_batch_calls=sum(m['resources']['native_total_batch_calls'] for m in models),
        interpretation='Every recorded fit and sweep checked, not sampled or numerically reexecuted. Raw dual gradient need not vanish; projected KKT and primal/dual gap certify final solves. Analytic zero-kernel fits store zero decision alpha with an equivalent analytic beta certificate. Process work sums differ from wall span; max process RSS is not aggregate simultaneous memory. Summed episode state bytes are cumulative snapshots excluding optimizer work arrays. Existing model and feature bytes are storage, not new transfer; extraction/forward costs are zero for this cache-reuse run. No predictions, scores, truth, checkpoints or feature values read.')


def audit_config(spec):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
    from cvsrffi.d92_branch_local_margin import FROZEN_CONFIG
    from cvsrffi.d92_branch_ridge import FROZEN_CONFIG as PRODUCER_CONFIG
    from export_d92_branch_features import FEATURE_CONTRACT
    conf,data=spec['confirmation'],spec['data']
    check(conf['candidate']==FROZEN_CONFIG and conf['candidate_method']=='D92-BranchLocalMargin-v1'
          and conf['candidate_folder']=='branch_local_margin','Frozen BranchLocalMargin configuration only')
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
               kernel_diagnostics,validate_certificate,validate_optimization,validate_fit,validate_resources,
               sweep_text_lines,sweep_text,audit_run,audit_runs)
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
    script=remote_script([audit_config(read_json(path)) for path in args.spec]);compile(script,'branch_local_margin_metadata_audit','exec')
    result=subprocess.run(['ssh',*FLAGS,'-T','N607','python3 -'],input=script.encode('utf-8'),capture_output=True,check=True)
    value=json.loads(result.stdout);finite_tree(value);check(value.get('status')=='VERIFIED','Audit incomplete')
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='runs'},allow_nan=False))


if __name__=='__main__':main()
