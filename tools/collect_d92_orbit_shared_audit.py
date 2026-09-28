"""Audit complete OSC fit metadata; never read scores, truth or feature arrays."""
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

from collect_d92_mvkme_audit import (check, finite_tree, read_json, json_lines,
    nonnegative, close, stats)
from collect_d92_bnna_audit import csv_matches, scalars


def validate_diagnostics(record, classes, old, physical=None):
    by_class = record['classwise_nll']
    check(len(by_class) == len(classes) and {v['class_id'] for v in by_class} == set(classes),
          'Diagnostic class coverage mismatch')
    for v in by_class:
        nonnegative(v['nll'], 'class diagnostic NLL')
        if physical is not None:
            losses = [p['nll'] for p in physical if p['class_id'] == v['class_id']]
            check(bool(losses), 'Missing physical OOF class')
            close(v['nll'], math.fsum(losses) / len(losses), 'physical/class OOF NLL')
    for key, chosen in [('macro_nll', by_class),
            ('old_nll', [v for v in by_class if v['class_id'] in old]),
            ('new_nll', [v for v in by_class if v['class_id'] not in old])]:
        if chosen:
            nonnegative(record[key], key)
            close(record[key], math.fsum(v['nll'] for v in chosen) / len(chosen), key)
        else:
            check(record[key] is None, 'Absent group must have null diagnostic')


def validate_training(training, ids, labels, classes, n, scope, fold):
    c = len(classes)
    check(training['scope'] == scope and training['fold'] == fold and training['train_k'] == n
          and training['train_physical_count'] == n*c and training['training_physical_ids'] == sorted(ids),
          'Training physical scope/count mismatch')
    check(training['optimizer_steps'] == 0 and training['all_states_estimated_from_trainfold_only'] is True,
          'Fit used non-training state or optimizer')
    medoids, shifts = training['medoid_physical_ids'], training['alignment_shifts']
    check(len(medoids) == c and {v['class_id'] for v in medoids} == set(classes), 'Medoid class coverage mismatch')
    check(len(shifts) == len(ids) and {v['physical_id'] for v in shifts} == set(ids), 'Alignment physical coverage mismatch')
    shift_by_id = {v['physical_id']: v['shift'] for v in shifts}
    check(all(type(s) is int and 0 <= s < 4 for s in shift_by_id.values()), 'Invalid cyclic alignment shift')
    for m in medoids:
        pid = m['physical_id']
        check(pid in ids and labels[pid] == m['class_id'] and shift_by_id[pid] == 0,
              'Medoid must be same-class training sample with zero shift')
    nu = c*(n-1)
    check(type(training['physical_df']) is int and training['physical_df'] == nu,
          'Physical residual degrees of freedom mismatch')
    close(training['shrinkage'], 256/(nu+256), 'physical shrinkage')
    trace, energy = training['covariance_trace'], training['feature_energy']
    nonnegative(trace, 'covariance trace'); nonnegative(energy, 'feature energy')
    tolerance = 64 * 2.220446049250313e-16 * energy
    close(training['zero_residual_tolerance'], tolerance, 'residual tolerance')
    zero = n == 1 or trace <= tolerance or energy == 0
    check(training['status'] == ('ZERO_PHYSICAL_RESIDUAL' if zero else 'SHARED_COVARIANCE_FIT'),
          'Residual branch mismatch')
    check(training['condition_bound'] == (1 if zero else nu+1), 'Condition bound mismatch')
    expected = dict(training_feature_bytes=n*c*4*256*8, aligned_feature_bytes=n*c*4*256*8,
        template_bytes=c*4*256*8, covariance_matrix_bytes=256*256*8,
        covariance_shape_bytes=256*256*8, cholesky_bytes=256*256*8,
        coefficient_bytes=8*c*4*256, intercept_bytes=8*c, persistent_state_bytes=8200*c)
    for key, value in expected.items():
        check(type(training[key]) is int and training[key] == value, 'Actual array byte mismatch: ' + key)
    for key in ('medoid_seconds', 'covariance_seconds', 'solve_seconds', 'fit_seconds'):
        nonnegative(training[key], key)
    check(sum(training[k] for k in ('medoid_seconds', 'covariance_seconds', 'solve_seconds')) <= training['fit_seconds'] + 1e-8,
          'Stage timing exceeds fit duration')


def validate_fit(row, compact, config):
    finite_tree(row); finite_tree(compact)
    k, c, classes, old = row['k'], row['classes'], row['registered_classes'], config['old_classes']
    check(row['method'] == 'D92-OSC-v1' and row['config'] == config['algorithm'], 'Frozen formula mismatch')
    check(type(k) is int and k in config['ks'] and type(c) is int and c-len(old) in config['new_counts'], 'K/class mismatch')
    check(len(classes) == len(set(classes)) == c and all(isinstance(v, str) and v for v in classes)
          and set(old).issubset(classes) and row['old_classes'] == old and row['new_count'] == c-len(old),
          'Class registry mismatch')
    check(row['selected'] == 'fixed_orbit_shared' and row['selection'] == 'none_cv_diagnostic_only'
          and row['candidate_count'] == 1, 'OSC must not select a candidate from CV')
    check(row['optimizer_steps'] == row['final_optimizer_steps'] == 0 and row['steps'] == [], 'OSC has no optimizer')
    check(row['query_rows_used_for_fit'] == row['source_rows_used_for_fit'] == row['new_source_payload_bytes'] == 0
          and row['ground_summary_used'] is row['encoder_updated'] is row['cross_row_adapted_state_reuse'] is False,
          'Source/query/adapted-state access violation')
    count = 0 if k == 1 else min(k, 3)
    check(row['fold_count'] == len(row['folds']) == count, 'Physical fold count mismatch')
    assignments = row['physical_fold_assignment']; ids = [v['physical_id'] for v in assignments]
    check(len(ids) == len(set(ids)) == k*c and all(isinstance(v, str) and v for v in ids), 'Support physical ID mismatch')
    check(all(v['class_id'] in classes for v in assignments), 'Unknown fold class')
    labels = {v['physical_id']: v['class_id'] for v in assignments}
    for cls in classes:
        members = sorted((v for v in assignments if v['class_id'] == cls), key=lambda v: v['physical_id'])
        check(len(members) == k and [v['fold'] for v in members] == ([-1]*k if k == 1 else [i%count for i in range(k)]),
              'Physical class fold assignment mismatch')
    seen = set()
    for fold in row['folds']:
        index = fold['fold']
        check(type(index) is int and index in range(count) and index not in seen, 'Duplicate/unknown fold')
        seen.add(index)
        train = sorted(v['physical_id'] for v in assignments if v['fold'] != index)
        held = sorted(v['physical_id'] for v in assignments if v['fold'] == index)
        check(fold['train_ids'] == train and fold['heldout_ids'] == held and not set(train)&set(held), 'Physical fold isolation mismatch')
        n, h = len(train)//c, len(held)//c
        check(fold['train_k'] == n and fold['heldout_k'] == h, 'Fold actual n mismatch')
        validate_training(fold['training'], train, labels, classes, n, 'fold', index)
        validate_diagnostics(fold['diagnostics'], classes, old)
        nonnegative(fold['heldout_score_seconds'], 'heldout scoring time')
    if k == 1:
        check(row['oof'] is None, 'K1 cannot use CV or validation')
    else:
        oof = row['oof']; physical = oof['physical_records']
        check(len(physical) == len(ids) and {v['physical_id'] for v in physical} == set(ids), 'OOF must cover each physical sample once')
        expected = {v['physical_id']: v for v in assignments}
        for record in physical:
            member = expected[record['physical_id']]
            check(record['class_id'] == member['class_id'] and record['fold'] == member['fold']
                  and record['predicted_class'] in classes, 'OOF physical/class binding mismatch')
            nonnegative(record['nll'], 'physical NLL')
        validate_diagnostics(oof, classes, old, physical)
        for fold in row['folds']:
            validate_diagnostics(fold['diagnostics'], classes, old, [v for v in physical if v['fold'] == fold['fold']])
        for key in ('macro_nll', 'old_nll', 'new_nll'):
            if oof[key] is not None:
                close(oof[key], sum(v['diagnostics'][key]*v['heldout_k']/k for v in row['folds']), 'Physical weighted OOF ' + key)
    validate_training(row['final_fit'], sorted(ids), labels, classes, k, 'final', None)
    for key, value in dict(head_bytes=8200*c, persistent_state_bytes=8200*c, support_feature_bytes=8*k*c*4*256).items():
        check(type(row[key]) is int and row[key] == value, 'Compiled/support bytes mismatch: ' + key)
    nonnegative(row['fit_seconds'], 'core fit time')
    stages = row['final_fit']['fit_seconds'] + sum(v['training']['fit_seconds']+v['heldout_score_seconds'] for v in row['folds'])
    check(stages <= row['fit_seconds'] + 1e-8, 'Nested fit time exceeds total')
    for key in ('split_id', 'k', 'classes', 'selected', 'selection', 'candidate_count', 'fold_count',
            'fit_seconds', 'head_bytes', 'persistent_state_bytes', 'receiver', 'scenario', 'support_seed', 'new_count',
            'fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds'):
        check(compact[key] == row[key], 'Compact/trace mismatch: ' + key)
    for key in ('fit_seconds', 'fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds', 'total_seconds'):
        nonnegative(compact[key], key)
    check(compact['fit_seconds'] <= compact['fit_call_seconds']+1e-8
          and sum(compact[k] for k in ('fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds')) <= compact['total_seconds']+1e-8,
          'Compact timing mismatch')
    check(compact['oof'] == (scalars(row['oof']) if row['oof'] is not None else None)
          and compact['final_fit'] == scalars(row['final_fit']), 'Compact diagnostics/final fit mismatch')
    check(compact['optimizer_steps'] == 0 and compact['learning_rate'] is compact['gradient_norm'] is compact['epoch'] is None
          and bool(compact['unavailable_reason']) and compact['source_validation'] is None
          and bool(compact['source_validation_reason'])
          and compact['query_rows_used_for_fit'] == compact['source_rows_used_for_fit'] == 0,
          'Compact optimizer/permission mismatch')


def validate_resources(row, startup, marker, config):
    root = Path(row['reuse_multiview_features_root'])
    feature = read_json(root/'features_complete.json')
    provenance = read_json(root/'checkpoint_provenance.json'); extraction = read_json(root/'startup.json')
    sha, seed = row['expected_checkpoint_sha256'], row['seeds']['model']
    check(feature['status'] == 'BNNA_FEATURES_COMPLETE' and feature['schema'] == 'd92_bnna_received_views_v1'
          and feature['algorithm'] == config['producer_algorithm'] and feature['capsule_id'] == config['capsule_id']
          and feature['checkpoint_sha256'] == sha and feature['model_seed'] == seed
          and feature['classes'] == config['old_classes'] and feature['dtype'] == 'float32', 'Frozen cache binding mismatch')
    check(feature['query_used_for_fitting'] is feature['source_data_access'] is feature['truth_read'] is feature['encoder_updated'] is False
          and feature['native_eval'] is feature['native_buffers_unchanged'] is True
          and feature['view_count_per_observation'] == 4 and feature['new_ground_statistics_bytes'] == 0
          and feature['native_forward_scope'] == config['forward_scope'], 'Frozen feature permission/view mismatch')
    check(provenance['checkpoint_sha256'] == sha and provenance['model_seed'] == seed
          and provenance['classes'] == config['old_classes'] and provenance['checkpoint_epoch'] == 200
          and provenance['checkpoint_inheritance'] == [] and provenance['target_access_before_freeze'] is False
          and provenance['verdict'] == 'MATCHED_SOURCE_ONLY_SCRATCH' and provenance['source_role_comparison'] == 'EXACT_MATCH'
          and provenance['model_file_bytes'] == feature['model_file_bytes'], 'Checkpoint origin mismatch')
    check(extraction['config'] == {'algorithm': config['producer_algorithm']} and extraction['provenance'] == provenance
          and extraction['capsule_id'] == config['capsule_id'] and extraction['checkpoint_sha256'] == sha
          and extraction['query_fit_access'] is extraction['source_data_access'] is extraction['truth_read'] is False,
          'Original extraction provenance mismatch')
    check(startup['config'] == {'algorithm': config['algorithm']} and startup['checkpoint_sha256'] == sha
          and startup['capsule_id'] == config['capsule_id'] and startup['model_seed'] == seed
          and Path(startup['orbit_features']) == root and Path(startup['baseline_row_root']) == Path(row['reuse_row_root']),
          'OSC startup binding mismatch')
    check(all(startup[k] is False for k in ('query_fit_access', 'truth_read', 'source_data_access', 'ground_summary_access',
          'cross_row_adapted_state_reuse', 'adapted_state_inherited', 'encoder_updated'))
          and startup['selection'] == 'none_cv_diagnostic_only' and startup['prediction_tie_policy'] == 'physical_class_id_ascending',
          'OSC startup input/selection mismatch')
    check(startup['feature_cache_schema'] == 'd92_bnna_received_views_v1' and startup['feature_cache_name'] == 'received_bnna_features.npz'
          and startup['feature_contract'] == config['feature_contract'] and startup['feature_cache_precision'] == 'float32'
          and startup['downstream_unit_norm_floor'] == 1e-12, 'Frozen feature semantics mismatch')
    check(startup['learning_rate'] is startup['gradient_norm'] is startup['epoch'] is startup['source_validation'] is None
          and bool(startup['unavailable_reason']) and bool(startup['source_validation_reason']), 'Analytical startup status mismatch')
    count = feature['count']; nonnegative(count, 'physical count', integer=True, positive=True)
    check(feature['identity_views_shape'] == [count, 4, 160] and feature['fft_shape'] == [count, 96]
          and feature['identity_views_bytes'] == count*4*160*4 and feature['fft_bytes'] == count*96*4
          and feature['feature_array_bytes'] == count*2944, 'Frozen cache shape/bytes mismatch')
    cache = root/'received_bnna_features.npz'
    check(feature['feature_file_bytes'] == cache.stat().st_size, 'Cache container bytes mismatch')
    payload = marker['payload_audit']
    check(startup['payload_audit'] == payload and payload['model_already_deployed'] is None
          and payload['model_incremental_transfer_bytes'] is None and bool(payload['model_deployment_unknown_reason']),
          'Unknown deployment/payload mismatch')
    check(payload['new_ground_statistics_bytes'] == payload['new_source_payload_bytes'] == payload['new_feature_extraction_seconds'] == 0
          and payload['reused_frozen_cache'] is True and payload['checkpoint_loaded'] is False
          and payload['model_file_bytes'] == feature['model_file_bytes']
          and payload['received_feature_array_bytes'] == feature['feature_array_bytes']
          and payload['received_feature_file_bytes'] == cache.stat().st_size, 'Reuse/resource accounting mismatch')
    nonnegative(feature['model_file_bytes'], 'training checkpoint bytes', integer=True, positive=True)
    nonnegative(feature['feature_seconds'], 'prior feature extraction seconds')
    return dict(feature_extraction_seconds=0, reused_frozen_cache=True, checkpoint_loaded=False,
        prior_producer_extraction_seconds=feature['feature_seconds'], prior_extraction_cost_scope='Historical cache construction, not this run compute',
        physical_observations=count, native_views_per_observation=4, new_native_forward_count=0,
        received_cache_array_bytes=feature['feature_array_bytes'], received_cache_file_bytes=cache.stat().st_size,
        existing_frozen_model_file_bytes=feature['model_file_bytes'], model_already_deployed=None, model_incremental_transfer_bytes=None,
        model_transfer_unavailable_reason='Deployment state unknown; complete training checkpoint package is not a measured minimum inference package',
        new_source_payload_bytes=0, ground_summary_used=False, frozen_cache_path=str(root),
        cache_scope='Existing received-only local cache, not source samples or an inherited adapted state')


def audit_run(config):
    root = Path(config['root'])
    check(read_json(root/'complete.json')['status'] == 'SCORED', 'Collection requires SCORED completion')
    state = read_json(root/'state.json')
    check(set(state) == {r['row_id'] for r in config['rows']} and all(v['status'] == 'PREDICTIONS_COMPLETE' for v in state.values()),
          'Incomplete model rows')
    expected = set(product(config['receivers'], config['scenarios'], config['ks'], config['new_counts'], config['support_seeds']))
    check(len(expected) == config['splits'], 'Matrix split count mismatch')
    models = []
    for row in config['rows']:
        folder = Path(row['output_root'])/'osc'; marker = read_json(folder/'predictions_complete.json')
        check(marker['status'] == 'PREDICTIONS_COMPLETE' and marker['split_count'] == marker['predictions'] == config['splits']
              and marker['capsule_id'] == config['capsule_id'] and marker['checkpoint_sha256'] == row['expected_checkpoint_sha256']
              and marker['truth_read'] is marker['source_data_access'] is marker['query_used_for_fitting'] is False
              and marker['new_ground_statistics_bytes'] == marker['new_source_payload_bytes'] == 0, 'Prediction completion mismatch')
        resources = validate_resources(row, read_json(folder/'startup.json'), marker, config)
        records = list(json_lines(folder/'compact.jsonl')); compact = {v['split_id']: v for v in records}
        check(len(records) == len(compact) == config['splits'], 'Missing/duplicate compact fits')
        csv_matches(folder/'compact.csv', records)
        csv_matches(folder/'fit_stages.csv', json_lines(folder/'fit_stages.jsonl'))
        stages = iter(json_lines(folder/'fit_stages.jsonl')); seen, cells = set(), set(); details = []
        for trace in json_lines(folder/'fit_trace.jsonl'):
            sid = trace['split_id']; check(sid in compact and sid not in seen, 'Unbound/duplicate fit trace'); seen.add(sid)
            validate_fit(trace, compact[sid], config)
            cell = tuple(trace[k] for k in ('receiver', 'scenario', 'k', 'new_count', 'support_seed'))
            check(cell in expected and cell not in cells, 'Unexpected/duplicate cell'); cells.add(cell)
            for stage in [f['training'] for f in trace['folds']] + [trace['final_fit']]:
                check(next(stages, None) == dict(split_id=sid, **scalars(stage)), 'Stage stream differs from full trace')
                details.append(stage)
        check(next(stages, None) is None and seen == set(compact) and cells == expected, 'Incomplete/extra analytical fit stages')
        check(all(v['total'] == config['splits'] for v in records)
              and {v['completed'] for v in records} == set(range(1, config['splits']+1)), 'Compact progress mismatch')
        final = [v for v in details if v['scope'] == 'final']; folds = [v for v in details if v['scope'] == 'fold']
        models.append(dict(row_id=row['row_id'], fits=len(records), fixed_k1_fits=sum(v['k'] == 1 for v in records),
            diagnostic_cv_fits=sum(v['k'] > 1 for v in records), analytical_fit_calls=len(details),
            fold_fit_calls=len(folds), final_fit_calls=len(final), optimizer_steps=0,
            final_status_counts=dict(Counter(v['status'] for v in final)),
            fold_status_counts=dict(Counter(v['status'] for v in folds)), resources=resources,
            timing={k: stats([v[k] for v in records]) for k in ('fit_seconds', 'fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds', 'total_seconds')},
            analytical_stage_timing={scope: {k: stats([v[k] for v in group]) for k in ('medoid_seconds', 'covariance_seconds', 'solve_seconds', 'fit_seconds')}
                                    for scope, group in [('all', details), ('fold', folds), ('final', final)]},
            numeric_state_byte_ranges={k: stats([v[k] for v in records]) for k in ('head_bytes', 'persistent_state_bytes')},
            working_array_byte_ranges={k: stats([v[k] for v in details]) for k in ('training_feature_bytes', 'aligned_feature_bytes',
                'template_bytes', 'covariance_matrix_bytes', 'covariance_shape_bytes', 'cholesky_bytes')},
            working_array_scope='Individual recorded allocations, not a sum of simultaneously live arrays or measured peak RAM; process RSS reported separately',
            peak_process_rss_bytes=marker.get('peak_process_rss_bytes'), peak_process_rss_reason=marker.get('peak_process_rss_reason'),
            trace_file=str(folder/'fit_trace.jsonl'), trace_file_bytes=(folder/'fit_trace.jsonl').stat().st_size,
            stage_file_bytes=(folder/'fit_stages.jsonl').stat().st_size))
    return dict(status='VERIFIED', method='D92-OSC-v1', models=models, total_fits=sum(v['fits'] for v in models),
        fixed_k1_fits=sum(v['fixed_k1_fits'] for v in models), diagnostic_cv_fits=sum(v['diagnostic_cv_fits'] for v in models),
        analytical_fit_calls=sum(v['analytical_fit_calls'] for v in models), optimizer_steps=0,
        selection='none_cv_diagnostic_only', query_rows_used_for_fit=0, source_rows_used_for_fit=0,
        new_source_payload_bytes=0, ground_summary_used=False, adapted_state_inherited=False,
        raw_traces_preserved=True, parameter_feedback_forbidden=True, optimizer_convergence_claim=False,
        scope='Complete physical fold/OOF/phase metadata consistency; not mathematical reexecution. No scores, truth, predictions, checkpoints or feature values read')


def audit_config(spec):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
    from cvsrffi.stage2_d92_orbit_shared import FROZEN_CONFIG
    from run_d92_confirmation import candidate_definition
    from d92_orbit_feature_cache import FEATURE_CONTRACT, FORWARD_SCOPE, PRODUCER_CONFIG
    conf, data = spec['confirmation'], spec['data']
    check(candidate_definition(conf)['candidate_method'] == 'D92-OSC-v1' and conf['candidate'] == FROZEN_CONFIG, 'Frozen OSC only')
    check(len(spec['rows']) == 4 and sorted(r['seeds']['model'] for r in spec['rows']) == list(range(2026092701, 2026092705)),
          'Four frozen model seeds required')
    check(data['k'] == [1, 5, 10, 20] and data['new_class_counts'] == [0, 2, 5, 10, 20]
          and len(data['scenarios']) == 3 and len(data['support_seeds']) == 5 and len(data['target_receivers']) in (1, 3),
          'Full preregistered cohort matrix required')
    count = math.prod(len(data[k]) for k in ('target_receivers', 'scenarios', 'k', 'new_class_counts', 'support_seeds'))
    check(count == conf['expected_split_count'] == conf['splits_per_model'], 'Spec split count mismatch')
    producer = read_json(PRODUCER_CONFIG)['algorithm']
    check(all(producer[k] == FROZEN_CONFIG[k] == v for k, v in FEATURE_CONTRACT.items()), 'Producer feature formula mismatch')
    return dict(root=spec['execution']['remote_run_root'], rows=spec['rows'], splits=count,
        receivers=data['target_receivers'], scenarios=data['scenarios'], ks=data['k'], new_counts=data['new_class_counts'],
        support_seeds=data['support_seeds'], old_classes=conf['old_classes'], algorithm=conf['candidate'],
        producer_algorithm=producer, feature_contract=FEATURE_CONTRACT, forward_scope=FORWARD_SCOPE,
        capsule_id=conf['reuse_validated_capsule_id'])


def remote_script(config):
    functions = (check, finite_tree, read_json, json_lines, nonnegative, close, stats, scalars, csv_matches,
                 validate_diagnostics, validate_training, validate_fit, validate_resources, audit_run)
    return ('import json,math,statistics,csv\nfrom pathlib import Path\nfrom collections import Counter\n'
            'from itertools import product,zip_longest\n' + '\n'.join(inspect.getsource(f) for f in functions)
            + '\nprint(json.dumps(audit_run(' + repr(config) + '),allow_nan=False))\n')


def main():
    from read_d92_run import FLAGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    script = remote_script(audit_config(read_json(args.spec))); compile(script, 'remote_audit', 'exec')
    result = subprocess.run(['ssh', *FLAGS, '-T', 'N607', 'python3 -'], input=script.encode('utf-8'), capture_output=True, check=True)
    value = json.loads(result.stdout); finite_tree(value); check(value.get('status') == 'VERIFIED', 'Audit incomplete')
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'models'}, allow_nan=False))


if __name__ == '__main__':
    main()
