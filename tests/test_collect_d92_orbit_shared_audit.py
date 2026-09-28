"""Only synthetic support and metadata; never real target scores."""
import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code'), str(ROOT/'tools')]
import collect_d92_orbit_shared_audit as tool
from cvsrffi.stage2_d92_orbit_shared import FROZEN_CONFIG, fit_orbit_shared


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def lines(path, records):
    path.write_text(''.join(json.dumps(v)+'\n' for v in records), encoding='utf-8')


def csvwrite(path, records):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader()
        for record in records:
            writer.writerow({k: 'N/A' if v is None else json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in record.items()})


def synthetic(k, zero=False, old_only=False):
    classes = ['old-a', 'old-b'] if old_only else ['old-a', 'new-b']
    rng = np.random.default_rng(749+k)
    z = rng.normal(size=(2*k, 4, 160)); fft = rng.normal(size=(2*k, 96))
    if zero:
        z *= 0; fft *= 0
    with threadpool_limits(limits=1):
        state = fit_orbit_shared(support_identity_views=z, support_fft=fft,
            support_labels=np.repeat(np.arange(2), k), support_ids=[f'p{i:03}' for i in range(2*k)],
            classes=classes, old_classes=classes if old_only else classes[:1])
    trace = state.audit_dict()
    trace.update(split_id=f'split-{k}', registered_classes=classes, old_classes=classes if old_only else classes[:1],
        receiver='rx', scenario='scene', support_seed=1, new_count=0 if old_only else 1,
        fit_call_seconds=trace['fit_seconds']+.001, query_score_seconds=.002, prediction_write_seconds=.003)
    return trace


def compact(trace, index=1, total=3):
    result = {k: trace[k] for k in ('split_id', 'k', 'classes', 'selected', 'selection', 'candidate_count', 'fold_count',
        'fit_seconds', 'head_bytes', 'persistent_state_bytes', 'receiver', 'scenario', 'support_seed', 'new_count',
        'fit_call_seconds', 'query_score_seconds', 'prediction_write_seconds')}
    result.update(completed=index, total=total, total_seconds=trace['fit_seconds']+.01,
        oof=tool.scalars(trace['oof']) if trace['oof'] is not None else None, final_fit=tool.scalars(trace['final_fit']),
        optimizer_steps=0, learning_rate=None, gradient_norm=None, epoch=None,
        unavailable_reason='Analytical fit', source_validation=None, source_validation_reason='No source access',
        query_rows_used_for_fit=0, source_rows_used_for_fit=0)
    return result


def config():
    return dict(algorithm=copy.deepcopy(FROZEN_CONFIG), old_classes=['old-a'], ks=[1, 2, 5], new_counts=[1])


@pytest.fixture(scope='module')
def measured():
    return [synthetic(k) for k in (1, 2, 5)]


def test_real_core_metadata_and_uneven_fold_df(measured):
    for record in measured:
        tool.validate_fit(record, compact(record), config())
    row = measured[-1]
    assert [v['training']['physical_df'] for v in row['folds']] == [4, 4, 6]
    assert row['oof'] is not None and measured[0]['oof'] is None
    assert row['head_bytes'] == 16400


@pytest.mark.parametrize('damage', ['formula', 'selection', 'candidate', 'optimizer', 'query', 'reuse', 'class', 'physical',
    'fold_class', 'fold_assignment', 'fold_leak', 'fold_n', 'fold_reestimate', 'medoid', 'medoid_shift', 'shift',
    'nu_views', 'shrinkage', 'zero_branch', 'condition', 'array_bytes', 'stage_time', 'oof_duplicate', 'oof_label',
    'oof_nll', 'oof_prediction', 'oof_macro', 'compiled_bytes', 'compact', 'timing', 'nonfinite'])
def test_reject_corruption(measured, damage):
    record = copy.deepcopy(measured[-1]); f = record['folds'][0]; training = f['training']
    if damage == 'formula': record['config']['norm_floor'] = .1
    if damage == 'selection': record['selection'] = 'best_oof'
    if damage == 'candidate': record['candidate_count'] = 2
    if damage == 'optimizer': record['optimizer_steps'] = 64
    if damage == 'query': record['query_rows_used_for_fit'] = 1
    if damage == 'reuse': record['cross_row_adapted_state_reuse'] = True
    if damage == 'class': record['registered_classes'][0] = 'unknown'
    if damage == 'physical': record['physical_fold_assignment'][0] = copy.deepcopy(record['physical_fold_assignment'][1])
    if damage == 'fold_class': record['physical_fold_assignment'][0]['class_id'] = 'unknown'
    if damage == 'fold_assignment': record['physical_fold_assignment'][0]['fold'] = 1
    if damage == 'fold_leak': training['training_physical_ids'][0] = f['heldout_ids'][0]
    if damage == 'fold_n': training['train_k'] = 5
    if damage == 'fold_reestimate': training['all_states_estimated_from_trainfold_only'] = False
    if damage == 'medoid': training['medoid_physical_ids'][0]['physical_id'] = f['heldout_ids'][0]
    if damage == 'medoid_shift':
        pid = training['medoid_physical_ids'][0]['physical_id']
        next(v for v in training['alignment_shifts'] if v['physical_id'] == pid)['shift'] = 1
    if damage == 'shift': training['alignment_shifts'][0]['shift'] = 4
    if damage == 'nu_views': training['physical_df'] *= 4
    if damage == 'shrinkage': training['shrinkage'] = 256/(256+4*training['physical_df'])
    if damage == 'zero_branch': training['status'] = 'ZERO_PHYSICAL_RESIDUAL'
    if damage == 'condition': training['condition_bound'] += 1
    if damage == 'array_bytes': training['covariance_matrix_bytes'] = 8
    if damage == 'stage_time': training['solve_seconds'] = training['fit_seconds']+1
    if damage == 'oof_duplicate': record['oof']['physical_records'][0] = copy.deepcopy(record['oof']['physical_records'][1])
    if damage == 'oof_label': record['oof']['physical_records'][0]['class_id'] = 'wrong'
    if damage == 'oof_nll': record['oof']['physical_records'][0]['nll'] += .1
    if damage == 'oof_prediction': record['oof']['physical_records'][0]['predicted_class'] = 'unknown'
    if damage == 'oof_macro': record['oof']['macro_nll'] += 1
    if damage == 'compiled_bytes': record['head_bytes'] = 8200
    if damage == 'nonfinite': training['covariance_trace'] = float('nan')
    small = compact(record)
    if damage == 'compact': small['selected'] = 'other'
    if damage == 'timing': small['total_seconds'] = 0
    with pytest.raises(ValueError): tool.validate_fit(record, small, config())


def test_k1_zero_and_old_only_are_legal():
    for k in (1, 2):
        record = synthetic(k, zero=True, old_only=True)
        cfg = config(); cfg.update(old_classes=record['old_classes'], new_counts=[0])
        tool.validate_fit(record, compact(record), cfg)
        assert record['final_fit']['status'] == 'ZERO_PHYSICAL_RESIDUAL'
        if k == 1:
            record['oof'] = {'illegal': True}
            with pytest.raises(ValueError, match='K1'): tool.validate_fit(record, compact(record), cfg)


def rewrite(folder, traces):
    records = [compact(t, i+1, len(traces)) for i, t in enumerate(traces)]
    stages = [dict(split_id=t['split_id'], **tool.scalars(stage)) for t in traces
              for stage in [f['training'] for f in t['folds']]+[t['final_fit']]]
    lines(folder/'fit_trace.jsonl', traces); lines(folder/'compact.jsonl', records); csvwrite(folder/'compact.csv', records)
    lines(folder/'fit_stages.jsonl', stages); csvwrite(folder/'fit_stages.csv', stages)


def make_run(tmp_path, measured):
    from d92_orbit_feature_cache import FEATURE_CONTRACT, FORWARD_SCOPE, PRODUCER_CONFIG
    producer = tool.read_json(PRODUCER_CONFIG)['algorithm']
    root = tmp_path/'run'; root.mkdir(); rows = []
    for model in (1, 2):
        output = root/f'model-{model}'; folder = output/'osc'; feature = tmp_path/f'old-{model}'/'bnna_features'
        baseline = tmp_path/f'baseline-{model}'
        folder.mkdir(parents=True); feature.mkdir(parents=True)
        row = dict(row_id=f'model-{model}', output_root=str(output), reuse_multiview_features_root=str(feature),
            reuse_row_root=str(baseline), expected_checkpoint_sha256='a'*64, seeds=dict(model=model)); rows.append(row)
        cache = feature/'received_bnna_features.npz'; cache.write_bytes(b'never-read-features')
        payload = dict(new_ground_statistics_bytes=0, new_source_payload_bytes=0, new_feature_extraction_seconds=0,
            reused_frozen_cache=True, checkpoint_loaded=False, model_file_bytes=123456,
            received_feature_array_bytes=100*2944, received_feature_file_bytes=cache.stat().st_size,
            model_already_deployed=None, model_incremental_transfer_bytes=None, model_deployment_unknown_reason='Unknown deployment')
        provenance = dict(checkpoint_sha256='a'*64, model_seed=model, classes=['old-a'], checkpoint_epoch=200,
            checkpoint_inheritance=[], target_access_before_freeze=False, verdict='MATCHED_SOURCE_ONLY_SCRATCH',
            source_role_comparison='EXACT_MATCH', model_file_bytes=123456)
        dump(feature/'features_complete.json', dict(status='BNNA_FEATURES_COMPLETE', schema='d92_bnna_received_views_v1', algorithm=producer,
            capsule_id='cap', checkpoint_sha256='a'*64, model_seed=model, classes=['old-a'], dtype='float32',
            query_used_for_fitting=False, source_data_access=False, truth_read=False, encoder_updated=False,
            native_eval=True, native_buffers_unchanged=True, view_count_per_observation=4, native_forward_scope=FORWARD_SCOPE,
            new_ground_statistics_bytes=0, count=100, identity_views_shape=[100, 4, 160], fft_shape=[100, 96],
            identity_views_bytes=100*4*160*4, fft_bytes=100*96*4, feature_array_bytes=100*2944,
            feature_file_bytes=cache.stat().st_size, model_file_bytes=123456, feature_seconds=1.25))
        dump(feature/'checkpoint_provenance.json', provenance)
        dump(feature/'startup.json', dict(config={'algorithm': producer}, provenance=provenance,
            capsule_id='cap', checkpoint_sha256='a'*64, query_fit_access=False, source_data_access=False, truth_read=False))
        dump(folder/'predictions_complete.json', dict(status='PREDICTIONS_COMPLETE', split_count=3, predictions=3, capsule_id='cap',
            checkpoint_sha256='a'*64, truth_read=False, source_data_access=False, query_used_for_fitting=False,
            new_ground_statistics_bytes=0, new_source_payload_bytes=0, payload_audit=payload, peak_process_rss_bytes=123456))
        dump(folder/'startup.json', dict(config={'algorithm': FROZEN_CONFIG}, checkpoint_sha256='a'*64,
            capsule_id='cap', model_seed=model, orbit_features=str(feature), baseline_row_root=str(baseline),
            query_fit_access=False, truth_read=False, source_data_access=False, ground_summary_access=False,
            cross_row_adapted_state_reuse=False, adapted_state_inherited=False, encoder_updated=False,
            selection='none_cv_diagnostic_only', prediction_tie_policy='physical_class_id_ascending',
            feature_cache_schema='d92_bnna_received_views_v1', feature_cache_name='received_bnna_features.npz',
            feature_contract=FEATURE_CONTRACT, feature_cache_precision='float32', downstream_unit_norm_floor=1e-12,
            learning_rate=None, gradient_norm=None, epoch=None, source_validation=None,
            unavailable_reason='Analytical', source_validation_reason='No source', payload_audit=payload))
        rewrite(folder, measured)
    dump(root/'complete.json', dict(status='SCORED'))
    dump(root/'state.json', {r['row_id']: dict(status='PREDICTIONS_COMPLETE') for r in rows})
    return dict(root=str(root), rows=rows, splits=3, receivers=['rx'], scenarios=['scene'], ks=[1, 2, 5],
        new_counts=[1], support_seeds=[1], old_classes=['old-a'], algorithm=copy.deepcopy(FROZEN_CONFIG),
        producer_algorithm=producer, feature_contract=FEATURE_CONTRACT, forward_scope=FORWARD_SCOPE, capsule_id='cap')


def test_complete_matrix_no_scores_features_or_adapted_state(tmp_path, measured, monkeypatch):
    cfg = make_run(tmp_path, measured); original = Path.open
    def guard(path, *a, **kw):
        assert path.name not in ('scores.json', 'truth.json', 'predictions.jsonl', 'received_bnna_features.npz', 'received.npz', 'final_ssdg.pth')
        if 'old-' in str(path): assert path.name in ('features_complete.json', 'checkpoint_provenance.json', 'startup.json')
        return original(path, *a, **kw)
    monkeypatch.setattr(Path, 'open', guard)
    result = tool.audit_run(cfg)
    assert result['total_fits'] == 6 and result['fixed_k1_fits'] == 2 and result['diagnostic_cv_fits'] == 4
    assert result['analytical_fit_calls'] == 16 and result['optimizer_steps'] == 0
    resources = result['models'][0]['resources']
    assert resources['feature_extraction_seconds'] == 0 and resources['prior_producer_extraction_seconds'] == 1.25
    assert resources['model_incremental_transfer_bytes'] is None and resources['checkpoint_loaded'] is False


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'stages', 'stagecsv', 'compactcsv', 'progress',
    'cachebytes', 'producer_formula', 'lineage', 'view_contract', 'deployment', 'model_bytes', 'adapted', 'marker', 'timing'])
def test_full_run_corruption_rejected(tmp_path, measured, fault):
    cfg = make_run(tmp_path, measured)
    row = cfg['rows'][0]; folder = Path(row['output_root'])/'osc'; feature = Path(row['reuse_multiview_features_root'])
    if fault in ('missing', 'duplicate'):
        traces = copy.deepcopy(measured)
        if fault == 'missing': traces.pop()
        else: traces[-1] = copy.deepcopy(traces[0])
        rewrite(folder, traces)
    elif fault in ('stages', 'stagecsv', 'compactcsv'):
        file = {'stages': 'fit_stages.jsonl', 'stagecsv': 'fit_stages.csv', 'compactcsv': 'compact.csv'}[fault]
        p = folder/file; content = p.read_text(encoding='utf-8')
        p.write_text(content.replace('fixed_orbit_shared', 'other') if fault == 'compactcsv'
            else '\n'.join(content.splitlines()[:-1])+'\n', encoding='utf-8')
    elif fault in ('progress', 'timing'):
        small = list(tool.json_lines(folder/'compact.jsonl'))
        small[0]['completed' if fault == 'progress' else 'total_seconds'] = 0
        lines(folder/'compact.jsonl', small); csvwrite(folder/'compact.csv', small)
    else:
        p = folder/'startup.json' if fault in ('deployment', 'adapted') else feature/'checkpoint_provenance.json' if fault == 'lineage' else folder/'predictions_complete.json' if fault == 'marker' else feature/'features_complete.json'
        value = tool.read_json(p)
        if fault == 'cachebytes': value['feature_file_bytes'] += 1
        if fault == 'producer_formula': value['algorithm']['phases_quarter_turns'] = [0, 2, 1, 3]
        if fault == 'lineage': value['checkpoint_inheritance'] = ['old-head']
        if fault == 'view_contract': value['native_forward_scope'] = 'different'
        if fault == 'deployment': value['payload_audit']['model_already_deployed'] = True
        if fault == 'model_bytes': value['model_file_bytes'] = 0
        if fault == 'adapted': value['adapted_state_inherited'] = True
        if fault == 'marker': value['split_count'] = 1
        dump(p, value)
    with pytest.raises(ValueError): tool.audit_run(cfg)


def test_remote_payload_same_audit(tmp_path, measured, capsys):
    cfg = make_run(tmp_path, measured)
    code = tool.remote_script(cfg); compile(code, 'synthetic_remote', 'exec')
    exec(code, {})
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'VERIFIED' and result['total_fits'] == 6


def test_incomplete_run_fails_before_logs(tmp_path, measured, monkeypatch):
    cfg = make_run(tmp_path, measured); dump(Path(cfg['root'])/'complete.json', dict(status='RUNNING'))
    monkeypatch.setattr(tool, 'json_lines', lambda p: pytest.fail('Nonterminal run opened logs'))
    with pytest.raises(ValueError, match='SCORED'): tool.audit_run(cfg)


def test_existing_output_rejected_before_ssh(tmp_path, monkeypatch):
    output = tmp_path/'existing.json'; output.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['collector', '--spec', str(tmp_path/'absent'), '--output', str(output)])
    monkeypatch.setattr(tool.subprocess, 'run', lambda *a, **kw: pytest.fail('No SSH permitted'))
    with pytest.raises(FileExistsError): tool.main()


@pytest.mark.parametrize('k', [1, 2, 5])
def test_actual_predictor_outputs(tmp_path, monkeypatch, k):
    from test_predict_d92_orbit_shared import inputs
    import predict_d92_orbit_shared as predictor
    from d92_orbit_feature_cache import FEATURE_CONTRACT, FORWARD_SCOPE, PRODUCER_CONFIG
    args = inputs(tmp_path, monkeypatch, k)
    with threadpool_limits(limits=1): predictor.predict(**args)
    out = args['output']; trace = tool.read_json(out/'fit_trace.jsonl'); small = tool.read_json(out/'compact.jsonl')
    marker = tool.read_json(out/'predictions_complete.json'); startup = tool.read_json(out/'startup.json')
    cfg = dict(algorithm=FROZEN_CONFIG, old_classes=trace['old_classes'], ks=[k], new_counts=[trace['new_count']],
        producer_algorithm=tool.read_json(PRODUCER_CONFIG)['algorithm'], capsule_id=args['expected_capsule_id'],
        feature_contract=FEATURE_CONTRACT, forward_scope=FORWARD_SCOPE)
    tool.validate_fit(trace, small, cfg)
    tool.csv_matches(out/'compact.csv', [small]); tool.csv_matches(out/'fit_stages.csv', tool.json_lines(out/'fit_stages.jsonl'))
    row = dict(reuse_multiview_features_root=str(args['orbit_features']), reuse_row_root=str(args['row_root']),
        expected_checkpoint_sha256=args['expected_checkpoint_sha256'], seeds=dict(model=startup['model_seed']))
    assert tool.validate_resources(row, startup, marker, cfg)['feature_extraction_seconds'] == 0
