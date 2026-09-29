"""Recorded full-fit metadata audit, using only synthetic NumPy cache evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT / 'code')]
import collect_d92_branch_local_ridge_audit as audit
import evaluate_d92_branch_local_ridge as predictor
import export_d92_branch_features as exporter
from test_evaluate_d92_branch_local_ridge import fixture, predict, dump, lines


def ready_fixture(tmp_path, constant=False):
    root = tmp_path / 'audit_run'; lane = root / 'lane'; lane.mkdir(parents=True)
    args, *_ = fixture(lane, constant=constant)
    args['output'] = lane / 'branch_local_ridge'
    features = args['branch_features']
    attrs = dict(emb_dim=160, t_dim=160, f_dim=160, active_defects=['pa'],
        id_feature_key='feat_joint', enable_dac=False, enable_pa=True, use_time_path=True, use_freq_path=True)
    provenance = predictor.read(features / 'checkpoint_provenance.json')
    provenance['active_branches'] = attrs
    dump(features / 'checkpoint_provenance.json', provenance)
    startup = predictor.read(features / 'startup.json'); startup['provenance'] = provenance
    dump(features / 'startup.json', startup)
    marker = predictor.read(features / 'features_complete.json')
    marker['active_branches'] = attrs
    marker['synthetic_smoke'].update(input='synthetic_PCG64_seed0', physical_forward_count=1,
        native_batch_calls=1, feature_shape=[1, 736], seconds=.001)
    marker['timing']['synthetic_smoke_seconds'] = .001
    dump(features / 'features_complete.json', marker)
    predict(args)
    row = dict(row_id='lane', output_root=str(lane), reuse_branch_features_root=str(features),
        expected_checkpoint_sha256=args['expected_checkpoint_sha256'], seeds=dict(model=provenance['model_seed']))
    config = dict(root=str(root), spec=dict(run_id='synthetic-local-audit'), rows=[row], splits=2,
        receivers=['RX'], scenarios=['scene'], ks=[1, 2], new_counts=[2], support_seeds=[7],
        old_classes=provenance['classes'], algorithm=predictor.local_core().FROZEN_CONFIG,
        producer_algorithm=exporter.local_core().FROZEN_CONFIG, feature_contract=exporter.FEATURE_CONTRACT,
        capsule_id=args['expected_capsule_id'])
    dump(root / 'complete.json', dict(status='SCORED', selection_feedback_forbidden=True,
                                    commit='synthetic', model_rows=1))
    dump(root / 'startup.json', dict(spec=config['spec'], commit='synthetic', timestamp=100.))
    dump(root / 'workflow_state.json', dict(status='SCORED', commit='synthetic', updated=120.))
    dump(root / 'state.json', dict(lane=dict(status='PREDICTIONS_COMPLETE')))
    return config, lane


@pytest.fixture
def ready(tmp_path):
    return ready_fixture(tmp_path)


def test_full_metadata_streams_and_recorded_resources_without_forbidden_reads(ready, monkeypatch):
    config, lane = ready
    original = Path.open
    def guard(path, *args, **kwargs):
        if 'r' in (args[0] if args else kwargs.get('mode', 'r')):
            assert path.name not in {'predictions.jsonl', 'scores.json', 'truth.json',
                                    'received_branch_features.npz', 'received.npz'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    result = audit.audit_run(config)
    assert result['status'] == 'VERIFIED' and result['total_fits'] == 2 and result['k1_fits'] == 1
    model = result['models'][0]; resource = model['resources']
    assert model['factorizations'] == model['full_fit_calls'] == 2
    assert model['degenerate_factorization_reductions'] == model['bandwidth_zero_fits'] == 0
    assert model['extra_diagnostic_triangular_solves'] == 4
    assert resource['new_source_payload_bytes'] == resource['model_incremental_transfer_bytes'] == 0
    assert resource['native_total_physical_forward_count'] == resource['extraction_timing']['total_seconds'] == 0
    assert resource['existing_received_native_forward_count'] == resource['count'] == 9
    assert resource['feature_array_bytes'] == 9 * 2944
    assert resource['checkpoint_loaded'] is False and resource['feature_cache_reused'] is True
    assert model['state_bytes']['head_bytes']['min'] == 8 * 3 * (736 + 3 + 2) + 32
    assert model['state_bytes']['head_bytes']['max'] == 8 * 6 * (736 + 3 + 2) + 32
    for key in ('bandwidth_tau', 'trace_scale', 'normal_equation_residual', 'trace_relative_error',
                'effective_degrees_of_freedom_seconds'):
        assert model['fit_measurements'][key]['min'] >= 0
    assert model['kernel_diagnostics']['training_offdiagonal_radial.q25']['min'] >= 0
    assert result['run_wall_seconds'] == 20.
    json.dumps(result, allow_nan=False)


def test_exact_degeneracy_has_zero_factorizations_and_smaller_scalar_state(tmp_path):
    config, _ = ready_fixture(tmp_path, constant=True)
    model = audit.audit_run(config)['models'][0]
    assert model['full_fit_calls'] == 2 and model['factorizations'] == 0
    assert model['degenerate_factorization_reductions'] == model['bandwidth_zero_fits'] == 2
    assert model['extra_diagnostic_triangular_solves'] == 0
    assert model['solver_counts'] == {'NO_FACTORIZATION_EXACT_ZERO': 2}
    assert model['degeneracy_counts'] == {'IDENTICAL_COMPLETE_FEATURES': 2}
    assert model['fit_measurements']['trace_scale'] is None
    assert model['state_bytes']['head_bytes']['min'] == 8 * 3 * (736 + 3 + 2) + 24


@pytest.mark.parametrize('fault', ['source', 'smoke', 'bytes', 'sha', 'frozen', 'branch', 'truth', 'modelbytes'])
def test_cache_metadata_corruption_rejected(ready, fault):
    config, lane = ready
    path = lane / 'branch_features/features_complete.json'; value = predictor.read(path)
    if fault == 'source': value['source_data_access'] = True
    elif fault == 'smoke': value['native_total_physical_forward_count'] -= 1
    elif fault == 'bytes': value['feature_file_bytes'] += 1
    elif fault == 'sha': value['checkpoint_sha256'] = 'bad'
    elif fault == 'frozen': value['native_parameters_unchanged'] = False
    elif fault == 'branch': value['active_branches']['enable_dac'] = True
    elif fault == 'truth': value['synthetic_smoke']['query_rows_read'] = 1
    elif fault == 'modelbytes': value['model_file_bytes'] += 1
    dump(path, value)
    with pytest.raises(ValueError): audit.audit_run(config)


@pytest.mark.parametrize('fault', ['held', 'ids', 'arraybytes', 'scalarbytes', 'objective', 'fold',
    'lambda', 'time', 'tau', 'gamma', 'residual', 'trace_error', 'factor', 'diagnostic_solves'])
def test_fit_contract_corruption_rejected(ready, fault):
    config, lane = ready
    row = lines(lane / 'branch_local_ridge/fit_trace.jsonl')[0]
    small = lines(lane / 'branch_local_ridge/compact.jsonl')[0]
    fit = row['final_fit']
    if fault == 'held': row['oof'] = {}
    elif fault == 'ids': row['support_records'][0]['physical_id'] = row['support_records'][1]['physical_id']
    elif fault == 'arraybytes': row['state_array_bytes']['alpha'] += 8
    elif fault == 'scalarbytes': row['state_scalar_bytes'] += 8
    elif fault == 'objective': fit['loss_total'] += 1
    elif fault == 'fold': row['fold_count'] = 1
    elif fault == 'lambda': fit['ridge_coefficient'] = .5
    elif fault == 'time': row['fit_seconds'] = 10000.
    elif fault == 'tau': fit['bandwidth_tau'] *= 2
    elif fault == 'gamma': fit['trace_scale'] *= 2
    elif fault == 'residual': fit['normal_equation_residual'] = fit['numerical_tolerance'] * 2
    elif fault == 'trace_error': fit['trace_relative_error'] = fit['numerical_tolerance'] * 2
    elif fault == 'factor': fit['factorization_calls'] = 0
    elif fault == 'diagnostic_solves': fit['effective_degrees_of_freedom_extra_triangular_solves'] = 0
    with pytest.raises(ValueError): audit.validate_fit(row, small, config)


@pytest.mark.parametrize('fault', ['compact_csv', 'stage_jsonl', 'completion_factors', 'terminal'])
def test_full_metadata_stream_and_terminal_consistency(ready, fault):
    config, lane = ready
    folder = lane / 'branch_local_ridge'
    if fault == 'compact_csv':
        path = folder / 'compact.csv'; path.write_text(path.read_text(encoding='utf-8').replace('fixed_no_selection', 'changed'), encoding='utf-8')
    elif fault == 'stage_jsonl':
        path = folder / 'fit_stages.jsonl'; path.write_text(path.read_text(encoding='utf-8').splitlines()[0] + '\n', encoding='utf-8')
    elif fault == 'completion_factors':
        path = folder / 'predictions_complete.json'; value = predictor.read(path); value['factorization_count'] = 1; dump(path, value)
    else:
        dump(Path(config['root']) / 'complete.json', dict(status='PREDICTIONS_COMPLETE'))
    with pytest.raises(ValueError): audit.audit_run(config)


@pytest.mark.parametrize('key,value', [('checkpoint_loaded', True), ('feature_cache_reused', False),
    ('native_total_physical_forward_count', 1), ('model_incremental_transfer_bytes', 100),
    ('model_already_deployed', True), ('feature_extraction_this_run_seconds', 1.)])
def test_storage_cannot_be_reported_as_new_forward_or_known_deployment(ready, key, value):
    config, lane = ready
    for name in ('startup.json', 'predictions_complete.json'):
        path = lane / 'branch_local_ridge' / name; record = predictor.read(path)
        record['payload_audit'][key] = value; dump(path, record)
    with pytest.raises(ValueError): audit.audit_run(config)


def test_joint_4800_fit_contract_allows_exact_factorization_reductions(ready, monkeypatch):
    config, _ = ready
    base = audit.audit_run(config)
    configs = [dict(receivers=['r1', 'r2', 'r3'], cohort=3), dict(receivers=['r4'], cohort=1)]
    def fake(c):
        value = deepcopy(base); value['models'] = []
        for seed in range(4):
            model = deepcopy(base['models'][0])
            model['fits'] = model['full_fit_calls'] = 300*c['cohort']
            model['k1_fits'] = 75*c['cohort']
            model['factorizations'] = 250*c['cohort']
            model['degenerate_factorization_reductions'] = 50*c['cohort']
            model['bandwidth_zero_fits'] = 50*c['cohort']
            model['extra_diagnostic_triangular_solves'] = 2*model['factorizations']
            model['resources']['checkpoint_sha256'] = str(seed)*64
            value['models'].append(model)
        return value
    monkeypatch.setattr(audit, 'audit_run', fake)
    result = audit.audit_runs(configs)
    assert result['total_fits'] == result['full_fit_calls'] == 4800
    assert result['factorizations'] == 4000 and result['degenerate_factorization_reductions'] == 800
    assert result['k1_fits'] == 1200 and result['extra_diagnostic_triangular_solves'] == 8000
    assert result['unique_model_count'] == 4 and result['existing_unique_model_file_bytes'] == 400
    assert result['model_already_deployed'] is None
    assert result['new_source_payload_bytes'] == result['model_incremental_transfer_bytes'] == 0
    assert result['joint_wall_span_seconds'] == 20 and result['summed_cohort_wall_seconds'] == 40
    with pytest.raises(ValueError): audit.audit_runs(configs[:1])
    configs[1]['receivers'] = ['r1']
    with pytest.raises(ValueError): audit.audit_runs(configs)


def synthetic_spec(receivers):
    data = dict(target_receivers=receivers, scenarios=['clear', 'low', 'rain'], k=[1, 5, 10, 20],
                new_class_counts=[0, 2, 5, 10, 20], support_seeds=list(range(5)))
    count = len(receivers)*300
    return dict(run_id='synthetic', execution=dict(remote_run_root='/synthetic/run'), data=data,
        rows=[dict(seeds=dict(model=seed)) for seed in range(2026092701, 2026092705)],
        confirmation=dict(candidate=predictor.local_core().FROZEN_CONFIG,
            candidate_method='D92-BranchLocalRidge-v1', candidate_folder='branch_local_ridge',
            expected_split_count=count, splits_per_model=count, old_classes=['old'],
            reuse_validated_capsule_id='synthetic-capsule'))


def test_remote_script_is_metadata_only_and_compiles_without_importing_numeric_core():
    configs = [audit.audit_config(synthetic_spec(['r1', 'r2', 'r3'])),
               audit.audit_config(synthetic_spec(['r4']))]
    source = audit.remote_script(configs)
    compile(source, 'local_ridge_metadata_audit', 'exec')
    assert 'np.load' not in source and 'import torch' not in source and 'import scipy' not in source
    assert "read_json(root/'scores.json')" not in source
    assert "read_json(folder/'predictions.jsonl')" not in source
    bad = synthetic_spec(['r4']); bad['data']['k'] = [1, 5]
    with pytest.raises(ValueError): audit.audit_config(bad)
