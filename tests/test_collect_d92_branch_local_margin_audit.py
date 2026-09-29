"""Metadata-only audit of all synthetic LocalMargin full fits and sweeps."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'tools'), str(ROOT/'code')]
import collect_d92_branch_local_margin_audit as audit
import evaluate_d92_branch_local_margin as predictor
import export_d92_branch_features as exporter
from test_evaluate_d92_branch_local_margin import fixture, predict, dump, lines


def ready_fixture(tmp_path, constant=False, single=False):
    root = tmp_path/'audit_run'; lane = root/'lane'; lane.mkdir(parents=True)
    args, *_ = fixture(lane, constant=constant)
    args['output'] = lane/'branch_local_margin'
    features = args['branch_features']
    attrs = dict(emb_dim=160, t_dim=160, f_dim=160, active_defects=['pa'],
        id_feature_key='feat_joint', enable_dac=False, enable_pa=True, use_time_path=True, use_freq_path=True)
    provenance = predictor.read(features/'checkpoint_provenance.json')
    provenance['active_branches'] = attrs
    dump(features/'checkpoint_provenance.json', provenance)
    startup = predictor.read(features/'startup.json'); startup['provenance'] = provenance
    dump(features/'startup.json', startup)
    marker = predictor.read(features/'features_complete.json'); marker['active_branches'] = attrs
    marker['synthetic_smoke'].update(input='synthetic_PCG64_seed0', physical_forward_count=1,
        native_batch_calls=1, feature_shape=[1,736], seconds=.001)
    marker['timing']['synthetic_smoke_seconds'] = .001
    dump(features/'features_complete.json', marker)
    if single:
        for path in (args['capsule']/'splits').glob('*.json'):
            split = predictor.read(path)
            split['support_indices'] = [i for i,y in zip(split['support_indices'],split['support_labels']) if y == 0]
            split['support_labels'] = [0]*split['k']; split['registered_classes'] = provenance['classes']
            dump(path, split)
    predict(args)
    row = dict(row_id='lane', output_root=str(lane), reuse_branch_features_root=str(features),
        expected_checkpoint_sha256=args['expected_checkpoint_sha256'], seeds=dict(model=provenance['model_seed']))
    config = dict(root=str(root), spec=dict(run_id='synthetic-margin-audit'), rows=[row], splits=2,
        receivers=['RX'], scenarios=['scene'], ks=[1,2], new_counts=[0 if single else 2], support_seeds=[7],
        old_classes=provenance['classes'], algorithm=predictor.local_core().FROZEN_CONFIG,
        producer_algorithm=exporter.local_core().FROZEN_CONFIG, feature_contract=exporter.FEATURE_CONTRACT,
        capsule_id=args['expected_capsule_id'])
    dump(root/'complete.json', dict(status='SCORED', selection_feedback_forbidden=True, commit='synthetic', model_rows=1))
    dump(root/'startup.json', dict(spec=config['spec'], commit='synthetic', timestamp=100.))
    dump(root/'workflow_state.json', dict(status='SCORED', commit='synthetic', updated=120.))
    dump(root/'state.json', dict(lane=dict(status='PREDICTIONS_COMPLETE')))
    return config, lane


@pytest.fixture
def ready(tmp_path):
    return ready_fixture(tmp_path)


def test_all_metadata_streams_and_resources_without_forbidden_reads(ready, monkeypatch):
    config, lane = ready
    original = Path.open
    def guard(path, *args, **kwargs):
        if 'r' in (args[0] if args else kwargs.get('mode','r')):
            assert path.name not in {'predictions.jsonl','scores.json','truth.json',
                'received_branch_features.npz','received.npz'}
            assert path.suffix not in {'.pth','.pt'}
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result = audit.audit_run(config); model = result['models'][0]; resource = model['resources']
    assert result['status'] == 'VERIFIED' and result['total_fits'] == 2 and result['k1_fits'] == 1
    assert model['factorizations'] == model['analytic_zero_fits'] == 0
    assert model['full_fit_calls'] == model['certified_fits'] == model['iterative_fits'] == 2
    assert model['optimizer_steps'] == model['row_block_solves'] > 0
    assert model['sweep_count'] == model['verified_sweep_records'] == model['verified_sweep_text_records'] > 2
    assert resource['new_source_payload_bytes'] == resource['model_incremental_transfer_bytes'] == 0
    assert resource['native_total_physical_forward_count'] == resource['extraction_timing']['total_seconds'] == 0
    assert resource['existing_received_native_forward_count'] == resource['count'] == 9
    assert resource['feature_array_bytes'] == 9*2944
    assert resource['checkpoint_loaded'] is False and resource['feature_cache_reused'] is True
    assert model['state_bytes']['head_bytes']['min'] == 8*3*(736+3+2)+32
    assert model['state_bytes']['head_bytes']['max'] == 8*6*(736+3+2)+32
    for key in ('bandwidth_tau','trace_scale','relative_kkt_residual','relative_duality_gap','trace_relative_error'):
        assert model['fit_measurements'][key]['min'] >= 0
    assert result['run_wall_seconds'] == 20.
    encoded = json.dumps(result,allow_nan=False)
    assert 'training_physical_ids' not in encoded and 'optimization_trace' not in encoded


@pytest.mark.parametrize('single',[False,True])
def test_analytic_certificate_has_zero_sweeps_and_no_factorization(tmp_path,single):
    config, _ = ready_fixture(tmp_path,constant=True,single=single)
    model = audit.audit_run(config)['models'][0]
    assert model['full_fit_calls'] == model['certified_fits'] == model['analytic_zero_fits'] == 2
    assert model['factorizations'] == model['optimizer_steps'] == model['sweep_count'] == model['iterative_fits'] == 0
    assert model['solver_counts'] == {'NO_OPTIMIZATION_ANALYTIC_ZERO':2}
    assert model['degeneracy_counts'] == {('SINGLE_REGISTERED_CLASS' if single else 'IDENTICAL_COMPLETE_FEATURES'):2}
    assert model['fit_measurements']['trace_scale'] is None
    assert model['state_bytes']['head_bytes']['min'] == (8*(736+1+2)+16 if single else 8*3*(736+3+2)+24)


@pytest.mark.parametrize('fault',['source','smoke','bytes','sha','frozen','branch','truth','modelbytes'])
def test_cache_metadata_corruption_rejected(ready,fault):
    config,lane = ready; path=lane/'branch_features/features_complete.json'; value=predictor.read(path)
    if fault=='source': value['source_data_access']=True
    elif fault=='smoke': value['native_total_physical_forward_count']-=1
    elif fault=='bytes': value['feature_file_bytes']+=1
    elif fault=='sha': value['checkpoint_sha256']='bad'
    elif fault=='frozen': value['native_parameters_unchanged']=False
    elif fault=='branch': value['active_branches']['enable_dac']=True
    elif fault=='truth': value['synthetic_smoke']['query_rows_read']=1
    elif fault=='modelbytes': value['model_file_bytes']+=1
    dump(path,value)
    with pytest.raises(ValueError): audit.audit_run(config)


@pytest.mark.parametrize('fault',['held','ids','arraybytes','scalarbytes','objective','fold','lambda',
    'time','tau','gamma','trace_error','factor','hessian','query','source','steps','sweeps',
    'uncertified','gap','kkt','early_sweep','sweep_geometry','sweep_change','sweep_steps','sweep_gradient'])
def test_fit_contract_and_every_sweep_corruption_rejected(ready,fault):
    config,lane=ready; folder=lane/'branch_local_margin'
    row=lines(folder/'fit_trace.jsonl')[0]; small=lines(folder/'compact.jsonl')[0]; fit=row['final_fit']
    if fault=='held': row['oof']={}
    elif fault=='ids': row['support_records'][0]['physical_id']=row['support_records'][1]['physical_id']
    elif fault=='arraybytes': row['state_array_bytes']['alpha']+=8
    elif fault=='scalarbytes': row['state_scalar_bytes']+=8
    elif fault=='objective': fit['loss_total']+=1
    elif fault=='fold': row['fold_count']=1
    elif fault=='lambda': fit['ridge_coefficient']=.5
    elif fault=='time': row['fit_seconds']=10000.
    elif fault=='tau': fit['bandwidth_tau']*=2
    elif fault=='gamma': fit['trace_scale']*=2
    elif fault=='trace_error': fit['trace_relative_error']=fit['numerical_tolerance']*2
    elif fault=='factor': fit['factorization_calls']=1
    elif fault=='hessian': fit['dual_hessian_materialized']=True
    elif fault=='query': row['query_rows_used_for_fit']=1
    elif fault=='source': row['source_rows_used']=1
    elif fault=='steps': fit['optimizer_steps']+=1
    elif fault=='sweeps': fit['optimization_trace'].pop(0)
    elif fault=='uncertified': fit['certified']=False
    elif fault=='gap': fit['relative_duality_gap']=.1
    elif fault=='kkt': fit['relative_kkt_residual']=.1
    elif fault=='early_sweep': fit['optimization_trace'][0]['certified']=True
    elif fault=='sweep_geometry': fit['optimization_trace'][0]['bandwidth_tau']*=2
    elif fault=='sweep_change': fit['optimization_trace'][0]['negative_dual_change']=1.
    elif fault=='sweep_steps': fit['optimization_trace'][0]['optimizer_steps']+=1
    elif fault=='sweep_gradient': fit['optimization_trace'][0]['gradient_coordinate']='dual_coefficients'
    with pytest.raises(ValueError): audit.validate_fit(row,small,config)


def test_raw_dual_gradient_is_a_measurement_not_a_stationarity_requirement(ready):
    config,lane=ready; folder=lane/'branch_local_margin'
    row=lines(folder/'fit_trace.jsonl')[0]; small=lines(folder/'compact.jsonl')[0]; fit=row['final_fit']
    # Scalar evidence alone cannot reconstruct the beta gradient; inactive dual
    # coordinates may have large positive gradients at a certified optimum.
    fit['gradient_norm']=fit['optimization_trace'][-1]['gradient_norm']=100.
    small['gradient_norm']=100.; small['final_fit']=audit.scalars(fit)
    audit.validate_fit(row,small,config)


@pytest.mark.parametrize('fault',['compact_csv','stage_jsonl','completion_factors','completion_steps',
    'completion_sweeps','terminal','failure','sweep_jsonl_missing','sweep_csv','sweep_text',
    'sweep_extra','sweep_text_extra','middle_sweep'])
def test_full_streams_and_terminal_consistency(ready,fault):
    config,lane=ready; folder=lane/'branch_local_margin'
    if fault=='compact_csv':
        path=folder/'compact.csv'; path.write_text(path.read_text(encoding='utf-8').replace('fixed_no_selection','changed'),encoding='utf-8')
    elif fault=='stage_jsonl':
        path=folder/'fit_stages.jsonl'; path.write_text(path.read_text(encoding='utf-8').splitlines()[0]+'\n',encoding='utf-8')
    elif fault.startswith('completion_'):
        key={'completion_factors':'factorization_count','completion_steps':'optimizer_steps','completion_sweeps':'sweep_count'}[fault]
        path=folder/'predictions_complete.json'; value=predictor.read(path); value[key]+=1; dump(path,value)
    elif fault=='terminal': dump(Path(config['root'])/'complete.json',dict(status='PREDICTIONS_COMPLETE'))
    elif fault=='failure': dump(folder/'technical_failure.json',dict(status='TECHNICAL_FAILURE'))
    elif fault=='sweep_csv':
        path=folder/'solver_sweeps.csv'; path.write_text(path.read_text(encoding='utf-8').replace('local_margin','changed'),encoding='utf-8')
    elif fault in ('sweep_text','sweep_text_extra'):
        path=folder/'solver_sweeps.log'; content=path.read_text(encoding='utf-8')
        path.write_text(content.replace('KKT=','BAD=',1) if fault=='sweep_text' else content+content.splitlines()[0]+'\n',encoding='utf-8')
    else:
        path=folder/'solver_sweeps.jsonl'; records=lines(path)
        if fault=='sweep_jsonl_missing': records.pop(0)
        elif fault=='sweep_extra': records.append(records[-1])
        else: records[len(records)//2]['bandwidth_tau']*=2
        path.write_text(''.join(json.dumps(v)+'\n' for v in records),encoding='utf-8')
        # Regenerate CSV so the independent all-trace binding, not merely CSV,
        # must detect missing/extra/intermediate records.
        csv_path=folder/'solver_sweeps.csv'; csv_path.unlink(); predictor.write_stage_csv(path,csv_path)
    with pytest.raises(ValueError): audit.audit_run(config)


@pytest.mark.parametrize('key,value',[('checkpoint_loaded',True),('feature_cache_reused',False),
    ('native_total_physical_forward_count',1),('model_incremental_transfer_bytes',100),
    ('model_already_deployed',True),('feature_extraction_this_run_seconds',1.)])
def test_storage_is_not_new_forward_transfer_or_known_deployment(ready,key,value):
    config,lane=ready
    for name in ('startup.json','predictions_complete.json'):
        path=lane/'branch_local_margin'/name; record=predictor.read(path)
        record['payload_audit'][key]=value; dump(path,record)
    with pytest.raises(ValueError): audit.audit_run(config)


def test_joint_4800_fits_1200_k1_and_actual_optimizer_totals(ready,monkeypatch):
    config,_=ready; base=audit.audit_run(config)
    configs=[dict(receivers=['r1','r2','r3'],cohort=3),dict(receivers=['r4'],cohort=1)]
    def fake(c):
        value=deepcopy(base); value['models']=[]
        for seed in range(4):
            model=deepcopy(base['models'][0]); count=300*c['cohort']
            model.update(fits=count,full_fit_calls=count,certified_fits=count,k1_fits=75*c['cohort'],
                factorizations=0,iterative_fits=250*c['cohort'],analytic_zero_fits=50*c['cohort'],
                optimizer_steps=1000*c['cohort'],row_block_solves=1000*c['cohort'],
                sweep_count=500*c['cohort'],verified_sweep_records=500*c['cohort'],verified_sweep_text_records=500*c['cohort'])
            model['resources']['checkpoint_sha256']=str(seed)*64; value['models'].append(model)
        return value
    monkeypatch.setattr(audit,'audit_run',fake)
    result=audit.audit_runs(configs)
    assert result['total_fits']==result['full_fit_calls']==result['certified_fits']==4800
    assert result['factorizations']==0 and result['k1_fits']==1200
    assert result['iterative_fits']==4000 and result['analytic_zero_fits']==800
    assert result['optimizer_steps']==result['row_block_solves']==16000
    assert result['sweep_count']==result['verified_sweep_records']==result['verified_sweep_text_records']==8000
    assert result['unique_model_count']==4 and result['existing_unique_model_file_bytes']==400
    assert result['model_already_deployed'] is None and result['model_incremental_transfer_bytes']==0
    assert result['joint_wall_span_seconds']==20 and result['summed_cohort_wall_seconds']==40
    with pytest.raises(ValueError): audit.audit_runs(configs[:1])
    configs[1]['receivers']=['r1']
    with pytest.raises(ValueError): audit.audit_runs(configs)


def synthetic_spec(receivers):
    data=dict(target_receivers=receivers,scenarios=['clear','low','rain'],k=[1,5,10,20],
        new_class_counts=[0,2,5,10,20],support_seeds=list(range(5)))
    count=len(receivers)*300
    return dict(run_id='synthetic',execution=dict(remote_run_root='/synthetic/run'),data=data,
        rows=[dict(seeds=dict(model=seed)) for seed in range(2026092701,2026092705)],
        confirmation=dict(candidate=predictor.local_core().FROZEN_CONFIG,
            candidate_method='D92-BranchLocalMargin-v1',candidate_folder='branch_local_margin',
            expected_split_count=count,splits_per_model=count,old_classes=['old'],reuse_validated_capsule_id='synthetic-capsule'))


def test_remote_script_is_self_contained_and_metadata_only():
    configs=[audit.audit_config(synthetic_spec(['r1','r2','r3'])),audit.audit_config(synthetic_spec(['r4']))]
    source=audit.remote_script(configs)
    compile(source,'margin_metadata_audit','exec')
    assert 'np.load' not in source and 'import torch' not in source and 'import scipy' not in source
    assert "read_json(root/'scores.json')" not in source and "read_json(folder/'predictions.jsonl')" not in source
    bad=synthetic_spec(['r4']); bad['data']['k']=[1,5]
    with pytest.raises(ValueError): audit.audit_config(bad)
