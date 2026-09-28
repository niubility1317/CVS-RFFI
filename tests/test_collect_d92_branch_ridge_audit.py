import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'code')]
import collect_d92_branch_ridge_audit as audit
import export_d92_branch_features as exporter
import evaluate_d92_branch_ridge as predictor
from test_export_d92_branch_features import full_fixture


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8')


def lines(path):return [json.loads(v) for v in path.read_text(encoding='utf-8').splitlines()]


@pytest.fixture
def ready(full_fixture,tmp_path):
    export_args,*_=full_fixture
    root=tmp_path/'audit_run';lane=root/'lane'
    export_args['output']=lane/'branch_features'
    with contextlib.redirect_stdout(io.StringIO()):
        exporter.export(**export_args)
        predictor.predict(**{k:export_args[k] for k in ('row_root','capsule','config','expected_capsule_id','expected_checkpoint_sha256')},
            branch_features=export_args['output'],output=lane/'branch_ridge')
    provenance=json.loads((lane/'branch_features/checkpoint_provenance.json').read_text())
    row=dict(row_id='lane',output_root=str(lane),expected_checkpoint_sha256=export_args['expected_checkpoint_sha256'],seeds=dict(model=export_args['seed']))
    config=dict(root=str(root),spec=dict(run_id='synthetic-audit'),rows=[row],splits=2,
        receivers=['target-rx'],scenarios=['test-scene'],ks=[1,2],new_counts=[1],support_seeds=[12],
        old_classes=provenance['classes'],algorithm=export_args['config']['algorithm'],feature_contract=exporter.FEATURE_CONTRACT,
        capsule_id=export_args['expected_capsule_id'])
    write(root/'complete.json',dict(status='SCORED',selection_feedback_forbidden=True,commit='test-head',model_rows=1))
    write(root/'startup.json',dict(spec=config['spec'],commit='test-head',timestamp=100.))
    write(root/'workflow_state.json',dict(status='SCORED',commit='test-head',updated=120.))
    write(root/'state.json',dict(lane=dict(status='PREDICTIONS_COMPLETE')))
    return config,lane


def test_actual_entry_streams_resources_without_forbidden_reads(ready,monkeypatch):
    config,lane=ready
    actual=Path.open
    def guard(path,*args,**kwargs):
        if 'r' in (args[0] if args else kwargs.get('mode','r')):
            assert path.name not in {'predictions.jsonl','scores.json','truth.json','received_branch_features.npz','received.npz'}
        return actual(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result=audit.audit_run(config)
    assert result['status']=='VERIFIED' and result['total_fits']==2 and result['k1_fits']==1
    assert result['run_wall_seconds']==20.
    model=result['models'][0];resources=model['resources']
    assert model['factorizations']==model['full_fit_calls']==2
    assert resources['smoke_forward_count']==resources['smoke_batch_calls']==1
    assert resources['native_total_physical_forward_count']==resources['count']+1
    assert resources['feature_array_bytes']==resources['count']*2944
    assert resources['new_source_payload_bytes']==0 and resources['model_incremental_transfer_bytes'] is None
    assert model['state_bytes']['head_bytes']['min']==model['state_bytes']['head_bytes']['max']==5896*3
    assert model['fit_measurements']['gradient_norm']['max']<1e-10
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('fault',['source','smoke','bytes','sha','frozen','branch','truth','modelbytes'])
def test_marker_provenance_resource_corruption_rejected(ready,fault):
    config,lane=ready;path=lane/'branch_features/features_complete.json';value=json.loads(path.read_text())
    if fault=='source':value['source_data_access']=True
    if fault=='smoke':value['native_total_physical_forward_count']-=1
    if fault=='bytes':value['feature_file_bytes']+=1
    if fault=='sha':value['checkpoint_sha256']='bad'
    if fault=='frozen':value['native_parameters_unchanged']=False
    if fault=='branch':value['active_branches']['enable_dac']=True
    if fault=='truth':value['synthetic_smoke']['query_rows_read']=1
    if fault=='modelbytes':value['model_file_bytes']+=1
    write(path,value)
    with pytest.raises(ValueError):audit.audit_run(config)


@pytest.mark.parametrize('fault',['k1_oof','label','ids','byte','objective','gradient','fold','lambda','config','time'])
def test_fit_semantics_corruption_rejected(ready,fault):
    config,lane=ready
    trace=lines(lane/'branch_ridge/fit_trace.jsonl')[0]
    compact=lines(lane/'branch_ridge/compact.jsonl')[0]
    if fault=='k1_oof':trace['oof']={}
    if fault=='label':trace['support_records'][0]['class_id']='invalid'
    if fault=='ids':trace['support_records'][0]['physical_id']=trace['support_records'][1]['physical_id']
    if fault=='byte':trace['head_bytes']+=8
    if fault=='objective':trace['final_fit']['loss_total']+=1
    if fault=='gradient':trace['final_fit']['gradient_norm']=float('nan')
    if fault=='fold':trace['fold_count']=1
    if fault=='lambda':trace['final_fit']['ridge_coefficient']=.5
    if fault=='config':trace['config']['branches'].append('dac')
    if fault=='time':trace['fit_seconds']=10000.
    with pytest.raises(ValueError):audit.validate_fit(trace,compact,config)


def test_csv_stream_mismatch_rejected(ready):
    config,lane=ready
    path=lane/'branch_ridge/fit_stages.csv'
    path.write_text(path.read_text().replace('CLOSED_FORM_SOLVED','changed'),encoding='utf-8')
    with pytest.raises(ValueError):audit.audit_run(config)


def test_missing_terminal_state_rejected_before_fit_read(ready):
    config,lane=ready
    write(Path(config['root'])/'complete.json',dict(status='PREDICTIONS_COMPLETE'))
    with pytest.raises(ValueError):audit.audit_run(config)


def test_full_joint_coverage_and_model_deduplication(ready,monkeypatch):
    config,_=ready;base=audit.audit_run(config)
    configs=[dict(receivers=['r1','r2','r3'],cohort=3),dict(receivers=['r4'],cohort=1)]
    def fake(c):
        value=copy.deepcopy(base);value['models']=[]
        for seed in range(4):
            model=copy.deepcopy(base['models'][0]);model['fits']=300*c['cohort'];model['k1_fits']=75*c['cohort']
            model['full_fit_calls']=model['factorizations']=model['fits']
            model['resources']['checkpoint_sha256']=str(seed)*64
            value['models'].append(model)
        return value
    monkeypatch.setattr(audit,'audit_run',fake)
    result=audit.audit_runs(configs)
    assert result['total_fits']==4800 and result['k1_fits']==1200
    assert result['unique_model_count']==4
    assert result['existing_unique_model_file_bytes']==4*base['models'][0]['resources']['existing_model_file_bytes']
    assert result['synthetic_smoke_forward_count']==8
    assert result['joint_wall_span_seconds']==20 and result['summed_cohort_wall_seconds']==40
    assert result['fold_fit_calls']==result['optimizer_steps']==0
    assert result['feature_dim']==736 and result['full_fit_calls']==result['factorizations']==4800
    with pytest.raises(ValueError):audit.audit_runs(configs[:1])
    configs[1]['receivers']=['r1']
    with pytest.raises(ValueError):audit.audit_runs(configs)


def test_bounded_remote_script_compile_and_no_executable_score_read():
    configs=[]
    for cohort in ('rx3','rx1'):
        spec=json.loads((ROOT/f'configs/d92_branch_ridge_repeat_{cohort}_20260929.json').read_text(encoding='utf-8'))
        configs.append(audit.audit_config(spec))
    source=audit.remote_script(configs)
    compile(source,'audit_script','exec')
    assert "read_json(root/'scores.json')" not in source
    assert "read_json(folder/'predictions.jsonl')" not in source
    assert 'np.load' not in source
