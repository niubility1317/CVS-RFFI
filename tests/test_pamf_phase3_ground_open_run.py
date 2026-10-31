"""Actual synthetic ground data -> frozen producer; no actual training/checkpoint."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def build_prediction_fixture(tmp_path, monkeypatch, *, produce=True):
    """Export real toy roles and evaluate; 2220-step metadata is nominal only.

    SourceDataset, split, G(clean) export, CAL, energy, class routing, frozen
    thresholds and all four prediction writers execute their production code.
    The old model loaders and new training/strict-checkpoint boundary are stubs.
    """
    from test_pamf_phase3_ground_open_data import fixture as data_fixture
    from experiments.pamf_phase3_open_world import fresh_world_score as fresh_score
    m = importlib.import_module('experiments.pamf_phase3_open_world.ground_open_run')
    d, dc, original, backend, data_events, counters = data_fixture.__wrapped__(tmp_path, monkeypatch)
    reference = dc['reference_config']; source = dc['source_config']
    native = dict(source['reader_config'], source_contract=source['source_contract'], source_cache=source['source_cache'])
    source_train = dict(native_config=native, IQ_root=str(tmp_path/'old_G10'), output_root=str(tmp_path/'r03_train'))
    reference.update(train_config=dict(output_root=str(tmp_path/'old_PCD_train'), U_public_root=str(tmp_path/'UNREAD_U/public')),
        source_control_config=dict(reference_config=dict(train_config=source_train)),
        Q_conditions={name:dict(public_root=str(tmp_path/name/'public'),capsule_id='toy-'+name,
            split_id='toy-split-'+name,expected_count=18,view_family=family) for name,family in
            [('clean_DURA','clean_cross_receiver'),('star_DURA','practical_cross_receiver')]})
    old_path=Path(reference['output_root'])/'complete.json';pm=json.loads(old_path.read_text());pm['config']=reference;_json(old_path,pm)
    data_done=d.build(dc,'cpu')
    train=m._train();monkeypatch.setattr(train,'COUNTS',dict(L=180,OE=60));monkeypatch.setattr(train,'US_COUNT',90)
    monkeypatch.setattr(m,'CAL_COUNT',90);monkeypatch.setattr(m,'SOURCE_Q_COUNT',135);monkeypatch.setattr(m,'TARGET_Q_COUNT',18)
    tc=dict(run_id=m.RUN_ID,code_commit=dc['code_commit'],reference_config=reference,data_root=dc['output_root'],
        output_root=str(tmp_path/'nominal_train'),steps=2220,lr=.0001)
    train.validate_config(tc)
    monkeypatch.setattr(train,'_reference',lambda c:(pm,None,None,None))
    monkeypatch.setattr(train,'_backend',lambda *a:backend)
    identities={}
    for arm in ('SUP','SSL'):
        ground,handles=train._ground_metadata(tc,pm,with_us=arm=='SSL')
        identities[arm]=train._identity(tc,arm,pm,dict(model='original_F0_E50',updates=11100),ground,handles)
    trained=dict(schema=train.SCHEMA,status=train.STATUS,run_id=m.RUN_ID,code_commit=tc['code_commit'],config=tc,
        steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=4440,real_student_forward_rows=1420800,
        identities=identities,source_L_OE_sampling_matched=True,original_models_unchanged=True,all_student_Sinc_unchanged=True,
        all_student_buffers_unchanged=True,target_U_reads=0,target_Q_reads=0,private_truth_reads=0,source_U_GT_reads=0,
        final_phase3_complete=False,parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],
        loaded_dura_identity=pm['loaded_dura_identity'],test_only_nominal_training_metadata=True,
        test_fixture_note='No training or real checkpoint in this fixture; strict training loader is tested by its owner')
    _json(Path(tc['output_root'])/'complete.json',trained)
    config=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],reference_config=reference,data_root=dc['output_root'],
        train_config=tc,output_root=str(tmp_path/'prediction'),batch_size=128)
    root=Path(config['output_root']);events=[];truth_reads=[];calibration_calls=[];model_rows=[]
    class Student(torch.nn.Module):
        def __init__(self, arm):
            super().__init__();self.arm=arm
            self.padding=torch.nn.Parameter(torch.zeros(232993,dtype=torch.float32),requires_grad=False)
            bias=torch.arange(6,dtype=torch.float32);bias[0 if arm=='SUP' else 1]=8
            self.register_buffer('bias',bias);self.eval()
        def forward(self,x):
            model_rows.append((self.arm,len(x)))
            raw=x.mean((1,2))[:,None].repeat(1,160)
            return raw,raw[:,:1]+self.bias
    models={arm:Student(arm) for arm in ('SUP','SSL')}
    monkeypatch.setattr(train,'load_trained_model',lambda c,arm,device:(models[arm],identities[arm]))
    role_loader=d.load_role;query_loader=d.load_query;calibrate=m.core.calibrate_known
    def load_role(root,role):
        assert role=='CAL','Frozen prediction must not open Us or any training role'
        events.append('CAL');return role_loader(root,role)
    def frozen_before_query():
        p=root/'calibration/complete.json';assert p.is_file() and (root/'thresholds.npz').is_file()
        report=json.loads(p.read_text());assert report['test_inputs_used'] is False and len(report['methods'])==3
    def query(root,name):
        frozen_before_query();events.append(name);return query_loader(root,name)
    def calibrator(ids,y,closed,baseline,raw,rx,day):
        calibration_calls.append(dict(ids=ids.copy(),y=y.copy(),closed=closed.copy(),baseline=baseline.copy(),raw=raw.copy()))
        return calibrate(ids,y,closed,baseline,raw,rx,day)
    monkeypatch.setattr(d,'load_role',load_role);monkeypatch.setattr(d,'load_query',query)
    monkeypatch.setattr(m.core,'calibrate_known',calibrator)
    ids=np.asarray(['future-%03d'%i for i in range(18)])
    target_y=np.asarray(list(range(10))+[0,1,2,4,6,7,8,9],np.int64)
    day=np.asarray(['2021_03_15']*10+['2021_03_23']*8)
    target_iq=np.random.default_rng(118).normal(size=(18,2,256)).astype(np.float32)
    target_baseline=np.tile(np.arange(6,dtype=np.float32),(18,1))
    def public(c,name):
        frozen_before_query();events.append(name)
        return dict(capsule_id='toy-'+name,split_id='toy-split-'+name,count=18),ids.copy(),target_iq.copy()
    monkeypatch.setattr(m,'_public',public)
    monkeypatch.setattr(m,'_old_target',lambda c,name:dict(ids=ids.copy(),default_logits=target_baseline.copy()))
    truth_dir=tmp_path/'synthetic_truth';truth_dir.mkdir()
    nv=original['novel'];nv_meta=nv['metadata']
    scenes=np.asarray(fresh_score.common.SCENES)
    nvtruth=truth_dir/'source_novel.npz'
    np.savez(nvtruth,ids=nv['ids'],tx=np.asarray(['19-15','19-16','7-1'])[np.arange(45)%3],
        rx_i=nv_meta['rx_i'],day_i=nv_meta['day_i'],scenario=scenes[np.arange(45)%len(scenes)])
    # Public reference evidence is separate from the scorer-only novel truth.
    old_nv=Path(reference['output_root'])/'source/source_DURA/novel_predictions.npz';old_nv.parent.mkdir(parents=True,exist_ok=True)
    np.savez(old_nv,ids=nv['ids'],rx_i=nv_meta['rx_i'],day_i=nv_meta['day_i'])
    truth_paths=dict(source_known=Path(dc['output_root'])/'private/known_test_truth.npz',source_novel=nvtruth)
    for name in ('clean','star'):
        p=truth_dir/(name+'.npz');truth_paths[name]=p
        scenario=np.asarray(['clean']*18) if name=='clean' else scenes[np.arange(18)%len(scenes)]
        np.savez(p,ids=ids,y=target_y,tx=np.asarray(fresh_score.TX_BY_LABEL)[target_y],
            receiver=np.full(18,34,np.int64),day=day,scenario=scenario)
    original_load=np.load
    def guarded(path,*args,**kwargs):
        resolved=Path(path).resolve() if isinstance(path,(str,Path)) else None
        if resolved in {v.resolve() for v in truth_paths.values()}:
            assert (root/'complete.json').is_file(),'Truth must follow all twelve frozen prediction points'
            truth_reads.append(str(resolved))
        if isinstance(path,(str,Path)) and any(part=='Us' for part in Path(path).parts):
            pytest.fail('Prediction/scoring attempted source Us array access')
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    counters_before=dict(counters)
    result=dict(module=m,config=config,root=root,data_root=Path(dc['output_root']),truths=truth_paths,events=events,
        truth_reads=truth_reads,calibration_calls=calibration_calls,models=models,backend=backend,
        data_done=data_done,train_done=trained,data_events=data_events,model_rows=model_rows,counters=counters,counters_before=counters_before)
    if produce:result['done']=m.evaluate(config,'cpu')
    return result


def test_actual_data_export_producer_all_twelve_before_truth(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch);m,c,done=f['module'],f['config'],f['done']
    assert done['status']==m.STATUS and done['all_12_points_complete'] and done['all_four_DURA_conditions_complete']
    assert done['all_thresholds_frozen_before_Q'] and done['source_Us_values_read_during_prediction']==0
    assert f['truth_reads']==[] and f['events']==['CAL',*m.CONDITIONS]
    assert done['real_student_forward_rows']==2*90+4*135+4*18 and done['G_forward_rows']==36
    assert f['counters']['G']-f['counters_before']['G']==38  # 36 actual future rows plus two smoke rows.
    assert f['counters']['F']==f['counters_before']['F']  # F0 CAL/query predictions are exact existing caches.
    assert sum(n for _,n in f['model_rows'])==done['real_student_forward_rows']+4
    with np.load(f['root']/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'].copy()
    reports=json.loads((f['root']/'calibration/complete.json').read_text())
    assert all(not v['constraints_passed'] for v in reports['methods'].values())
    calls=f['calibration_calls'];assert len(calls)==3
    assert set(calls[0]['closed'])=={5} and set(calls[1]['closed'])=={0} and set(calls[2]['closed'])=={1}
    assert all(np.array_equal(a['baseline'],calls[0]['closed']) for a in calls)
    for name in m.CONDITIONS:
        with subtests.test(condition=name):
            rows=m._rows(f['root'],name)
            expected=np.column_stack([m.core.energy(rows['default_logits'][:,j])-tau[j,rows['closed'][:,j]] for j in range(3)])
            np.testing.assert_array_equal(rows['scores'],expected)
            # The inherited synthetic G10 logits are tied, unlike G(clean).
            baseline_class=0 if name=='source_star_DURA' else 5
            assert np.all(rows['closed'][:,0]==baseline_class) and np.all(rows['closed'][:,1]==0) and np.all(rows['closed'][:,2]==1)
    for first,second in zip(m.CONDITIONS[::2],m.CONDITIONS[1::2]):
        np.testing.assert_array_equal(m._rows(f['root'],first)['ids'],m._rows(f['root'],second)['ids'])
    with pytest.raises(FileExistsError):m.evaluate(c,'cpu')


def test_prediction_rejects_private_config_and_preserves_partial_outputs(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch,produce=False);m,c=f['module'],f['config']
    for bad in (dict(c,truth_path='forbidden'),dict(c,batch_size=64)):
        with subtests.test(config=True):
            with pytest.raises(ValueError):m.validate_config(bad)
    public=m._public
    def fail(c,name):
        if name=='cross_star_DURA':raise ValueError('synthetic final query unavailable')
        return public(c,name)
    monkeypatch.setattr(m,'_public',fail)
    with pytest.raises(ValueError,match='final query'):m.evaluate(c,'cpu')
    assert not (f['root']/'complete.json').exists() and f['truth_reads']==[]
    assert (f['root']/'queries/cross_clean_DURA/predictions.npz').is_file()


def test_missing_known_class_is_scientific_failure_without_shrinking_roster():
    from experiments.pamf_phase3_open_world import ground_open_core as core
    y=np.asarray([0,1,2,4,5],np.int64);accepted=np.ones(5,bool)
    groups=core._known_groups(y,y,y,accepted,{'overall':np.ones(5,bool)})
    assert groups['overall']['coverage_complete'] is False
    assert groups['overall']['C_K'] is None and groups['overall']['A_K'] is None
    assert groups['overall']['passed'] is False
