"""Actual source export/old scoring/new tail predictions, with frozen toy NNs."""
from copy import deepcopy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def build_prediction_fixture(tmp_path,mp,produce=True):
    """Only NN/checkpoint loading and nominal training metadata are mocked.

    The old builder, old producer and historical CPU scoring run first. After
    that history is sealed, the four same synthetic truth files are inaccessible
    until the new producer completes all twelve points. No new toy training.
    """
    from test_pamf_phase3_ground_open_run import build_prediction_fixture as inherited
    from experiments.pamf_phase3_open_world import ground_open_score as oldscore
    old=inherited(tmp_path,mp)
    oldscore.score(old['root'],*[old['truths'][k] for k in ('source_known','source_novel','clean','star')],output=old['root'].parent/'score')
    assert len(old['truth_reads'])==4
    old['truth_reads'].clear()
    m=importlib.import_module('experiments.pamf_phase3_open_world.ground_tail_run');t=m._train()
    oldcfg=old['config'];mp.setattr(m,'REFERENCE_CODE',oldcfg['code_commit']);mp.setattr(t,'REFERENCE_CODE',oldcfg['code_commit'])
    mp.setattr(t,'COUNTS',dict(m.previous._train().COUNTS))
    mp.setattr(m,'CAL_COUNT',90);mp.setattr(m,'SOURCE_Q_COUNT',135);mp.setattr(m,'TARGET_Q_COUNT',18)
    tc=dict(run_id=t.RUN_ID,code_commit='e'*40,reference_config=oldcfg,output_root=str(tmp_path/'tail_train'),steps=2220,lr=.0001)
    c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],reference_config=oldcfg,train_config=tc,output_root=str(tmp_path/'tail_prediction'),batch_size=128)
    t.validate_config(tc)
    pm,ground,handles,reference=t._reference(tc)
    parent=dict(model='original_F0_E50',updates=11100)
    identities={arm:t._identity(tc,arm,pm,parent,ground,handles,reference) for arm in t.ARMS}
    rows={arm:dict(schema=t.SCHEMA,status=t.ROW_STATUS,run_id=t.RUN_ID,code_commit=tc['code_commit'],config=tc,arm=arm,
        identity=identities[arm],updates=2220,source_only=True,ground_clean_training=True,test_only_nominal_training_metadata=True) for arm in t.ARMS}
    trained=dict(schema=t.SCHEMA,status=t.STATUS,run_id=t.RUN_ID,code_commit=tc['code_commit'],config=tc,arms=list(t.ARMS),rows=rows,identities=identities,
        steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=4440,real_student_forward_rows=1136640,
        real_F_forward_rows=0,G_forward_rows=0,
        source_L_OE_sampling_matched=True,source_sampling_matched=True,teacher_source_sampling_matched=True,same_input_teacher=True,
        original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        all_student_heads_changed=True,all_student_bodies_changed=True,all_students_initialized_from_original_F0=True,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],
        original_parent_identity=parent,ground_data_identity=ground,source_handles=handles,source_ground_reference_identity=reference,
        source_only=True,ground_clean_training=True,source_L_reads=180,source_OE_reads=60,new_target_UDA_updates=0,
        test_only_nominal_training_metadata=True,test_fixture_note='No training here; true two-step NativeEnergy training/strict reload is covered by its owned test',
        final_phase3_complete=False,**{k:0 for k in t.ZEROS})
    _json(Path(tc['output_root'])/'complete.json',trained)
    root=Path(c['output_root']);events=[];calibration_calls=[];truth_reads=[];model_rows=[]
    class Student(torch.nn.Module):
        def __init__(self,arm):
            super().__init__();self.arm=arm;self.padding=torch.nn.Parameter(torch.zeros(232993),requires_grad=False)
            bias=torch.arange(6,dtype=torch.float32);bias[2 if arm=='DMEAN' else 3]=8
            self.register_buffer('bias',bias);self.eval()
        def forward(self,x):
            model_rows.append((self.arm,len(x)));raw=x.mean((1,2))[:,None].repeat(1,160)
            return raw,raw[:,:1]+self.bias
    models={arm:Student(arm) for arm in t.ARMS}
    mp.setattr(t,'load_trained_model',lambda c,arm,device:(models[arm],identities[arm]))
    def frozen():
        assert (root/'thresholds.npz').is_file() and (root/'calibration/complete.json').is_file()
        report=json.loads((root/'calibration/complete.json').read_text())
        assert len(report['methods'])==3 and report['test_inputs_used'] is False
    role=m.data.load_role;query=m.data.load_query;public=m._public;calibrate=m.calibration.calibrate_known
    def load_role(path,name):
        assert name=='CAL','New prediction cannot read L/OE/Us training values'
        events.append('CAL');return role(path,name)
    def load_query(path,name):
        frozen();events.append(name);return query(path,name)
    def target(c,name):
        frozen();events.append(name);return public(c,name)
    def calibration(ids,y,closed,baseline,raw,rx,day):
        calibration_calls.append(dict(ids=ids.copy(),closed=closed.copy(),baseline=baseline.copy()))
        return calibrate(ids,y,closed,baseline,raw,rx,day)
    mp.setattr(m.data,'load_role',load_role);mp.setattr(m.data,'load_query',load_query)
    mp.setattr(m,'_public',target);mp.setattr(m.calibration,'calibrate_known',calibration)
    original_load=np.load;truth_paths={Path(v).resolve() for v in old['truths'].values()}
    def guarded(path,*args,**kwargs):
        p=Path(path).resolve() if isinstance(path,(Path,str)) else None
        if p in truth_paths:
            assert (root/'complete.json').is_file(),'Current four truths must wait for the new tail producer'
            truth_reads.append(str(p))
        return original_load(path,*args,**kwargs)
    mp.setattr(np,'load',guarded)
    result=dict(module=m,config=c,root=root,truths=old['truths'],oldfixture=old,train_done=trained,events=events,
        calibration_calls=calibration_calls,truth_reads=truth_reads,models=models,model_rows=model_rows,
        data_root=old['data_root'],data_done=old['data_done'],backend=old['backend'],counters_before=dict(old['counters']))
    if produce:result['done']=m.evaluate(c,'cpu')
    return result


def test_actual_old_history_new_twelve_points_reuse_baseline_and_only_two_calibrations(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch);m,c,done=f['module'],f['config'],f['done']
    assert done['all_12_points_complete'] and done['source_controls_reused_exactly']
    assert done['original_F0_baseline_independent_of_method_columns'] and done['all_thresholds_frozen_before_Q']
    assert f['truth_reads']==f['oldfixture']['truth_reads']==[] and f['events']==['CAL',*m.CONDITIONS]
    assert len(f['calibration_calls'])==2
    assert set(f['calibration_calls'][0]['closed'])=={2} and set(f['calibration_calls'][1]['closed'])=={3}
    assert all(set(a['baseline'])=={5} for a in f['calibration_calls'])
    assert done['real_student_forward_rows']==2*90+4*135+4*18
    assert sum(n for _,n in f['model_rows'])==done['real_student_forward_rows']+4
    assert done['G_forward_rows']==36 and f['oldfixture']['counters']['G']-f['counters_before']['G']==38
    assert done['real_F_forward_rows']==0 and f['oldfixture']['counters']['F']==f['counters_before']['F']
    with np.load(f['root']/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'].copy()
    with np.load(f['oldfixture']['root']/'thresholds.npz',allow_pickle=False) as z:old_tau=z['conditional'].copy()
    np.testing.assert_array_equal(tau[0],old_tau[2])
    for condition in m.CONDITIONS:
        with subtests.test(condition=condition):
            a=m._rows(f['root'],condition);old=m.previous._rows(f['oldfixture']['root'],condition)
            for name in ('default_logits','closed','raw_scores','scores'):
                np.testing.assert_array_equal(a[name][:,0],old[name][:,2])
            np.testing.assert_array_equal(a['baseline_logits'],old['default_logits'][:,0])
            np.testing.assert_array_equal(a['baseline_closed'],old['baseline_closed'])
            for j in (1,2):np.testing.assert_array_equal(a['scores'][:,j],m.calibration.energy(a['default_logits'][:,j])-tau[j,a['closed'][:,j]])
            assert 'y' not in a and not any('known' in k or 'unknown' in k for k in a)
    for a,b in zip(m.CONDITIONS[::2],m.CONDITIONS[1::2]):
        np.testing.assert_array_equal(m._rows(f['root'],a)['ids'],m._rows(f['root'],b)['ids'])
    with pytest.raises(FileExistsError):m.evaluate(c,'cpu')


def test_bad_metadata_private_overrides_and_partial_final_query_keep_truth_closed(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch,produce=False);m,c=f['module'],f['config']
    for change in (dict(truth_path='hidden'),dict(batch_size=1),dict(output_root=c['reference_config']['data_root']),
                   dict(reference_config=dict(c['reference_config'],code_commit='0'*40))):
        with subtests.test(change=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,**change))
    p=Path(c['train_config']['output_root'])/'complete.json';before=p.read_bytes();bad=json.loads(before);bad['source_Us_reads']=90;_json(p,bad)
    with pytest.raises(ValueError):m.evaluate(c,'cpu')
    assert f['events']==[] and not f['root'].exists();p.write_bytes(before)
    public=m._public
    def fail(c,name):
        if name=='cross_star_DURA':raise ValueError('synthetic final condition unavailable')
        return public(c,name)
    monkeypatch.setattr(m,'_public',fail)
    with pytest.raises(ValueError,match='final condition'):m.evaluate(c,'cpu')
    assert not (f['root']/'complete.json').exists() and f['truth_reads']==[]
    assert (f['root']/'queries/cross_clean_DURA/predictions.npz').is_file()
