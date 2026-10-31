"""Synthetic UNIFORM prediction preparation; no real CP or CUDA is used."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.ground_uniform_run')


def _json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')


def test_four_readouts_reuse_CPAIR_two_columns_and_independent_F0(monkeypatch,subtests):
    m=api();monkeypatch.setattr(m,'TARGET_Q_COUNT',2)
    d=np.asarray([[0,0,8,0,0,0]]*2,np.float32);p=np.asarray([[0,0,0,0,9,0]]*2,np.float32);b=np.asarray([[0,0,0,0,0,7]]*2,np.float32)
    logits=np.stack((d,d,p,p),axis=1);raw=np.column_stack((m.calibration.energy(d),m._centered(d),m.calibration.energy(p),m._centered(p)))
    old=dict(ids=np.asarray(['a','b']),default_logits=logits,closed=logits.argmax(2).astype(np.int64),raw_scores=raw,scores=raw.copy(),baseline_logits=b,baseline_closed=b.argmax(1).astype(np.int64))
    students={'UNIFORM':np.asarray([[0,10,0,0,0,0]]*2,np.float32)};a=m._stack(old,students,np.zeros((4,6),np.float64));m._validate_rows(a,'cross_clean_DURA')
    assert np.all(a['closed'][:,:2]==4) and np.all(a['closed'][:,2:]==1) and np.all(a['baseline_closed']==5)
    for key in ('default_logits','closed','raw_scores','scores'):np.testing.assert_array_equal(a[key][:,:2],old[key][:,2:4])
    for defect in ('private','baseline','identity','raw'):
        with subtests.test(defect=defect),pytest.raises(ValueError):
            wrong=deepcopy(a)
            if defect=='private':wrong['unknown']=np.zeros(2,np.int64)
            elif defect=='baseline':wrong['baseline_logits']=old['default_logits'][:,0]
            elif defect=='identity':wrong['default_logits'][:,3,0]+=1
            else:wrong['raw_scores'][0,3]+=.1
            m._validate_rows(wrong,'cross_clean_DURA')
    old['scores'][0,3]+=.1
    with pytest.raises(ValueError,match='control'):m._stack(old,students,np.zeros((4,6),np.float64))

def build_prediction_fixture(tmp_path,mp,produce=True):
    """Real toy historical predictions/scores; nominal new-training/CP boundary."""
    from test_pamf_phase3_ground_centered_run import build_prediction_fixture as inherited
    from experiments.pamf_phase3_open_world import ground_centered_score as history_score
    old=inherited(tmp_path,mp,produce=False)
    # Extend the nominal checkpoint boundary with the current reference's
    # sampling/final-state declarations; no synthetic training is asserted.
    old['train_done']['source_indices_match_DMEAN']=True
    old['train_done']['rows']['CPAIR'].update(source_indices_match_DMEAN=True,final_phase3_complete=False)
    old_train=Path(old['config']['train_config']['output_root'])
    _json(old_train/'complete.json',old['train_done']);_json(old_train/'rows/CPAIR/complete.json',old['train_done']['rows']['CPAIR'])
    # Independent centered prediction/score parent leaves the tail score intact.
    old['config']['output_root']=str(tmp_path/'centered_history/predict');old['root']=Path(old['config']['output_root'])
    old['done']=old['module'].run(old['config'],'cpu')
    history_score.score(old['root'],*[old['truths'][k] for k in ('source_known','source_novel','clean','star')],output=old['root'].parent/'score')
    assert len(old['truth_reads'])==4
    for historical in (old,old['oldfixture'],old['oldfixture']['oldfixture']):historical['truth_reads'].clear()
    m=api();t=m._train();r=old['config'];mp.setattr(m,'REFERENCE_CODE',r['code_commit']);mp.setattr(t,'REFERENCE_CODE',r['code_commit'])
    mp.setattr(t.ancestry,'REFERENCE_CODE',r['code_commit'])
    mp.setattr(t,'COUNTS',dict(old['module']._train().COUNTS))
    mp.setattr(m,'CAL_COUNT',90);mp.setattr(m,'SOURCE_Q_COUNT',135);mp.setattr(m,'TARGET_Q_COUNT',18);mp.setattr(m,'COUNTS',dict(CAL=90,source_Q=135,cross_Q=18))
    tc=dict(run_id=t.RUN_ID,code_commit='7'*40,reference_config=r,output_root=str(tmp_path/'uniform_train'),steps=2220,lr=.0001)
    c=dict(run_id=m.RUN_ID,code_commit=tc['code_commit'],reference_config=r,train_config=tc,output_root=str(tmp_path/'uniform_prediction'),batch_size=128)
    t.validate_config(tc)
    # This metadata-only placeholder is never consumed as a training trace.
    trace=Path(r['train_config']['output_root'])/'rows/CPAIR/steps.jsonl';trace.parent.mkdir(parents=True,exist_ok=True);trace.write_text('',encoding='utf-8')
    pm,ground,handles,reference=t._reference(tc);parent=dict(model='original_F0_E50',updates=11100)
    identities={arm:t._identity(tc,arm,pm,parent,ground,handles,reference) for arm in m.ARMS}
    rows={arm:dict(schema=t.SCHEMA,status=t.ROW_STATUS,run_id=t.RUN_ID,code_commit=tc['code_commit'],config=tc,arm=arm,identity=identities[arm],updates=2220,
        source_only=True,ground_clean_training=True,test_only_nominal_training_metadata=True) for arm in m.ARMS}
    trained=dict(schema=t.SCHEMA,status=t.STATUS,run_id=t.RUN_ID,code_commit=tc['code_commit'],config=tc,arms=list(m.ARMS),rows=rows,identities=identities,
        steps_per_arm=2220,updates_per_arm=2220,total_optimizer_updates=2220,real_student_forward_rows=568320,real_F_forward_rows=0,G_forward_rows=0,
        source_L_OE_sampling_matched=True,source_sampling_matched=True,teacher_source_sampling_matched=True,same_input_teacher=True,
        source_indices_match_CPAIR=True,matched_source_arm='CPAIR',original_models_unchanged=True,all_student_Sinc_unchanged=True,all_student_buffers_unchanged=True,
        all_student_heads_changed=True,all_student_bodies_changed=True,all_students_initialized_from_original_F0=True,
        parent_F0_identity=pm['parent_F0_identity'],dura_identity=pm['dura_identity'],loaded_dura_identity=pm['loaded_dura_identity'],original_parent_identity=parent,
        ground_data_identity=ground,source_handles=handles,source_centered_reference_identity=reference,
        source_tail_reference_identity=old['done']['source_tail_reference_identity'],source_ground_reference_identity=old['done']['source_ground_reference_identity'],
        source_only=True,ground_clean_training=True,new_target_UDA_updates=0,OE_objective='equal_domain_uniform_posterior',pair_risk_in_objective=False,test_only_nominal_training_metadata=True,
        test_fixture_note='No training here; strict native uniform training/loading is covered by its owner',final_phase3_complete=False,**{k:0 for k in t.ZEROS})
    _json(Path(tc['output_root'])/'complete.json',trained)
    for arm,row in rows.items():_json(Path(tc['output_root'])/'rows'/arm/'complete.json',row)
    events=[];calibration_calls=[];truth_reads=[];model_rows=[]
    class Student(torch.nn.Module):
        def __init__(self,arm):
            super().__init__();self.arm=arm;self.padding=torch.nn.Parameter(torch.zeros(232993),requires_grad=False)
            bias=torch.arange(6,dtype=torch.float32);bias[3]=10;self.register_buffer('bias',bias);self.eval()
        def forward(self,x):
            model_rows.append((self.arm,len(x)));raw=x.mean((1,2))[:,None].repeat(1,160);return raw,raw[:,:1]+self.bias
    models={arm:Student(arm) for arm in m.ARMS};mp.setattr(t,'load_trained_model',lambda config,arm,device:(models[arm],identities[arm]))
    mp.setattr(m.previous._train(),'load_trained_model',lambda *a,**k:pytest.fail('New prediction must never load or forward old CPAIR'))
    role=m.data.load_role;query=m.data.load_query;public=m._public;calibrate=m.calibration.calibrate_known
    def frozen():
        root=Path(c['output_root']);assert (root/'thresholds.npz').is_file();report=json.loads((root/'calibration/complete.json').read_text())
        assert len(report['methods'])==4 and report['test_inputs_used'] is False
    def load_role(path,name):
        assert name=='CAL','Only frozen known CAL values may be opened';assert Path(c['output_root'],'smoke.json').is_file();events.append('CAL');return role(path,name)
    def load_query(path,name):frozen();events.append(name);return query(path,name)
    def target(config,name):frozen();events.append(name);return public(config,name)
    def calibrator(ids,y,closed,baseline,raw,rx,day):
        calibration_calls.append(dict(ids=ids.copy(),closed=closed.copy(),baseline=baseline.copy(),raw=raw.copy()));return calibrate(ids,y,closed,baseline,raw,rx,day)
    mp.setattr(m.data,'load_role',load_role);mp.setattr(m.data,'load_query',load_query);mp.setattr(m,'_public',target);mp.setattr(m.calibration,'calibrate_known',calibrator)
    original_load=np.load;truth_paths={Path(p).resolve() for p in old['truths'].values()}
    def guarded(path,*a,**kw):
        p=Path(path).resolve() if isinstance(path,(str,Path)) else None
        if p in truth_paths:
            assert Path(c['output_root'],'complete.json').is_file(),'All sixteen points must complete before current truth';truth_reads.append(str(p))
        return original_load(path,*a,**kw)
    mp.setattr(np,'load',guarded)
    result=dict(module=m,config=c,root=Path(c['output_root']),oldfixture=old,train_done=trained,events=events,calibration_calls=calibration_calls,truth_reads=truth_reads,
        truths=old['truths'],models=models,model_rows=model_rows,backend=old['backend'],data_done=old['data_done'],data_root=old['data_root'],
        counters=old['counters'],counters_before=dict(old['counters']))
    if produce:result['done']=m.run(c,'cpu')
    return result


def test_actual_sixteen_points_CAL_before_Q_zero_legacy_and_shared_logits(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch);m,c,done=f['module'],f['config'],f['done']
    assert done['all_16_points_complete'] and done['all_thresholds_frozen_before_Q'] and done['paired_readouts_share_logits']
    assert f['events']==['CAL',*m.CONDITIONS] and f['truth_reads']==[] and len(f['calibration_calls'])==2
    assert [set(v['closed']) for v in f['calibration_calls']]==[{3},{3}] and all(np.all(v['baseline']==5) for v in f['calibration_calls'])
    assert done['real_student_forward_rows']==90+2*135+2*18 and done['real_legacy_forward_rows']==done['real_F_forward_rows']==0
    assert sum(n for _,n in f['model_rows'])==done['real_student_forward_rows']+2
    assert f['counters']['G']-f['counters_before']['G']==done['G_forward_rows']+2==38 and f['counters']['F']-f['counters_before']['F']==2
    with np.load(f['root']/'thresholds.npz',allow_pickle=False) as z:tau=z['conditional'].copy()
    oldcal,oldtau=m._old_calibration(c);cal=m._read(f['root']/'calibration/complete.json');np.testing.assert_array_equal(tau[:2],oldtau[2:4])
    assert cal['methods'][m.METHODS[0]]==oldcal['methods']['CPAIR_energy'] and cal['methods'][m.METHODS[1]]==oldcal['methods']['CPAIR_centered']
    for condition in m.CONDITIONS:
        with subtests.test(condition=condition):
            a=m._rows(f['root'],condition);old=m._legacy(c,condition)
            for key in ('ids','baseline_logits','baseline_closed'):np.testing.assert_array_equal(a[key],old[key])
            for key in ('default_logits','closed','raw_scores','scores'):np.testing.assert_array_equal(a[key][:,:2],old[key][:,2:4])
            assert not np.array_equal(a['baseline_logits'],old['default_logits'][:,0])
            for j in range(4):np.testing.assert_array_equal(a['scores'][:,j],a['raw_scores'][:,j]-tau[j,a['closed'][:,j]])
            assert 'y' not in a
    with pytest.raises(FileExistsError):m.run(c,'cpu')


def test_unfrozen_training_private_options_wrong_old_row_and_partial_final_Q(tmp_path,monkeypatch,subtests):
    f=build_prediction_fixture(tmp_path,monkeypatch,produce=False);m,c=f['module'],f['config']
    for change in (dict(private_truth='hidden'),dict(U_true_K=3),dict(batch_size=64),dict(output_root=f['data_root'].as_posix())):
        with subtests.test(option=next(iter(change))),pytest.raises(ValueError):m.validate_config(dict(c,**change))
    p=Path(c['train_config']['output_root'])/'complete.json';before=p.read_bytes();wrong=json.loads(before);wrong['all_student_buffers_unchanged']=False;_json(p,wrong)
    with pytest.raises(ValueError,match='freeze'):m.run(c,'cpu')
    assert f['events']==[] and not f['root'].exists();p.write_bytes(before);legacy=m._legacy
    def damaged(config,name):
        a=legacy(config,name)
        if name=='source_clean_DURA':a['raw_scores'][0,2]+=.1;a['scores'][0,2]+=.1
        return a
    monkeypatch.setattr(m,'_legacy',damaged)
    with pytest.raises(ValueError,match='control|reconstruct'):m.run(c,'cpu')
    assert not (f['root']/'complete.json').exists() and not f['truth_reads']
    monkeypatch.setattr(m,'_legacy',legacy);c['output_root']=str(tmp_path/'partial_uniform');public=m._public
    def partial(config,name):
        value=public(config,name);return (value[0],value[1][:-1],value[2][:-1]) if name=='cross_star_DURA' else value
    monkeypatch.setattr(m,'_public',partial)
    with pytest.raises(ValueError,match='canonical'):m.run(c,'cpu')
    root=Path(c['output_root']);assert not (root/'complete.json').exists() and (root/'queries/cross_clean_DURA/predictions.npz').is_file() and not f['truth_reads']
