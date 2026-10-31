"""Synthetic teacher/center math and matched two-arm training; no real data."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():
    try:return importlib.import_module('experiments.pamf_phase3_open_world.momentum_prototype_train')
    except ModuleNotFoundError:pytest.fail('momentum prototype training is not implemented')


def test_cosine_soft_targets_center_old_order_and_stop_gradient():
    c=api().core
    z=torch.tensor([[2.,0.],[0.,3.]],requires_grad=True);head=torch.eye(2,requires_grad=True)
    logits=c.cosine_logits(z,head);torch.testing.assert_close(logits,torch.eye(2))
    center=torch.tensor([.2,-.2]);before=center.clone();q=c.teacher_targets(logits,center)
    torch.testing.assert_close(q,torch.softmax((torch.eye(2)-before)/.04,dim=1));assert not q.requires_grad
    c.update_center_(center,logits);torch.testing.assert_close(center,.9*before+.1*torch.tensor([.5,.5]))
    value=c.soft_target_ce(logits,q);value.backward()
    assert z.grad is not None and head.grad is not None and center.grad is None
    assert torch.isfinite(z.grad).all()


def test_ema_formula_skips_frozen_sinc_and_never_copies_buffers_or_gradients():
    c=api().core
    student=torch.nn.Module();student.register_parameter('body',torch.nn.Parameter(torch.tensor([2.,-3.])))
    student.register_parameter('sinc',torch.nn.Parameter(torch.tensor([1.234567]),requires_grad=False));student.register_buffer('buffer',torch.tensor([9.]))
    teacher=deepcopy(student).requires_grad_(False)
    with torch.no_grad():teacher.body.copy_(torch.tensor([1.,4.]));teacher.buffer.fill_(2)
    old=teacher.body.clone();frozen=teacher.sinc.clone();buffer=teacher.buffer.clone()
    count=c.ema_update_(teacher,student,.996)
    expected=old.mul(.996).add(student.body.detach(),alpha=.004)
    assert torch.equal(teacher.body,expected) and torch.equal(teacher.sinc,frozen) and torch.equal(teacher.buffer,buffer)
    assert count==2 and all(p.grad is None for p in teacher.parameters())
    th=torch.tensor([[2.,0.]]);sh=torch.tensor([[0.,1.]])
    c.ema_tensor_(th,sh,.996);torch.testing.assert_close(th,torch.tensor([[1.992,.004]]));assert not torch.allclose(th.norm(dim=1),torch.ones(1))


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_fresh_pcd_train import fixture as previous
    with previous(root,mp) as (fresh,t,pm,b,events,manifest):
        m=api();world=m.reference
        mp.setattr(m,'STEPS',2);mp.setattr(m,'U_COUNT',16);mp.setattr(m,'U_BATCH',4);mp.setattr(m,'COUNTS',dict(fresh.COUNTS))
        mp.setattr(world,'U_COUNT',16)
        t['code_commit']=m.REFERENCE_CODE
        done=fresh.train(t,'cpu');events.clear()
        r=dict(run_id=world.RUN_ID,code_commit=m.REFERENCE_CODE,train_config=t,source_control_config=t['reference_config'],
            Q_conditions={name:dict(public_root=str(root/name/'public'),capsule_id='never-open-'+name,split_id='q-'+name,expected_count=world.Q_COUNT,view_family=family)
                for name,family in [('clean_DURA','clean_cross_receiver'),('star_DURA','practical_cross_receiver')]},
            output_root=str(root/'completed_fresh_eval'),batch_size=128)
        out=Path(r['output_root']);out.mkdir()
        control=json.loads((Path(t['reference_config']['train_config']['output_root'])/'SOURCE/complete.json').read_text())
        meta=dict(schema=world.SCHEMA,status=world.STATUS,run_id=world.RUN_ID,code_commit=m.REFERENCE_CODE,config=r,
            all_three_DURA_conditions_complete=True,source_states_unchanged=True,frozen_parameters_buffers_unchanged=True,
            classification_unchanged=True,source_models_frozen_before_V=True,all_thresholds_frozen_before_novel_and_Q=True,private_truth_reads=0,
            model_identities=dict(E0=pm['model_identity'],SOURCE=control['identity'],TARGET_PCD=done['identity']))
        (out/'complete.json').write_text(json.dumps(meta),encoding='utf-8')
        c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=r,output_root=str(root/'momentum_train'),steps=2,lr=.0001)
        yield m,c,pm,b,events


def test_toy_two_arms_match_source_and_u_then_strict_load_without_aux_or_u(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events):
        old_cache=Path(c['reference_config']['train_config']['output_root'])/'U_features.npz'
        with np.load(old_cache,allow_pickle=False) as z:cache={k:z[k].copy() for k in ('ids','views','parent_z')}
        np.savez(old_cache,**cache,neighbors=np.asarray([dict(forbidden_to_consume=True)],dtype=object))
        original_load=torch.load
        def no_target_initialization(path,*args,**kwargs):
            assert Path(path)!=old_cache.parent/'final.pt'
            return original_load(path,*args,**kwargs)
        monkeypatch.setattr(torch,'load',no_target_initialization)
        initial=m.body._snapshot(b['model']);done=m.train(c,'cpu');root=Path(c['output_root'])
        assert done['total_optimizer_updates']==4 and done['updates_per_arm']==2 and done['target_U_reads']==16
        assert done['source_control_source_only'] and done['source_sampling_matched'] and done['U_sampling_matched']
        assert done['real_parent_signal_forward_rows']==sum(m.COUNTS.values()) and done['G_forward_rows']==done['real_F_forward_rows']==0
        traces={a:[json.loads(x) for x in (root/a/'steps.jsonl').read_text().splitlines()] for a in m.ARMS}
        source_path=Path(c['reference_config']['train_config']['reference_config']['train_config']['output_root'])/'SOURCE/steps.jsonl'
        control=[json.loads(x) for x in source_path.read_text().splitlines()]
        for row,old in zip(traces[m.ARMS[0]],control):assert all(row[k]==old[k] for k in ('L_indices','OE_indices'))
        for left,right in zip(traces[m.ARMS[0]],traces[m.ARMS[1]]):
            assert all(left[k]==right[k] for k in ('L_indices','OE_indices','U_indices'))
            assert left['teacher_gradients_none'] and right['teacher_gradients_none']
            assert len(left['teacher_slot_mass'])==len(left['center_before'])==len(left['center_after'])==16
            assert left['teacher_parameter_updates']==0 and right['teacher_parameter_updates']>0
            assert left['gradient_norms']['body']>0 and right['gradient_norms']['prototype']>0
        fixed=torch.load(root/m.ARMS[0]/'aux_teacher.pt',weights_only=False)
        ema=torch.load(root/m.ARMS[1]/'aux_teacher.pt',weights_only=False)
        assert all(torch.equal(v,fixed['network'][k]) for k,v in initial.items())
        assert any(not torch.equal(v,ema['network'][k]) for k,v in initial.items() if k.startswith('projection.'))
        for k,v in initial.items():
            if k.startswith(m.body.signalcore.SINC_PREFIX):assert torch.equal(v,ema['network'][k])
        original_np_load=np.load;original_torch_load=torch.load
        def no_u(path,*args,**kwargs):
            assert Path(path).name!='U_features.npz';return original_np_load(path,*args,**kwargs)
        def no_aux(path,*args,**kwargs):
            assert Path(path).name!='aux_teacher.pt';return original_torch_load(path,*args,**kwargs)
        with monkeypatch.context() as scoped:
            scoped.setattr(np,'load',no_u);scoped.setattr(torch,'load',no_aux)
            model,i=m.load_trained_model(c,m.ARMS[1],'cpu')
        assert i==done['identities'][m.ARMS[1]] and i['temporary_head_parameters']==2560
        assert all(not p.requires_grad for p in model.parameters()) and not i['source_only']
        other=root/m.ARMS[0]/'complete.json';saved=other.read_bytes();row=json.loads(saved);row['status']='PARTIAL';other.write_text(json.dumps(row))
        with pytest.raises(ValueError):m.load_trained_model(c,m.ARMS[1],'cpu')
        other.write_bytes(saved)
        cp=root/m.ARMS[1]/'final.pt';payload=torch.load(cp,weights_only=False)
        sinc=next(k for k in payload['network'] if k.startswith(m.body.signalcore.SINC_PREFIX));payload['network'][sinc]+=1;torch.save(payload,cp)
        with pytest.raises(ValueError,match='Sinc'):m.load_trained_model(c,m.ARMS[1],'cpu')


def test_private_config_and_wrong_source_control_never_open_u_or_publish(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events):
        for update in (dict(Q_public_root='forbidden'),dict(truth='forbidden'),dict(steps=1),dict(reference_config=dict(c['reference_config'],code_commit='b'*40))):
            with pytest.raises(ValueError):m.validate_config(dict(c,**update))
        path=Path(c['reference_config']['train_config']['reference_config']['train_config']['output_root'])/'SOURCE/complete.json'
        row=json.loads(path.read_text());row['source_only']=False;path.write_text(json.dumps(row))
        monkeypatch.setattr(m,'_load_U',lambda *a,**k:pytest.fail('U must not open for target-contaminated SOURCE'))
        with pytest.raises(ValueError):m.train(c,'cpu')
        assert not Path(c['output_root'],'complete.json').exists()


def test_failed_smoke_is_first_and_does_not_read_source_or_u(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,events):
        events.clear();monkeypatch.setattr(m.body.parent,'_smoke',lambda *a:dict(status='FAIL'))
        monkeypatch.setattr(m.body,'_source_cache',lambda *a:pytest.fail('No source arrays before smoke'))
        monkeypatch.setattr(m,'_load_U',lambda *a:pytest.fail('No U arrays before smoke'))
        with pytest.raises(ValueError,match='smoke'):m.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root']).exists()
