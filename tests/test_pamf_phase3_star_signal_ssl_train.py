"""Source replay then star-U two-view learning with a real synthetic r03 CP."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_ssl_train')


def test_exact_configuration_cannot_accept_truth_query_or_unknown_count():
    m=api()
    assert m.TRAIN_FIELDS=={'run_id','code_commit','source_train_config','reference_prediction_root','U_public_root','U_capsule_id','U_split_id','output_root','steps','lr'}
    for extra in ('truth','Q_public_root','K'):
        with pytest.raises(ValueError):m.validate_config({extra:'forbidden'})


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_mixture_uda_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (mix,mc,pm,backend,u,events,seen,public):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'U_COUNT',16);mp.setattr(m,'U_BATCH',4);mp.setattr(m,'COUNTS',dict(m.base.COUNTS))
        c=dict(run_id=m.RUN_ID,code_commit='f'*40,source_train_config=deepcopy(pm['config']['train_config']),reference_prediction_root=mc['reference_prediction_root'],
            U_public_root=mc['U_public_root'],U_capsule_id=m.U_CAPSULE_ID,U_split_id=m.U_SPLIT_ID,output_root=str(root/'ssl_train'),steps=2,lr=.0001)
        role_loader=m.base.load_role
        def role(path,name):
            assert name in ('L','OE') and (Path(c['output_root'])/'smoke.json').is_file()
            assert 'U' not in events;events.append(name);return role_loader(path,name)
        mp.setattr(m.base,'load_role',role)
        def U(config):
            assert (Path(c['output_root'])/'SOURCE/complete.json').is_file()
            assert not (Path(c['output_root'])/'TARGET_SSL/final.pt').exists()
            events.append('U');return u
        mp.setattr(m,'_public',U)
        created=[];student=m._student
        def clone(parent):
            child=student(parent);created.append(child)
            assert m._snapshot(child).keys()==m._snapshot(parent).keys()
            assert all(torch.equal(v,m._snapshot(parent)[k]) for k,v in m._snapshot(child).items())
            assert not {p.data_ptr() for p in parent.parameters()}&{p.data_ptr() for p in child.parameters()}
            return child
        mp.setattr(m,'_student',clone)
        yield m,c,pm,backend,u,events,created


def test_two_steps_matched_source_isolated_U_rng_live_signal_gradients_and_strict_reload(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,u,events,created):
        original=m.views.phase_gain_view;draws=[]
        def view(x,rng):
            value=original(x,rng);draws.append(value.detach().clone());return value
        monkeypatch.setattr(m.views,'phase_gain_view',view)
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No ndarray ABI bridge'))
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('No ndarray ABI bridge'))
        done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','OE','U']
        assert done['original_models_unchanged'] and done['G_forward_rows']==16 and done['real_F_forward_rows']==0
        assert done['real_student_signal_forward_rows']==2*256*2+2*4*2
        assert len(created)==2 and all(not module.training for student in created for module in student.modules())
        assert all(not torch.equal(draws[i],draws[i+1]) for i in range(0,len(draws),2))
        smoke=m._read(out/'smoke.json')['gradient']
        for key in ('sinc','middle','last'):assert smoke['U_gradient_norms'][key]>0
        assert smoke['optimizer_updates']==0 and smoke['parameters_buffers_unchanged']
        traces={arm:[json.loads(line) for line in (out/arm/'steps.jsonl').read_text().splitlines()] for arm in m.ARMS}
        for left,right in zip(traces['SOURCE'],traces['TARGET_SSL']):
            for key in ('L_indices','OE_indices'):assert left[key]==right[key]
            assert left['U_indices']==[] and right['source_indices_equal']
            assert len(right['U_indices'])==len(set(right['U_indices']))==4
            assert right['U_SSL_weight']==1 and left['U_SSL_weight']==0
            for row in (left,right):assert row['gradient_norms']['sinc']>0 and row['gradient_norms']['last']>0
        for arm in m.ARMS:
            row=m._read(out/arm/'complete.json');identity=row['identity']
            assert row['source_only']==(arm=='SOURCE') and identity['target_U_used']==(arm=='TARGET_SSL')
            assert row['target_U_reads']==(0 if arm=='SOURCE' else 16)
            assert row['optimizer']['sinc_steps']==row['optimizer']['physical_clip_calls']==2
            assert row['model_buffers_unchanged'] and identity['parent_signal_identity']==pm['model_identity']
            model,loaded=m.load_trained_model(c,arm,'cpu');assert loaded==identity and not model.training and all(not p.requires_grad for p in model.parameters())
            assert all('cls_head.weight' not in key for key in model.state_dict())
        with np.load(out/'U_restored.npz') as z:assert set(z.files)=={'ids','restored'} and z['restored'].shape==(16,2,256)
        cp=out/'TARGET_SSL/final.pt';payload=torch.load(cp,weights_only=False);assert 'optimizer' not in payload and 'U_restored' not in payload
        payload['network'].pop('projection.network.2.bias');torch.save(payload,cp)
        with pytest.raises(ValueError,match='Strict'):m.load_trained_model(c,'TARGET_SSL','cpu')
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_invalid_roles_and_failed_gradient_smoke_never_open_source_or_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,u,events,created):
        for delta in (dict(truth='private'),dict(steps=1),dict(lr=.001),dict(U_capsule_id='clean'),dict(U_public_root=str(tmp_path/'evaluation/public')),dict(output_root=c['source_train_config']['IQ_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        monkeypatch.setattr(m,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('synthetic gradient failure')))
        with pytest.raises(ValueError,match='gradient failure'):m.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root'],'complete.json').exists()


def test_U_failure_preserves_SOURCE_but_does_not_complete_or_allow_partial_loader(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,pm,b,u,events,created):
        def failure(config):
            assert Path(c['output_root'],'SOURCE/complete.json').is_file();raise ValueError('synthetic U open failure')
        monkeypatch.setattr(m,'_public',failure)
        with pytest.raises(ValueError,match='U open failure'):m.train(c,'cpu')
        assert Path(c['output_root'],'SOURCE/final.pt').is_file() and not Path(c['output_root'],'complete.json').is_file()
        assert not Path(c['output_root'],'TARGET_SSL/final.pt').is_file()
        with pytest.raises(ValueError):m.load_trained_model(c,'SOURCE','cpu')
