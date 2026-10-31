"""Tiny actual parent checkpoint -> frozen-body projection learning and reload."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_target_prototype_train')


@contextmanager
def fixture(root,mp):
    # The historical toy CP helper has a zero-last initial projector. Give its
    # synthetic r03 endpoint learned weights before any ancestral predictions.
    import test_pamf_phase3_star_signal_open_train as source_test
    original_checkpoint=source_test.frozen_checkpoint
    def trained_checkpoint(c,m):
        model,identity=original_checkpoint(c,m);generator=torch.Generator().manual_seed(812)
        with torch.no_grad():model.projection.network[-1].weight.copy_(.003*torch.randn(model.projection.network[-1].weight.shape,generator=generator))
        path=Path(c['output_root'])/'final.pt';payload=torch.load(path,weights_only=False);payload['network']={k:v.detach().clone() for k,v in model.state_dict().items()};torch.save(payload,path)
        return model,identity
    mp.setattr(source_test,'frozen_checkpoint',trained_checkpoint)
    from test_pamf_phase3_star_class_density_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (parent,pc,_,backend,_,_,_):
        t=api();pc['code_commit']=t.PARENT_CODE;pm=parent.run(pc,'cpu');mp.setattr(t,'STEPS',2);mp.setattr(t,'U_COUNT',16);mp.setattr(t,'U_BATCH',4)
        mp.setattr(t.ssl,'U_COUNT',16);mp.setattr(t,'COUNTS',dict(t.base.COUNTS))
        c=dict(run_id=t.RUN_ID,code_commit='d'*40,reference_config=pc,U_public_root=str(root/'star/discovery/public'),U_capsule_id=t.U_CAPSULE_ID,
            U_split_id=t.U_SPLIT_ID,output_root=str(root/'prototype_train'),steps=2,lr=.0001)
        events=[];load=t.parent._backend
        def checkpoint(*a):events.append('checkpoint');return load(*a)
        mp.setattr(t.parent,'_backend',checkpoint);role=t.base.load_role
        def source_role(path,name):
            assert Path(c['output_root'],'smoke.json').is_file() and 'U' not in events;events.append(name);return role(path,name)
        mp.setattr(t.base,'load_role',source_role)
        rng=np.random.default_rng(73);u=(dict(count=16,capsule_id=t.U_CAPSULE_ID,split_id=t.U_SPLIT_ID,registered_classes=t.base.native.CLASSES),
            np.asarray(['U-'+str(i) for i in range(16)]),rng.normal(size=(16,2,256)).astype(np.float32))
        def public(config):
            assert Path(c['output_root'],'SOURCE/complete.json').is_file() and not Path(c['output_root'],'TARGET_PCD/final.pt').exists()
            events.append('U');return u
        mp.setattr(t.ssl,'_public',public);created=[];student=t._student
        def independent(parent):
            child=student(parent);created.append(child)
            assert not {p.data_ptr() for p in parent.parameters()}&{p.data_ptr() for p in child.parameters()}
            return child
        mp.setattr(t,'_student',independent)
        yield t,c,pm,backend,u,events,created


def test_tiny_projection_only_training_matches_source_then_U_and_strict_reload(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,u,events,created):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No NumPy ABI bridge'));monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('No NumPy ABI bridge'))
        before=t._snapshot(b['model']) if b else None;done=t.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','OE','U'] and len(created)==2
        assert done['all_student_signal_encoders_unchanged'] and done['all_student_buffers_unchanged'] and done['original_models_unchanged']
        assert done['real_parent_signal_forward_rows']==sum(t.COUNTS.values())+3*16 and done['G_forward_rows']==16 and done['real_F_forward_rows']==0
        assert done['projection_forward_rows']==2*256*2+(16+2*4)*2
        assert not Path(out/'SOURCE/prototype_state.npz').exists() and Path(out/'TARGET_PCD/prototype_state.npz').is_file()
        smoke=t._read(out/'smoke.json')['gradient'];assert all(v>0 for v in smoke['gradient_norms'].values()) and smoke['optimizer_updates']==0
        traces={arm:[json.loads(line) for line in Path(out/arm/'steps.jsonl').read_text().splitlines()] for arm in t.ARMS}
        for left,right in zip(traces['SOURCE'],traces['TARGET_PCD']):
            assert left['L_indices']==right['L_indices'] and left['OE_indices']==right['OE_indices']
            assert left['U_indices']==[] and len(set(right['U_indices']))==4
            assert right['U_PCD_weight']==.1 and left['U_PCD_weight']==0
            assert left['gradient_norms']['hidden']>0 and right['gradient_norms']['last']>0 and right['gradient_norms']['prototype']>0
        with np.load(out/'U_features.npz') as z:
            assert set(z.files)=={'ids','u','c','parent_z','neighbors'} and z['u'].shape==(3,16,160) and z['c'].shape==(3,16,640)
            assert not np.array_equal(z['u'][1],z['u'][2]) and not np.any(z['neighbors']==np.arange(16))
        for arm in t.ARMS:
            identity=done['identities'][arm];assert identity['signal_feature_cumulative_updates']==13320 and identity['projection_cumulative_updates']==2222
            assert identity['encoder_frozen'] and identity['source_only']==(arm=='SOURCE') and identity['prototype_deployed'] is False
            loaded,i=t.load_trained_model(c,arm,'cpu');assert i==identity and all(not p.requires_grad for p in loaded.parameters())
            assert sum(p.numel() for p in loaded.parameters())==334721 and all(not m.training for m in loaded.modules())
            assert all(torch.equal(v,loaded.signal.state_dict()[k].detach().cpu()) for k,v in created[0].signal.state_dict().items())
        cp=out/'TARGET_PCD/final.pt';payload=torch.load(cp,weights_only=False);key=next(k for k in payload['network'] if k.startswith('signal.') and payload['network'][k].is_floating_point())
        payload['network'][key]+=1;torch.save(payload,cp)
        with pytest.raises(ValueError,match='encoder'):t.load_trained_model(c,'TARGET_PCD','cpu')


def test_private_inputs_or_failed_probe_never_open_real_source_or_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,u,events,created):
        for delta in (dict(truth='private'),dict(Q_public_root='forbidden'),dict(steps=1),dict(U_capsule_id='clean'),dict(output_root=c['reference_config']['train_config']['IQ_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_config(dict(c,**delta))
        monkeypatch.setattr(t,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('projection probe failure')))
        with pytest.raises(ValueError,match='probe failure'):t.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root']).exists()


def test_failed_U_preserves_SOURCE_but_no_partial_loader_or_target_checkpoint(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,u,events,created):
        def fail(config):
            assert Path(c['output_root'],'SOURCE/complete.json').is_file();raise ValueError('U read failed')
        monkeypatch.setattr(t.ssl,'_public',fail)
        with pytest.raises(ValueError,match='U read failed'):t.train(c,'cpu')
        assert Path(c['output_root'],'SOURCE/final.pt').is_file() and not Path(c['output_root'],'complete.json').exists()
        assert not Path(c['output_root'],'TARGET_PCD/final.pt').exists()
        with pytest.raises(ValueError):t.load_trained_model(c,'SOURCE','cpu')
