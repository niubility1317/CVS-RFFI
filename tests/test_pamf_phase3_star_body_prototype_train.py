"""Synthetic live-IQ body adaptation; no real checkpoint or observations."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_body_prototype_train')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_target_prototype_train import fixture as previous
    with previous(root,mp) as (_,c,pm,b,u,events,_):
        t=api();mp.setattr(t,'STEPS',2);mp.setattr(t,'U_COUNT',16);mp.setattr(t,'U_BATCH',4);mp.setattr(t,'COUNTS',dict(t.base.COUNTS))
        c['run_id']=t.RUN_ID;created=[];student=t._student
        def independent(parent):
            child=student(parent);created.append(child)
            assert not {p.data_ptr() for p in parent.parameters()}&{p.data_ptr() for p in child.parameters()}
            return child
        mp.setattr(t,'_student',independent)
        yield t,c,pm,b,u,events,created


def test_live_IQ_gradients_leave_Sinc_and_buffers_frozen_without_step():
    torch.set_num_threads(1)
    t=api();from test_pamf_phase3_star_signal_open_model import return_aux_source
    native,_=return_aux_source();parent=t.signalcore.SignalOpen(t.signalcore.HeadlessSignal(native.id_backbone.encoder))
    with torch.no_grad():parent.projection.network[-1].weight.normal_(0,.003)
    parent.eval().requires_grad_(False);student=t._student(parent);before=t._snapshot(student)
    assert sum(p.numel() for p in student.parameters() if p.requires_grad)==334673
    sinc=[(n,p) for n,p in student.named_parameters() if n.startswith(t.signalcore.SINC_PREFIX)]
    assert len(sinc)==2 and all(p.shape==(24,1) and not p.requires_grad for n,p in sinc)
    result=t._gradient_smoke(student,'cpu');assert result['status']=='PASS' and result['optimizer_updates']==0
    assert result['gradient_norms']['body']>0 and result['gradient_norms']['hidden']>0 and result['gradient_norms']['last']>0 and result['gradient_norms']['prototype']>0
    assert result['Sinc_gradients_none'] and all(p.grad is None for p in student.parameters())
    assert all(torch.equal(v,student.state_dict()[k].detach().cpu()) for k,v in before.items())
    assert not any(m.training for m in student.modules()) and all(not p.requires_grad for p in parent.parameters())


def test_tiny_actual_IQ_body_updates_matched_source_and_strict_reload(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,u,events,created):
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('NumPy ABI forbidden'));monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('NumPy ABI forbidden'))
        monkeypatch.setattr(t.signalcore,'optimizer',lambda *a,**k:pytest.fail('Sinc CENTER optimizer forbidden'))
        monkeypatch.setattr(t.signalcore,'clip',lambda *a,**k:pytest.fail('Physical clip forbidden'))
        initial=t._snapshot(b['model']);done=t.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','OE','U'] and len(created)==2
        assert done['all_student_Sinc_unchanged'] and done['all_student_bodies_changed'] and done['all_student_projections_changed']
        assert done['all_student_buffers_unchanged'] and done['original_models_unchanged'] and done['SOURCE_completed_before_U_open']
        assert done['real_parent_signal_forward_rows']==sum(t.COUNTS.values())+16
        assert done['real_student_signal_forward_rows']==2*256*2+(16+2*4)*2 and done['G_forward_rows']==16 and done['real_F_forward_rows']==0
        traces={arm:[json.loads(line) for line in Path(out/arm/'steps.jsonl').read_text().splitlines()] for arm in t.ARMS}
        for left,right in zip(traces['SOURCE'],traces['TARGET_PCD']):
            assert left['L_indices']==right['L_indices'] and left['OE_indices']==right['OE_indices']
            assert not left['U_indices'] and len(set(right['U_indices']))==4
            assert left['gradient_norms']['body']>0 and right['gradient_norms']['hidden']>0 and right['gradient_norms']['prototype']>0
            assert left['ordinary_clip_calls']==left['step'] and left['Sinc_optimizer_steps']==left['physical_clip_calls']==0
        with np.load(out/'U_features.npz') as a:
            assert set(a.files)=={'ids','views','parent_z','neighbors'} and a['views'].shape==(3,16,2,256) and a['parent_z'].shape==(16,160)
            assert not np.array_equal(a['views'][1],a['views'][2]) and not np.any(a['neighbors']==np.arange(16))
        for arm in t.ARMS:
            row=done['rows'][arm];i=row['identity'];assert i['signal_feature_cumulative_updates']==13322 and i['projection_cumulative_updates']==2222
            assert not i['encoder_frozen'] and i['Sinc_frozen'] and i['input_path']=='cached_P1_IQ_only' and i['trainable_parameters']==334673
            assert row['student_encoder_unchanged'] is False and row['student_Sinc_unchanged'] and row['student_body_changed'] and row['student_projection_changed']
            loaded,identity=t.load_trained_model(c,arm,'cpu');assert identity==i and all(not p.requires_grad for p in loaded.parameters())
            assert not any(m.training for m in loaded.modules())
            assert any(not torch.equal(initial[k],loaded.state_dict()[k].detach().cpu()) for k in initial if k.startswith('signal.') and not k.startswith(t.signalcore.SINC_PREFIX))
            assert all(torch.equal(initial[k],loaded.state_dict()[k].detach().cpu()) for k in initial if k.startswith(t.signalcore.SINC_PREFIX))
        cp=out/'TARGET_PCD/final.pt';payload=torch.load(cp,weights_only=False);key=next(k for k in payload['network'] if k.startswith(t.signalcore.SINC_PREFIX))
        payload['network'][key]+=1;torch.save(payload,cp)
        with pytest.raises(ValueError,match='Sinc'):t.load_trained_model(c,'TARGET_PCD','cpu')


def test_config_or_failed_probe_never_open_source_or_U(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,u,events,created):
        for delta in (dict(truth='private'),dict(Q_public_root='forbidden'),dict(steps=1),dict(U_capsule_id='clean'),dict(output_root=c['reference_config']['train_config']['IQ_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):t.validate_config(dict(c,**delta))
        monkeypatch.setattr(t,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('body probe failure')))
        with pytest.raises(ValueError,match='body probe failure'):t.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root']).exists()


def test_failed_U_keeps_source_but_never_publishes_partial_global(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,u,events,created):
        def fail(config):
            assert Path(c['output_root'],'SOURCE/complete.json').is_file();raise ValueError('U failure')
        monkeypatch.setattr(t.ssl,'_public',fail)
        with pytest.raises(ValueError,match='U failure'):t.train(c,'cpu')
        assert Path(c['output_root'],'SOURCE/final.pt').is_file() and not Path(c['output_root'],'complete.json').exists()
        assert not Path(c['output_root'],'TARGET_PCD/final.pt').exists()
        with pytest.raises(ValueError):t.load_trained_model(c,'SOURCE','cpu')


def test_reload_omits_nonpersistent_cache_but_rejects_saved_buffer_and_Sinc_changes(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (t,c,pm,b,u,events,created):
        done=t.train(c,'cpu');original_loader=t.base.load_trained_model
        def rebuilt_parent(*args,**kwargs):
            model,identity=original_loader(*args,**kwargs)
            model.register_buffer('runtime_feature_cache',torch.tensor([3.]),persistent=False)
            return model,identity
        monkeypatch.setattr(t.base,'load_trained_model',rebuilt_parent)
        loaded,identity=t.load_trained_model(c,'TARGET_PCD','cpu')
        assert identity==done['identities']['TARGET_PCD']
        assert 'runtime_feature_cache' in dict(loaded.named_buffers()) and 'runtime_feature_cache' not in loaded.state_dict()
        path=Path(c['output_root'])/'TARGET_PCD/final.pt';saved=path.read_bytes();payload=torch.load(path,weights_only=False)
        buffer_key=next(k for k in dict(loaded.named_buffers()) if k in loaded.state_dict() and not k.startswith(t.signalcore.SINC_PREFIX))
        sinc_key=next(k for k in dict(loaded.named_parameters()) if k.startswith(t.signalcore.SINC_PREFIX))
        for key in (buffer_key,sinc_key):
            with subtests.test(frozen_key=key):
                payload['network'][key]+=1;torch.save(payload,path)
                with pytest.raises(ValueError,match='Frozen source Sinc/buffers'):t.load_trained_model(c,'TARGET_PCD','cpu')
                payload['network'][key]-=1;path.write_bytes(saved)
