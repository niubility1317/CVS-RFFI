"""Fixed-density source learning, same-pass gradients and strict checkpoints."""
from contextlib import contextmanager
from copy import deepcopy
import importlib
import json
from pathlib import Path
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_density_boundary_train')


def test_exact_source_only_fields_cannot_accept_target_inputs():
    m=api();assert m.FIELDS=={'run_id','code_commit','reference_config','output_root','steps','lr'}
    for key in ('U_public_root','Q_truth','K'):
        with pytest.raises(ValueError):m.validate_config({key:'forbidden'})


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_local_cov_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (local,lc,counts,_,public,backend):
        m=api();mp.setattr(m,'STEPS',2);mp.setattr(m,'COUNTS',dict(m.base.COUNTS))
        c=dict(run_id=m.RUN_ID,code_commit='d'*40,reference_config=deepcopy(lc['reference_config']),output_root=str(root/'boundary_train'),steps=2,lr=.0001)
        events=[];created=[];load_backend=m._backend;load_role=m.base.load_role;student=m._student
        def load(*args):events.append('checkpoint');return load_backend(*args)
        def role(path,name):
            assert name in ('L','OE') and Path(c['output_root'],'smoke.json').is_file();events.append(name);return load_role(path,name)
        def clone(parent):
            child=student(parent);created.append(child)
            assert not {p.data_ptr() for p in parent.parameters()}&{p.data_ptr() for p in child.parameters()}
            assert all(torch.equal(v,child.state_dict()[k].cpu()) for k,v in m._snapshot(parent).items())
            return child
        mp.setattr(m,'_backend',load);mp.setattr(m.base,'load_role',role);mp.setattr(m,'_student',clone)
        yield m,c,backend,events,created


def test_two_steps_same_source_frozen_teachers_live_BCE_and_strict_model_reload(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (m,c,b,events,created):
        monkeypatch.setattr(torch.Tensor,'numpy',lambda *a:pytest.fail('No NumPy ABI bridge'))
        monkeypatch.setattr(torch,'from_numpy',lambda *a:pytest.fail('No NumPy ABI bridge'))
        load=np.load;read=m._read
        def source_npz(path,*a,**kw):
            assert Path(path).name not in ('thresholds.npz','known_predictions.npz','novel_predictions.npz','predictions.npz')
            return load(path,*a,**kw)
        def source_json(path):
            assert 'calibration' not in Path(path).parts;return read(path)
        monkeypatch.setattr(np,'load',source_npz);monkeypatch.setattr(m,'_read',source_json)
        done=m.train(c,'cpu');out=Path(c['output_root'])
        assert events==['checkpoint','L','OE'] and len(created)==2
        assert done['source_only'] and done['target_U_reads']==0 and done['fixed_density_state_unchanged'] and done['point_teacher_unchanged']
        assert done['real_student_signal_forward_rows']==1024 and done['real_parent_signal_forward_rows']==done['F_head_only_rows']==450
        assert done['real_F_forward_rows']==done['G_forward_rows']==0
        smoke=m._read(out/'smoke.json')['gradient']
        for key in ('sinc','middle','last'):assert smoke['BCE_gradient_norms'][key]>0
        assert smoke['optimizer_updates']==0 and smoke['parameters_buffers_unchanged']
        traces={arm:[json.loads(v) for v in (out/arm/'steps.jsonl').read_text().splitlines()] for arm in m.ARMS}
        for left,right in zip(traces['REPLAY'],traces['BOUNDARY']):
            for key in ('L_indices','OE_indices'):assert left[key]==right[key]
            assert left['boundary_weight']==0 and right['boundary_weight']==1
            for row in (left,right):
                assert row['source_indices_equal'] and row['point_anchor_weight']==1
                np.testing.assert_allclose(row['point_anchor'],.5*(row['anchor_L']+row['anchor_OE']))
                np.testing.assert_allclose(row['loss_total'],row['source_loss']+row['point_anchor']+row['boundary_weight']*row['boundary_loss'],rtol=1e-6)
                assert row['gradient_norms']['sinc']>0 and row['gradient_norms']['last']>0
        assert traces['REPLAY'][0]['point_anchor']<1e-10 and traces['BOUNDARY'][-1]['point_anchor']>0
        with np.load(out/'reference_signal.npz') as z:
            assert set(z.files)=={'L_ids','L_z','L_closed','OE_ids','OE_z','OE_closed'}
            for role in ('L','OE'):
                teacher=m.base.parent.load_role(c['reference_config']['train_config']['feature_root'],role)
                with torch.no_grad():expected=b['reader'].classify_features(torch.tensor(teacher['u'].tolist())).argmax(1)
                np.testing.assert_array_equal(z[role+'_closed'],expected.tolist());np.testing.assert_array_equal(z[role+'_ids'],teacher['ids'])
                if role=='L':assert np.any(z['L_closed']!=teacher['metadata']['y'])
        (out/'reference_signal.npz').unlink()  # private teacher rows are not a deployment requirement
        for arm in m.ARMS:
            model,identity=m.load_trained_model(c,arm,'cpu');row=done['rows'][arm]
            assert identity==row['identity'] and identity['source_only'] and not identity['target_U_used']
            assert identity['signal_feature_cumulative_updates']==13322 and identity['projection_cumulative_updates']==2222
            assert not model.training and all(not p.requires_grad for p in model.parameters())
            assert row['optimizer']['sinc_steps']==row['optimizer']['physical_clip_calls']==2
        cp=out/'BOUNDARY/final.pt';value=torch.load(cp,weights_only=False)
        assert not {'optimizer','reference_signal','teacher_bank'}&set(value)
        value['network'].pop('projection.network.2.bias');torch.save(value,cp)
        with pytest.raises(ValueError,match='Strict'):m.load_trained_model(c,'BOUNDARY','cpu')
        with pytest.raises(FileExistsError):m.train(c,'cpu')


def test_roles_config_and_failed_smoke_do_not_open_real_source(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,b,events,created):
        for delta in (dict(U_public_root='forbidden'),dict(steps=1),dict(lr=.001),dict(output_root=c['reference_config']['train_config']['IQ_root'])):
            with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))
        monkeypatch.setattr(m,'_gradient_smoke',lambda *a:(_ for _ in ()).throw(ValueError('synthetic gradient failure')))
        with pytest.raises(ValueError,match='gradient failure'):m.train(c,'cpu')
        assert events==['checkpoint'] and not Path(c['output_root'],'complete.json').is_file()


def test_private_OE_labels_and_partial_second_arm_cannot_complete_or_load(tmp_path,monkeypatch,subtests):
    with fixture(tmp_path,monkeypatch) as (m,c,b,events,created):
        loader=m.base.load_role
        def bad(path,role):
            a=deepcopy(loader(path,role))
            if role=='OE':a['metadata']['y']=np.zeros(len(a['ids']),np.int64)
            return a
        with monkeypatch.context() as mp:
            mp.setattr(m.base,'load_role',bad)
            with pytest.raises(ValueError):m.train(c,'cpu')
        cc=dict(c,output_root=str(tmp_path/'partial'));events.clear();loop=m._loop
        def failure(*args,**kwargs):
            if args[4]=='BOUNDARY':raise ValueError('synthetic second arm failure')
            return loop(*args,**kwargs)
        monkeypatch.setattr(m,'_loop',failure)
        with pytest.raises(ValueError,match='second arm'):m.train(cc,'cpu')
        assert Path(cc['output_root'],'REPLAY/final.pt').is_file() and not Path(cc['output_root'],'complete.json').is_file()
        with pytest.raises(ValueError):m.load_trained_model(cc,'REPLAY','cpu')
