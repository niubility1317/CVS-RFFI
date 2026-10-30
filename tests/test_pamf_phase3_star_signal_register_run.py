"""Owned outputs and source-radius -> unlabeled U -> V -> future-Q boundary."""
from contextlib import contextmanager
from pathlib import Path
import importlib
import numpy as np
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_signal_register_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_class_density_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (parent,pc,_,backend,_,_,_):
        m=api();pc['code_commit']=m.PARENT_CODE;parent.run(pc,'cpu')
        counts=dict(parent.COUNTS,L=6300);mp.setattr(m,'COUNTS',counts);mp.setattr(m,'Q_COUNT',parent.Q_COUNT);mp.setattr(m,'U_COUNT',48)
        c=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=pc,output_root=str(root/'signal_register'),batch_size=128,
               U_public_root=str(root/'star/discovery/public'),U_capsule_id=m.U_reader.U_CAPSULE_ID,U_split_id=m.U_reader.U_SPLIT_ID)
        rng=np.random.default_rng(631);events=[];load=m.parent._backend;radius=m.core.source_radius;discover=m.core.discover
        rows=m._rows;target=m._target_rows;calibrate=m.core.calibrate
        def checkpoint(*a):events.append('checkpoint');return load(*a)
        def source_L(*a):
            assert events==['checkpoint'] and Path(c['output_root'],'smoke.json').is_file();events.append('L')
            state=m._reference(c)[1];centers=state['known_means']@np.linalg.inv(state['whitener'])+state['origin']
            values=np.repeat(centers,70,axis=0)+rng.normal(scale=.001,size=(6300,160));values/=np.linalg.norm(values,axis=1,keepdims=True)
            groups=[(y,r,d) for y in range(6) for r in (1,3,4,6,8) for d in (1,2,3)]
            meta={k:np.repeat(np.asarray([g[j] for g in groups],np.int64),70) for j,k in enumerate(('y','rx','day'))}
            return values.astype(np.float32),meta,np.asarray(['sourceL-'+str(i) for i in range(6300)]),dict(role='L',count=6300)
        def source_radius(*a):assert events==['checkpoint','L'];events.append('radius');return radius(*a)
        U_public=(dict(count=48,capsule_id=c['U_capsule_id'],split_id=c['U_split_id'],registered_classes=parent.signal_train.native.CLASSES),
                  np.asarray(['U-'+str(i).zfill(3) for i in range(48)]),rng.normal(size=(48,2,256)).astype(np.float32))
        def public(*a):
            assert events==['checkpoint','L','radius'] and Path(c['output_root'],'models/complete.json').is_file()
            events.append('U');return U_public
        def discover_U(*a):assert events==['checkpoint','L','radius','U'];events.append('discover');return discover(*a)
        def source_rows(path,role):
            if not Path(c['output_root'],'complete.json').exists():
                if role=='known':assert events==['checkpoint','L','radius','U','discover'] and Path(c['output_root'],'registration/complete.json').is_file()
                else:assert Path(c['output_root'],'calibration/complete.json').is_file()
                events.append(role)
            return rows(path,role)
        def calibration(*a):
            assert events[-1]=='known' and Path(c['output_root'],'registration/complete.json').is_file()
            return calibrate(*a)
        def query_rows(path,name):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'novel_predictions.npz').is_file();events.append(name)
            return target(path,name)
        mp.setattr(m.parent,'_backend',checkpoint);mp.setattr(m,'_source_L',source_L);mp.setattr(m.core,'source_radius',source_radius)
        mp.setattr(m.U_reader,'_public',public);mp.setattr(m.core,'discover',discover_U);mp.setattr(m.core,'calibrate',calibration)
        mp.setattr(m,'_rows',source_rows);mp.setattr(m,'_target_rows',query_rows)
        mp.setattr(m.parent,'_public',lambda *a:pytest.fail('Q uses frozen cached predictions, never opens query IQ'))
        yield m,c,counts,events,backend,U_public


def test_config_rejects_truth_K_clean_U_missing_DURA_and_output_overlap(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    p=dict(code_commit=m.PARENT_CODE,output_root=str(tmp_path/'parent'),reference_prediction_root=str(tmp_path/'ref'),
           train_config=dict(IQ_root=str(tmp_path/'IQ'),output_root=str(tmp_path/'train')),dura_release=str(tmp_path/'G'),dura_checkpoint=str(tmp_path/'G.pt'),
           Q_conditions={name:dict(public_root=str(tmp_path/name/'evaluation/public')) for name in m.CONDITIONS})
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=p,output_root=str(tmp_path/'new'),batch_size=128,
           U_public_root=str(tmp_path/'star/discovery/public'),U_capsule_id=m.U_reader.U_CAPSULE_ID,U_split_id=m.U_reader.U_SPLIT_ID)
    assert m.validate_config(c) is c
    for change in (dict(U_truth='private'),dict(true_K=2),dict(U_capsule_id='clean'),dict(U_public_root=p['Q_conditions']['clean_DURA']['public_root']),
                   dict(batch_size=True),dict(output_root=p['dura_release']),dict(output_root=p['train_config']['IQ_root']),dict(reference_config=dict(p,code_commit='b'*40))):
        with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))


def test_existing_output_never_loads_models_or_registers_twice(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'validate_config',lambda c:c)
    monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('Existing output must not load or fit'))
    p=tmp_path/'sentinel';p.write_text('keep',encoding='utf-8')
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')
    assert p.read_text(encoding='utf-8')=='keep'


def test_failed_actual_checkpoint_smoke_never_opens_source_or_U(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'validate_config',lambda c:c)
    models=[torch.nn.Linear(2,2).eval().requires_grad_(False) for _ in range(3)]
    monkeypatch.setattr(m.parent,'_backend',lambda *a:dict(reader=models[0],model=models[1],dura=(models[2],None)))
    monkeypatch.setattr(m.parent,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('actual smoke failure')))
    monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('No source aggregate before smoke'))
    monkeypatch.setattr(m,'_U',lambda *a:pytest.fail('No U before smoke'))
    with pytest.raises(ValueError,match='smoke failure'):m.run(dict(output_root=str(tmp_path/'new'),reference_config={}),'cpu')
    assert not (tmp_path/'new').exists()
