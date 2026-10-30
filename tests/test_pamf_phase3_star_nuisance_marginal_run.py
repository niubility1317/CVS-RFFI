"""Cached prediction permissions and source-aggregate-only nuisance derivation."""
from contextlib import contextmanager
from pathlib import Path
import importlib
import pytest
import torch


def api():return importlib.import_module('experiments.pamf_phase3_open_world.star_nuisance_marginal_run')


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_class_density_run import fixture as parent_fixture
    with parent_fixture(root,mp) as (parent,pc,_,backend,public,_,_):
        m=api();pc['code_commit']=m.PARENT_CODE;parent.run(pc,'cpu')
        counts=dict(parent.COUNTS);mp.setattr(m,'COUNTS',counts);mp.setattr(m,'Q_COUNT',parent.Q_COUNT)
        c=dict(run_id=m.RUN_ID,code_commit='e'*40,reference_config=pc,output_root=str(root/'nuisance_marginal'),batch_size=128)
        events=[];load=m.parent._backend;build=m.core.build_prior;rows=m._rows;target=m._target_rows
        def checkpoint(*args):events.append('checkpoint');return load(*args)
        def prior(state):
            assert events==['checkpoint'] and Path(c['output_root'],'smoke.json').is_file()
            events.append('prior');return build(state)
        def source_rows(path,role):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'models/complete.json').is_file()
                if role=='known':assert events==['checkpoint','prior']
                else:assert Path(c['output_root'],'calibration/complete.json').is_file() and Path(c['output_root'],'known_predictions.npz').is_file()
                events.append(role)
            return rows(path,role)
        def query_rows(path,name):
            if not Path(c['output_root'],'complete.json').exists():
                assert Path(c['output_root'],'novel_predictions.npz').is_file();events.append(name)
            return target(path,name)
        mp.setattr(m.parent,'_backend',checkpoint);mp.setattr(m.core,'build_prior',prior)
        mp.setattr(m,'_rows',source_rows);mp.setattr(m,'_target_rows',query_rows)
        mp.setattr(m.signal_eval,'_iq_role',lambda *a:pytest.fail('No L/OE/V/novel IQ is consumed by this producer'))
        mp.setattr(m.parent,'_public',lambda *a:pytest.fail('Only cached target predictions may be read'))
        yield m,c,counts,events,backend


def test_exact_config_rejects_supervision_unknown_inputs_and_output_overlap(tmp_path,monkeypatch,subtests):
    m=api();monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    p=dict(code_commit=m.PARENT_CODE,output_root=str(tmp_path/'parent'),reference_prediction_root=str(tmp_path/'reference'),
        train_config=dict(IQ_root=str(tmp_path/'IQ'),output_root=str(tmp_path/'train')))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=p,output_root=str(tmp_path/'new'),batch_size=128)
    assert m.validate_config(c) is c
    for change in (dict(truth='private'),dict(target_labels='forbidden'),dict(U_public_root='forbidden'),dict(batch_size=True),
            dict(output_root=p['train_config']['IQ_root']),dict(reference_config=dict(p,code_commit='b'*40))):
        with subtests.test(change=change),pytest.raises(ValueError):m.validate_config(dict(c,**change))


def test_existing_output_never_loads_checkpoints_or_rederives_prior(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'validate_config',lambda c:c)
    monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('No checkpoint restart'))
    monkeypatch.setattr(m.core,'build_prior',lambda *a:pytest.fail('No duplicate prior derivation'))
    p=tmp_path/'sentinel';p.write_text('keep',encoding='utf-8')
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')
    assert p.read_text(encoding='utf-8')=='keep'


def test_failed_checkpoint_smoke_precedes_aggregate_and_any_records(tmp_path,monkeypatch):
    m=api();monkeypatch.setattr(m,'validate_config',lambda c:c)
    models=[torch.nn.Linear(2,2).eval().requires_grad_(False) for _ in range(3)]
    monkeypatch.setattr(m.parent,'_backend',lambda *a:dict(reader=models[0],model=models[1],dura=(models[2],None)))
    monkeypatch.setattr(m.parent,'_smoke',lambda *a:(_ for _ in ()).throw(ValueError('actual checkpoint smoke failure')))
    monkeypatch.setattr(m,'_reference',lambda *a:pytest.fail('No aggregate access before smoke'))
    monkeypatch.setattr(m,'_rows',lambda *a:pytest.fail('No source rows before smoke'))
    monkeypatch.setattr(m,'_target_rows',lambda *a:pytest.fail('No query rows before smoke'))
    out=tmp_path/'failed'
    with pytest.raises(ValueError,match='smoke failure'):m.run(dict(output_root=str(out),reference_config={}),'cpu')
    assert not out.exists()
