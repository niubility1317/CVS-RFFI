"""Source-fit/calibration/query boundaries; core Gaussian algebra is tested separately."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_class_density_run import fixture as parent_fixture
    from experiments.pamf_phase3_open_world import star_local_cov_run as m
    with parent_fixture(root,mp) as (parent,pc,_,backend,public,old_events,_):
        pc['code_commit']=m.PARENT_CODE
        parent.run(pc,'cpu')
        events=[];counts=dict(parent.COUNTS);mp.setattr(m,'COUNTS',counts);mp.setattr(m,'Q_COUNT',parent.Q_COUNT)
        c=dict(run_id=m.RUN_ID,code_commit='b'*40,reference_config=pc,output_root=str(root/'local_cov'),batch_size=128)
        original_rows=m._rows;original_target=m._target_rows
        def L(*args):
            events.append('L');n=counts['L'];x=np.ones((n,160),np.float32)/np.sqrt(160)
            return x,dict(y=np.zeros(n,np.int64),rx=np.ones(n,np.int64),day=np.ones(n,np.int64)),np.asarray(['L'+str(i) for i in range(n)]),dict(role='L',count=n)
        def fit(state,*args):
            assert events==['L'];events.append('fit')
            v=state['known_variances'];transform=np.eye(160)[None]/np.sqrt(v[:,None,None])
            local=dict(schema=np.asarray(m.core.SCHEMA),transforms=transform,log_determinants=160*np.log(v),trace_variances=v.copy(),alpha=np.asarray(70/230,np.float64))
            m.core.validate_local(local,state);return local,dict(synthetic_fit=True)
        def rows(path,role):
            if not (Path(c['output_root'])/'complete.json').exists():
                if role=='known':assert events==['L','fit']
                else:assert (Path(c['output_root'])/'calibration/complete.json').is_file()
                events.append(role)
            return original_rows(path,role)
        def target(path,name):
            assert (Path(c['output_root'])/'calibration/complete.json').is_file()
            if not (Path(c['output_root'])/'complete.json').exists():events.append(name)
            return original_target(path,name)
        mp.setattr(m,'_source_L',L);mp.setattr(m.core,'fit',fit);mp.setattr(m,'_rows',rows);mp.setattr(m,'_target_rows',target)
        yield m,c,counts,events,public,backend


def test_config_rejects_extra_target_supervision_wrong_parent_and_output_collision(tmp_path,monkeypatch,subtests):
    from experiments.pamf_phase3_open_world import star_local_cov_run as m
    monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    pc=dict(code_commit=m.PARENT_CODE,output_root=str(tmp_path/'parent'),reference_prediction_root=str(tmp_path/'ref'),train_config=dict(IQ_root=str(tmp_path/'IQ'),output_root=str(tmp_path/'train')))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=pc,output_root=str(tmp_path/'new'),batch_size=128)
    assert m.validate_config(c) is c
    for delta in (dict(U_public_root='forbidden'),dict(truth='private'),dict(batch_size=True),dict(output_root=pc['train_config']['IQ_root']),dict(reference_config=dict(pc,code_commit='b'*40))):
        with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))


def test_existing_output_does_not_load_models_or_repeat_fit(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_local_cov_run as m
    monkeypatch.setattr(m,'validate_config',lambda c:c)
    monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('Do not reload models or fit an existing run'))
    c=dict(output_root=str(tmp_path));sentinel=tmp_path/'sentinel';sentinel.write_text('kept')
    with pytest.raises(FileExistsError):m.run(c,'cpu')
    assert sentinel.read_text()=='kept'


def test_source_L_uses_only_existing_restored_values_and_closes_role(monkeypatch):
    from experiments.pamf_phase3_open_world import star_local_cov_run as m
    received=np.full((2,2,256),99,np.float32);restored=np.full_like(received,3);events=[]
    role=dict(ids=np.asarray(['a','b']),received=received,restored=restored,
              metadata=dict(y=np.asarray([0,1]),rx_i=np.asarray([1,3]),day_i=np.asarray([1,2])),manifest=dict(role='L'))
    monkeypatch.setattr(m.signal_eval,'_iq_role',lambda c,r,i:role if r=='L' else pytest.fail('L only'))
    def project(model,values,device):
        assert values is restored;events.append('P1');return np.ones((2,160),np.float32)
    monkeypatch.setattr(m.signal_eval,'_project',project);monkeypatch.setattr(m.signal_eval,'_close_role',lambda r:events.append('close'))
    result=m._source_L(dict(reference_config=dict(train_config={})),dict(model=object()),{},'cpu')
    assert events==['P1','close'] and result[2].tolist()==['a','b']
