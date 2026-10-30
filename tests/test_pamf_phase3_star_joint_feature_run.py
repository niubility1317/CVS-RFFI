"""Synthetic checkpoint execution and source-fit/query role ordering."""
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pytest


@contextmanager
def fixture(root,mp):
    from test_pamf_phase3_star_class_density_run import fixture as parent_fixture
    from experiments.pamf_phase3_open_world import star_joint_feature_run as m
    with parent_fixture(root,mp) as (parent,pc,_,backend,public,_,seen):
        pc['code_commit']=m.frozen.PARENT_CODE;parent.run(pc,'cpu')
        counts=dict(parent.COUNTS);mp.setattr(m,'COUNTS',counts);mp.setattr(m,'Q_COUNT',parent.Q_COUNT);events=[]
        c=dict(run_id=m.RUN_ID,code_commit='c'*40,reference_config=pc,output_root=str(root/'joint_feature'),batch_size=128)
        rows=m._rows;load_public=m.parent._public
        def pairs(*args):
            result={}
            for role in ('L','OE'):
                events.append(role);n=counts[role];z=np.ones((n,160),np.float32)/np.sqrt(160)
                meta=dict(rx_i=np.ones(n,np.int64),day_i=np.ones(n,np.int64))
                if role=='L':meta['y']=np.zeros(n,np.int64)
                result[role]=dict(u=z,z=z.copy(),metadata=meta,ids=np.asarray(['fit'+role+str(i) for i in range(n)]))
            return result,dict(synthetic=True)
        def fit(*args):
            assert events==['L','OE'];events.append('fit');d=m.core.density
            state=dict(schema=np.asarray(d.JOINT_SCHEMA),origin=np.zeros(320),whitener=np.eye(320),log_abs_det_whitener=np.asarray(0.),
                pooled_L_variance=np.asarray(1.),known_means=np.zeros((90,320)),known_variances=np.ones(90),known_weights=np.full(90,1/90),
                background_means=np.zeros((1,320)),background_variances=np.full(1,2.),background_weights=np.ones(1))
            m.core.validate_joint(state);return state,dict(synthetic_fit=True)
        def source_rows(path,role):
            if not (Path(c['output_root'])/'complete.json').exists():
                if role=='known':assert events==['L','OE','fit']
                else:assert (Path(c['output_root'])/'calibration/complete.json').is_file()
                events.append(role)
            return rows(path,role)
        def opened(config,name):
            assert (Path(c['output_root'])/'calibration/complete.json').is_file();events.append(name);return load_public(config,name)
        mp.setattr(m,'_source_pairs',pairs);mp.setattr(m.core,'fit',fit);mp.setattr(m,'_rows',source_rows);mp.setattr(m.parent,'_public',opened)
        yield m,c,counts,events,public,backend,seen


def test_config_rejects_target_supervision_wrong_parent_and_role_output_collision(tmp_path,monkeypatch,subtests):
    from experiments.pamf_phase3_open_world import star_joint_feature_run as m
    monkeypatch.setattr(m.parent,'validate_config',lambda c:c)
    pc=dict(code_commit=m.frozen.PARENT_CODE,output_root=str(tmp_path/'parent'),reference_prediction_root=str(tmp_path/'ref'),train_config=dict(IQ_root=str(tmp_path/'IQ'),feature_root=str(tmp_path/'teacher'),output_root=str(tmp_path/'train')))
    c=dict(run_id=m.RUN_ID,code_commit='a'*40,reference_config=pc,output_root=str(tmp_path/'new'),batch_size=128)
    assert m.validate_config(c) is c
    for delta in (dict(truth='private'),dict(U_public_root='discovery'),dict(batch_size=True),dict(output_root=pc['train_config']['feature_root']),dict(reference_config=dict(pc,code_commit='b'*40))):
        with subtests.test(delta=delta),pytest.raises(ValueError):m.validate_config(dict(c,**delta))


def test_existing_output_never_reloads_models_or_fits(tmp_path,monkeypatch):
    from experiments.pamf_phase3_open_world import star_joint_feature_run as m
    monkeypatch.setattr(m,'validate_config',lambda c:c);monkeypatch.setattr(m.parent,'_backend',lambda *a:pytest.fail('No reload or refit'))
    with pytest.raises(FileExistsError):m.run(dict(output_root=str(tmp_path)),'cpu')


def test_query_current_uses_same_DURA_output_and_avoids_signal_forward(monkeypatch):
    import torch
    from experiments.pamf_phase3_open_world import star_joint_feature_run as m
    events=[]
    class Reader:
        def features(self,x):
            events.append(('F0',x.clone()));return x.flatten(1)[:,:160]
        def classify_features(self,f):return f[:,:6]
    def G(x):
        y=x.flip(-1)*.7+.03;events.append(('G',y.clone()));return (y,)
    b=dict(reader=Reader(),dura=(None,G),model=lambda x:pytest.fail('Cached query z must be reused'))
    a=m._current(b,np.ones((2,2,256),np.float32),'cpu')
    assert a['unit_features'].shape==(2,160) and [v[0] for v in events]==['G','F0']
    torch.testing.assert_close(events[0][1],events[1][1],rtol=0,atol=0)
