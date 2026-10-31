import numpy as np
import pytest
import torch
from experiments.pamf_phase2_practical import phase_orbit as orbit,teacher_features


def fake_capture(reader,x):
    flat=x.flatten(1)
    value={key:flat[:,:d]+2 for key,d in (('z_id',160),('t_emb',160),('f_emb',160),('pa_local',160))}
    value['sig_views']=flat[:,:336]+3
    return value,flat[:,:6]


def test_time_orbit_RF_invariance_and_unchanged_head_prior(monkeypatch):
    monkeypatch.setattr(orbit,'capture',fake_capture)
    x=np.random.default_rng(81).normal(size=(4,2,256)).astype(np.float32)
    result,logits=teacher_features.extract_time(None,x,'cpu',batch=3)
    shifted,_=teacher_features.extract_time(None,np.roll(x,64,axis=2),'cpu',batch=2)
    ordinary,old_logits=teacher_features.extract(None,x,'cpu',batch=2)
    assert all(np.allclose(v,shifted[k],atol=1e-7) for k,v in result.items() if k!='teacher_phase_logits')
    assert np.array_equal(result['teacher_phase_logits'],ordinary['teacher_phase_logits'])
    assert np.array_equal(logits,old_logits)
    one=teacher_features.extract_time(None,x[1:2],'cpu')[0]
    assert all(np.allclose(v[1],one[k][0]) for k,v in result.items())


def test_G_runs_on_original_record_once_before_time_and_phase_orbit(monkeypatch):
    monkeypatch.setattr(orbit,'capture',fake_capture)
    x=np.random.default_rng(82).normal(size=(3,2,256)).astype(np.float32);calls=[]
    def engine(value):calls.append(value.clone());return (value+.2,)
    got,_=teacher_features.extract_time(None,x,'cpu',engine,batch=2)
    direct,_=teacher_features.extract_time(None,x+.2,'cpu',batch=2)
    assert len(calls)==2 and torch.equal(torch.cat(calls),torch.tensor(x))
    assert all(np.allclose(v,direct[k]) for k,v in got.items())
    with pytest.raises(ValueError,match='time orbit'):orbit.extract(None,x,'cpu',time_shifts=(0,32))
    with pytest.raises(ValueError,match='sixteen'):orbit.spacetime_moments([{}]*4)


def test_time_method_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen
    from experiments.pamf_phase2_practical.milkr.prior_fusion import fit
    assert screen.fit_function({'kernel_method':'time_orbit'}) is fit
    sentinel=object();monkeypatch.setattr(teacher_features,'extract_time',lambda *a:sentinel)
    assert screen.extract_for_method({'kernel_method':'time_orbit'},None,None,'cpu') is sentinel
