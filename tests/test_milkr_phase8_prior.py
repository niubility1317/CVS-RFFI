import numpy as np
import pytest
import torch
from experiments.pamf_phase2_practical import phase_orbit as orbit,teacher_features
from tests.test_milkr_phase_orbit import fake_capture


def test_C8_RF_moments_keep_original_C4_head_function_and_batch_independence(monkeypatch):
    calls=[]
    def capture(reader,value):calls.append(value.clone());return fake_capture(reader,value)
    monkeypatch.setattr(orbit,'capture',capture)
    x=np.random.default_rng(111).normal(size=(3,2,256)).astype(np.float32)
    result,logits=teacher_features.extract_phase8(None,x,'cpu',batch=3)
    assert len(calls)==8
    for index in range(4):assert torch.equal(calls[2*index],torch.tensor(orbit.rotate(x,index)))
    baseline,old_logits=teacher_features.extract(None,x,'cpu',batch=3)
    assert np.array_equal(result['teacher_phase_logits'],baseline['teacher_phase_logits']) and np.array_equal(logits,old_logits)
    shifted,_=teacher_features.extract_phase8(None,orbit.rotate_eighth(x,1),'cpu',batch=2)
    assert all(np.allclose(value,shifted[key],rtol=2e-6,atol=2e-7) for key,value in result.items() if key!='teacher_phase_logits')
    one=teacher_features.extract_phase8(None,x[1:2],'cpu')[0]
    assert all(np.allclose(value[1],one[key][0]) for key,value in result.items())


def test_C8_G_once_precedes_all_phases(monkeypatch):
    monkeypatch.setattr(orbit,'capture',fake_capture);x=np.random.default_rng(112).normal(size=(3,2,256)).astype(np.float32);calls=[]
    def engine(value):calls.append(value.clone());return (value+.2,)
    got,_=teacher_features.extract_phase8(None,x,'cpu',engine,batch=2);direct,_=teacher_features.extract_phase8(None,x+.2,'cpu',batch=2)
    assert len(calls)==2 and torch.equal(torch.cat(calls),torch.tensor(x))
    assert all(np.allclose(value,direct[key]) for key,value in got.items())
    with pytest.raises(ValueError,match='eighth'):orbit.rotate_eighth(x,8)
    with pytest.raises(ValueError,match='eight per-record'):orbit.eighth_moments([{}]*4)
    with pytest.raises(ValueError,match='phase-only'):orbit.extract(None,x,'cpu',time_shifts=(0,64,128,192),phase_eighths=True)


def test_C8_method_routes(monkeypatch):
    from experiments.pamf_phase2_practical import milkr_cv_screen as screen
    from experiments.pamf_phase2_practical.milkr.prior_fusion import fit
    assert screen.fit_function({'kernel_method':'phase8_prior'}) is fit
    sentinel=object();monkeypatch.setattr(teacher_features,'extract_phase8',lambda *a:sentinel)
    assert screen.extract_for_method({'kernel_method':'phase8_prior'},None,None,'cpu') is sentinel
