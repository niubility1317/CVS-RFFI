from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from experiments.pamf_dura_direct_train import exact_reference as m


def test_teacher_constructor_cannot_advance_matched_G_probe_rng(monkeypatch):
    def load(*args):
        torch.randn(1000)
        return torch.nn.Linear(2,2)
    monkeypatch.setattr(m,'load_reader',load)
    torch.manual_seed(42);before=torch.get_rng_state().clone()
    teacher,info=m.load_teacher_preserving_rng(dict(reader='reader',source_cache='source'),'cpu')
    assert torch.equal(before,torch.get_rng_state()) and info['F_load_RNG_restored'] and info['CPU_rng_restored']
    assert info['CUDA_rng_restored'] is None and not teacher.training and all(not p.requires_grad for p in teacher.parameters())


def test_exact_guide_is_already_weighted_variance_stays_diagnostic(monkeypatch):
    received=torch.ones(2,2,256,requires_grad=True)
    physical={key:torch.full((2,),value) for key,value in dict(iq_nmse=1.,prior_nmse=2.,spectral_error=3.,action_probe_nmse=4.,log_variance_mse=1e6).items()}
    monkeypatch.setattr(m,'source_objective',lambda *args,**kwargs:(torch.full((2,),1000.),physical,received))
    def guide(reader,r,*args):
        torch.randn(13)
        value=r.sum((1,2))*.001
        return value,dict(feature_error=torch.ones(2),weight=torch.ones(2),raw_pressure=torch.ones(2)*.5,bounded_pressure=torch.ones(2)*.5,pressure_defined=torch.ones(2,dtype=torch.bool))
    monkeypatch.setattr(m,'exact_guide',guide)
    before=torch.get_rng_state().clone()
    loss,terms,_=m.objective(None,None,received,received.detach(),None,None,None,None,None)
    assert torch.equal(before,torch.get_rng_state())
    torch.testing.assert_close(loss,torch.full((2,),1.+.1*2+.05*3+.05*4+.512))
    torch.testing.assert_close(terms[m.GUIDE],torch.full((2,),.512))
    loss.mean().backward();torch.testing.assert_close(received.grad,torch.full_like(received,.0005))


def test_target_permissions_and_qualification_are_not_silently_defaulted():
    common=dict(decision=dict(eligible=True),F_unchanged=True,F_teacher=m.f_guided.teacher_reference(),optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0)
    nonlinear=dict(common,G_state_unchanged=True,iterations=8,half_step=.5,dimension=10)
    cost=dict(common,G_unchanged=True,source_views=['1','20'],batch=256,timed_per_case_view=12)
    m.validate_qualifications(nonlinear,cost)
    with pytest.raises(ValueError):m.validate_qualifications(nonlinear,dict(cost,TX_labels_read=1))
    with pytest.raises(ValueError):m.validate_qualifications(dict(nonlinear,F_teacher={}),cost)
    with pytest.raises(ValueError):m.validate_qualifications(nonlinear,dict(cost,decision=dict(eligible=False)))


def test_single_E30_online_F_budget_and_worker_score_routes():
    m.configure();cfg=m.base.configuration()
    assert (cfg['epochs'],cfg['updates'],cfg['L_exposures'],cfg['U_exposures'])==(30,7410,189000,1701000)
    assert cfg['F_loaded_in_training'] and cfg['F_training_forward_after_cache']==cfg['F_input_VJP_inputs']==1890000
    assert cfg['source_teacher_target_shape']==[63000,10] and not cfg['training_reads_J_cache'] and cfg['F_J_reference_inputs']==0
    assert 'source_reference_offset_shape' not in cfg and 'F_clean_J_shape' not in cfg
    args=SimpleNamespace(root=Path('/project'),commit='commit')
    for mode,role in [('worker',None),('score','source'),('score','target')]:
        command=m.base.command(args,mode,'scratch_joint',role)
        assert m.MODULE in command and mode in command and m.parent.MODULE not in command
