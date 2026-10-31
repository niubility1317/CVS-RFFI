import copy
import pytest
import torch
from experiments.pamf_dura_direct_train import balanced_tangent as candidate


def test_objective_balances_only_extra_term_without_F(monkeypatch):
    model=torch.nn.Conv1d(2,2,1).double();y=torch.randn(3,2,256,dtype=torch.float64);x=torch.randn_like(y);jac=torch.randn(3,5,2,256,dtype=torch.float64)*10
    def source(student,received,*args,**kwargs):
        restored=student(received);z=restored.square().mean((1,2))
        return z,dict(iq_nmse=z,prior_nmse=2*z,spectral_error=3*z,action_probe_nmse=4*z,log_variance_mse=0*z),restored
    monkeypatch.setattr(candidate,'source_objective',source)
    loss,terms,restored=candidate.objective(model,y,x,None,None,None,jac)
    torch.testing.assert_close(loss,1.55*restored.square().mean((1,2))+candidate.WEIGHT*terms[candidate.METRIC])
    assert torch.all(terms['F_balanced_pressure']<=1+2e-5) and (terms['F_balance_limited']>0).any()
    loss.mean().backward();assert model.weight.grad is not None and torch.isfinite(model.weight.grad).all()


def test_cache_reuse_rejects_wrong_source_F_or_supervision():
    value=dict(status='SOURCE_F_CLEAN_J_COMPLETE',count=63000,shape=[63000,5,2,256],numeric_bytes=645120000,
        F_teacher=candidate.f_guided.teacher_reference(),F_inputs=63000,F_J_VJP_inputs=315000,F_updates=0,F_unchanged=True,TX_labels_read=0,V_reads=0,target_reads=0)
    candidate.verify_cache_metadata(value)
    for key,bad in [('target_reads',1),('F_updates',1),('count',512),('F_unchanged',False),('status','SOURCE_BOUNDED_METRIC_COMPLETE')]:
        wrong=copy.deepcopy(value);wrong[key]=bad
        with pytest.raises(ValueError):candidate.verify_cache_metadata(wrong)


def test_E30_cache_only_configuration_and_owner_registration():
    candidate.configure();config=candidate.base.configuration()
    assert (config['epochs'],config['updates'],config['parameters'])==(30,7410,266951)
    assert config['training_F_access'] and config['new_F_cache_inputs']==0 and config['F_restored_training_inputs']==0
    assert config['fixed_source_weight']==candidate.WEIGHT and config['step_weight_detached']
    assert config['max_total_visible_GPU_processes']==3 and config['source_corruption_schedule']=='01..20then01..10'
    import inspect
    assert 'pamf_dura_direct_train.balanced_tangent worker' in inspect.getsource(candidate.base.dispatch)
