import copy
import pytest
import torch
from experiments.pamf_dura_direct_train import clean_tangent as candidate


def test_objective_adds_only_fixed_clean_metric_without_F(monkeypatch):
    student=torch.nn.Conv1d(2,2,1).double();y=torch.randn(3,2,256,dtype=torch.float64);x=torch.randn_like(y)
    jac=torch.randn(3,5,2,256,dtype=torch.float64)
    def source(model,received,*args,**kwargs):
        restored=model(received);z=restored.square().mean((1,2))
        return z,dict(iq_nmse=z,prior_nmse=z*2,spectral_error=z*3,action_probe_nmse=z*4,log_variance_mse=z*0),restored
    monkeypatch.setattr(candidate,'source_objective',source)
    loss,terms,restored=candidate.objective(student,y,x,None,None,None,jac)
    expected=(1+.2+.15+.2)*restored.square().mean((1,2))+candidate.WEIGHT*candidate.tangent_loss(restored,x,jac)
    torch.testing.assert_close(loss,expected);loss.mean().backward()
    assert student.weight.grad is not None and torch.isfinite(student.weight.grad).all()


def test_source_qualification_checks_frozen_source_identity():
    result=dict(status='ANALYZED_SOURCE_ONLY',decision=dict(eligible=True,views=[dict(view='1',passed=True),dict(view='20',passed=True)],fixed_source_weight=candidate.WEIGHT,automatic_training=False),
        F_teacher=candidate.f_guided.teacher_reference(),source_checkpoint=candidate.CONTROL,source_commit=candidate.CONTROL_COMMIT,
        F_unchanged=True,optimizer_updates=0,TX_labels_read=0,V_reads=0,target_reads=0,new_G_inputs=0)
    candidate.verify_qualification(result)
    for key,value in [('target_reads',1),('F_unchanged',False),('source_commit','other')]:
        bad=copy.deepcopy(result);bad[key]=value
        with pytest.raises(ValueError):candidate.verify_qualification(bad)
    bad=copy.deepcopy(result);bad['decision']['fixed_source_weight']*=2
    with pytest.raises(ValueError):candidate.verify_qualification(bad)


def test_E30_configuration_and_resource_owner_registration():
    candidate.configure();config=candidate.base.configuration()
    assert config['epochs']==30 and config['updates']==7410 and config['training_F_access']
    assert config['F_restored_training_inputs']==0 and config['F_clean_J_cache_count']==63000
    assert config['parameters']==266951 and config['source_corruption_schedule']=='01..20then01..10'
    assert config['allowed_GPUs']==[4,6,7] and config['max_total_visible_GPU_processes']==3
    import inspect
    assert 'pamf_dura_direct_train.clean_tangent worker' in inspect.getsource(candidate.base.dispatch)
