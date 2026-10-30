import copy
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from experiments.pamf_dura_direct_train import native_iq_hint as trial


def test_hint_uses_final_iq_once_and_clean_has_no_teacher_gradient(monkeypatch):
    restored=torch.tensor([[[2.,3.]],[[4.,5.]]],requires_grad=True)
    clean=torch.ones_like(restored,requires_grad=True)
    reference=torch.zeros_like(restored,requires_grad=True)
    calls=[]
    def source(*args,**kwargs):
        calls.append(kwargs)
        terms={k:torch.ones(2) for k in ('iq_nmse','prior_nmse','spectral_error','action_probe_nmse','log_variance_mse')}
        return None,terms,restored
    monkeypatch.setattr(trial,'source_objective',source)
    loss,terms=trial.objective(None,None,clean,None,None,None,reference,torch.tensor([False,True]))
    torch.testing.assert_close(loss,torch.tensor([1.2,21.7]))
    loss.sum().backward()
    assert len(calls)==1 and calls[0]=={'return_restored':True}
    assert restored.grad[0].abs().sum()==0 and restored.grad[1].abs().sum()>0
    assert clean.grad is None and reference.grad is None
    torch.testing.assert_close(terms['native_hint_active'],torch.tensor([0.,1.]))


def test_two_iq_targets_equal_midpoint_plus_constant_on_nonclean():
    generator=torch.Generator().manual_seed(12)
    z=torch.randn(3,2,256,generator=generator);x=torch.randn(3,2,256,generator=generator)
    teacher=torch.randn(3,2,256,generator=generator);active=torch.tensor([True,False,True])
    energy=x.square().mean((1,2)).clamp_min(1e-4)
    loss=(z-x).square().mean((1,2))/energy+trial.final_hint(z,x,teacher,active)
    midpoint=(x+teacher)/2
    expected=torch.where(active,2*(z-midpoint).square().mean((1,2))/energy+
        .5*(x-teacher).square().mean((1,2))/energy,(z-x).square().mean((1,2))/energy)
    torch.testing.assert_close(loss,expected)
    with pytest.raises(ValueError):trial.final_hint(z,x,teacher,active.float())


def cache_metadata():
    return dict(status='SOURCE_NATIVE_IQ_CACHE_COMPLETE',source_contract='original7/63/30',
        code_commit=trial.CACHE_COMMIT,source_action_run=trial.base.ACTION,teacher_run=trial.TEACHER_RUN,teacher_commit=trial.TEACHER_COMMIT,
        teacher_component='native_map_prior.stage3',teacher_used_parameters=4665073,
        teacher_coarse_epochs=20,teacher_action_epochs=20,teacher_prior_epochs=20,teacher_bundle_component_epochs=60,
        teacher_source_selection_before_target=True,teacher_unchanged=True,views=list(range(1,21)),count_per_view=63000,
        teacher_inputs=1260000,teacher_smoke_inputs=2,L_visits=126000,U_visits=1134000,numeric_bytes=2580480000,
        teacher_updates=0,F_reads=0,TX_labels_read=0,V_reads=0,target_reads=0,model_optimizer_updates=0)


def test_rejects_unfinished_wrong_teacher_and_hidden_data():
    record=cache_metadata();launch=dict(commit=trial.CACHE_COMMIT)
    plan=dict(run=trial.CACHE_RUN,source_contract='original7/63/30');complete=dict(status='ANALYZED',failed=[])
    trial.validate_cache_metadata(record,launch,plan,complete)
    for key,value in [('teacher_component','frozen.coarse'),('teacher_commit','bad'),('teacher_unchanged',False),
                      ('F_reads',1),('TX_labels_read',1),('V_reads',1),('target_reads',1),('teacher_inputs',63000)]:
        changed=copy.deepcopy(record);changed[key]=value
        with pytest.raises(ValueError):trial.validate_cache_metadata(changed,launch,plan,complete)
    with pytest.raises(ValueError):trial.validate_cache_metadata(record,launch,plan,dict(status='RUNNING',failed=[]))


def test_mask_uses_source_scene_and_rejects_unknown_scene(tmp_path):
    values=np.full(63000,'practical_low_urban',dtype='U32');values[:9000]='clean'
    np.savez(tmp_path/'groups.npz',scene=values)
    assert trial.nonclean_mask(tmp_path).sum()==54000
    values[0]='unknown';np.savez(tmp_path/'groups.npz',scene=values)
    with pytest.raises(ValueError):trial.nonclean_mask(tmp_path)


def test_fixed_budget_routing_and_checkpoint_roundtrip(monkeypatch,tmp_path):
    for key,value in vars(trial.base).copy().items():monkeypatch.setattr(trial.base,key,value)
    trial.configure();base=trial.base;c=base.configuration()
    assert (c['epochs'],c['updates'],c['parameters'],c['training_stages'])==(30,7410,266951,1)
    assert c['teacher_training_visits']==1890000 and c['teacher_cache_inputs']==0
    assert c['source_clean_teacher_weight']==0 and c['source_nonclean_teacher_weight']==1
    assert not c['inference_needs_scene_or_teacher'] and not c['training_F_access']
    args=SimpleNamespace(root=tmp_path,commit='a'*40)
    assert all(base.command(args,m,'scratch_joint','target')[3]==trial.MODULE for m in ('worker','score'))
    model=base.scratch_model(2026092701).eval();x=torch.ones(1,2,256)
    with torch.no_grad():expected=model(x)
    checkpoint=tmp_path/'model.pt';base.save_model(checkpoint,model,'scratch_joint',0,0,args.commit)
    loaded=base.load_model(checkpoint,'cpu',arm='scratch_joint',commit=args.commit,epoch=0)
    with torch.no_grad():torch.testing.assert_close(loaded(x),expected,rtol=0,atol=0)
    with pytest.raises(ValueError):base.load_model(checkpoint,'cpu',arm='scratch_joint',commit='b'*40,epoch=0)
