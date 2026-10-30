import pytest
import torch
from experiments.pamf_dura_gradient import module_roles as trial


def test_source_rejects_target_teacher_or_wrong_training_contract():
    source=dict(status='SOURCE_TRAINED',epoch=20,updates=4940,L_exposures=126000,U_exposures=1134000,
        training_TX_labels_read=0,training_F_access=False,V_reads=0,target_reads=0,teacher_updates=0,
        learned_teacher_inputs=0,initialization_checkpoint_loads=0,ancestor_epochs=0,head_fit_updates=0,parameters=266951)
    trial.validate_source(source)
    for key,value in [('target_reads',1),('training_F_access',True),('training_TX_labels_read',1),('epoch',30),('teacher_updates',1)]:
        with pytest.raises(ValueError):trial.validate_source(dict(source,**{key:value}))


def test_groups_cover_unique_named_parameters_and_reject_unowned_modules():
    names=['frozen.coarse.conv.weight','frozen.estimator.conv.weight','prior.0.conv.weight']
    assert trial.module_groups(names)==dict(coarse=[0],action=[1],prior=[2])
    with pytest.raises(ValueError):trial.module_groups(names+['unknown.weight'])


def test_new_coarse_clean_term_retains_only_its_real_gradient(monkeypatch):
    coarse=torch.tensor([[[2.,3.]]],requires_grad=True);clean=torch.ones_like(coarse,requires_grad=True)
    calls=[]
    def forward(*args,**kwargs):
        calls.append(kwargs)
        terms={k:torch.ones(1) for k in ('iq_nmse','prior_nmse','spectral_error','action_probe_nmse')}
        return None,terms,coarse,coarse
    monkeypatch.setattr(trial,'direct_objective',forward)
    terms=trial.source_losses(None,None,clean,None,None,None)
    terms['coarse_clean_IQ'].sum().backward()
    assert len(calls)==1 and calls[0]==dict(return_coarse=True)
    assert coarse.grad.abs().sum()>0 and clean.grad is None
    assert terms['coarse_clean_IQ'].item()==pytest.approx(2.5)
