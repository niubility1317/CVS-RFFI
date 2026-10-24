import pytest
import torch
from experiments.pamf_ssl_research.source_branch_gradients import (
    GROUPS,labeled_components,unlabeled_components,gradient_facts,labeled_pairs,source_loaders)
from experiments.pamf_ssl_research.hard_pl_objective import mechanism_loss
from experiments.pamf_ssl_research.u_gradprotect_objective import protected_loss


def test_original_U_coefficients_and_rejected_examples_are_preserved():
    torch.manual_seed(7)
    logits=[torch.randn(8,6,dtype=torch.float64,requires_grad=True) for _ in range(6)]
    y=torch.arange(8).remainder(6);accepted=torch.tensor([1,1,0,1,1,0,1,1],dtype=torch.bool)
    physical=torch.arange(8).remainder(2).bool();aux=logits[0].square().mean()
    total=6*mechanism_loss(*logits,y,accepted,physical,aux)[0]
    parts=unlabeled_components(logits,y,accepted,physical,total)
    assert parts['U_aux'].item()==pytest.approx((6*.1/1.25*aux).item())
    expected=torch.autograd.grad(total,logits,retain_graph=True)
    actual=torch.autograd.grad(sum(parts.values()),logits,retain_graph=True)
    assert all(torch.allclose(a,b,rtol=1e-12,atol=1e-12) for a,b in zip(actual,expected))
    assert all(torch.equal(g[~accepted],torch.zeros_like(g[~accepted])) for g in torch.autograd.grad(parts['U_As'],logits[2:4]))


def test_complete_L_pairs_include_RDrop_and_source_aux_without_changing_dose():
    p=torch.tensor([.3,.7],dtype=torch.float64,requires_grad=True)
    pairs=[(p.square().sum()*i,p.sum()*i,p.sum()*0) for i in (1,2,3,4)]
    total=.5*pairs[0][0]+.25*sum(v[0] for v in pairs[1:])+.1*p.exp().sum()
    parts=labeled_components(pairs,total)
    assert parts['L_aux'].item()==pytest.approx((.1*p.exp().sum()).item())
    assert torch.allclose(torch.autograd.grad(sum(parts.values()),p,retain_graph=True)[0],torch.autograd.grad(total,p)[0])


def test_full_L_nonnegative_dot_does_not_imply_each_branch_is_protected():
    parameters=[torch.nn.Parameter(torch.tensor([.2,.4],dtype=torch.float64)) for _ in GROUPS]
    named=[(name,parameter) for name,parameter in zip(('root.t2.weight','root.cls_head.weight','root.other.weight'),parameters)]
    a=sum(p[0] for p in parameters);b=sum(p[1] for p in parameters);zero=(a+b)*0
    losses=dict(L_clean=a,L_As=2*b,L_aux=zero,U_clean=-2*a+b,U_As=zero,U_aux=zero)
    facts,expected=gradient_facts(named,losses)
    _,_,actual=protected_loss(a+2*b,-2*a+b,named)
    assert all(torch.allclose(g,h,rtol=1e-12,atol=1e-12) for g,h in zip(expected,actual))
    for group in GROUPS:
        anchors=facts['groups'][group]['anchors']
        assert anchors['L_full']['after']['dot']==pytest.approx(0.,abs=1e-12)
        assert anchors['L_clean']['after']['dot']==pytest.approx(-2.)
        assert anchors['L_As']['after']['dot']==pytest.approx(2.)
    assert facts['component_autograd_calls']==6
    assert all(p.grad is None and torch.equal(p.detach(),torch.tensor([.2,.4],dtype=torch.float64)) for p in parameters)


def test_missing_unused_group_has_undefined_cosine_and_zero_norm():
    p=torch.nn.Parameter(torch.tensor([1.,2.],dtype=torch.float64));named=[('root.t2.weight',p)]
    z=p.sum()*0;losses={k:z for k in ('L_clean','L_As','L_aux','U_clean','U_As','U_aux')}
    facts,_=gradient_facts(named,losses)
    assert facts['groups']['cls_head']['norms']['L_full']==0
    assert facts['groups']['t2']['anchors']['L_full']['after']['cosine'] is None


@pytest.mark.parametrize('kind',['missing','nonfinite','nonscalar'])
def test_invalid_loss_component_cannot_produce_source_conclusion(kind):
    p=torch.nn.Parameter(torch.ones(2));losses={k:p.sum() for k in ('L_clean','L_As','L_aux','U_clean','U_As','U_aux')}
    if kind=='missing':losses.pop('U_As')
    if kind=='nonfinite':losses['U_As']=p.sum()*float('nan')
    if kind=='nonscalar':losses['U_As']=p
    with pytest.raises(ValueError):gradient_facts([('root.t2.weight',p)],losses)


def test_pair_wrapper_is_restored_after_exception_and_rejects_nested_capture():
    from experiments.cric_dura_comparison import t1_model
    original=t1_model.pair
    with pytest.raises(RuntimeError,match='probe'):
        with labeled_pairs():
            with pytest.raises(ValueError,match='One disposable'):
                with labeled_pairs():pass
            raise RuntimeError('probe')
    assert t1_model.pair is original
    with labeled_pairs() as captured:
        t1_model.pair(torch.zeros(2,6),torch.zeros(2,6),torch.tensor([0,1]))
    assert len(captured)==1 and t1_model.pair is original


def test_wrong_pair_count_and_unbalanced_physical_half_are_rejected():
    with pytest.raises(ValueError,match='four'):labeled_components([],torch.tensor(0.))
    with pytest.raises(ValueError,match='six'):unlabeled_components([],None,None,None,None)
    with pytest.raises(ValueError,match='physical half'):
        unlabeled_components([torch.zeros(4,6)]*6,torch.zeros(4,dtype=torch.long),torch.ones(4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool),torch.tensor(0.))


def test_diagnostic_loader_never_opens_validation_or_target(monkeypatch):
    from experiments.cric_dura_comparison import f_data
    roles=[]
    def dataset(root,role,contract):
        roles.append(role);return role
    monkeypatch.setattr(f_data,'SourceDataset',dataset)
    monkeypatch.setattr(f_data,'FixedEpochLoader',lambda dataset,batch,seed,drop_last:dict(role=dataset,batch=batch,drop_last=drop_last))
    loaders=source_loaders(dict(source_cache='existing-source',model_seed=7),{})
    assert roles==['L_s','U_s'] and set(loaders)=={'train_loader','unlabeled_loader'}
    assert loaders['train_loader']==dict(role='L_s',batch=128,drop_last=True)
    assert loaders['unlabeled_loader']==dict(role='U_s',batch=256,drop_last=False)
