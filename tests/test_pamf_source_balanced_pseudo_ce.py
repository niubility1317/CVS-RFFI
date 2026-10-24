import pytest
import torch
from experiments.pamf_ssl_research.source_balanced_pseudo_ce import contribution_weights,cv,evaluate


def test_weights_preserve_accepted_mass_and_upweight_underrepresented_pseudo_class():
    p=torch.tensor([0]*8+[1]*2);a=torch.ones(10,dtype=torch.bool);rx=torch.tensor([1,3]*5)
    w=contribution_weights(p,a,rx)
    assert w.mean()==pytest.approx(1) and float(w[p==1].mean())>float(w[p==0].mean())
    assert float(w.min())>=.5 and float(w.max())<=1.5+1e-12 and not w.requires_grad


def test_source_RX_and_class_both_affect_contribution_without_truth():
    p=torch.zeros(10,dtype=torch.long);a=torch.ones(10,dtype=torch.bool);rx=torch.tensor([1]*8+[3]*2)
    w=contribution_weights(p,a,rx)
    assert w[rx==3].mean()>w[rx==1].mean() and w.sum()==pytest.approx(10)


def test_extremely_rare_group_does_not_exceed_predeclared_cap():
    p=torch.tensor([0]*255+[1]);a=torch.ones(256,dtype=torch.bool);rx=torch.ones(256,dtype=torch.long)
    w=contribution_weights(p,a,rx)
    assert w[-1]==pytest.approx(1.5) and float(w.min())>=.5 and w.sum()==pytest.approx(256)


def test_rejected_entries_cannot_influence_the_weights_of_accepted_samples():
    p=torch.tensor([0,0,1,2]);a=torch.tensor([True,True,True,False]);rx=torch.tensor([1,3,1,99])
    full=contribution_weights(p,a,rx);small=contribution_weights(p[a],torch.ones(3,dtype=torch.bool),rx[a])
    torch.testing.assert_close(full[a],small)


def test_empty_or_balanced_batches_are_neutral():
    p=torch.tensor([0,0,1,1]);rx=torch.tensor([1,3,1,3])
    for a in (torch.zeros(4,dtype=torch.bool),torch.ones(4,dtype=torch.bool)):
        torch.testing.assert_close(contribution_weights(p,a,rx),torch.ones(4,dtype=torch.float64))


@pytest.mark.parametrize('case',['float_acceptance','bad_pseudo','unaligned_RX'])
def test_malformed_inputs_rejected(case):
    p=torch.tensor([0,1]);a=torch.ones(2,dtype=torch.bool);rx=torch.tensor([1,3])
    if case=='float_acceptance':a=a.float()
    elif case=='bad_pseudo':p[0]=-1
    else:rx=rx[:1]
    with pytest.raises(ValueError):contribution_weights(p,a,rx)


def test_population_cv_and_incomplete_source_matrix_are_explicit():
    assert cv(torch.tensor([1.,3.]))==pytest.approx(.5) and cv(torch.zeros(2)) is None
    with pytest.raises(ValueError):evaluate(torch.tensor([0,1]),torch.ones(2,dtype=torch.bool),torch.tensor([1,3]),torch.arange(2),[1,3,4,6,8])
