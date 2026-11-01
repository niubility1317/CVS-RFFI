import math
import pytest
import torch

from experiments.pamf_phase1_flow.phase_orbit import phase_candidates, select_phase


def test_phase_candidates_preserve_energy_and_zero_is_identity():
    torch.manual_seed(5)
    x=torch.randn(4,2,256)
    candidate=phase_candidates(x,[0.,math.pi/4,-math.pi/2])
    assert torch.equal(candidate[:,0],x)
    torch.testing.assert_close(candidate.square().sum((2,3)),x.square().sum((1,2))[:,None].expand(-1,3),rtol=1e-6,atol=1e-4)


def test_soft_consensus_returns_actual_phase_or_baseline_when_no_candidate_agrees():
    logits=torch.tensor([[[3.,0.,0.],[0.,3.,0.],[0.,4.,0.]],[[2.,0.,1.8],[0.,2.,1.8],[0.,2.,1.8]]])
    chosen=select_phase(logits,rule='soft_mean')
    assert chosen.tolist()==[1,0]
    no_winner=torch.tensor([[[2.,0.,1.8],[0.,2.,1.8]]])
    assert select_phase(no_winner,rule='soft_mean').tolist()==[0]


def test_majority_ties_keep_baseline_and_max_confidence_is_actual_candidate():
    logits=torch.tensor([[[0.,3.,0.],[0.,4.,0.],[0.,0.,5.],[0.,0.,2.]]])
    assert select_phase(logits,rule='majority').tolist()==[0]
    assert select_phase(logits,rule='max_confidence').tolist()==[2]


def test_all_selection_ties_preserve_anchor():
    confidence_tie=torch.tensor([[[0.,0.],[2.,0.],[0.,2.]]])
    assert select_phase(confidence_tie,rule='max_confidence').tolist()==[0]
    mean_tie=torch.tensor([[[0.,2.],[2.,0.]]])
    assert select_phase(mean_tie,rule='soft_mean').tolist()==[0]


def test_phase_order_must_increase_in_absolute_value():
    with pytest.raises(ValueError):
        phase_candidates(torch.ones(1,2,256),[0.,math.pi/2,math.pi/4])
