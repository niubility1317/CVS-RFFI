import torch
from torch import nn
import pytest

from experiments.pamf_lora_main import w24_joint_pamf as experiment


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 10)

    def forward(self, iq):
        return self.head(iq.mean((-1,1))),


class Tracker:
    def aperture(self, raw):
        return raw.unflatten(-1,(8,1024))[..., :432]


def test_unlabeled_iq_changes_only_restoration_losses(monkeypatch):
    torch.manual_seed(9)
    raw = torch.randn(4,8192,dtype=torch.complex64)
    labels = torch.tensor([0,6])
    g = nn.Parameter(torch.tensor(.2))
    f = Reader()
    anchor = Reader().requires_grad_(False)
    monkeypatch.setattr(experiment,'wide_delta',lambda g,tracker,iq: g * iq[..., :384])
    generic = lambda x,rng: x + .1 * torch.randn(x.shape, dtype=x.dtype, generator=rng)

    def parts(iq):
        return experiment.loss_parts(g,f,anchor,Tracker(),generic,iq,labels,torch.Generator().manual_seed(3))

    baseline = parts(raw)
    changed = raw.clone()
    changed[2:] *= 1j
    changed[2:, :4000] = -8 * changed[2:, :4000]
    replacement = parts(changed)
    torch.testing.assert_close(baseline[:3],replacement[:3],rtol=0.,atol=0.)
    assert not torch.isclose(baseline[3],replacement[3])
    gradients = torch.autograd.grad(baseline[0],(g,f.head.weight))
    assert all(x.abs().sum() > 0 for x in gradients)

    with pytest.raises(ValueError,match='equal labeled and unlabeled'):
        parts(raw[:3])


def test_only_the_frozen_pamf_candidate_can_open_the_formal_gate():
    fit = {k:k for k in experiment.FIT_FIELDS}
    fit.update(native_target_root='target', validation='V')
    out = experiment.allowed_fit_runtime(fit,'out')
    assert set(out) == set(experiment.FIT_FIELDS)
    assert out['run_root'] == 'out'
    scores = {m:dict(six_correct=1400,low_urban_correct=170,by_scene={'clean':297}) for m in experiment.MODES}
    scores['joint']['six_correct'] = 1600
    assert not experiment.qualifies(scores)
    scores[experiment.CANDIDATE]['six_correct'] = 1401
    assert experiment.qualifies(scores)
    for case in experiment.FULL_CASES:
        proof = experiment.formal_receipt(case)
        assert proof['modes'][-1] == experiment.CANDIDATE and 'joint' not in proof['modes']
        assert proof['processing_route'] == 'full_native' and not proof['target_truth_read']
