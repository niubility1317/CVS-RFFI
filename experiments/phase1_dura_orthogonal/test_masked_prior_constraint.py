import torch
import numpy as np
import pytest

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss, load_prior, SOURCE_COMMIT
from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior


def test_frozen_prior_loss_has_output_gradient_but_no_teacher_gradient():
    torch.manual_seed(1)
    prior = MaskedIQPrior().eval().requires_grad_(False)
    clean = torch.randn(3, 2, 256)
    starts = torch.tensor([3, 80, 220])
    same = masked_prior_loss(prior, clean, clean, starts)
    assert same == 0
    output = (clean + .1*torch.randn_like(clean)).requires_grad_()
    loss = masked_prior_loss(prior, output, clean, starts)
    loss.backward()
    assert loss > 0 and output.grad is not None and output.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in prior.parameters())


def test_prior_loader_rejects_target_ancestry_and_mismatched_ids(monkeypatch, tmp_path):
    fit = np.arange(32768)
    summary = dict(status='SOURCE_MASKED_PRIOR_COMPLETE', commit=SOURCE_COMMIT, fit_ids=32768,
                   epochs=5, TX_labels_read=0, F_weights_read=0, F_outputs_read=0, V_reads=0, target_reads=0)
    state = dict(commit=SOURCE_COMMIT, fit_ids=fit, model=MaskedIQPrior().state_dict())
    monkeypatch.setattr(core, 'read', lambda _: summary)
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: state)
    loaded = load_prior(tmp_path, fit, 'cpu')
    assert all(not p.requires_grad for p in loaded.parameters())
    summary['target_reads'] = 1
    with pytest.raises(ValueError, match='provenance'):
        load_prior(tmp_path, fit, 'cpu')
    summary['target_reads'] = 0
    with pytest.raises(ValueError, match='fit IDs'):
        load_prior(tmp_path, fit[::-1].copy(), 'cpu')
