import torch
import numpy as np
import pytest

from experiments.pamf_phase1_flow import experiment as core
from experiments.phase1_dura_orthogonal.masked_prior_constraint import masked_prior_loss, temporal_prior_loss, load_prior, SOURCE_COMMIT
from experiments.phase1_dura_orthogonal.source_masked_prior import MaskedIQPrior, masked_inputs, SPAN


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


def test_loss_and_output_gradient_match_original_mask_preparation():
    torch.manual_seed(12)
    prior = MaskedIQPrior().eval().requires_grad_(False)
    clean = torch.randn(3, 2, 256)
    output = (clean + .1*torch.randn_like(clean)).requires_grad_()
    starts = torch.tensor([1, 100, 223])
    observed, mask, scale, _ = masked_inputs(output, starts)
    clean_observed, clean_mask, clean_scale, _ = masked_inputs(clean, starts)
    with torch.no_grad():
        target = prior(clean_observed, clean_mask, clean_scale)
    prediction = prior(observed, mask, scale)
    energy = clean.square().mean((1, 2)).clamp_min(1e-8)
    reference = (((prediction-target).square()*mask[:, None]).sum((1, 2))/(2*SPAN*energy)).mean()
    reference_grad = torch.autograd.grad(reference, output)[0]
    candidate = masked_prior_loss(prior, output, clean, starts)
    candidate_grad = torch.autograd.grad(candidate, output)[0]
    assert torch.equal(reference, candidate)
    assert torch.equal(reference_grad, candidate_grad)


def test_temporal_prior_loss_preserves_teacher_and_backpropagates():
    torch.manual_seed(13)
    prior = MaskedIQPrior().eval().requires_grad_(False)
    clean = torch.randn(3, 2, 256)
    starts = torch.tensor([1, 90, 223])
    assert temporal_prior_loss(prior, clean, clean, starts) == 0
    output = (clean + .1*torch.randn_like(clean)).requires_grad_()
    loss = temporal_prior_loss(prior, output, clean, starts)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0 and output.grad.abs().sum() > 0
    assert all(p.grad is None and not p.requires_grad for p in prior.parameters())
