import pytest
import torch
import json
import numpy as np

from experiments.pamf_dura_prior_refine.model import objective, learning_rate


def test_prior_auxiliary_has_exact_declared_weight_and_no_teacher_gradient():
    torch.manual_seed(11)
    clean = torch.randn(2, 2, 256)
    outputs = [clean.clone().requires_grad_(), clean.clone().requires_grad_()]
    priors = [(clean+.2).requires_grad_(), (clean-.1).requires_grad_()]
    teacher = [clean.clone().requires_grad_(), clean.clone().requires_grad_()]
    control, control_terms = objective(outputs, priors, teacher, clean, 20, 'continuation')
    anchored, anchor_terms = objective(outputs, priors, teacher, clean, 20, 'prior_anchor')
    torch.testing.assert_close(anchored-control, .1*anchor_terms['prior_nmse'])
    anchored.mean().backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in priors)
    assert all(value.grad is None for value in teacher)
    assert torch.equal(control, torch.zeros_like(control))


def test_budget_endpoints_and_reject_undeclared_arm():
    assert learning_rate(246) == pytest.approx(2e-4)
    assert learning_rate(4939) == pytest.approx(2e-5)
    with pytest.raises(ValueError):
        learning_rate(4940)
    clean = torch.zeros(1, 2, 256)
    with pytest.raises(ValueError):
        objective([clean]*2, [clean]*2, [clean]*2, clean, 1, 'oracle')


def test_truthlast_rejects_incomplete_last_view_and_wrong_ids(tmp_path):
    from experiments.pamf_dura_prior_refine.evaluation import complete_arrays
    from experiments.pamf_dura_prior_refine.runner import VIEWS, F_RUN
    ids=np.array(['one','two'])
    complete=dict(status='PREDICTIONS_COMPLETE',count=2,views=list(VIEWS),role='target',
        arm='prior_anchor',F_run=F_RUN,F_seed=2026092701,original_default_head=True,
        TX_truth_reads=0,state_updates=0,head_fit_updates=0,code_commit='a'*40)
    (tmp_path/'complete.json').write_text(json.dumps(complete),encoding='utf-8')
    np.save(tmp_path/'physical_ids.npy',ids)
    for view in VIEWS:
        np.save(tmp_path/(view+'.iq.npy'),np.zeros((2,2,256),np.float32))
        scores=np.zeros((2,6),np.float32)
        if view==VIEWS[-1]:scores[-1,-1]=np.nan
        np.save(tmp_path/(view+'.logits.npy'),scores)
    with pytest.raises(ValueError,match='truth remains closed'):
        complete_arrays(tmp_path,ids,arm='prior_anchor',role='target',commit='a'*40)
    with pytest.raises(ValueError,match='physical IDs differ'):
        complete_arrays(tmp_path,ids[::-1],arm='prior_anchor',role='target',commit='a'*40)
