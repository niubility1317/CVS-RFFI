"""Whole-model revision boundaries; no datasets or remote experiments required."""
from copy import deepcopy
from unittest.mock import patch
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_dual_cvsincnet import build_dual_model


def model(version=None, **config):
    return build_dual_model(num_classes=3, num_domains=2, model_size='M', dataset='wisig',
        input_len=64, model_variant='lite_d', branch_ablation='no_dac',
        domain_branch_ablation='no_stats', id_feature_key='feat_joint', dom_feature_key='feat_imp',
        use_ecrs=version is not None, ecrs_config={'version': version, **config} if version else {},
        fast_infer_when_no_aux=False)


@pytest.mark.parametrize('version', ['v1r', 'v2'])
def test_revision_initialization_preserves_all_baseline_parameters_and_rng(version):
    torch.manual_seed(314)
    baseline = model()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(314)
    revised = model(version)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
    state = revised.state_dict()
    for name, expected in baseline.state_dict().items():
        torch.testing.assert_close(state[name], expected, atol=0, rtol=0, msg=name)


@pytest.mark.parametrize('version', ['v1r', 'v2'])
def test_compute_only_preserves_raw_gradients_buffers_rng_and_optimizer_step(version):
    torch.manual_seed(772)
    baseline = model().train()
    torch.manual_seed(772)
    revised = model(version, compute_only=True).train()
    x, labels = torch.randn(3,2,64), torch.tensor([0,1,2])
    optimizers = [torch.optim.AdamW(m.parameters(), lr=.001) for m in (baseline, revised)]
    outputs, rngs = [], []
    for m, opt in zip((baseline, revised), optimizers):
        torch.manual_seed(19)
        out = m(x, y_tx=labels, return_aux=True)
        # Domain losses are included because side-branch RNG must not change them.
        loss = F.cross_entropy(out['tx_logits'], labels) + out['dom_logits'].square().mean()
        loss.backward()
        opt.step()
        outputs.append(out)
        rngs.append(torch.get_rng_state().clone())
    torch.testing.assert_close(outputs[0]['tx_logits'], outputs[1]['tx_logits'], atol=0, rtol=0)
    torch.testing.assert_close(outputs[0]['z_id'], outputs[1]['z_id'], atol=0, rtol=0)
    torch.testing.assert_close(rngs[0], rngs[1], atol=0, rtol=0)
    revised_parameters = dict(revised.named_parameters())
    for name, parameter in baseline.named_parameters():
        actual = revised_parameters[name]
        assert (parameter.grad is None) == (actual.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, actual.grad, atol=0, rtol=0, msg=name)
        torch.testing.assert_close(parameter, actual, atol=0, rtol=0, msg=name)
    revised_state = revised.state_dict()
    for name, expected in baseline.state_dict().items():
        torch.testing.assert_close(revised_state[name], expected, atol=0, rtol=0, msg=name)
    assert all(p.grad is None for p in revised.ecrs.parameters())


@pytest.mark.parametrize('version', ['v1', 'v1r', 'v2'])
def test_identity_only_never_executes_domain_backbone(version):
    m = model(version).eval()
    x = torch.randn(2,2,64)
    with torch.no_grad(), patch.object(m.dom_backbone, 'forward', side_effect=AssertionError('domain executed')):
        out = m.forward_identity(x)
    assert out['z_id'].shape == (2,160)
    assert 'dom_logits' not in out


def test_v2_response_only_exposes_exact_ema_input_without_either_backbone():
    m = model('v2').train()
    with patch.object(m.dom_backbone, 'forward', side_effect=AssertionError('domain executed')), \
         patch.object(m.id_backbone, 'forward', side_effect=AssertionError('identity executed')):
        out = m.forward_response(torch.randn(3,2,64))
    assert out['encoder_input'].shape == (3,48)
    assert not out['encoder_input'].requires_grad
    teacher = deepcopy(m.response_encoder()).eval()
    expected = F.normalize(teacher(out['encoder_input']), dim=-1)
    torch.testing.assert_close(out['z_resp'], expected)
    F.cross_entropy(out['resp_tx_logits'], torch.tensor([0,1,2])).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.response_encoder().parameters())
    assert all(p.grad is None for p in m.id_backbone.parameters())


def test_v2_fusion_zero_projection_has_first_step_gradient_and_raw_isolated():
    m = model('v2', fusion_mode='fixed', fixed_rho=.05).train()
    m.initialize_ecrs_fusion_head()
    m.ecrs.set_active_fusion(.05)
    before = m.ecrs.response_projection.weight.detach().clone()
    optimizer = torch.optim.SGD(m.parameters(), lr=.01)
    out = m(torch.randn(3,2,64), return_aux=True)
    torch.testing.assert_close(out['z_id_fused'], F.normalize(out['z_id_raw'], dim=-1))
    F.cross_entropy(out['tx_logits_fused'], torch.tensor([0,1,2])).backward()
    assert m.ecrs.response_projection.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in m.id_backbone.parameters())
    assert all(p.grad is None for p in m.response_encoder().parameters())
    optimizer.step()
    assert not torch.equal(before, m.ecrs.response_projection.weight)


def test_v2_checkpoint_extra_state_and_fused_identity_roundtrip(tmp_path):
    m = model('v2', fusion_mode='fixed').eval()
    m.initialize_ecrs_fusion_head()
    m.ecrs.set_active_fusion(.025)
    with torch.no_grad():
        m.ecrs.response_projection.weight.normal_(std=.02)
    x = torch.randn(2,2,64)
    expected = m.forward_identity(x)
    state = m.state_dict()
    assert state['ecrs._extra_state']['fusion_mode'] == 'fixed'
    path = tmp_path/'model.pth'
    torch.save(state, path)
    restored = model('v2', fusion_mode='off').eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    actual = restored.forward_identity(x)
    torch.testing.assert_close(actual['tx_logits'], expected['tx_logits'], atol=0, rtol=0)
    torch.testing.assert_close(actual['z_id'], expected['z_id'], atol=0, rtol=0)
    assert float(restored.ecrs.active_rho) == pytest.approx(.025)
