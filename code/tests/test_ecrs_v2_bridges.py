"""Attributable bridge wiring, not claims of historical solver equivalence."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_dual_cvsincnet import (ResponseBasis, FrozenLegacyECRSReference,
                                  build_dual_model, complex_to_iq, iq_to_complex)
from cvsrffi.ecrs_v2 import ECRSV2PhysicalEstimator, ECRSV2Branch


VARIANTS = ('legacy28_old_reference','legacy28_estimated_reference','compact8')


def physical(variant):
    return ECRSV2PhysicalEstimator(estimator_variant=variant, anchor_mode='real8',
        legacy_reference_provider=FrozenLegacyECRSReference() if variant==VARIANTS[0] else None,
        legacy_basis=ResponseBasis() if variant!='compact8' else None)


def model(variant):
    return build_dual_model(num_classes=3,num_domains=2,model_size='M',dataset='wisig',
        input_len=64,model_variant='lite_d',branch_ablation='no_dac',domain_branch_ablation='no_stats',
        use_ecrs=True,ecrs_config={'version':'v2','estimator_variant':variant,'anchor_mode':'real8'},
        fast_infer_when_no_aux=False)


@pytest.mark.parametrize('variant',VARIANTS)
def test_bridge_shapes_frozen_physics_and_fit_first_reference_scale(variant):
    torch.manual_seed(76)
    p = physical(variant)
    x = torch.randn(2,2,64)
    out = p(x,return_diagnostics=True)
    assert out['resp_coef'].shape == (2,8 if variant=='compact8' else 28)
    assert out['resp_anchor'].shape == (2,8,2)
    assert torch.isfinite(out['resp_anchor']).all()
    assert all(not v.requires_grad for v in p.parameters())
    old = p.cross_fit(x)[0]
    # At T64 even the largest guard7 puts the full evaluation block at t>=39.
    changed = x.clone(); changed[:,:,39:] += 250*torch.randn_like(changed[:,:,39:])
    new = p.cross_fit(changed)[0]
    for key in ('resp_coef','resp_anchor'):
        torch.testing.assert_close(old['fit'][key],new['fit'][key],atol=0,rtol=0)
    if variant!='compact8':
        torch.testing.assert_close(old['fit']['diagnostics']['basis_amplitude_scale'],
                                   new['fit']['diagnostics']['basis_amplitude_scale'],atol=0,rtol=0)
    assert not torch.equal(old['nmse_full'],new['nmse_full'])


def test_legacy_default_basis_and_reference_reuse_original_formula():
    torch.manual_seed(172)
    s = torch.randn(2,64,dtype=torch.complex64)
    basis = ResponseBasis()
    old = basis(s)
    scale = torch.quantile(s.abs(),.95,dim=1,keepdim=True).clamp_min(1e-4)
    h = torch.nn.functional.pad(s[:,:-1],(1,0))
    h2 = torch.nn.functional.pad(s[:,:-2],(2,0))
    explicit = basis(s,amplitude_scale=scale,history=h,history2=h2)
    torch.testing.assert_close(explicit,old,atol=1e-6,rtol=1e-6)
    provider = FrozenLegacyECRSReference()
    iq = complex_to_iq(s)
    nuisance = provider.nuisance_estimator(iq)
    canonical = provider.canonicalizer(iq,nuisance)
    reference,_ = provider.content_estimator(canonical)
    actual = provider(s,s)
    torch.testing.assert_close(actual['reference'],reference,atol=0,rtol=0)
    torch.testing.assert_close(actual['observation'],iq_to_complex(canonical),atol=0,rtol=0)


def test_bridge_shared_encoder_head_initialization_and_identity_bundle():
    models=[]
    for variant in VARIANTS:
        torch.manual_seed(765)
        models.append(model(variant))
    ref = models[-1]
    for m in models:
        for name,value in ref.response_encoder().state_dict().items():
            torch.testing.assert_close(m.response_encoder().state_dict()[name],value,atol=0,rtol=0)
        for name,value in ref.ecrs_response_head.state_dict().items():
            torch.testing.assert_close(m.ecrs_response_head.state_dict()[name],value,atol=0,rtol=0)
        out = m.forward_response(torch.randn(3,2,64))
        torch.nn.functional.cross_entropy(out['resp_tx_logits'],torch.tensor([0,1,2])).backward()
        assert any(v.grad is not None and v.grad.abs().sum()>0 for v in m.response_encoder().parameters())
        assert all(v.grad is None for v in m.ecrs.physical.parameters())
        bundle = m.export_ecrs_bundle()
        assert bundle['identity_backbone_state']
        assert not any('dom_backbone' in k for k in bundle['identity_backbone_state'])
        assert 'bridge_claim_boundary' in bundle['metadata']


def test_old28_probe_permutation_uses_explicit_history_and_fit_scale():
    p = physical('legacy28_estimated_reference')
    s = torch.randn(2,8,dtype=torch.complex64)
    history = torch.randn_like(s)
    scale = torch.tensor([[.6],[1.2]])
    order = torch.randperm(8)
    actual = p._basis(s,scale,history,history)
    permuted = p._basis(s[:,order],scale,history[:,order],history[:,order])
    torch.testing.assert_close(permuted,actual[:,order],atol=0,rtol=0)


def test_plan_rho_cap_quarter_is_supported():
    branch=ECRSV2Branch(fusion_mode='fixed',fixed_rho=.25)
    branch.set_active_fusion(.25)
    assert float(branch.active_rho)==.25
