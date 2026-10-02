import pytest
import torch
from torch.nn import functional as F
from experiments.cvs_adaptive_volterra_identity.model import build as control
from experiments.cvs_neural_residual_identity.model import build,VARIANTS,neural_contract,ComplexDepthwise

torch.set_num_threads(2)

@pytest.mark.parametrize('causal',[False,True])
def test_complex_depthwise_pairs_channels_and_matches_four_real_convolutions(causal):
    torch.manual_seed(13);m=ComplexDepthwise(3,causal);z=torch.randn(2,2,3,19)
    r,i=z.unbind(1);pad=(4,0) if causal else (2,2)
    conv=lambda x,w:F.conv1d(F.pad(x,pad),w,groups=3)
    expected=torch.stack([conv(r,m.weight_real)-conv(i,m.weight_imag),conv(r,m.weight_imag)+conv(i,m.weight_real)],1)
    torch.testing.assert_close(m(z),expected,atol=1e-6,rtol=1e-5)
    if causal:
        changed=z.clone();changed[...,10:]+=3
        torch.testing.assert_close(m(z)[...,:10],m(changed)[...,:10],rtol=0,atol=0)

@pytest.mark.parametrize('variant,parameters',zip(VARIANTS,[220987,239419]))
def test_initial_function_exact_rng_preserved_and_real_ce_trains_hidden_weights(variant,parameters):
    torch.manual_seed(17);base=control('adaptive_volterra_lag4');expected_rng=torch.get_rng_state()
    torch.manual_seed(17);m=build(variant)
    assert torch.equal(torch.get_rng_state(),expected_rng)
    assert m.contract()==neural_contract(variant)
    assert sum(p.numel() for p in m.parameters())==parameters
    x=torch.randn(4,2,256);y=torch.tensor([0,1,2,3]);m.eval();base.eval()
    torch.testing.assert_close(m(x),base(x),rtol=0,atol=0)
    optimizer=torch.optim.AdamW(m.parameters(),lr=.0002,weight_decay=.0001)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);loss=F.cross_entropy(m(x),y);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        if step==0:
            assert all(torch.count_nonzero(b.expand.weight_real.grad)==0 for _,b in m.residual_blocks())
            assert all(b.project.weight_real.grad.norm()>0 for _,b in m.residual_blocks())
        else:
            assert all(b.expand.weight_real.grad.norm()>0 and b.temporal.weight_real.grad.norm()>0 for _,b in m.residual_blocks())
        optimizer.step()
    # Per-sample path: no inference state fitted across a batch.
    m.eval();z=m(x)
    torch.testing.assert_close(z,torch.cat([m(t[None]) for t in x]),atol=2e-4,rtol=2e-4)
    theta=.43;r,i=x.unbind(1)
    rotated=torch.stack([r*torch.cos(torch.tensor(theta))-i*torch.sin(torch.tensor(theta)),r*torch.sin(torch.tensor(theta))+i*torch.cos(torch.tensor(theta))],1)
    torch.testing.assert_close(z,m(rotated),atol=2e-3,rtol=2e-3)
    diag=m.diagnostics(x)
    assert len(diag['neural_residual']['records'])==(2 if variant==VARIANTS[0] else 4)
    assert all(r['relative_output_change_mean']>0 for r in diag['neural_residual']['records'])
