import pytest
import torch

from experiments.pamf_lora_main.packet_context_prior import PacketContextPrior, TOKENS, WIDTH
from experiments.pamf_lora_main.shared_gain_dura import SharedGainDura


def test_global_context_mixes_distant_tokens_with_equal_capacity():
    torch.manual_seed(8)
    global_prior = PacketContextPrior(global_context=True).eval()
    local_prior = PacketContextPrior(global_context=False).eval()
    local_prior.load_state_dict(global_prior.state_dict(),strict=True)
    tokens = torch.randn(1,2*TOKENS,WIDTH)
    changed = tokens.clone()
    changed[:,TOKENS+25] += torch.arange(WIDTH)/WIDTH
    with torch.no_grad():
        gl = global_prior.mix(tokens)
        after_global = global_prior.mix(changed)
        lc = local_prior.mix(tokens)
        after_local = local_prior.mix(changed)
    torch.testing.assert_close(lc[:,:8],after_local[:,:8],rtol=0.,atol=0.)
    assert (gl[:,:8]-after_global[:,:8]).square().sum() > 1e-8
    assert sum(p.numel() for p in global_prior.parameters()) == sum(p.numel() for p in local_prior.parameters())


@pytest.mark.parametrize('length, bins',[(256,20),(8192,1024)])
def test_time_and_frequency_readouts_both_connect_to_native_dura(length,bins):
    torch.manual_seed(17)
    model = SharedGainDura(retain_received=True,packet_context=True,clean_noise_floor=torch.zeros(bins))
    raw = torch.randn(2,length,dtype=torch.complex64)
    stage, *_ = model.restoration(raw)
    assert stage[-1].shape==raw.shape
    torch.testing.assert_close(stage[-1],raw,atol=3e-6,rtol=3e-6)
    target = raw + torch.randn_like(raw)*.2
    loss = (stage[-1]-target).abs().square().mean()
    parameters = [weight for prior in model.priors for weight in prior.readout_weights()]
    gradients = torch.autograd.grad(loss,parameters)
    assert all(gradient.isfinite().all() and gradient.norm()>0. for gradient in gradients)
