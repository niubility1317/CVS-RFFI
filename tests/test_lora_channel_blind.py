import torch

from experiments.pamf_lora_main.channel_blind import ChannelBlind, generic_observation
from experiments.pamf_lora_main.channel_blind_source import render
from experiments.pamf_lora_main.dura_repair import PacketRestorer


def test_clean_u_corruption_and_identity_initialization():
    torch.manual_seed(38)
    raw = torch.randn(2,6,256,dtype=torch.complex64)
    rng = torch.Generator().manual_seed(25)
    a,b = (generic_observation(raw,rng) for _ in range(2))
    energy = raw.abs().square().sum(-1)
    assert not torch.equal(a,b)
    assert ((a-raw).abs().square().sum(-1)<energy+1e-5).all()
    model = ChannelBlind()
    q = torch.ones(2,6,1)
    assert all(torch.equal(y,a) for y in model(a,q))
    with torch.no_grad():
        model.prior.out.bias.fill_(.3)
    for y in model(a,q):
        torch.testing.assert_close(y.mean(-1), a.mean(-1))


def test_candidate_render_ignores_reference_noise():
    torch.manual_seed(49)
    raw = torch.randn(2,6,2,256)
    baseline = lambda x:(x,torch.zeros(len(x),dtype=torch.bool))
    model = ChannelBlind().eval()
    tracker = PacketRestorer('consensus').eval()
    a,n = render(raw,torch.randn_like(raw),baseline,tracker,(model,),threshold=0.)
    b,k = render(raw,torch.zeros_like(raw),baseline,tracker,(model,),threshold=0.)
    assert a.shape==(5,*raw.shape) and n==k==len(raw)
    assert torch.equal(a,b)
