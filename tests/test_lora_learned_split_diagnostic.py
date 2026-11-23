import torch

from experiments.pamf_lora_main import joint_spectrum_source as parent
from experiments.pamf_lora_main.dura_repair import PacketRestorer, real_iq
from experiments.pamf_lora_main.learned_split_diagnostic import MODES, component_edits, render
from experiments.pamf_lora_main.variance_transport import physical_norm


def test_equal_private_rms_and_carrier():
    rng = torch.Generator().manual_seed(19)
    raw = torch.randn((3,6,256),dtype=torch.complex64,generator=rng)
    d = torch.randn(raw.shape,dtype=raw.dtype,generator=rng)
    level = .3+torch.rand((3,6,1),generator=rng)
    edits,_ = component_edits(raw,raw+d-d.mean(-1,keepdim=True),level)
    torch.testing.assert_close(edits[0],edits[2]+edits[3],atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(physical_norm(edits[6]-edits[2],level),physical_norm(edits[7]-edits[2],level))
    assert edits.mean(-1).abs().max()<2e-6


def test_render_agrees_with_archived_parent_readout_and_clean_guard():
    iq = real_iq(torch.randn((3,6,256),dtype=torch.complex64))
    tracker = PacketRestorer('consensus').eval()
    model = parent.RECIPES['exchange'].model().eval()
    original = lambda batch:(batch,None)
    with torch.no_grad():
        output,active = render(iq,None,original,tracker,(model,),threshold=-1.)
        assert active==len(iq) and output.shape==(len(MODES),*iq.shape)
        waves,_,_ = parent.recover(iq,tracker,(model,),-1.)
        assert torch.equal(output[2:4],waves[1:3])
        bypass,active = render(iq,None,original,tracker,(model,),threshold=float('inf'))
        assert active==0 and torch.equal(bypass,iq.expand_as(bypass))
