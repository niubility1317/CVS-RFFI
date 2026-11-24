import torch

from experiments.pamf_lora_main import gated_slope, gated_slope_source
from experiments.pamf_lora_main.dura_repair import PacketRestorer, complex_iq, real_iq
from experiments.pamf_lora_main.joint_unroll_source import coordinates


def packets(tracker):
    time = torch.linspace(-1,1,256)
    carrier = tracker.reference.chirp
    base = carrier * torch.polar(torch.ones_like(time),.21*time)
    iq = base[None,None,:].expand(2,6,-1).clone()
    noise = torch.randn(iq.shape,generator=torch.Generator().manual_seed(12),dtype=iq.dtype)
    iq[1] += .8 * noise[1]
    return real_iq(iq)


def test_guarded_training_frame_changes_only_active_rows_and_round_trips_iq(monkeypatch):
    tracker = PacketRestorer('consensus')
    iq = packets(tracker)
    raw = coordinates(tracker,iq)
    proxy = torch.tensor([gated_slope.THRESHOLD,.15])
    monkeypatch.setattr(gated_slope,'corruption_score',lambda _wave: proxy)
    guarded = gated_slope.guarded_coordinates(tracker,iq)
    for before,after in zip(raw,guarded,strict=True):
        assert torch.equal(before[0],after[0])
    assert torch.equal(raw[3],guarded[3])
    assert not torch.equal(raw[1][1],guarded[1][1])
    wave,rotation,quality,rms = guarded
    assert (quality>=0).all() and (quality<=1).all()
    torch.testing.assert_close(wave*rotation*rms,complex_iq(iq),rtol=1e-5,atol=1e-6)


def test_frozen_source_modes_keep_bypassed_iq_and_dose(monkeypatch):
    tracker = PacketRestorer('consensus')
    iq = packets(tracker)
    monkeypatch.setattr(gated_slope_source,'corruption_score',lambda _wave: torch.tensor([.0,.12]))

    def model(y, _q, _rotation):
        change = .05j*(y-y.mean(-1,keepdim=True))
        return y,y+change

    output,n = gated_slope_source.render(iq,None,lambda x:(x,None),tracker,(model,model))
    modes = gated_slope_source.MODES
    assert n==1
    assert torch.equal(output[:,0],iq[0].expand_as(output[:,0]))
    for name in ('matched','pooled'):
        half = output[modes.index(name+'_half'),1]-iq[1]
        full = output[modes.index(name+'_full'),1]-iq[1]
        torch.testing.assert_close(2*half,full,atol=5e-7,rtol=1e-5)
