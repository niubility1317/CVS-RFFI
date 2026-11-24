import numpy as np
import torch

from experiments.pamf_lora_main import carrier_factorial_source as m


def inputs():
    t = torch.linspace(-1, 1, 32)
    start = torch.polar(torch.ones_like(t), 2*t)
    end = torch.polar(torch.ones_like(t), -3*t + .1)
    shape0 = torch.complex(.04*t, .05*t)
    shape1 = torch.complex(-.08*t, .10*t)
    a = start*(1 + shape0)
    b = end*(.8 + shape1)
    return a[None,None], b[None,None], start, end, shape0, shape1


def test_independent_component_swaps_use_own_tracking_rotation():
    a,b,first,second,shape0,shape1 = inputs()
    outputs = m.factorial_readouts(a,b,(first,second,first))
    assert torch.equal(outputs[0],a) and torch.equal(outputs[-2],b)
    expected = (first*(1+shape1), first*(.8+shape0), second*(1+shape0))
    for pos,wave in zip((1,2,4),expected,strict=True):
        torch.testing.assert_close(outputs[pos],m.normalize_to(wave[None,None],a))
    for x in (*outputs[1:-2],outputs[-1]):
        torch.testing.assert_close(x.abs().square().mean(),a.abs().square().mean())


def test_received_basis_projection_is_retained_up_to_shared_gain():
    a,b,first,second,_,_ = inputs()
    readouts = m.factorial_readouts(a,b,(first,second,first))
    output = readouts[-1]
    frame = output*first.conj()
    original = a*first.conj()
    ratio = frame.mean(-1)/original.mean(-1)
    torch.testing.assert_close(ratio.imag,torch.zeros_like(ratio.imag),atol=1e-6,rtol=0)
    unbiased = b + (original - b*first.conj()).mean(-1,keepdim=True)*first
    torch.testing.assert_close(output,m.normalize_to(unbiased,a))


def test_factorial_effect_uses_all_four_contexts():
    outcomes = np.asarray([a+2*d+4*t+a*d for t,a,d in m.LEVELS])[:,None,None]
    assert m.mean_effect(outcomes,0).item()==4
    assert m.mean_effect(outcomes,1).item()==1.5
    assert m.mean_effect(outcomes,2).item()==2.5
