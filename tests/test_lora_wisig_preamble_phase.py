import numpy as np
import pytest
import torch

from experiments.pamf_lora_main import gain_dura_screen as screen
from experiments.pamf_lora_main.preamble_phase import apply_curvature, paired_curvature, receive_curvature
from experiments.pamf_lora_main.repeated_edit import PREAMBLE


@pytest.mark.parametrize('length',(256,8192))
def test_detects_nonaffine_phase_and_leaves_affine_phase_alone(length):
    torch.manual_seed(12)
    start,count,period = PREAMBLE[length]
    stop = start+count*period
    position = torch.arange(count*period)/period
    raw = torch.randn(1,length,dtype=torch.complex64)
    raw[:,start:stop] = torch.randn(period,dtype=torch.complex64).repeat(count)
    affine, observed = raw.clone(), raw.clone()
    affine[:,start:stop] *= torch.exp(1j*(.21 + .17*position))
    observed[:,start:stop] = affine[:,start:stop]*torch.exp(1j*.065*position.square())
    torch.testing.assert_close(receive_curvature(affine)[0],torch.zeros(1),atol=1e-6,rtol=0.)
    estimate, confidence = receive_curvature(observed)
    torch.testing.assert_close(estimate,torch.full((1,),.065),atol=1e-6,rtol=0.)
    assert confidence.item()>.9
    torch.testing.assert_close(paired_curvature(observed,raw),estimate,atol=.001,rtol=0.)
    result = apply_curvature(observed,estimate)
    torch.testing.assert_close(result.abs(),observed.abs())
    assert torch.equal(result[:,:start],observed[:,:start])
    assert torch.equal(result[:,stop:],observed[:,stop:])


def test_paired_clean_reference_is_never_used_by_receiver_branches():
    torch.manual_seed(4)
    received = torch.randn(2,256,dtype=torch.complex64)
    clean = received.clone()
    other = received.clone()
    other[:,80:160] *= torch.exp(-1j*.08*(torch.arange(80)/20).square())
    calibration = dict(center=[0]*10,precision=(1e6*torch.eye(10)).tolist(),cutoff=1.)
    def model(iq, *, clean_cut):
        return iq*2
    first = screen.phase_audit_outputs([(model,calibration)],received,clean,None)
    second = screen.phase_audit_outputs([(model,calibration)],received,other,None)
    assert all(torch.equal(a,b) for a,b in zip(first[:3],second[:3],strict=True))
    assert not torch.equal(first[3],second[3])


def test_reference_is_classified_but_never_promotion_eligible():
    decisions = np.ones((2,len(screen.PHASE_AUDIT_MODES),160),dtype=np.int16)
    result = screen.score_phase_audit(decisions,np.ones(160,dtype=np.int16))
    assert 'phase_then_dura' in result['candidates']
    assert 'reference_phase' not in result['candidates']
    reference = result['isolated_phase_diagnostics']['reference_phase']
    assert 'promote' not in reference
    assert reference['correct']['clean']['reference_phase'] == 160
