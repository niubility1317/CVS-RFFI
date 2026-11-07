import torch
from experiments.phase1_dura_orthogonal.signal_gate_source_probe import alpha_and_active


def test_alpha_tracks_nonzero_correction_and_ignores_zero_direction():
    anchor=torch.ones(2,2,256)
    proposal=anchor.clone()
    proposal[0]+=0.1
    output=anchor+.8*(proposal-anchor)
    alpha,active=alpha_and_active(anchor,proposal,output)
    assert active.tolist()==[True,False]
    torch.testing.assert_close(alpha[0],torch.tensor(.8),rtol=1e-5,atol=1e-5)
