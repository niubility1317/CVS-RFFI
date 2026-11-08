import torch
from experiments.phase1_dura_orthogonal.source_direct_dura_mix import mixture_metrics


def test_exact_identity_action_line_and_waveform_oracle():
    anchor=torch.zeros(2,2,256)
    direct=torch.ones_like(anchor)
    received=torch.zeros_like(anchor)
    clean=torch.ones_like(anchor)
    received[0]=.25; received[1]=2.
    clean[0]=.75; clean[1]=.5
    values=mixture_metrics(received,anchor,direct,clean,lambda z:z)
    torch.testing.assert_close(values[:,5],torch.tensor([.25,1.]))
    torch.testing.assert_close(values[:,6],torch.tensor([.75,.5]))
    assert torch.all(values[:,4] <= values[:,0])
    assert torch.all(values[:,4] <= values[:,1])
