from types import SimpleNamespace

import numpy as np
import torch

from experiments.pamf_lora_main.noise_replicas import DRAWS,cumulative_observations,receiver_noise
from experiments.pamf_lora_main.noise_replicas_source import components


def test_receiver_noise_matches_complex_and_complementary_covariance():
    rng=np.random.default_rng(86)
    count=80000
    a=.74+.33j
    b=.16-.06j
    var=.8
    w=receiver_noise(np.full(count,a),np.full(count,b),var,rng)
    np.testing.assert_allclose((np.abs(w)**2).mean(),var*(abs(a)**2+abs(b)**2),rtol=.02)
    np.testing.assert_allclose((w**2).mean(),2*a*b*var,rtol=.06)


def test_fresh_same_action_average_retains_signal_and_runs_down_as_inverse_k():
    rng=np.random.default_rng(576)
    n=50000
    op=SimpleNamespace(main=np.ones(n),image=np.full(n,.13j))
    signal=np.full(n,1.2-.9j)
    variance=.3
    noisy=signal+receiver_noise(op.main,op.image,variance,rng)
    averaged=cumulative_observations(noisy,signal,op,variance,rng)
    initial=np.mean(np.abs(noisy-signal)**2)
    for count,wave in zip(DRAWS,averaged,strict=True):
        assert abs(wave.mean()-signal[0])<.015
        np.testing.assert_allclose(np.mean(np.abs(wave-signal)**2)/initial,1/count,rtol=.035)


def test_split_common_and_private_recombines_in_nonuniform_iq_coordinates():
    gen=torch.Generator().manual_seed(32)
    edit=torch.randn(3,6,2,256,generator=gen)
    phase=torch.randn(3,6,256,generator=gen)
    gain=.4+torch.rand(3,6,1,generator=gen)
    basis=torch.polar(gain.expand_as(phase),phase)
    common,private=components(edit,basis)
    torch.testing.assert_close(common+private,edit,atol=1e-6,rtol=1e-5)
    def dechirp(x): return torch.complex(x[:,:,0],x[:,:,1])/basis
    c=dechirp(common)
    p=dechirp(private)
    torch.testing.assert_close(c,c[:,:1].expand_as(c),atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(p.mean(1),torch.zeros_like(p[:,0]),atol=1e-6,rtol=0)
