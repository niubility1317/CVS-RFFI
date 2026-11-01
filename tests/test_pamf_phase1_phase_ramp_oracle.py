import numpy as np
import pytest

from experiments.pamf_phase1_flow.phase_ramp_source_oracle import phase_ramp_oracle


def test_oracle_recovers_grid_aligned_phase_ramp_and_complex_gain():
    rng=np.random.default_rng(7)
    clean=rng.normal(size=(3,256))+1j*rng.normal(size=(3,256))
    t=np.arange(256)
    received=clean*(0.4+0.7j)*np.exp(2j*np.pi*7*t/512)
    restored=phase_ramp_oracle(received,clean,nfft=512)
    np.testing.assert_allclose(restored,clean,rtol=1e-11,atol=1e-11)


def test_oracle_rejects_zero_energy_and_invalid_grid():
    x=np.ones((2,256),dtype=np.complex128)
    with pytest.raises(ValueError):phase_ramp_oracle(np.zeros_like(x),x)
    with pytest.raises(ValueError):phase_ramp_oracle(x,x,nfft=128)
