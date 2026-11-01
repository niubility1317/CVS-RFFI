import numpy as np
import pytest

from experiments.pamf_phase1_flow.equalizer_source_oracle import scalar_oracle


def test_complex_scalar_oracle_recovers_known_gain_and_phase():
    rng=np.random.default_rng(7)
    clean=rng.normal(size=(4,256))+1j*rng.normal(size=(4,256))
    received=clean*(0.31+0.72j)
    restored=scalar_oracle(received,clean)
    np.testing.assert_allclose(restored,clean,rtol=1e-12,atol=1e-12)


def test_complex_scalar_oracle_rejects_zero_or_mismatched_input():
    x=np.ones((2,256),dtype=np.complex128)
    with pytest.raises(ValueError):scalar_oracle(np.zeros_like(x),x)
    with pytest.raises(ValueError):scalar_oracle(x,x[:1])
