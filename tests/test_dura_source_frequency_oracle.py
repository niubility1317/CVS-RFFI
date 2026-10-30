import numpy as np
import pytest
from experiments.pamf_dura_efficiency.source_frequency_oracle import replace_bins


def test_complex_partition_replacement_removes_exact_partition_error():
    rng=np.random.default_rng(7);x=rng.normal(size=(3,2,256)).astype(np.float32)
    z=x+.2*rng.normal(size=x.shape).astype(np.float32)
    removed=[]
    for q in range(4):
        iq,s=replace_bins(x,z,q)
        assert np.all(s['replaced_bin_count']==64)
        assert np.all(s['iq_nmse']<=s['native_iq_nmse']+1e-7)
        removed.append(s['native_iq_nmse']-s['iq_nmse'])
    np.testing.assert_allclose(np.sum(removed,axis=0),s['native_iq_nmse'],rtol=2e-6,atol=1e-7)


def test_phase_probe_preserves_spectral_magnitude_and_reference_identity():
    rng=np.random.default_rng(9);x=rng.normal(size=(2,2,256)).astype(np.float32);z=x[:,::-1].copy()
    iq,s=replace_bins(x,z,3,'phase')
    fft=lambda a:np.fft.fft(a[:,0]+1j*a[:,1],norm='ortho')
    np.testing.assert_allclose(abs(fft(iq)),abs(fft(z)),atol=3e-6)
    assert np.all(s['iq_nmse']<=s['native_iq_nmse']+1e-7)
    for component in ('complex','phase','magnitude'):
        iq,_=replace_bins(x,x,3,component);np.testing.assert_allclose(iq,x,atol=2e-7)


def test_invalid_boundary_is_rejected():
    with pytest.raises(ValueError):replace_bins(np.zeros((2,2,255)),np.zeros((2,2,255)),0)
    with pytest.raises(ValueError):replace_bins(np.zeros((2,2,256)),np.zeros((2,2,256)),4)
