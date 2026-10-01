import numpy as np
import pytest
from experiments.cvs_rff_physics.known_excitation import template,packet_statistics,TONES,REFERENCE,validate_config


def iq(z):
    return np.stack([np.asarray(z).real,np.asarray(z).imag],axis=1)


def test_sampling_and_known_spectral_coefficients():
    n=np.arange(256)
    assert np.max(abs(template(n+20)-template(n)))<1e-13
    x=np.fft.fft(template(np.arange(20)))
    mask=np.zeros(20,dtype=bool);mask[TONES//4%20]=True
    assert np.max(abs(x[~mask]))<1e-13
    assert np.std(abs(x[mask]))<1e-13
    assert np.mean(abs(template(np.arange(20)))**2)==pytest.approx(1)


@pytest.mark.parametrize('conjugate',[False,True])
def test_gain_phase_fractional_timing_cfo_correspondence(conjugate):
    n=np.arange(256);z=template(n,3.1875)
    if conjugate:z=z.conjugate()
    z=2.3*np.exp(.73j)*z*np.exp(2j*np.pi*130_000*n/25_000_000)
    v=packet_statistics(iq(z[None]))
    assert v['w80_160_template_correlation'][0]==pytest.approx(1,abs=1e-12)
    assert v['w80_160_occupied_tone_fraction'][0]==pytest.approx(1,abs=1e-12)
    assert v['w80_160_wrapped_offset_hz'][0]==pytest.approx(130_000,abs=1e-8)


def test_conjugation_is_unidentifiable_from_time_and_complex_gain():
    n=np.arange(256)
    assert np.max(abs(template(n).conjugate()+1j*template(n-10)))<1e-13


def test_periodicity_and_tone_support_alone_do_not_prove_standard_waveform():
    rng=np.random.default_rng(41);coeff=np.exp(2j*np.pi*rng.random(12))
    n=np.arange(256)
    z=np.sum(coeff*np.exp(2j*np.pi*n[:,None]*TONES/80),axis=1)
    v=packet_statistics(iq(z[None]))
    assert v['w80_160_lag20_coherence'][0]==pytest.approx(1)
    assert v['w80_160_occupied_tone_fraction'][0]==pytest.approx(1)
    assert v['w80_160_template_correlation'][0]<.9


def test_lti_can_confuse_known_template_residual_with_hardware():
    n=np.arange(256);s=template(n)
    received=s+.55j*template(n-2)
    v=packet_statistics(iq(received[None]))
    assert v['w80_160_lag20_coherence'][0]==pytest.approx(1)
    assert v['w80_160_occupied_tone_fraction'][0]==pytest.approx(1)
    assert v['w80_160_template_correlation'][0]<.99


def test_zero_and_nonfinite_input_behavior():
    values=packet_statistics(np.zeros((2,2,256)))
    assert all(np.isfinite(v).all() for v in values.values())
    with pytest.raises(ValueError):packet_statistics(np.full((2,2,256),np.nan))
    with pytest.raises(ValueError):packet_statistics(np.zeros((2,2,128)))


def test_source_only_input_guards():
    c=dict(method='rff_known_excitation',reference=REFERENCE,split_seed=392005,roles=['L_s','V'],
        source_receivers=[1,3,4,6,8],source_days=[1,2,3],equalized=1,out_len=256,
        rms_normalize=True,augmentation=False,model=False)
    validate_config(c)
    for k,v in [('target_inputs','query'),('checkpoint','old.pt'),('reference','unverified'),('augmentation',True)]:
        with pytest.raises(ValueError):validate_config(dict(c,**{k:v}))
