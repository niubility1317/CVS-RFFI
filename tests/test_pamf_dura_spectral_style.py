import pytest
import torch
from experiments.pamf_dura_efficiency.spectral_style import bank_from_psd,spectral_reference,paired_spectral
from experiments.pamf_restoration.models import complex_iq,pair


def test_source_bank_is_positive_reciprocal_and_has_identity_diagonal():
    p=torch.arange(1,49,dtype=torch.float64).reshape(3,16)
    bank,meta=bank_from_psd(p)
    assert meta['gain_min']>0 and meta['condition_number_max']>=1
    torch.testing.assert_close(bank[0,0],torch.ones(16,dtype=torch.float64))
    torch.testing.assert_close(bank[0,1]*bank[1,0],torch.ones(16,dtype=torch.float64))
    torch.testing.assert_close(bank[0,1]*bank[1,2],bank[0,2])
    for invalid in (torch.zeros_like(p),p*float('nan'),-p):
        with pytest.raises(ValueError):bank_from_psd(invalid)


def test_spectral_change_preserves_RMS_phase_and_is_invertible():
    generator=torch.Generator().manual_seed(21)
    x=torch.randn(2,2,16,generator=generator,dtype=torch.float64)
    gain=torch.linspace(.5,2,16,dtype=torch.float64).expand(2,-1)
    reference=spectral_reference(x,gain)
    torch.testing.assert_close(reference.square().mean((1,2)),x.square().mean((1,2)))
    first=torch.fft.fft(complex_iq(x),norm='ortho');second=torch.fft.fft(complex_iq(reference),norm='ortho')
    torch.testing.assert_close(first/first.abs(),second/second.abs(),atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(spectral_reference(reference,1/gain),x,atol=1e-12,rtol=1e-12)


def test_same_noncommuting_action_and_noise_are_preserved():
    generator=torch.Generator().manual_seed(22)
    x=torch.randn(2,2,16,generator=generator,dtype=torch.float64)
    noise=.05*torch.randn(x.shape,generator=generator,dtype=x.dtype)
    phase=torch.exp(1j*torch.linspace(0,3,16,dtype=x.dtype))
    def action(z):return phase*z+.2*z.conj()
    y=pair(action(complex_iq(x)))+noise
    gain=torch.linspace(.5,2,16,dtype=x.dtype).expand(2,-1)
    observed,reference=paired_spectral(y,x,action,gain)
    torch.testing.assert_close(observed-pair(action(complex_iq(reference))),noise,atol=1e-15,rtol=1e-12)
