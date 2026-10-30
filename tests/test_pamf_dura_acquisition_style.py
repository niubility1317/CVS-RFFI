import torch
from experiments.pamf_dura_efficiency.acquisition_style import Style,sample_style,linear_style,apply_style,paired_style
from experiments.pamf_restoration.models import complex_iq,pair


def test_bounded_linear_core_is_invertible_and_RMS_is_preserved():
    generator=torch.Generator().manual_seed(17)
    style=sample_style(1,generator,dtype=torch.float64)
    assert style.taps.abs().sum()<=.08+1e-12 and style.image.abs().max()<=.015+1e-12
    basis=torch.eye(16,dtype=torch.float64).reshape(16,2,8)
    expanded=Style(style.taps.expand(16,-1),style.image.expand(16))
    matrix=linear_style(basis,expanded).reshape(16,16)
    assert torch.linalg.svdvals(matrix).min()>=.8274-1e-12
    value=torch.randn(1,2,256,generator=generator,dtype=torch.float64)
    actual=apply_style(value,style)
    torch.testing.assert_close(actual.square().mean((1,2)),value.square().mean((1,2)),atol=1e-12,rtol=1e-12)


def test_action_composition_preserves_exact_noise_for_noncommuting_A():
    generator=torch.Generator().manual_seed(18)
    clean=torch.randn(2,2,256,generator=generator,dtype=torch.float64)
    style=sample_style(2,generator,dtype=torch.float64)
    phase=torch.exp(1j*torch.linspace(0,3,256,dtype=torch.float64))
    def action(value):return phase*value+.2*value.conj()
    noise=torch.randn(2,2,256,generator=generator,dtype=torch.float64)*.05
    received=pair(action(complex_iq(clean)))+noise
    observed,reference=paired_style(received,clean,action,style)
    torch.testing.assert_close(observed-pair(action(complex_iq(reference))),noise,atol=2e-15,rtol=2e-13)
    assert (apply_style(received,style)-observed).abs().max()>1e-3


def test_local_generator_does_not_change_model_or_probe_random_stream():
    torch.manual_seed(19);before=torch.get_rng_state().clone()
    local=torch.Generator().manual_seed(20);sample_style(32,local)
    assert torch.equal(before,torch.get_rng_state())
