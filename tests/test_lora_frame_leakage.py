import torch

from experiments.pamf_lora_main.frame_leakage import frame_factors, private_terms, slope_refinements


def test_absolute_center_slope_and_rms_reconstruct_measured_coordinate():
    time = torch.linspace(-1,1,256)
    truth = torch.exp(1j*(3*time.square()+.07*time))
    offset = torch.tensor([[.2,.4]])[...,None]
    slope = torch.tensor([[.4,-.3]])[...,None]
    observed = truth * torch.exp(1j*(offset + slope*time))
    ideal_rms = torch.tensor([[[.8],[1.2]]])
    observed_rms = torch.tensor([[[1.4],[.9]]])
    a,b,c = frame_factors((observed,observed_rms),(truth,ideal_rms))
    torch.testing.assert_close(b[...,128],torch.ones_like(b[...,128]))
    torch.testing.assert_close(a*b*c,truth*observed.conj()*ideal_rms/observed_rms)


def test_private_error_is_exact_sum_of_common_leakage_and_private_deformation():
    gen = torch.Generator().manual_seed(9)
    nominal = torch.randn(5,6,35,dtype=torch.complex64,generator=gen)
    multiplier = torch.randn(5,6,35,dtype=torch.complex64,generator=gen)
    expected,leak,shift = private_terms(nominal,multiplier)
    actual = nominal*multiplier
    private = actual-actual.mean(1,keepdim=True)
    torch.testing.assert_close(private-expected,leak+shift)
    torch.testing.assert_close(leak.mean(1),torch.zeros_like(leak[:,0]),atol=1e-6,rtol=0.)


def test_identical_chirps_only_leak_when_coordinate_distortion_differs():
    common = torch.arange(1.,31.).to(torch.complex64)[None,None,:].expand(2,6,-1)
    shared = torch.tensor([1.2+.15j, .8j])[:,None,None].expand_as(common)
    expected,leak,shape_error = private_terms(common,shared)
    for value in (expected,leak,shape_error):
        torch.testing.assert_close(value,torch.zeros_like(value),atol=1e-5,rtol=0.)
    mixed = shared.clone()
    mixed[:,1] *= .9
    _,leak,shape_error = private_terms(common,mixed)
    assert leak.abs().amax()>1.
    torch.testing.assert_close(shape_error,torch.zeros_like(shape_error))


def test_tapered_refinement_corrects_small_residual_slope_in_same_capture_basis():
    time = torch.linspace(-1,1,256)
    carrier = torch.exp(1j * (3*time.square()))
    wave = torch.exp(1j * (-.35*time + .42)).reshape(1,1,-1).expand(2,6,-1)
    initial = carrier.expand_as(wave)
    local,pooled = slope_refinements(wave,initial,carrier)
    target = initial*wave
    for result in (local,pooled):
        torch.testing.assert_close(result,target,atol=2e-4,rtol=0.)
