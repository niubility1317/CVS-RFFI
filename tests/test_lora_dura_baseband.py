import torch

from experiments.pamf_lora_main.dura_baseband_source import ARM, RECIPES
from experiments.pamf_lora_main.dura_identity_action import IdentityActionDura
from experiments.pamf_lora_main.dura_stage_noise_audit import direction_parts


def test_fit_and_inference_use_the_same_unit_frame_and_original_state():
    torch.manual_seed(227)
    native = IdentityActionDura().eval()
    candidate = RECIPES[ARM].model().eval()
    candidate.load_state_dict(native.state_dict(), strict=True)
    # Read out both native frame channels even at the cold-start zero output.
    with torch.no_grad():
        for prior in native.restorer.prior:
            prior.out.weight.normal_(std=.05)
        candidate.load_state_dict(native.state_dict(), strict=True)
        observed = torch.randn(1,6,256,dtype=torch.complex64)
        phi = torch.linspace(0,11,256).expand_as(observed.real)
        rotation = torch.polar(torch.ones_like(phi), phi)
        expected = native(observed, None, torch.ones_like(rotation))
        actual = candidate(observed, None, rotation)
        fit_route = RECIPES[ARM].restore(candidate, observed, None, rotation)
        for wave, fit, reference in zip(actual, fit_route, expected, strict=True):
            torch.testing.assert_close(wave, reference, rtol=0., atol=0.)
            torch.testing.assert_close(fit, reference, rtol=0., atol=0.)
            torch.testing.assert_close(wave.abs().square().mean(), observed.abs().square().mean())



def test_noise_direction_separates_aligned_and_orthogonal_energy():
    noise = torch.tensor([[[1.+0.j, 0.+0.j]]])
    edit = torch.tensor([[[-.25+0.j, 0.+.5j]]])
    removed, orthogonal, remaining = direction_parts(edit, noise)
    torch.testing.assert_close(removed, torch.tensor([.25]))
    torch.testing.assert_close(orthogonal, torch.tensor([.25]))
    torch.testing.assert_close(remaining, torch.tensor([.8125]))
