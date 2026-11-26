import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases, evaluation_modes
from experiments.pamf_lora_main.w24_source_bias import coherent_error
from experiments.pamf_lora_main.w24_third_prior import trial


def test_signed_cross_error_responds_to_repeatable_complex_detail_errors():
    center = torch.ones((3, 8, 384), dtype=torch.complex64)
    i = torch.arange(384)
    tone = torch.polar(torch.ones_like(i, dtype=torch.float32), i * (2 * torch.pi * 9 / 384))
    shared = tone.expand_as(center) * .02
    detail = shared * torch.tensor([1, -1] * 4)[None, :, None]
    frame = torch.ones_like(center[..., :1])

    def risk(a, b):
        prediction = torch.stack((center + a, center + b, center + 100 * detail), 1)
        return coherent_error(prediction, center, frame, with_identity=False)

    expected = risk(detail, detail)
    assert expected > 0
    torch.testing.assert_close(risk(detail, -detail), -expected)
    torch.testing.assert_close(risk(detail, 1j * detail), torch.zeros_like(expected), atol=1e-7, rtol=0)
    torch.testing.assert_close(risk(shared, shared), torch.zeros_like(expected))
    torch.testing.assert_close(risk(torch.full_like(center, .02j), torch.full_like(center, .02j)),
                               torch.zeros_like(expected), atol=1e-7, rtol=0)


def test_clean_exposures_and_probe_do_not_enter_signed_error_gradient():
    torch.manual_seed(83)
    center = torch.ones((4, 8, 384), dtype=torch.complex64)
    frame = torch.ones_like(center[..., :1])
    observed = (center[:, None, ...] + .01 * torch.randn((4, 3, 8, 384), dtype=torch.complex64))
    observed.requires_grad_(True)
    coherent_error(observed, center, frame, with_identity=True).backward()
    assert observed.grad[:2, :2].abs().sum() > 0
    assert (observed.grad[2:] == 0).all()
    assert (observed.grad[:, 2] == 0).all()


def test_coherent_error_full_only_contract():
    spec = trial('coherent-error')
    assert spec.arms == ('coherent_control', 'coherent_error')
    assert set(cases('coherent-error')) == {f'{FULL}-{i}' for i in range(3)}
    assert evaluation_modes(FULL, True, third_prior='coherent-error')[-2:] == spec.source_modes[2:4]
