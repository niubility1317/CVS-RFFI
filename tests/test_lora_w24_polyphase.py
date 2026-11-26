import torch

from experiments.pamf_lora_main.w24_formal import FULL, cases
from experiments.pamf_lora_main.w24_polyphase import NativePolyphaseUNet, orthogonal_taps
from experiments.pamf_lora_main.w24_third_prior import fit_inputs, trial


def test_shifted_taps_have_zero_real_overlap_and_equal_white_noise_gain():
    x = torch.arange(54, dtype=torch.float64)[None, :]
    taps = torch.stack([(torch.sinc(.384 * (x - phase)) * .384).squeeze(0) for phase in (25., 25.4)])
    basis = orthogonal_taps(taps)
    assert basis.shape == (2, 54, 4)
    torch.testing.assert_close((basis * taps[..., None]).sum(-2), torch.zeros(2, 4, dtype=taps.dtype),
                               atol=1e-14, rtol=0)
    torch.testing.assert_close(basis.square().sum(-2), taps.square().sum(-1, keepdim=True).expand(-1, 4))


def test_native_candidate_and_aperture_control_have_equal_parameters_and_preserve_center_dc():
    from types import SimpleNamespace

    torch.manual_seed(22)
    idx = torch.arange(384)[:, None] + torch.arange(54)[None, :]
    idx = idx[None, ...] + 1024 * torch.arange(8)[:, None, None]
    taps = (.384 * torch.sinc(.384 * (torch.arange(54.) - 25.))).expand(384, -1)
    tracker = SimpleNamespace(indices=idx, taps=taps)
    y = torch.randn(1, 8, 384, dtype=torch.complex64)
    phase = torch.ones_like(y)
    rms = torch.ones(1, 8, 1)
    edges = torch.randn(1, 8, 48, dtype=torch.complex64)
    wave = torch.randn(1, 8192, dtype=torch.complex64)
    control = NativePolyphaseUNet(native_context=False).eval()
    native = NativePolyphaseUNet(native_context=True).eval()
    native.load_state_dict(control.state_dict())
    with torch.no_grad():
        for model in (control, native):
            result = model.propose_native(tracker, y, phase, rms, rms, edges, wave)[0][-1]
            scaled = model.propose_native(tracker, y, phase, rms, rms, edges, wave * 2.)[0][-1]
            assert result.shape == y.shape and torch.isfinite(result).all()
            torch.testing.assert_close(result.mean(-1), y.mean(-1), rtol=0, atol=3e-7)
            torch.testing.assert_close(result, scaled)
        unchanged = control.propose_native(tracker, y, phase, rms, rms, edges, wave * 0)[0][-1]
        torch.testing.assert_close(unchanged, control.propose_native(tracker, y, phase, rms, rms, edges, wave)[0][-1])


def test_polyphase_matches_full_route_and_has_only_source_u_fit_handles():
    spec = trial('polyphase')
    assert spec.steps == 48000 and spec.routes == (FULL,)
    assert set(cases('polyphase')) == {f'{FULL}-{i}' for i in range(3)}
    rt = dict(source_root='U', run_root='runs/new', classifier_root='F', native_target_root='target')
    for arm in spec.arms:
        assert fit_inputs(rt, arm) == dict(source_root='U', run_root=f'runs/new/{arm}')
