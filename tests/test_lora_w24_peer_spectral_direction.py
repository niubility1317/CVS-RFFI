from pathlib import Path

import pytest
import torch

from experiments.pamf_lora_main.w24_peer_spectral_direction import PeerSpectralDirection
from experiments.pamf_lora_main import w24_peer_spectral_direction_source as recipe


def test_local_tokens_are_independent_and_peer_uses_other_chirps():
    model = PeerSpectralDirection().eval()
    gen = torch.Generator().manual_seed(47)
    tokens = torch.randn(1, 8, 16, model.width, generator=gen)
    changed = tokens.clone()
    changed[:, 1, :, 0] += .7
    assert not torch.allclose(model.peer_context(tokens)[:, 0], model.peer_context(changed)[:, 0])
    torch.testing.assert_close(model.local_coefficients(tokens)[:, 0], model.local_coefficients(changed)[:, 0])


def test_native_direction_heads_keep_capture_and_clean_zero_edit():
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        gen = torch.Generator().manual_seed(17)
        model = PeerSpectralDirection().eval()
        y = torch.randn(1, 8192, dtype=torch.complex64, generator=gen)
        zero = torch.zeros_like(y)
        with torch.no_grad():
            for path in model(y, zero):
                torch.testing.assert_close(path, zero)
            reference = .1*y
            original = model(y, reference)
            rotated = model(1j*y, 1j*reference)
        for value, turned in zip(original, rotated, strict=True):
            assert value.shape == y.shape and value.dtype == y.dtype
            assert torch.isfinite(value).all()
            torch.testing.assert_close(turned, 1j*value, atol=2e-5, rtol=2e-5)
        local, cross = model.train()(y, reference)
        (local.abs().square().mean() + cross.abs().square().mean()).backward()
        assert model.attn.in_proj_weight.grad is not None
        assert torch.isfinite(model.attn.in_proj_weight.grad).all()
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('key', ['target_root', 'classifier_root', 'V_metadata'])
def test_training_rejects_non_source_handles(key):
    project = Path('test-project')
    cfg = dict(project=str(project), run_root=str(project/'runs'/recipe.RUN),
        source_root=str(project/'releases'/f'{recipe.DATA_RUN}-release'/'data'/'source'),
        training_release_root=str(project/'releases'/f'{recipe.TRAIN_RUN}-release'),
        practical_cache=str(project/'runs'/recipe.PRACTICAL_CACHE_RUN/'source_practical'))
    recipe.fit_config(cfg)
    with pytest.raises(ValueError, match='source-only handles'):
        recipe.fit_config({**cfg, key: 'extra'})
