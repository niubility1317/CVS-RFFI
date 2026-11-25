import numpy as np
import pytest

from experiments.pamf_lora_main.w24_residual_aperture import middle_centers


def test_residual_adapter_only_places_the_six_observed_center_windows():
    observed = np.zeros((2, 6, 2, 256), dtype=np.float32)
    observed[:, :, 0] = np.arange(1, 7)[None, :, None]
    observed[:, :, 1] = np.arange(6, 0, -1)[None, :, None]
    result = middle_centers(observed)
    assert result.shape == (2, 8, 2, 432)
    assert not result[:, (0, 7)].any()
    assert not result[..., 384:].any()
    np.testing.assert_allclose(np.angle(result[:, 1:7, 0, 100] + 1j * result[:, 1:7, 1, 100]),
        np.broadcast_to(np.angle(np.arange(1, 7) + 1j * np.arange(6, 0, -1)), (2, 6)), atol=1e-7)
    with pytest.raises(ValueError):
        middle_centers(observed[..., :-1])
