import numpy as np
import pytest

from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.packet_physics import aligned_view
from experiments.pamf_lora_main.physical_inverse import ReplayAction, convolution, full_replay, residual_replay


def test_sparse_widely_linear_operator_transposes_in_real_coordinates():
    rng = np.random.default_rng(513)
    h = rng.normal(size=(70, 7)) + 1j*rng.normal(size=(70, 7))
    main, image = rng.normal(size=(2, 70)) + 1j*rng.normal(size=(2, 70))
    action = ReplayAction(convolution(np.arange(4, 74), h, 76), main, image).operator()
    x, y = rng.normal(size=152), rng.normal(size=140)
    np.testing.assert_allclose(np.dot(action @ x, y), np.dot(x, action.rmatvec(y)), atol=1e-10)


@pytest.mark.parametrize(('route', 'replay'), [('full', full_replay), ('residual', residual_replay)])
def test_replay_keeps_the_original_received_iq(route, replay):
    clock = np.arange(8192) % 1024
    rng = np.random.default_rng(91)
    iq = (np.exp(2j*np.pi*clock*(clock/1024 - .5)*.055) + .014*rng.normal(size=len(clock)))
    scene = VIEWS[-1]
    original = aligned_view(iq, scene, SOURCE[0], 'physical-inverse-test', 'source-V', processing_route=route)
    values, error = replay(iq, scene, SOURCE[0], 'physical-inverse-test')
    assert np.array_equal(values[0], original)
    assert values.shape == (4, 6, 2, 256) and np.isfinite(values).all()
    assert error < 2e-9
