import numpy as np
import pytest

from experiments.pamf_lora_main.data import SOURCE, VIEWS
from experiments.pamf_lora_main.packet_physics import aligned_view, chirp_samples
from experiments.pamf_lora_main.physical_inverse import ReplayAction, convolution
from experiments.pamf_lora_main.registered_inverse import (
    inverse_full, inverse_residual, projected_operator, sampling_matrix,
)


def test_registered_sampler_matches_forward_and_real_adjoint():
    rng = np.random.default_rng(813)
    width, offset = 8064, -104
    sample = sampling_matrix(width, offset)
    observed = rng.normal(size=width)+1j*rng.normal(size=width)
    np.testing.assert_allclose(sample@observed, chirp_samples(observed, offset).ravel(), atol=2e-14)
    rows = np.arange(width)
    kernels = rng.normal(size=(width, 4))+1j*rng.normal(size=(width, 4))
    op = ReplayAction(convolution(rows,kernels,width), np.full(width,.8), np.full(width,.1j))
    operator = projected_operator(op,sample)
    x, y = rng.normal(size=operator.shape[1]), rng.normal(size=operator.shape[0])
    np.testing.assert_allclose(np.dot(operator@x,y), np.dot(x,operator.rmatvec(y)), atol=1e-9)


@pytest.mark.parametrize(('route','solve'), [('full',inverse_full),('residual',inverse_residual)])
def test_zero_learned_correction_matches_raw_inverse(route, solve):
    clock = np.arange(8192) % 1024
    rng = np.random.default_rng(19)
    packet = np.exp(2j*np.pi*clock*(clock/1024-.5)*.055)+.014*rng.normal(size=len(clock))
    scene, receiver = VIEWS[-1], SOURCE[0]
    identity = 'registered-inverse-test'
    original = aligned_view(packet,scene,receiver,identity,'source-V',processing_route=route)
    prediction, relative = solve(packet,scene,receiver,identity,original,
                                 np.zeros((6,256),dtype=np.complex64), np.full(6,.25))
    assert prediction.shape==(5,6,2,256) and np.isfinite(prediction).all()
    assert relative<2e-9
    np.testing.assert_array_equal(prediction[1],prediction[0])
    np.testing.assert_array_equal(prediction[2],prediction[0])
