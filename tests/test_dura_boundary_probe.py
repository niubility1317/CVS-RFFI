import numpy as np
import pytest
from experiments.pamf_dura_efficiency.boundary_probe import tail_cases


def test_tail_repair_and_global_comparator_have_equal_energy_and_exact_support():
    rng=np.random.default_rng(101);native=rng.normal(size=(7,2,256)).astype(np.float32);clean=(native+.1*rng.normal(size=native.shape)).astype(np.float32)
    before=native.copy();values=tail_cases(native,clean)
    np.testing.assert_array_equal(values[0],native);np.testing.assert_array_equal(values[1,:,:,:232],native[:,:,:232])
    np.testing.assert_array_equal(values[1,:,:,232:],clean[:,:,232:]);np.testing.assert_array_equal(values[3],clean)
    norms=np.sum((values.astype(np.float64)-native[None])**2,(2,3))
    np.testing.assert_allclose(norms[1],norms[2],atol=1e-7,rtol=1e-6)
    errors=np.sum((values.astype(np.float64)-clean[None])**2,(2,3))
    assert np.all(errors[2]<=errors[1]+1e-7) and np.all(errors[1]<=errors[0]+1e-7)
    np.testing.assert_array_equal(native,before)


def test_zero_error_or_zero_tail_does_not_add_an_unrelated_correction():
    rng=np.random.default_rng(102);native=rng.normal(size=(3,2,256)).astype(np.float32)
    for value in tail_cases(native,native):np.testing.assert_array_equal(value,native)
    clean=native.copy();clean[:,:,:232]+=.1
    values=tail_cases(native,clean);np.testing.assert_array_equal(values[1],native);np.testing.assert_array_equal(values[2],native)
    with pytest.raises(ValueError):tail_cases(native[:,:,:128],clean[:,:,:128])
