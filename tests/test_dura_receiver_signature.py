import numpy as np
import pytest
from experiments.pamf_dura_efficiency.receiver_signature import receiver_basis,feature_cases


def sample():
    rng=np.random.default_rng(71);rx=np.repeat([1,3,4,6,8],30)
    values=rng.normal(size=(150,12))+.4*rx[:,None]*rng.normal(size=(5,12))[np.repeat(np.arange(5),30)]
    held=np.arange(0,150,30);return values,rx,held


def test_basis_uses_only_remaining_source_records_and_exact_centered_rx_span():
    values,rx,held=sample();fit=receiver_basis(values,rx,held);basis=fit['basis']
    np.testing.assert_allclose(basis.T@basis,np.eye(4),atol=1e-12)
    centered=fit['means']-fit['center']
    np.testing.assert_allclose((centered@basis)@basis.T,centered,atol=1e-12)
    changed=values.copy();changed[held]=1e10
    again=receiver_basis(changed,rx,held)
    np.testing.assert_allclose(again['basis']@again['basis'].T,basis@basis.T,atol=1e-12)
    assert fit['fit_records']==145 and fit['held_out_records']==5 and np.all(fit['counts']==29)


def test_counterfactual_splits_full_signature_correction_into_orthogonal_parts():
    values,rx,held=sample();basis=receiver_basis(values,rx,held)['basis'];rng=np.random.default_rng(72)
    native=rng.normal(size=(8,12));signature=rng.normal(size=(8,12));target=rng.normal(size=(8,12))
    before=native.copy();cases=feature_cases(native,signature,target,basis)
    np.testing.assert_allclose(cases[1],cases[2]+cases[3]-cases[0],atol=1e-6)
    np.testing.assert_allclose((cases[2]-native)@basis,0,atol=1e-6)
    removed=cases[3]-native;np.testing.assert_allclose(removed,(removed@basis)@basis.T,atol=1e-6)
    np.testing.assert_array_equal(native,before)
    equal=feature_cases(native,signature,signature,basis)
    for value in equal:np.testing.assert_array_equal(value,native.astype(np.float32))


def test_invalid_or_incomplete_source_fit_fails_without_silent_rank_fallback():
    values,rx,held=sample()
    with pytest.raises(ValueError):receiver_basis(values,rx,np.array([0,0]))
    with pytest.raises(ValueError):receiver_basis(values,rx,np.array([-1]))
    with pytest.raises(ValueError):receiver_basis(values,rx,np.flatnonzero(rx==8))
    with pytest.raises(ValueError,match='independent'):receiver_basis(np.ones_like(values),rx,held)
    invalid=values.copy();invalid[1,0]=np.nan
    with pytest.raises(FloatingPointError):receiver_basis(invalid,rx,held)
