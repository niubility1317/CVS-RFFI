import numpy as np
import pytest
import torch
from experiments.pamf_action_restore.operators import Action
from experiments.pamf_dura_efficiency.known_action_reference import (
    real_operator,noise_covariance,gaussian_reference,fit_prior)


def test_dense_operator_matches_native_reflect_and_widely_linear_image():
    rng=np.random.default_rng(11)
    kernel=rng.normal(size=(2,64))*.02;coefficients=rng.normal(size=(4,256))*.1;coefficients[0]+=1
    x=rng.normal(size=(2,256))
    A=real_operator(kernel,coefficients)
    complex_tensor=lambda v:torch.tensor(v[0]+1j*v[1],dtype=torch.complex128)[None]
    native=Action(complex_tensor(kernel),complex_tensor(coefficients[:2]),complex_tensor(coefficients[2:]))
    z=native(complex_tensor(x))[0]
    expected=torch.stack((z.real,z.imag)).numpy()
    np.testing.assert_allclose((A@x.reshape(512)).reshape(2,256),expected,atol=2e-15,rtol=2e-13)


def test_delay_example_has_unobserved_tail_without_claiming_actual_channel_nullity():
    kernel=np.zeros((2,64));kernel[0,24]=1
    coefficients=np.zeros((4,256));coefficients[0]=1
    A=real_operator(kernel,coefficients)
    delta=np.zeros((2,256));delta[:,232:]=np.arange(24)
    np.testing.assert_array_equal(A@delta.reshape(512),np.zeros(512))


def test_noise_covariance_matches_complex_simulation():
    rng=np.random.default_rng(27)
    coefficients=rng.normal(size=(4,256))*.2;coefficients[0]+=1
    N=noise_covariance(coefficients);idx=17
    a=coefficients[0,idx]+1j*coefficients[1,idx];b=coefficients[2,idx]+1j*coefficients[3,idx]
    w=np.sqrt(.0005)*(rng.normal(size=400000)+1j*rng.normal(size=400000))
    z=a*w+b*w.conj();actual=np.cov(np.stack((z.real,z.imag)),bias=True)
    np.testing.assert_allclose(N[np.ix_([idx,idx+256],[idx,idx+256])],actual,rtol=.01,atol=1e-6)


def test_gaussian_estimate_has_expected_known_noise_shrinkage():
    rng=np.random.default_rng(31);y=rng.normal(size=512);mean=rng.normal(size=512)
    actual=gaussian_reference(y,np.eye(512),.25*np.eye(512),mean,2*np.eye(512))
    np.testing.assert_allclose(actual,mean+2/2.25*(y-mean),atol=1e-14)
    with pytest.raises(ValueError):real_operator(np.zeros((2,63)),np.ones((4,256)))


def test_source_prior_excludes_probe():
    rng=np.random.default_rng(4);clean=rng.normal(size=(13,2,256))
    first=fit_prior(clean,np.arange(10))
    clean[10:]=1e6
    second=fit_prior(clean,np.arange(10))
    for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
