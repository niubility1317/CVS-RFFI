import numpy as np
from scipy.sparse.linalg import aslinearoperator

from experiments.pamf_lora_main import discrepancy_inverse
from experiments.pamf_lora_main.discrepancy_inverse import (
    discrepancy_solutions,
    solve_packet,
    weight_operator,
)


def test_weighted_operator_preserves_the_real_adjoint():
    rng = np.random.default_rng(96)
    a = rng.normal(size=(18,12))
    w = rng.uniform(.4,2,size=18)
    x,y = rng.normal(size=12),rng.normal(size=18)
    weighted=weight_operator(aslinearoperator(a),w)
    np.testing.assert_allclose(weighted.matvec(x),w*(a@x))
    np.testing.assert_allclose(weighted.matvec(x)@y,x@weighted.rmatvec(y))


def test_discrepancy_selects_nested_iterates_from_one_path():
    a=aslinearoperator(np.diag([1.,.1,1.,.1]))
    observation=np.array([2+1j,1+.8j])
    (coarse,fine),steps=discrepancy_solutions(a,observation,complex_variance=1.)
    assert steps==[1,2]
    assert abs(coarse[1])<.2
    np.testing.assert_allclose(fine,[2+1j,10+8j])


def test_no_signal_above_estimated_noise_budget_preserves_a_zero_inverse():
    a=aslinearoperator(np.eye(4))
    observation=np.array([.1+.1j,.1+.1j])
    result,steps=discrepancy_solutions(a,observation,complex_variance=1.)
    assert steps==[0,0]
    for estimate in result:
        np.testing.assert_array_equal(estimate,np.zeros(2))


def test_exact_reference_cannot_change_any_raw_inverse():
    a=aslinearoperator(np.diag([1.2,.9,.3,.7,1.2,.9,.3,.7]))
    y=np.array([1+1j, .5-.5j, 2+0j, -.5j])
    sigma=np.array([.3,.8])
    first,it1=solve_packet(a,y,y*.1,sigma)
    second,it2=solve_packet(a,y,y*.6,sigma)
    assert it1==it2
    for x,z in zip(first[:-1],second[:-1],strict=True):
        np.testing.assert_array_equal(x,z)
    assert not np.array_equal(first[-1],second[-1])


def test_weighted_budget_propagates_variance_through_clipped_weights(monkeypatch):
    budgets=[]

    def capture(_operator,observation,variance):
        budgets.append(variance)
        return (observation,observation),[1,1]

    monkeypatch.setattr(discrepancy_inverse,'discrepancy_solutions',capture)
    operator=aslinearoperator(np.eye(12))
    y=np.ones(6,dtype=np.complex128)
    solve_packet(operator,y,y,np.array([0.,0.,0.,0.,0.,1.]))
    np.testing.assert_allclose(budgets,[1/6,1/36])
