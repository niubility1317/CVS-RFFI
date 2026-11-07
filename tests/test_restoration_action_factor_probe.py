import torch

from experiments.pamf_phase3_restoration.covariance_inverse import dense_action
from experiments.pamf_phase3_restoration.source_action_factor_probe import aligned_factors


def apply_gauge(factors, q):
    h,main,image=factors
    return h*q,main/q,image/q.conj()


def test_gauge_alignment_preserves_original_actions_and_hybrid_operators():
    estimated = (torch.tensor([[1.+.2j,.3+.5j]],dtype=torch.complex128),
        torch.tensor([[1.+.1j, .8+.4j, 1.1+.3j]],dtype=torch.complex128),
        torch.tensor([[.05+.02j,.02-.03j,-.01+.05j]],dtype=torch.complex128))
    true = (torch.tensor([[.7-.4j,.3+.2j]],dtype=torch.complex128),
        torch.tensor([[1.1-.2j, .9+.3j, .7-.3j]],dtype=torch.complex128),
        torch.tensor([[.03-.05j,.04+.04j,-.02+.01j]],dtype=torch.complex128))
    a,b,correlation=aligned_factors(estimated,true)
    torch.testing.assert_close(dense_action(*a),dense_action(*estimated))
    torch.testing.assert_close(dense_action(*b),dense_action(*true))
    assert correlation.min()>0
    inner=(a[0].conj()*b[0]).sum(1)
    torch.testing.assert_close(inner.imag,torch.zeros_like(inner.imag),atol=1e-14,rtol=0)
    assert inner.real.min()>0
    c,d,_=aligned_factors(apply_gauge(estimated,torch.tensor(1.8-.3j,dtype=torch.complex128)),
                          apply_gauge(true,torch.tensor(-.2+1.4j,dtype=torch.complex128)))
    torch.testing.assert_close(dense_action(b[0],*a[1:]),dense_action(d[0],*c[1:]))
    torch.testing.assert_close(dense_action(a[0],*b[1:]),dense_action(c[0],*d[1:]))
