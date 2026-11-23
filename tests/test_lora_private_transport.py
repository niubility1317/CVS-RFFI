import torch

from experiments.pamf_lora_main.private_transport import (
    contrast_basis, modal_transport, replace_private,
)


def test_helmert_basis_is_an_orthonormal_zero_sum_contrast():
    basis=contrast_basis(6,dtype=torch.float64,device='cpu')
    torch.testing.assert_close(basis.T@basis,torch.eye(5,dtype=basis.dtype))
    torch.testing.assert_close(basis.sum(0),torch.zeros(5,dtype=basis.dtype),atol=1e-8,rtol=0)


def test_transport_matches_private_covariance_and_preserves_eigenvector_orientation():
    q=contrast_basis(6,dtype=torch.float64,device='cpu')
    noise=torch.tensor([.2,.4,.15,.7,.5,1.],dtype=q.dtype)
    var=torch.tensor(.031,dtype=q.dtype)
    covariance=q.T@torch.diag(noise)@q
    factor=torch.linalg.cholesky(covariance+var*torch.eye(5,dtype=q.dtype))
    residual=(q@factor).T[:,None,:].expand(-1,1,-1)
    eigenvalues,directions=torch.linalg.eigh(covariance)
    variance=(var/(var+eigenvalues))[None,None,:].expand(len(residual),-1,-1)
    axes=(q@directions)[None,None,:,:].expand(len(residual),-1,-1,-1)
    mmse,match=modal_transport(residual,axes,variance)
    torch.testing.assert_close(match[:,0].T@match[:,0],var*(q@q.T))
    cross=(match[:,0]@(q@directions)) * (mmse[:,0]@(q@directions))
    assert cross.min() >= 0
    assert match.square().mean() > mmse.square().mean()


def test_zero_noise_gives_input_and_replacing_private_leaves_common_fixed():
    g=torch.Generator().manual_seed(12)
    basis=contrast_basis(6,dtype=torch.float64,device='cpu')
    residual=torch.randn(4,8,6,generator=g,dtype=torch.float64)
    residual=residual-residual.mean(-1,keepdim=True)
    axes=basis[None,None,:,:].expand(4,8,-1,-1)
    full=torch.ones(4,8,5,dtype=basis.dtype)
    one,two=modal_transport(residual,axes,full)
    torch.testing.assert_close(one,residual)
    torch.testing.assert_close(two,residual)
    original=torch.randn(4,6,12,generator=g,dtype=torch.complex128)
    replacement=torch.randn(4,6,4,generator=g,dtype=torch.complex128)
    index=torch.tensor([1,2,10,11])
    result=replace_private(original,index,replacement)
    torch.testing.assert_close(result.mean(1),original.mean(1))
    torch.testing.assert_close(result[...,0],original[...,0])
    torch.testing.assert_close(result[...,3:10],original[...,3:10])
