from unittest.mock import patch

import torch

from experiments.pamf_phase3_restoration.covariance_inverse import gaussian_posterior
from experiments.pamf_phase3_restoration.structured_noise_posterior import iq_noise_blocks, posterior


def fixture():
    rng = torch.Generator().manual_seed(9)
    action = torch.randn(2, 6, 6, generator=rng, dtype=torch.float64)
    y = torch.randn(2, 6, generator=rng, dtype=torch.float64)
    z = torch.randn(6, 6, generator=rng, dtype=torch.float64)
    cov = torch.eye(6, dtype=torch.float64) + z @ z.T
    mean = torch.randn(6, generator=rng, dtype=torch.float64)
    main = torch.tensor([[1.2+.4j, .5+1j, .8-.3j], [.6-.4j, .8+.2j, .9+.1j]], dtype=torch.complex128)
    image = torch.tensor([[.25+.1j, .15+.3j, .05+.2j], [.15+.1j, .2-.2j, .05+.2j]], dtype=torch.complex128)
    return action, y, mean, cov, main, image, torch.tensor([.05,.3],dtype=torch.float64)


def test_widely_linear_noise_covariance_matches_real_transform():
    _, _, _, _, main, image, variance = fixture()
    native = variance/(main.abs().square()+image.abs().square()).mean(1)
    proper_transform = torch.stack((torch.stack(((main+image).real,-(main-image).imag),dim=-1),
                                    torch.stack(((main+image).imag,(main-image).real),dim=-1)),dim=-2)
    expected = native[:,None,None,None]*(proper_transform @ proper_transform.transpose(-1,-2))
    di, dq, cross = iq_noise_blocks(main,image,variance)
    torch.testing.assert_close(di, expected[:,:,0,0])
    torch.testing.assert_close(dq, expected[:,:,1,1])
    torch.testing.assert_close(cross, expected[:,:,1,0])
    torch.testing.assert_close((di+dq).mean(1)*.5,variance)
    assert (di*dq-cross.square()).min()>0


def test_structured_posterior_and_risk_match_precision_identity():
    a, y, mean, cov, main, image, variance = fixture()
    di,dq,cross = iq_noise_blocks(main,image,variance)
    noise = torch.cat((torch.cat((torch.diag_embed(di),torch.diag_embed(cross)),dim=-1),
                       torch.cat((torch.diag_embed(cross),torch.diag_embed(dq)),dim=-1)),dim=-2)
    noise_inv = torch.linalg.inv(noise)
    cov_inv = torch.linalg.inv(cov)
    precision = cov_inv + a.transpose(-1,-2) @ noise_inv @ a
    rhs = cov_inv @ mean + (a.transpose(-1,-2) @ noise_inv @ y[:,:,None]).squeeze(-1)
    expected = torch.linalg.solve(precision,rhs[:,:,None]).squeeze(-1)
    expected_trace = torch.linalg.inv(precision).diagonal(dim1=-2,dim2=-1).sum(-1)
    with patch.object(torch.linalg,'cholesky',wraps=torch.linalg.cholesky) as cholesky:
        output, relative_residual, trace = posterior(a,y,mean,cov,variance,blocks=(di,dq,cross),risk=True)
        assert cholesky.call_count==1
    torch.testing.assert_close(output,expected,rtol=1e-9,atol=1e-9)
    torch.testing.assert_close(trace,expected_trace,rtol=1e-8,atol=1e-10)
    assert relative_residual.max()<1e-10


def test_scalar_control_matches_existing_gaussian_posterior():
    a, y, mean, cov, _, _, variance = fixture()
    expected, _ = gaussian_posterior(a,y,mean,cov,variance)
    output, relative_residual, trace = posterior(a,y,mean,cov,variance)
    torch.testing.assert_close(output,expected,rtol=0,atol=0)
    assert relative_residual.max()<1e-10 and trace is None
